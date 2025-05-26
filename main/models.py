import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
import argparse
import sys
import os
from ultralytics import YOLO
from typing import Callable, Optional
from stable_baselines3.common.utils import polyak_update
from tqdm import trange
import time
import gc
import lpips
import zarr
from zarr.codecs import BloscCodec

from stable_baselines3 import SAC

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import GymStepReturn, DictReplayBufferSamples
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader, test_dataloader, val_dataloader, denormalize_batch, renormalize_batch
from environment import DataloaderEnv

from utils import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

parser = argparse.ArgumentParser(description='Command Line Args for Image Classifier')
parser.add_argument("--save_lc", action="store_false", help="Save all files")
args = parser.parse_args()

gc.collect()
torch.cuda.empty_cache()

class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"): 
        super().__init__()
        
        # TODO fix code for if weights are not pretrained
        # 166 MB VRAM
        model_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Yields 512 dim vector 
        self.encoder = nn.Sequential(*list(model_base.children())[:-1]).to(device) # remove head

        for param in self.encoder.parameters(): 
            param.requires_grad = False

        for module in self.encoder.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

    def forward(self, x): 
        x = self.encoder(x).squeeze(-1).squeeze(-1)

        return x

# Actor
class PerturbationModel(nn.Module): 
    def __init__(self, latent_dim, batch_size, low_rank=4, l_inf_norm = 0.05, l2_norm=0.1, device="cuda"): 
        super().__init__()

        self.l_inf_norm = l_inf_norm
        self.l2_norm = l2_norm
        self.device = device
        self.latent_dim = latent_dim
        self.low_rank = low_rank
        self.batch_size = batch_size

        self.downsampled_enc = nn.Sequential(
            nn.Linear(512, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(inplace=False)
        ).to(device)

        # Gaussian distribution in latent space
        # Note, mu and cov will error with Nan/Inf in backprop if float16 used
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), 
            nn.LeakyReLU(inplace=True)
        ).to(device)
        num_lower_triangular = (latent_dim * (latent_dim + 1)) // 2
        self.log_cholesky_layer = nn.Sequential(
            nn.Linear(latent_dim, num_lower_triangular), 
            nn.LeakyReLU(inplace=True)
        ).to(device)
        
        # 154 MB VRAM for fc and deconv combined
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14), 
            nn.BatchNorm1d(256 * 14 * 14),
            nn.ReLU(inplace=True)
        ).to(device)

        # TODO optimize memory with groups, fusing conv and batchnorm, etc if possible
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(8, 128), 
            nn.ReLU(inplace=True), 

            nn.Dropout2d(p=0.3),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(8, 64), 
            nn.ReLU(inplace=True), 

            nn.Dropout2d(p=0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(8, 32), 
            nn.ReLU(inplace=True)
        ).to(device)

        self.residual_out = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False).to(device), 
            nn.Tanh()
        )

        # Gating mask (same spatial resolution as output)
        self.gate = nn.Sequential(
            nn.ConvTranspose2d(256, 3, kernel_size=32, stride=16, padding=8),
            nn.Sigmoid()  # soft gate ∈ (0,1)
        ).to(device)

    # a = pi(s), for latent state s
    def forward(self, x): 
        x = self.downsampled_enc(x)
        mus = self.mu(x).float()
        lower_triang_elts = self.log_cholesky_layer(x).float()
        lower_triang = torch.zeros(lower_triang_elts.shape[0], self.latent_dim, self.latent_dim, device=self.device)

        indices = torch.tril_indices(self.latent_dim, self.latent_dim)
        lower_triang[:, indices[0], indices[1]] = lower_triang_elts

        # Exponentiate diagonal elements for stability
        diag_indices = torch.arange(self.latent_dim, device=self.device)
        lower_triang[:, diag_indices, diag_indices] = torch.nn.functional.softplus(lower_triang[:, diag_indices, diag_indices]).clamp(min=1e-3, max=1e2) + 1e-2

        cov_mtxs = torch.matmul(lower_triang, lower_triang.transpose(-1, -2)) 
        cov_mtxs += torch.eye(self.latent_dim, device=self.device, dtype=torch.float32) * 1e-2

        return x, mus, cov_mtxs
        
    def set_training_mode(self, mode: bool):
        self.train(mode)

    # TODO check this, may be bugged b/c shapes are incorrect
    def spatial_entropy(self, mask):
        # mask: [B, 1, H, W], assumed normalized to [0, 1]
        eps = 1e-8
        flat = mask.view(mask.size(0), -1)
        probs = flat / (flat.sum(dim=1, keepdim=True) + eps)
        entropy = -(probs * (probs + eps).log()).sum(dim=1)  # [B]
        return entropy
        
    def adaptive_area(self, entropy, min_ratio=0.02, max_ratio=0.20):
        norm_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-6)  # [B]
        return min_ratio + (max_ratio - min_ratio) * norm_entropy  # [B]
    
    def decode(self, x):
        x = self.fc(x)

        x = x.view(x.size(0), 256, 14, 14)
        main_deconv = self.deconv(x) # [B, 32, H, W]
        residual = self.residual_out(main_deconv) # [B, 3, H, W]

        gate_mask = self.gate(x)
        gate_mask = gate_mask.clamp(min=1e-8)  # prevent log(0)

        area_px = self.adaptive_area(self.spatial_entropy(gate_mask)) * 224 * 224 * 3
        gate_sum = gate_mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        scale = area_px.view(-1, 1, 1, 1) / gate_sum
        gate_mask = gate_mask * scale

        sparse_residual = residual * gate_mask

        return sparse_residual, gate_mask, area_px
    
    # Bound L2 norm
    def bound_l2(self, perturbation): 
        norm = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True).clamp(min=1e-6)
        factor = torch.clamp(self.l2_norm / norm, max=1.0)
        return perturbation * factor.view(-1, 1, 1, 1)
    
class Critic(nn.Module): 
    def __init__(self, latent_dim, device="cuda"):
        super(Critic, self).__init__()
        self.device = device

        # Input: concatenated state and action latent vectors
        self.critic_one = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim // 2), 
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 2, latent_dim // 8), 
            nn.BatchNorm1d(latent_dim // 8),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 8, 1)
        ).to(self.device)

        self.critic_two = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim // 2), 
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 2, latent_dim // 8), 
            nn.BatchNorm1d(latent_dim // 8),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 8, 1)
        ).to(self.device)

    def forward(self, combined): 
        return self.critic_one(combined), self.critic_two(combined)
    
    def set_training_mode(self, mode: bool):
        self.train(mode)


class CustomSACPolicy(SACPolicy):
    """
    implements both actor and critic in one model
    """
    def __init__(self, observation_space, action_space, lr_schedule, encoder, latent_dim, batch_size, target_entropy=-8.0, device="cuda", **kwargs):
        super().__init__(observation_space, action_space, lr_schedule)
        self.feature_encoder = encoder
        self.batch_size = batch_size

        self.observation_space = observation_space
        self.action_space = action_space

        self.actor = PerturbationModel(latent_dim, device=device, batch_size=batch_size)

        # Input: concatenated state and action latent vectors
        self.critic = Critic(latent_dim=latent_dim, device=self.device)
        self.critic_target = Critic(latent_dim=latent_dim, device=self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=4e-5, weight_decay=1e-5)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-5)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=6e-5)

        self.target_entropy = target_entropy

    # TODO update to work
    def forward(self, *args, **kwargs):
        """
        Forward pass for SAC policy (actor and critic).
        
        :param obs: The input observation.
        :param deterministic: Whether to use deterministic actions or sample from the action distribution.
        :return: action, log_prob, Q
        """
        obs = args[0]
        deterministic = args[1] if len(args) > 1 else False
        
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # If it's already a tensor, leave it alone
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        resnet_state = self.feature_encoder(obs_tensor)

        latent_state, mus, cov_mtxs = self.actor(resnet_state)

        # Note: old, unused
        dist = SquashedDiagGaussianDistribution(mus, cov_mtxs)

        # Determine action (deterministic or stochastic)
        if deterministic:
            action = dist.get_mean()  # For deterministic action
        else:
            action = dist.sample()  # For stochastic action

        combined = torch.cat([latent_state, action], dim=1)

        # Note: old, unused
        values = self.critic(combined)

        return self.actor.decode(action).squeeze(0), dist.log_prob(action), values

    # Note: unused, required
    def _predict(self, observation, deterministic = False):
        with torch.no_grad(): 
            observation = self.feature_encoder(observation)

            observation, mus, cov_mtxs = self.actor(observation)
            
            if deterministic: 
                actions = mus
            else: 
                dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

                actions = dist.sample()

            return actions
        
    def pred_upsampled_action(self, observation, deterministic=False): 
        with torch.no_grad(): 
            resnet_state = self.feature_encoder(observation)

            # Actor: Forward pass through the actor network (returns logits for actions)
            latent_state, mus, cov_mtxs = self.actor(resnet_state)

            if deterministic: 
                deltas = mus
            else: 
                dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

                deltas = dist.sample()

            decoded, gate_mask, _ = self.actor.decode(latent_state + deltas)
            decoded = decoded.squeeze(0)

        return deltas, decoded, gate_mask

    # Note: observation is already latent vector
    def predict_action_with_prob(self, observation, deterministic = False):
        observation, mus, cov_mtxs = self.actor(observation)
        log_prob = None
        
        if deterministic: 
            deltas = mus
        else: 
            dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

            deltas = dist.sample()
            log_prob = dist.log_prob(deltas).unsqueeze(1)

        return deltas, torch.clamp(log_prob, min=-20, max=0)

    # Note: observation is already latent vector
    def predict_action_with_prob_upsampling(self, observation, deterministic = False):
        observation, mus, cov_mtxs = self.actor(observation)
        log_prob = None
        
        if deterministic: 
            deltas = mus
        else: 
            dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

            deltas = dist.sample()
            log_prob = dist.log_prob(deltas)

        actions_upsampled, gate_mask, target_area = self.actor.decode(observation + deltas)
        actions_upsampled = actions_upsampled.squeeze(0)

        return observation, actions_upsampled, gate_mask, target_area, deltas, log_prob.unsqueeze(1)
    
    @property
    def alpha(self): 
        return self.log_alpha.exp()

class ZarrReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: torch.device,
        store_path: str = "zarr_buffer",
        dtype: np.dtype = np.float32,
        compressor: str = "zstd",
        **kwargs
    ):
        self.store = zarr.open_group(store_path, mode='w')

        self.device = device
        self.buffer_size = buffer_size

        self.obs_shape = observation_space.shape
        self.action_shape = action_space.shape

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        self.pos = 0
        self.full = False

        self.obs = self.store.create_array(
            "observations",
            shape=(buffer_size,) + self.obs_shape,
            chunks=(1,) + self.obs_shape,
            dtype=dtype,
            compressors=compressor
        )
        self.next_obs = self.store.create_array(
            "next_observations",
            shape=(buffer_size,) + self.obs_shape,
            chunks=(1,) + self.obs_shape,
            dtype=dtype,
            compressors=compressor
        )
        self.actions = self.store.create_array(
            "actions",
            shape=(buffer_size,) + self.action_shape,
            chunks=(1,) + self.action_shape,
            dtype=dtype,
            compressors=compressor
        )
        self.rewards = self.store.create_array(
            "rewards",
            shape=(buffer_size,),
            dtype=dtype, 
            compressors=compressor
        )
        self.dones = self.store.create_array(
            "dones",
            shape=(buffer_size,),
            dtype=np.bool_,
            compressors=compressor
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[GymStepReturn] = None
    ) -> None:
        # Ensure batch dimension
        if obs.ndim == len(self.obs_shape):  # single sample
            obs = np.expand_dims(obs, axis=0)
            next_obs = np.expand_dims(next_obs, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = np.expand_dims(done, axis=0)

        batch_size = obs.shape[0]

        insert_indices = np.arange(self.pos, self.pos + batch_size) % self.buffer_size

        # Insert using slicing
        self.obs[insert_indices] = obs.astype(self.obs.dtype)
        self.next_obs[insert_indices] = next_obs.astype(self.next_obs.dtype)
        self.actions[insert_indices] = action.astype(np.float32)
        self.rewards[insert_indices] = reward.astype(np.float32)
        self.dones[insert_indices] = done.astype(np.bool_)

        self.pos = (self.pos + batch_size) % self.buffer_size
        self.full = self.full or (self.pos == 0 or self.pos < batch_size)

    def sample(self, batch_size: int, env=None) -> DictReplayBufferSamples:
        store = zarr.open(self.store.store, mode="r")  # Open in read-only mode
        
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        # Load and convert to torch
        obs = torch.tensor(store["observations"][indices], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(store["next_observations"][indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(store["actions"][indices], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(store["rewards"][indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(store["dones"][indices], dtype=torch.float32, device=self.device)

        return DictReplayBufferSamples(obs, actions, next_obs, dones, rewards)
    
    def length(self): 
        return self.pos
    
class ZarrSAC(SAC):
    replay_buffer_class = None  # disable internal ReplayBuffer creation

    def __init__(self, *args, **kwargs):
        super(ZarrSAC, self).__init__(*args, **kwargs)

        # Ensure that ent_coef_tensor is initialized
        if not hasattr(self, 'ent_coef_tensor'):
            self.ent_coef_tensor = self.policy.alpha

        if not hasattr(self, 'actor'):
            self.actor = self.policy.actor  # Use the policy's actor
        if not hasattr(self, 'critic'):
            self.critic = self.policy.critic  # Use the policy's critic
        if not hasattr(self, 'target_critic'):
            self.target_critic = self.policy.critic_target  # Use the policy's target critic
        if not hasattr(self, 'ent_coef_optimizer'):
            self.ent_coef_optimizer = self.policy.alpha_optimizer

    def _setup_model(self):
        # Don't call super()._setup_model() directly, it tries to use replay_buffer_class
        super(SAC, self)._setup_model()  # skip SAC-specific buffer setup

        self.replay_buffer = ZarrReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            store_path="zarr_sac_buffer"
        )

    def save(self, path, **kwargs):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.target_critic.state_dict(),
            'alpha': self.policy.log_alpha,
            'ent_coef_tensor': self.ent_coef_tensor,
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
            'alpha_optimizer': self.policy.alpha_optimizer.state_dict(),
        }

        # Save to the path
        torch.save(save_dict, path)

    def load(self, path, env=None):
        state_dict = torch.load(path)

        # Load the parameters for actor, critic, target critic, and alpha
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.target_critic.load_state_dict(state_dict['critic_target'])
        self.policy.log_alpha = state_dict['alpha']
        self.ent_coef_tensor = state_dict['ent_coef_tensor']
        
        # Load optimizers
        self.actor.optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic.optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.policy.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])

        return self

class ResultConnector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, perturbed_imgs, yolo_results):
        # Save tensors for backward
        # Perturbed_img has a lot behind it (x + delta x), [B, 3, W, H]
        # Yolo_results is a detached tensor with the probability of the true classification, batched [B]
        ctx.save_for_backward(perturbed_imgs, yolo_results)

        return yolo_results

    @staticmethod
    def backward(ctx, grad_output):
        perturbed_imgs, _ = ctx.saved_tensors

        # # perturbed_imgs = saved[0]
        # # yolo_res = saved[1:]

        # Pass value up to perturbed image
        grad_pert_img = grad_output.view(perturbed_imgs.shape[0], 1, 1, 1).expand(perturbed_imgs.shape)

        # Placeholder because need to have this
        grad_yolo_res = grad_output

        return (grad_pert_img, grad_yolo_res)
    
def results_to_tensor(true_results, pred_results, device="cuda"): 
    result_list = []

    for true, pred in zip(true_results, pred_results): 
        result_list.append(pred.probs.data[true.probs.top1].item())

    return torch.tensor(result_list, device=device)


def apply_action_no_grad(img_classifier, obs_tensor, actions): 
    perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

    orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
    perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

    perturbed_normalized_clamp = renormalize_batch(perturbed_denormalized).cpu()

    with torch.no_grad():
        orig_results = img_classifier(orig_denormalized, verbose=False)
        perturbed_results = img_classifier(perturbed_denormalized, verbose=False)

    return perturbed_normalized_clamp, orig_results, perturbed_results

def apply_action_grad(img_classifier, obs_tensor, actions): 
    perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

    orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
    perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

    orig_results = img_classifier(orig_denormalized, verbose=False)
    perturbed_results = img_classifier(perturbed_denormalized, verbose=False)

    return orig_results, perturbed_results, orig_denormalized, perturbed_denormalized

def differentiable_topk_mask(tensor, k_percent=0.01, temperature=0.01):
    """
    Approximate top-k% using a sharp softmax to get a soft mask.
    Args:
        tensor: [B, C, H, W] input magnitude
        k_percent: fraction of values to keep high (e.g., 0.01 = top 1%)
        temperature: lower = sharper selection (default: 0.01)
    Returns:
        soft_mask: [B, C, H, W] values in [0, 1], sum approx equal to k%
    """
    B, C, H, W = tensor.shape
    flat = tensor.view(B, -1)  # [B, N]
    
    # Apply softmax with temperature to get a soft top-k-like distribution
    weights = torch.softmax(flat / temperature, dim=1)  # [B, N]

    # Scale to preserve sparsity level (optional)
    N = flat.shape[1]
    target_mass = k_percent * N
    soft_mask = weights * (target_mass * N)  # Optional: match scale
    soft_mask = torch.clamp(soft_mask, 0, 1)

    return soft_mask.view(B, C, H, W)

# x in [0,1]
def brightness(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
