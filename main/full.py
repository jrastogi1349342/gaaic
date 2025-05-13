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

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()

gc.collect()
torch.cuda.empty_cache()

# To run: python3 main/full.py

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
            nn.Linear(latent_dim, 1024 * 7 * 7), 
            nn.BatchNorm1d(1024 * 7 * 7),
            nn.ReLU(inplace=True)
        ).to(device)

        # TODO optimize memory with groups, fusing conv and batchnorm, etc if possible
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 

            nn.Dropout2d(p=0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.Tanh()
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
        
    def decode(self, x):
        x = self.fc(x)

        x = x.view(x.size(0), 1024, 7, 7)
        x = self.deconv(x) # [-1,1] b/c tanh

        return x
    
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

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-5)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=5e-4)

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
                actions = mus
            else: 
                dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

                actions = dist.sample()

        return actions, self.actor.decode(actions).squeeze(0)

    # Note: observation is already latent vector
    def predict_action_with_prob(self, observation, deterministic = False):
        observation, mus, cov_mtxs = self.actor(observation)
        log_prob = None
        
        if deterministic: 
            actions = mus
        else: 
            dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

            actions = dist.sample()
            log_prob = dist.log_prob(actions).unsqueeze(1)

        return actions, torch.clamp(log_prob, min=-20, max=0)

    # Note: observation is already latent vector
    def predict_action_with_prob_upsampling(self, observation, deterministic = False):
        observation, mus, cov_mtxs = self.actor(observation)
        log_prob = None
        
        if deterministic: 
            actions = mus
        else: 
            dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

            actions = dist.sample()
            log_prob = dist.log_prob(actions)

        actions_upsampled = self.actor.decode(actions).squeeze(0)

        return observation, actions_upsampled, actions, log_prob.unsqueeze(1)
    
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


def make_env_fn(dataset, obj_classifier, idx, latent_dim):
    def _init():
        return DataloaderEnv(dataset, obj_classifier, idx, latent_dim, batch_size=1)
    return _init

def make_vec_env(dataset, obj_classifier, num_envs, latent_dim):
    return DummyVecEnv([make_env_fn(dataset, obj_classifier, idx, latent_dim) for idx in range(num_envs)])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.5
latent_dim = 32
batch_size = 1 # must be 1, use multiple environments for parallel episodes
training_batch_size = 128
num_train_envs = 64
num_timesteps = 200
gradient_update_freq = 64
train_data = train_dataloader(batch_size=batch_size, num_workers=0)
obj_classifier = YOLO("yolo11n-cls.pt").to(device).eval()
# raw_obj_classifier = obj_classifier.model

# for param in raw_obj_classifier.parameters():
#     param.requires_grad = False

train_envs = make_vec_env(train_data, obj_classifier, num_train_envs, latent_dim)

num_test_envs = 10
test_data = test_dataloader(batch_size=batch_size, num_workers=0)
eval_envs = make_vec_env(test_data, obj_classifier, num_test_envs, latent_dim)
test_freq = 5 # every x timesteps in training
time_save = time.time()


encoder = Encoder(latent_dim=latent_dim, device=device)

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=train_envs,
    buffer_size=50_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size * num_train_envs,
        device=device
    ),
    verbose=1,
)

def apply_action_no_grad(obs_tensor, actions): 
    perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

    orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
    perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

    perturbed_normalized_clamp = renormalize_batch(perturbed_denormalized).cpu()

    with torch.no_grad():
        orig_results = obj_classifier(orig_denormalized, verbose=False)
        perturbed_results = obj_classifier(perturbed_denormalized, verbose=False)

    return perturbed_normalized_clamp, orig_results, perturbed_results

def apply_action_grad(obs_tensor, actions): 
    perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

    orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
    perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

    orig_results = obj_classifier(orig_denormalized, verbose=False)
    perturbed_results = obj_classifier(perturbed_denormalized, verbose=False)
    # perturbed_probs, _ = raw_obj_classifier(perturbed_denormalized)

    return orig_results, perturbed_results, perturbed_denormalized

def rollout(
    envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    gamma: float,
    max_steps: int = 50, 
): 
    envs.reset()

    total_rewards = np.zeros((envs.num_envs))
    step_num = 0
    curr_gamma = 1

    while True:
        obs_batch = np.stack([env.batch for env in envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            _, actions = policy.pred_upsampled_action(obs_tensor, deterministic=True)
        
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_clamp, orig_results, perturbed_results = apply_action_no_grad(obs_tensor, actions)

        for i, env in enumerate(envs.envs):
            if step_num == 0: 
                env.set_results(perturbed_normalized_clamp[i], orig_results[i], perturbed_results[i], False)
            else: 
                env.set_results(perturbed_normalized_clamp[i], orig_results[i], perturbed_results[i], dones[i])

        _, rewards, dones, _ = envs.step(actions_npy)
        total_rewards += curr_gamma * rewards
        curr_gamma *= gamma

        step_num += 1

        if step_num > max_steps or all(dones): 
            break
        
    return total_rewards.mean()

def train_model(
    train_envs: DummyVecEnv,
    test_envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    replay_buffer, 
    total_timesteps: int = 20,
    batch_size: int = 32,
    gradient_update_freq: int = 1,
    gamma: float = 0.95,
    tau: float = 0.005, 
    test_freq: float = 5, 
    visualize_lc: bool = True, 
):
    train_envs.reset() 
    avg_rewards = []
    max_rollout_found = -np.inf
    actor_losses = []
    critic_losses = []
    alpha_losses = []
    classification_losses = []
    l1_norms = []
    l2_norms = []
    smoothness_vals = []

    for i in trange(total_timesteps):
        obs_batch = np.stack([env.batch for env in train_envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            latent_actions, actions = policy.pred_upsampled_action(obs_tensor, deterministic=False)
        
        latent_actions = latent_actions.cpu().numpy()
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_clamp, orig_results, perturbed_results = apply_action_no_grad(obs_tensor, actions)

        for j, env in enumerate(train_envs.envs):
            env.set_results(perturbed_normalized_clamp[j], orig_results[j], perturbed_results[j])

        next_obs, rewards, dones, _ = train_envs.step(actions_npy)

        replay_buffer.add(obs_batch, next_obs, latent_actions, rewards, dones)

        for idx, done in enumerate(dones):
            if done.item():
                train_envs.env_method("reset", indices=idx)

        if replay_buffer.length() >= 10_000:
            for _ in range(gradient_update_freq):
                batch = replay_buffer.sample(batch_size)

                policy.critic.optimizer.zero_grad()
                policy.actor.optimizer.zero_grad()
                policy.alpha_optimizer.zero_grad()

                with torch.no_grad():
                    latent_obs = policy.feature_encoder(batch.observations)
                    latent_next_obs = policy.feature_encoder(batch.next_observations)

                # downsampled_obs = policy.actor.downsampled_enc(latent_obs)
                downsampled_next_obs = policy.actor.downsampled_enc(latent_next_obs)

                with torch.no_grad():
                    next_actions, next_log_probs = policy.predict_action_with_prob(latent_next_obs, deterministic=False)
                    q_next_a, q_next_b = policy.critic_target(torch.cat([downsampled_next_obs, next_actions], dim=1))
                    q_next = torch.min(q_next_a, q_next_b)

                    # TODO check if I need to detach alpha here, since in torch.no_grad()
                    target_q = batch.rewards.unsqueeze(1) + gamma * (1 - batch.dones.unsqueeze(1)) * (q_next - policy.alpha.detach() * next_log_probs)

                # Actor update
                downsampled_obs, new_actions_upsampled, new_actions, log_probs = policy.predict_action_with_prob_upsampling(latent_obs, deterministic=False)

                # Assume the prediction from the classifier on the original image is the true result, even if that's not true
                orig_results, perturbed_results, perturbed_denormalized = apply_action_grad(batch.observations, new_actions_upsampled)
                
                # Detached tensor with the probability of the true classification, batched
                formatted_probs = results_to_tensor(orig_results, perturbed_results)
                # Formatted_probs but connected to computation graph for perturbed_denormalized
                perturbed_outputs_grads = ResultConnector.apply(perturbed_denormalized, formatted_probs)


                # In [0, 1]
                classification_loss = perturbed_outputs_grads.mean()
                # classification_loss = perturbed_results.gather(1, batched_true_classes.view(-1, 1)).mean()

                q_new_action_a, q_new_action_b = policy.critic(torch.cat([downsampled_obs, new_actions], dim=1))
                # Note: these norms are not on [0, 1] images
                l2_norm_loss = torch.norm(new_actions_upsampled, p=2, dim=(1, 2, 3)).mean()
                smoothness_loss = torch.abs(torch.diff(new_actions_upsampled, dim=2)).mean() + \
                                  torch.abs(torch.diff(new_actions_upsampled, dim=1)).mean() 
                l1_norm_loss = torch.norm(new_actions_upsampled, p=1, dim=(1, 2, 3)).mean()

                # L2, smoothness, L1, classification weights
                # 1e-2, 1e-3, 1e-2 for Learned_main_1745897869.9799478.zip
                # 1e-1, 1e-3, 5e-4 for Learned_main_1745940359.0014925.zip
                # 1e-2, 0, 1e-4 for Learned_main_1745976187.284214.zip, with smooth top-k k=50, temp=0.2
                # 1e-2, 0, 1e-4 for Learned_main_1745978720.9735777.zip, with smooth top-k k=50000, temp=0.75
                # 1e-2, 1e-5, 1e-5 for Learned_main_1746032822.896086.zip, with smooth top-k k=50000, temp=0.9
                # 1e-2, 0, 1e-5, 100 for Learned_main_1746154653.8339.zip: too much noise but perfect classifications
                # 1e-2, 1e-5, 1e-2, 200 for Learned_main_1746198230.5772636.zip: too much noise but perfect classifications
                # 1e-2, 1e-4, 5e-2, 300 for Learned_main_1746245278.3612838.zip: invisible noise for each step but worse classifications
                # 1e-2, 1e-4, 3e-2, 500 for Learned_main_1746248347.6195297.zip: semi visible noise for each step, not much worse classifications
                # 1e-2, 1e-4, 4e-2, 1000 for Learned_main_1746288507.925695.zip: semi visible noise for each step, decent classifications
                # 1e-2, 1e-0, 1e-5, 100 for Learned_main_1747093838.8063166.zip: looks like shader, decent classifications, good numerical results, loss increases (good b/c learning, not overfitting)
                actor_loss = (policy.alpha.detach() * log_probs - torch.min(q_new_action_a, q_new_action_b)).mean() + \
                             100 * classification_loss + \
                             1e-2 * l2_norm_loss + \
                             1e-0 * smoothness_loss + \
                             1e-5 * l1_norm_loss
                actor_losses.append(actor_loss.item())
                classification_losses.append(classification_loss.item())
                l2_norms.append(l2_norm_loss.item())
                l1_norms.append(l1_norm_loss.item())
                smoothness_vals.append(smoothness_loss.item())

                # if i == 9: 
                #     display_comp_graph(classification_loss, "perturbed_probs_comp_graph")
                
                actor_loss = actor_loss.to(policy.device)

                # Critic update
                current_q_a, current_q_b = policy.critic(torch.cat([downsampled_obs, batch.actions], dim=1))
                critic_loss = F.mse_loss(current_q_a, target_q) + F.mse_loss(current_q_b, target_q)
                critic_losses.append(critic_loss.item())

                critic_loss = critic_loss.to(policy.device)

                # Alpha update
                alpha_loss = -(policy.log_alpha * (log_probs.detach() + policy.target_entropy)).mean()
                alpha_losses.append(alpha_loss.item())

                alpha_loss = alpha_loss.to(policy.device)

                critic_loss.backward(retain_graph=True)
                actor_loss.backward()
                alpha_loss.backward()

                # if i % 40 == 0: 
                #     print(f"Mem summary at iter {i}: \n{torch.cuda.memory_summary()}\n\n")

                policy.critic.optimizer.step()
                policy.actor.optimizer.step()
                policy.alpha_optimizer.step()

                # Soft update target
                polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

        if i % test_freq == 0: 
            rollout_val = rollout(test_envs, policy, gamma)
            if rollout_val > max_rollout_found: 
                model.save(f"Learned_main_intermed_{time_save}.zip")
                max_rollout_found = rollout_val
            avg_rewards.append(rollout_val)

    if visualize_lc: 
        plot_learning_curve(avg_rewards, test_freq, f"Learned_main_{time_save}_LC.png")
        plot_per_step(actor_losses, 1, f"Learned_main_{time_save}_actor.png", "Actor Loss")
        plot_per_step(critic_losses, 1, f"Learned_main_{time_save}_critic.png", "Critic Loss")
        plot_per_step(alpha_losses, 1, f"Learned_main_{time_save}_alpha.png", "Alpha Loss")
        plot_per_step(l1_norms, 1, f"Learned_main_{time_save}_l1.png", "L1 Loss")
        plot_per_step(l2_norms, 1, f"Learned_main_{time_save}_l2.png", "L2 Loss")
        plot_per_step(smoothness_vals, 1, f"Learned_main_{time_save}_smoothness.png", "Smoothness Loss")
        plot_per_step(classification_losses, 1, f"Learned_main_{time_save}_cls.png", "Classification Loss")


# 1000 timesteps, 32 envs, batch size 256 takes 1 hr to run
train_model(model.env, eval_envs, model.policy, model.replay_buffer, total_timesteps=num_timesteps, batch_size=training_batch_size, gradient_update_freq=gradient_update_freq, gamma=gamma, test_freq=test_freq)
model.save(f"Learned_main_{time_save}.zip")
