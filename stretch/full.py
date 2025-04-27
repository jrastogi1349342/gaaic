import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
from collections import namedtuple
import argparse
from itertools import count
import sys
import os
from ultralytics import YOLO
from typing import Callable, Optional
from stable_baselines3.common.utils import polyak_update
from tqdm import trange
from torch.amp import GradScaler
import time

import zarr
from numcodecs import Blosc
from zarr.codecs import BloscCodec

from stable_baselines3 import SAC

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, GymStepReturn, DictReplayBufferSamples
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

# torch.autograd.set_detect_anomaly(True)

# To run: python3 stretch/full.py

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
        assert torch.min(x).positive() 
        assert not torch.isnan(x).any(), "State contains NaNs in Encoder!"
        assert not torch.isinf(x).any(), "State contains Infs in Encoder!"
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        assert not torch.isnan(x).any(), "NaN detected after resnet encoder!"
        assert not torch.isinf(x).any(), "Inf detected after resnet encoder!"
        # print("Min of latent embedding:", torch.min(x), "Max of latent embedding:", torch.max(x), "Nan:", torch.any(torch.isnan(x)), "Inf:", torch.any(torch.isinf(x)))
        # print("Encoder fc:", self.fc.weight)
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
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), 
            nn.LeakyReLU(inplace=True)
        ).to(device)
        num_lower_triangular = (latent_dim * (latent_dim + 1)) // 2
        self.log_cholesky_layer = nn.Sequential(
            nn.Linear(latent_dim, num_lower_triangular), 
            nn.LeakyReLU(inplace=True)
        ).to(device)
        
        # nn.init.kaiming_uniform_(self.mu.weight, a=5)
        # nn.init.kaiming_uniform_(self.log_cholesky_layer.weight, a=2)

        # 154 MB VRAM for fc and deconv combined
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 15 * 15), 
            nn.BatchNorm1d(1024 * 15 * 15),
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
        # with torch.autograd.detect_anomaly():

        assert not torch.isnan(x).any(), "State contains NaNs after Resnet!"
        assert not torch.isinf(x).any(), "State contains NaNs after Resnet!"
        # print(f"After resnet, min: {torch.min(x)}\tmax: {torch.max(x)}")

        # print("Mu min:", torch.min(self.mu.weight), "Max:", torch.max(self.mu.weight), "Nan:", torch.any(torch.isnan(self.mu.weight)), "Inf:", torch.any(torch.isinf(self.mu.weight)))
        # print("Log std min:", torch.min(self.log_std.weight), "Max:", torch.max(self.log_std.weight), "Nan:", torch.any(torch.isnan(self.log_std.weight)), "Inf:", torch.any(torch.isinf(self.log_std.weight)))
        # print("Low rank factor min:", torch.min(self.low_rank_factor.weight), "Max:", torch.max(self.low_rank_factor.weight), "Nan:", torch.any(torch.isnan(self.low_rank_factor.weight)), "Inf:", torch.any(torch.isinf(self.low_rank_factor.weight)))
        # print("x min:", torch.min(x), "Max:", torch.max(x), "Nan:", torch.any(torch.isnan(x)), "Inf:", torch.any(torch.isinf(x)))

        x = self.downsampled_enc(x)
        assert not torch.isnan(x).any() and not torch.isinf(x).any(), \
                f"""NaNs/Infs detected after downsampled encoder fc! {count_nan_in_weights(self.downsampled_enc)} NaNs model, 
                {count_inf_in_weights(self.downsampled_enc)} Infs model, 
                {torch.sum(torch.isnan(x)).item()} NaNs downsampled, {torch.sum(torch.isinf(x)).item()} Infs downsampled"""
        # assert not torch.isinf(x).any(), f"Inf detected after downsampled encoder fc! {count_inf_in_weights(self.downsampled_enc)} Infs in model, {torch.sum(torch.isinf(x)).item()} in downsampled" 

        mus = self.mu(x).float()

        assert not torch.isnan(mus).any() and not torch.isinf(mus).any(), \
                f"""NaNs/Infs detected after mu calc! {count_nan_in_weights(self.mu)} NaNs model, 
                {count_inf_in_weights(self.mu)} Infs model, 
                {torch.sum(torch.isnan(mus)).item()} NaNs calc, {torch.sum(torch.isinf(mus)).item()} Infs calc"""

        # assert not torch.isnan(x).any(), f"NaN detected after mu calc! {self.downsampled_enc.weight}"
        # assert not torch.isinf(x).any(), f"Inf detected after mu calc! {self.mu.weight}"

        lower_triang_elts = self.log_cholesky_layer(x).float()

        assert not torch.isnan(lower_triang_elts).any() and not torch.isinf(lower_triang_elts).any(), \
                f"""NaNs/Infs detected after cov calc! {count_nan_in_weights(self.log_cholesky_layer)} NaNs model, 
                {count_inf_in_weights(self.log_cholesky_layer)} Infs model, 
                {torch.sum(torch.isnan(lower_triang_elts)).item()} NaNs calc, {torch.sum(torch.isinf(lower_triang_elts)).item()} Infs calc"""

        # assert not torch.isnan(x).any(), f"NaN detected after Lower triang covariance calc! {self.log_cholesky_layer.weight}"
        # assert not torch.isinf(x).any(), f"Inf detected after Lower triang covariance calc! {self.log_cholesky_layer.weight}"

        lower_triang = torch.zeros(lower_triang_elts.shape[0], self.latent_dim, self.latent_dim, device=self.device)

        indices = torch.tril_indices(self.latent_dim, self.latent_dim)
        lower_triang[:, indices[0], indices[1]] = lower_triang_elts

        # Exponentiate diagonal elements for stability
        diag_indices = torch.arange(self.latent_dim, device=self.device)
        lower_triang[:, diag_indices, diag_indices] = torch.nn.functional.softplus(lower_triang[:, diag_indices, diag_indices]).clamp(min=1e-3, max=1e2) + 1e-2
        # lower_triang = lower_triang.to(torch.float32)

        # with torch.autocast(device_type='cuda', enabled=False): 
        cov_mtxs = torch.matmul(lower_triang, lower_triang.transpose(-1, -2)) 
        # print(f"Before sum: {lower_triang.dtype}, {lower_triang.transpose(-1, -2).dtype}, {cov_mtxs.dtype}")
        cov_mtxs += torch.eye(self.latent_dim, device=self.device, dtype=torch.float32) * 1e-2

        # print(f"After sum: {lower_triang.dtype}, {cov_mtxs.dtype}")
        if torch.isnan(cov_mtxs).any() or torch.isinf(cov_mtxs).any():
            print("NaN or Inf detected in cov_mtxs!")

        # log_std = self.log_std(x).float()
        # log_std = torch.clamp(log_std, min=-5, max=2) # prevent exploding values
        # std = torch.exp(log_std)
        # print(f"Std min: {std.min()}, max: {std.max()}")
        # # print(mus.shape, log_std.shape, std.shape)
        # assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 2!"
        # assert not torch.isinf(mus).any(), "Inf detected Perturbation Model 2!"
        # assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 3!"
        # assert not torch.isinf(log_std).any(), "Inf detected Perturbation Model 3!"
        # assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 4!"
        # assert not torch.isinf(log_std).any(), "Inf detected Perturbation Model 4!"

        # diag_cov = torch.diag_embed(std ** 2 + 1e-2)

        # # print(x.shape) 
        # # print(self.low_rank_factor(x).shape)
        # U = self.low_rank_factor(x).view(self.batch_size, self.latent_dim, self.low_rank).float()
        # low_rank_cov = torch.matmul((U * 15), (U * 15).transpose(-1, -2)) # PSD

        # ranks = torch.linalg.matrix_rank(U)
        # print(f"Ranks of U: {ranks}\tDesired rank: {self.low_rank}")
        # # print(diag_cov.shape, U.shape, low_rank_cov.shape)

        # print(f"Mean norm of U: {torch.norm(U, dim=[-2, -1]).mean().item()}")

        # cov_mtxs = diag_cov + low_rank_cov + torch.eye(self.latent_dim, device=x.device) * 2e-1 # for stability
        # cov_mtxs *= 0.1 # For stability
        # print(f"Covs min: {cov_mtxs.min()}, max: {cov_mtxs.max()}, type: {cov_mtxs.dtype}")

        # ranks = torch.linalg.matrix_rank(cov_mtxs)
        # print(f"Ranks of Cov mtxs: {ranks}\tDesired rank: {self.latent_dim}")
        
        # print(f"Min mu: {torch.min(mus)}\tMax mu: {torch.max(mus)}")

        # eigenvalues = torch.linalg.eigvalsh(cov_mtxs) 
        # print(f"Min eigenvalues: {eigenvalues.min(dim=-1)[0]}")  # Get the minimum eigenvalue per batch
        # # assert torch.all(eigenvalues >= 0), "Some covariance matrices are not PSD!"

        # print(f"Trace of covariance: {cov_mtxs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)}")

        # logdet = torch.slogdet(cov_mtxs)
        # print(f"Sign of determinant of covariance matrices: {logdet.sign}\tLog of det: {logdet.logabsdet}")

        return x, mus, cov_mtxs
        
    def set_training_mode(self, mode: bool):
        self.train(mode)
        
    def decode(self, x):
        # with torch.autograd.detect_anomaly():
        assert not torch.isnan(x).any(), "Latent action contains NaNs!"
        assert not torch.isinf(x).any(), "Latent action contains Infs!"

        x = self.fc(x)

        assert not torch.isnan(x).any(), "Upsampled latent action contains NaNs!"
        assert not torch.isinf(x).any(), "Upsampled latent action contains Infs!"

        x = x.view(x.size(0), 1024, 15, 15)
        x = self.deconv(x)

        assert not torch.isnan(x).any(), "Deconvolved action contains NaNs!"
        assert not torch.isinf(x).any(), "Deconvolved action contains Infs!"

        x = torch.clamp(x, min=-1.0, max=1.0)

        # Alternative
        # return self.deconv(x.unsqueeze(-1).unsqueeze(-1))  # Add dimensions for ConvTranspose2d

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
    def __init__(self, observation_space, action_space, lr_schedule, encoder, latent_dim, batch_size, target_entropy=-32.0, device="cuda", **kwargs):
        super().__init__(observation_space, action_space, lr_schedule)
        self.feature_encoder = encoder
        self.batch_size = batch_size

        self.observation_space = observation_space
        self.action_space = action_space

        self.actor = PerturbationModel(latent_dim, device=device, batch_size=batch_size)
        assert isinstance(self.actor, nn.Module), "Actor must be nn.Module"
        assert hasattr(self.actor, "train"), "Actor must have train function"

        # Input: concatenated state and action latent vectors
        self.critic = Critic(latent_dim=latent_dim, device=self.device)
        self.critic_target = Critic(latent_dim=latent_dim, device=self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # TODO implement alpha 
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.target_entropy = target_entropy

    def forward(self, *args, **kwargs):
        """
        Forward pass for SAC policy (actor and critic).
        
        :param obs: The input observation.
        :param deterministic: Whether to use deterministic actions or sample from the action distribution.
        :return: action, log_prob, Q
        """
        obs = args[0]
        deterministic = args[1] if len(args) > 1 else False
        # print(f"[DEBUG] type(obs): {type(obs)}")
        # print(f"[DEBUG] hasattr(obs, 'shape'): {hasattr(obs, 'shape')}")
        # print(f"[DEBUG] obs: {obs}")
        
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # If it's already a tensor, leave it alone
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        resnet_state = self.feature_encoder(obs_tensor)

        # Actor: Forward pass through the actor network (returns logits for actions)
        latent_state, mus, cov_mtxs = self.actor(resnet_state)

        # Action distribution: Squashed Gaussian distribution for continuous actions
        # Note: old, unused
        dist = SquashedDiagGaussianDistribution(mus, cov_mtxs)

        # Determine action (deterministic or stochastic)
        if deterministic:
            action = dist.get_mean()  # For deterministic action
        else:
            action = dist.sample()  # For stochastic action

        # half_actions_latent = action.half()
        combined = torch.cat([latent_state, action], dim=1)
        # print(half_actions_latent.dtype, combined.dtype)

        # Note: old, unused
        values = self.critic(combined)

        return self.actor.decode(action).squeeze(0), dist.log_prob(action), values

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

            # # Action distribution: Squashed Gaussian distribution for continuous actions
            # dist = SquashedDiagGaussianDistribution(mus, cov_mtxs)

            # # Determine action (deterministic or stochastic)
            # if deterministic:
            #     action = dist.get_mean()  # For deterministic action
            # else:
            #     action = dist.sample()  # For stochastic action

            if deterministic: 
                actions = mus
            else: 
                dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

                actions = dist.sample()


        # half_actions_latent = actions.half()

        return actions, self.actor.decode(actions).squeeze(0)

    # Note: observation is already latent vector
    def predict_action_with_prob(self, observation, deterministic = False):
        # observation = self.feature_encoder(observation)

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
        # observation = self.feature_encoder(observation)

        observation, mus, cov_mtxs = self.actor(observation)
        log_prob = None
        
        if deterministic: 
            actions = mus
        else: 
            dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

            actions = dist.sample()
            log_prob = dist.log_prob(actions)

        actions_upsampled = self.actor.decode(actions).squeeze(0)

        return actions_upsampled, actions, log_prob.unsqueeze(1)

    def evaluate_actions(self, obs, actions): 
        combined = torch.cat([self.feature_encoder(obs), actions], dim=1)
        q1, q2 = self.critic(combined)

        mu, std = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return q1, q2, log_prob
    
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
        # super().__init__(buffer_size, observation_space, action_space, device, **kwargs)

        self.store = zarr.open_group(store_path, mode='w')

        self.device = device
        self.buffer_size = buffer_size

        self.obs_shape = observation_space.shape
        self.action_shape = action_space.shape

        # compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
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

    def _setup_model(self):
        # Don't call super()._setup_model() directly, it tries to use replay_buffer_class
        super(SAC, self)._setup_model()  # skip SAC-specific buffer setup

        # Now manually set your custom buffer
        self.replay_buffer = ZarrReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            store_path="zarr_sac_buffer"
        )

def make_env_fn(dataset, obj_detector, idx, latent_dim):
    def _init():
        return DataloaderEnv(dataset, obj_detector, idx, latent_dim, batch_size=1)
    return _init

def make_vec_env(dataset, obj_detector, num_envs, latent_dim):
    return DummyVecEnv([make_env_fn(dataset, obj_detector, idx, latent_dim) for idx in range(num_envs)])

# def linear_schedule(initial_value):
#     def func(progress_remaining):
#         return progress_remaining * initial_value
#     return func

# lr_schedule = get_linear_fn(3e-4, end=1e-5, end_fraction=0.1)

    
# SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.95
latent_dim = 32
batch_size = 1 # must be 1, use multiple environments for parallel episodes
training_batch_size = 64
num_train_envs = 32
num_timesteps = 1000
train_data = train_dataloader(batch_size=batch_size, num_workers=0)
obj_detector = YOLO("yolo11n.pt").to(device).eval()
train_envs = make_vec_env(train_data, obj_detector, num_train_envs, latent_dim)

num_test_envs = 10
test_data = test_dataloader(batch_size=batch_size, num_workers=0)
eval_envs = make_vec_env(test_data, obj_detector, num_test_envs, latent_dim)
test_freq = 5 # every x timesteps in training
# DataloaderEnv(train_data, obj_detector=obj_detector, batch_size=batch_size) # TODO consider SubProcEnv

# TODO look at: 
# Consider SAC envs vs dataloader stuff
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
# https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html


encoder = Encoder(latent_dim=latent_dim, device=device)
# model = CustomSACPolicy(encoder=encoder, latent_dim=latent_dim, batch_size=batch_size, 
#                      observation_space=env.observation_space, action_space=env.action_space, 
#                      device=device)
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

# optimizer = optim.Adam(model.parameters(), lr=1e-7)
# eps = np.finfo(np.float32).eps.item()

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=train_envs,
    buffer_size=10_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size * num_train_envs,
        device=device
    ),
    # replay_buffer_class=ZarrReplayBuffer,
    # replay_buffer_kwargs={
    #     "store_path": "my_zarr_buffer",
    #     "compressor": "zstd"
    # },
    verbose=1,
    # tensorboard_log="./sac_custom/",
)

# TODO write this
def rollout(
    envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    gamma: float,
    max_steps: int = 50, 
): 
    envs.reset()

    total_rewards = np.zeros((envs.num_envs))
    step_num = 0
    curr_gamma = gamma

    while True:
        obs_batch = np.stack([env.batch for env in envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            _, actions = policy.pred_upsampled_action(obs_tensor, deterministic=True)
        
        # latent_actions = latent_actions.cpu().numpy()
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

        orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
        perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

        perturbed_normalized_clamp = renormalize_batch(perturbed_denormalized).cpu()
    
        with torch.no_grad():
            orig_results = obj_detector(orig_denormalized, verbose=False)
            perturbed_results = obj_detector(perturbed_denormalized, verbose=False)

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

# model.learn(total_timesteps=5)

def train_model(
    train_envs: DummyVecEnv,
    test_envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    replay_buffer, 
    total_timesteps: int = 20,
    batch_size: int = 32,
    gradient_updates: int = 1,
    gamma: float = 0.95,
    tau: float = 0.005, 
    test_freq: float = 5, 
    visualize_lc: bool = True
):
    train_envs.reset() 
    avg_rewards = []
    # obs, _ = envs.reset()  # (num_envs, C, H, W)
    # obs = torch.tensor(obs, dtype=torch.float16, device=device)
    # num_envs = envs.num_envs
    # print(num_envs)

    # scaler = GradScaler("cuda")

    for i in trange(total_timesteps):
        obs_batch = np.stack([env.batch for env in train_envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            # print(f"697: {obs_tensor.device}")
            latent_actions, actions = policy.pred_upsampled_action(obs_tensor, deterministic=False)
        
        latent_actions = latent_actions.cpu().numpy()
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

        orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
        perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

        perturbed_normalized_clamp = renormalize_batch(perturbed_denormalized).cpu()
    
        with torch.no_grad():
            # print(f"711: {orig_denormalized.device}")
            # print(f"712: {perturbed_denormalized.device}")
            orig_results = obj_detector(orig_denormalized, verbose=False)
            perturbed_results = obj_detector(perturbed_denormalized, verbose=False)

        for j, env in enumerate(train_envs.envs):
            env.set_results(perturbed_normalized_clamp[j], orig_results[j], perturbed_results[j])

        next_obs, rewards, dones, _ = train_envs.step(actions_npy)
        # obs = obs.cpu()
        # print("Obs and next obs shapes:", obs_batch.shape, next_obs.shape, rewards.shape, dones.shape)
        # next_obs = torch.tensor(next_obs, dtype=torch.float16, device=device)

        # rewards = torch.tensor(rewards, dtype=torch.float16, device=device).unsqueeze(-1)
        # dones = torch.tensor(dones, dtype=torch.float16, device=device).unsqueeze(-1)

        replay_buffer.add(obs_batch, next_obs, latent_actions, rewards, dones)
        # obs = next_obs

        if replay_buffer.length() >= batch_size:
            for _ in range(gradient_updates):
                batch = replay_buffer.sample(batch_size)

                # print(batch.observations.shape, batch.actions.shape, batch.next_observations.shape, batch.dones.shape, batch.rewards.shape)

                with torch.no_grad():
                    # print(f"737: {batch.observations.device}")
                    # print(f"738: {batch.next_observations.device}")
                    latent_obs = policy.feature_encoder(batch.observations)
                    latent_next_obs = policy.feature_encoder(batch.next_observations)

                # print(f"742: {latent_obs.device}")
                # print(f"743: {latent_next_obs.device}")
                downsampled_obs = policy.actor.downsampled_enc(latent_obs)
                downsampled_next_obs = policy.actor.downsampled_enc(latent_next_obs)

                with torch.no_grad():
                    next_actions, next_log_probs = policy.predict_action_with_prob(latent_next_obs, deterministic=False)
                    # print(latent_next_obs.shape, next_actions.shape)
                    # print(f"750: {downsampled_next_obs.device}")
                    # print(f"751: {next_actions.device}")
                    q_next_a, q_next_b = policy.critic_target(torch.cat([downsampled_next_obs, next_actions], dim=1))
                    q_next = torch.min(q_next_a, q_next_b)

                    # print(batch.rewards.unsqueeze(1).dtype, type(gamma), batch.dones.unsqueeze(1).dtype, q_next.dtype, policy.alpha.dtype, next_log_probs.dtype)

                    # TODO check if I need to detach alpha here, since in torch.no_grad()
                    target_q = batch.rewards.unsqueeze(1) + gamma * (1 - batch.dones.unsqueeze(1)) * (q_next - policy.alpha.detach() * next_log_probs)

                policy.critic.optimizer.zero_grad()
                policy.actor.optimizer.zero_grad()
                policy.alpha_optimizer.zero_grad()

                # with torch.autocast(device_type='cuda'): 
                # Actor update
                new_actions_upsampled, new_actions, log_probs = policy.predict_action_with_prob_upsampling(latent_obs, deterministic=False)
                # print("obs:", downsampled_obs.min(), downsampled_obs.max(), downsampled_obs.mean())
                # print("actions:", new_actions.min(), new_actions.max(), new_actions.mean())

                # print(f"750: {downsampled_obs.device}")
                # print(f"751: {new_actions.device}")
                q_new_action_a, q_new_action_b = policy.critic(torch.cat([downsampled_obs, new_actions], dim=1))
                l2_norm_loss = torch.norm(new_actions_upsampled, p=2, dim=(1, 2, 3)).mean()
                smoothness_loss = torch.mean(torch.abs(new_actions_upsampled[:, :, :-1] - new_actions_upsampled[:, :, 1:])) + \
                                  torch.mean(torch.abs(new_actions_upsampled[:, :-1, :] - new_actions_upsampled[:, 1:, :])) 
                l1_norm_loss = torch.norm(new_actions_upsampled, p=1, dim=(1, 2, 3)).mean()
                actor_loss = (policy.alpha.detach() * log_probs - torch.min(q_new_action_a, q_new_action_b)).mean() + \
                             0.1 * l2_norm_loss + 0.01 * smoothness_loss + 0.1 * l1_norm_loss
                
                actor_loss = actor_loss.to(policy.device)

                # Critic update
                # print(f"783: {batch.actions.device}")
                current_q_a, current_q_b = policy.critic(torch.cat([downsampled_obs, batch.actions], dim=1))
                critic_loss = F.mse_loss(current_q_a, target_q) + F.mse_loss(current_q_b, target_q)
                critic_loss = critic_loss.to(policy.device)

                # Alpha update
                alpha_loss = -(policy.log_alpha * (log_probs.detach() + policy.target_entropy)).mean().to(policy.device)

                # if i == 18: 
                #     display_comp_graph(q_new_action_a, "q_new_action_a_iter18")
                #     display_comp_graph(current_q_a, "current_q_a_iter18")

                # print(target_q.dtype, current_q_a.dtype)

                # print(critic_loss, actor_loss, alpha_loss)

                # with torch.autograd.set_detect_anomaly(True):
                critic_loss.backward(retain_graph=True)
                actor_loss.backward()
                alpha_loss.backward()

                    # scaler.scale(actor_loss).backward()
                    # scaler.scale(critic_loss).backward()
                    # scaler.scale(alpha_loss).backward()

                policy.critic.optimizer.step()
                policy.actor.optimizer.step()
                policy.alpha_optimizer.step()

                # scaler.step(policy.critic.optimizer)
                # scaler.step(policy.actor.optimizer)
                # scaler.step(policy.alpha_optimizer)

                # Soft update target
                polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

        for idx, done in enumerate(dones):
            if done.item():
                # print(f"Reset env {idx}")
                train_envs.env_method("reset", indices=idx)
            #     obs[idx] = torch.tensor(envs.reset_single(idx), dtype=torch.float16, device=device)
            # else:
            #     obs[idx] = next_obs[idx]

        if i % test_freq == 0: 
            rollout_val = rollout(test_envs, policy, gamma)
            # print(f"Running rollout: {rollout_val}")
            avg_rewards.append(rollout_val)

    if visualize_lc: 
        plot_learning_curve(avg_rewards, test_freq)

# 1000 timesteps, 16 envs, batch size 64 takes 1.5 hrs to run
train_model(model.env, eval_envs, model.policy, model.replay_buffer, total_timesteps=num_timesteps, batch_size=training_batch_size, gamma=gamma, test_freq=test_freq)


model.policy.save(f"Learned_{time.time()}.zip")