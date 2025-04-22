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

import zarr
from numcodecs import Blosc
from zarr.codecs import BloscCodec

from stable_baselines3 import SAC

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, GymStepReturn, DictReplayBufferSamples
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader
from environment import DataloaderEnv
from utils import count_nan_in_weights, count_inf_in_weights

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

# To run: python3 s_baselines/full.py

class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"): 
        super().__init__()
        
        # TODO fix code for if weights are not pretrained
        # 166 MB VRAM
        model_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Yields 512 dim vector 
        self.encoder = nn.Sequential(*list(model_base.children())[:-1]).half().to(device) # remove head

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
            nn.LeakyReLU(inplace=True)
        ).half().to(device)

        # Gaussian distribution in latent space
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), 
            nn.LeakyReLU(inplace=True)
        ).half().to(device)
        num_lower_triangular = (latent_dim * (latent_dim + 1)) // 2
        self.log_cholesky_layer = nn.Sequential(
            nn.Linear(latent_dim, num_lower_triangular), 
            nn.LeakyReLU(inplace=True)
        ).half().to(device)
        
        # nn.init.kaiming_uniform_(self.mu.weight, a=5)
        # nn.init.kaiming_uniform_(self.log_cholesky_layer.weight, a=2)

        # 154 MB VRAM for fc and deconv combined
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 15 * 15), 
            nn.BatchNorm1d(1024 * 15 * 15),
            nn.ReLU(inplace=True)
        ).half().to(device)

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
        ).half().to(device)

    # a = pi(s), for latent state s
    def forward(self, x): 
        with torch.autograd.detect_anomaly():

            assert not torch.isnan(x).any(), "State contains NaNs after Resnet!"
            assert not torch.isinf(x).any(), "State contains NaNs after Resnet!"
            print(f"After resnet, min: {torch.min(x)}\tmax: {torch.max(x)}")

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

            mus = torch.tanh(self.mu(x).float())

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

            lower_triang = torch.zeros(self.batch_size, self.latent_dim, self.latent_dim, device=self.device)

            indices = torch.tril_indices(self.latent_dim, self.latent_dim)
            lower_triang[:, indices[0], indices[1]] = lower_triang_elts

            # Exponentiate diagonal elements for stability
            diag_indices = torch.arange(self.latent_dim, device=self.device)
            lower_triang[:, diag_indices, diag_indices] = torch.nn.functional.softplus(lower_triang[:, diag_indices, diag_indices]) + 1e-2

            cov_mtxs = torch.matmul(lower_triang, lower_triang.transpose(-1, -2)) + torch.eye(self.latent_dim, device=x.device) * 1e-2

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
            
            print(f"Min mu: {torch.min(mus)}\tMax mu: {torch.max(mus)}")

            eigenvalues = torch.linalg.eigvalsh(cov_mtxs) 
            print(f"Min eigenvalues: {eigenvalues.min(dim=-1)[0]}")  # Get the minimum eigenvalue per batch
            # assert torch.all(eigenvalues >= 0), "Some covariance matrices are not PSD!"

            print(f"Trace of covariance: {cov_mtxs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)}")

            logdet = torch.slogdet(cov_mtxs)
            print(f"Sign of determinant of covariance matrices: {logdet.sign}\tLog of det: {logdet.logabsdet}")

            return x, mus, cov_mtxs
        
    def decode(self, x):
        with torch.autograd.detect_anomaly():
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
        ).half().to(self.device)

        self.critic_two = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim // 2), 
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 2, latent_dim // 8), 
            nn.BatchNorm1d(latent_dim // 8),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 8, 1)
        ).half().to(self.device)

    def forward(self, combined): 
        return self.critic_one(combined), self.critic_two(combined)


class CustomSACPolicy(SACPolicy):
    """
    implements both actor and critic in one model
    """
    def __init__(self, observation_space, action_space, lr_schedule, encoder, latent_dim, batch_size, device="cuda", **kwargs):
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

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def forward(self, *args, **kwargs):
        """
        Forward pass for SAC policy (actor and critic).
        
        :param obs: The input observation.
        :param deterministic: Whether to use deterministic actions or sample from the action distribution.
        :return: action, log_prob, Q
        """
        obs = args[0]
        deterministic = args[1] if len(args) > 1 else False
        print(f"[DEBUG] type(obs): {type(obs)}")
        print(f"[DEBUG] hasattr(obs, 'shape'): {hasattr(obs, 'shape')}")
        print(f"[DEBUG] obs: {obs}")
        
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float16, device=self.device)
        # If it's already a tensor, leave it alone
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        resnet_state = self.feature_encoder(obs_tensor)

        # Actor: Forward pass through the actor network (returns logits for actions)
        latent_state, mus, cov_mtxs = self.actor(resnet_state)

        # Action distribution: Squashed Gaussian distribution for continuous actions
        dist = SquashedDiagGaussianDistribution(mus, cov_mtxs)

        # Determine action (deterministic or stochastic)
        if deterministic:
            action = dist.get_mean()  # For deterministic action
        else:
            action = dist.sample()  # For stochastic action

        half_actions_latent = action.half()
        combined = torch.cat([latent_state, half_actions_latent], dim=1)
        # print(half_actions_latent.dtype, combined.dtype)
        values = self.critic(combined)

        return self.actor.decode(half_actions_latent).squeeze(0), dist.log_prob(action), values

    def _predict(self, observation, deterministic = False):
        with torch.no_grad(): 
            observation = self.feature_encoder(observation)

            observation, mus, cov_mtxs = self.actor(observation)
            
            if deterministic: 
                actions = torch.tanh(mus)
            else: 
                dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

                actions = torch.tanh(dist.sample())

            return actions.half()

    def evaluate_actions(self, obs, actions): 
        combined = torch.cat([self.feature_encoder(obs), actions.half()], dim=1)
        q1, q2 = self.critic(combined)

        mu, std = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return q1, q2, log_prob
    

class ZarrReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: torch.device,
        store_path: str = "zarr_buffer",
        dtype: np.dtype = np.float16,
        compressor: str = "zstd",
        **kwargs
    ):
        # super().__init__(buffer_size, observation_space, action_space, device, **kwargs)

        self.store = zarr.open_group(store_path, mode='w')

        self.device = device

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
        reward: float,
        done: bool,
        infos: Optional[GymStepReturn] = None
    ) -> None:
        # Normalize data types if needed
        self.obs[self.pos] = obs.astype(self.obs.dtype)
        self.next_obs[self.pos] = next_obs.astype(self.next_obs.dtype)
        self.actions[self.pos] = action.astype(np.float32)
        self.rewards[self.pos] = np.float16(reward)
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, env=None) -> DictReplayBufferSamples:
        store = zarr.open(self.store.store, mode="r")  # Open in read-only mode
        
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        # Load and convert to torch
        obs = torch.tensor(store["observations"][indices], dtype=torch.float16, device=self.device)
        next_obs = torch.tensor(store["next_observations"][indices], dtype=torch.float16, device=self.device)
        actions = torch.tensor(store["actions"][indices], dtype=torch.float16, device=self.device)
        rewards = torch.tensor(store["rewards"][indices], dtype=torch.float16, device=self.device)
        dones = torch.tensor(store["dones"][indices], dtype=torch.float16, device=self.device)

        return DictReplayBufferSamples(obs, actions, rewards, next_obs, dones)
    
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

# def linear_schedule(initial_value):
#     def func(progress_remaining):
#         return progress_remaining * initial_value
#     return func

# lr_schedule = get_linear_fn(3e-4, end=1e-5, end_fraction=0.1)

    
# SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 32
batch_size = 2
num_episodes = 5
train_data = train_dataloader(batch_size=batch_size)
obj_detector = YOLO("yolo11n.pt").to(device).eval()
env = DataloaderEnv(train_data, obj_detector=obj_detector, batch_size=batch_size)

encoder = Encoder(latent_dim=latent_dim, device=device)
# model = CustomSACPolicy(encoder=encoder, latent_dim=latent_dim, batch_size=batch_size, 
#                      observation_space=env.observation_space, action_space=env.action_space, 
#                      device=device)
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

# optimizer = optim.Adam(model.parameters(), lr=1e-7)
# eps = np.finfo(np.float32).eps.item()

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=env,
    buffer_size=10_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size,
        device=device
    ),
    # replay_buffer_class=ZarrReplayBuffer,
    # replay_buffer_kwargs={
    #     "store_path": "my_zarr_buffer",
    #     "compressor": "zstd"
    # },
    verbose=1,
    tensorboard_log="./sac_custom/",
)

model.learn(total_timesteps=5)
