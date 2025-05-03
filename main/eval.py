import torch
from stable_baselines3 import SAC
import sys
import os
from ultralytics import YOLO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader, test_dataloader, val_dataloader, denormalize_batch, renormalize_batch, display_batch
from environment import DataloaderEnv
from full import Encoder, ZarrSAC, CustomSACPolicy

from utils import *



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
num_val_envs = 8
num_timesteps = 1000
val_data = val_dataloader(batch_size=batch_size, num_workers=0)
obj_classifier = YOLO("yolo11n-cls.pt").to(device).eval()
train_envs = make_vec_env(val_data, obj_classifier, num_val_envs, latent_dim)

encoder = Encoder(latent_dim=latent_dim, device=device)

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=train_envs,
    buffer_size=10_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size * num_val_envs,
        device=device
    ),
    verbose=1,
)

model.load(f"Learned_main_1746288507.925695.zip")

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

        perturbed_normalized_to_s = obs_tensor + actions # not [0,1]

        orig_denormalized = denormalize_batch(obs_tensor) # [0,1] range
        perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s) # [0,1] range

        perturbed_normalized_clamp = renormalize_batch(perturbed_denormalized)
        perturbed_normalized_clamp_cpu = renormalize_batch(perturbed_denormalized).cpu()
    
        with torch.no_grad():
            orig_results = obj_classifier(orig_denormalized, verbose=False)
            perturbed_results = obj_classifier(perturbed_denormalized, verbose=False)

        for i, env in enumerate(envs.envs):
            if step_num == 0: 
                env.set_results(perturbed_normalized_clamp_cpu[i], orig_results[i], perturbed_results[i], False)
            else: 
                env.set_results(perturbed_normalized_clamp_cpu[i], orig_results[i], perturbed_results[i], dones[i])

        next_obs, rewards, dones, _ = envs.step(actions_npy)
        total_rewards += curr_gamma * rewards
        curr_gamma *= gamma

        display_batch(obs_tensor)
        display_batch(perturbed_normalized_clamp)

        step_num += 1

        if step_num > max_steps: 
            break
        
    return total_rewards.mean()

rollout(train_envs, model.policy, gamma=gamma)