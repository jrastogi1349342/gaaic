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
num_val_envs = 32
num_timesteps = 1000
val_data = train_dataloader(batch_size=batch_size, num_workers=0)
obj_classifier = YOLO("yolo11n-cls.pt").to(device).eval()
val_envs = make_vec_env(val_data, obj_classifier, num_val_envs, latent_dim)

encoder = Encoder(latent_dim=latent_dim, device=device)

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=val_envs,
    buffer_size=10_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size * num_val_envs,
        device=device
    ),
    verbose=1,
)

model.load(f"Learned_main_1747093838.8063166.zip")

def rollout(
    envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    gamma: float,
    max_steps: int = 50, 
): 
    envs.reset()

    # for [0,1] normalized images
    l1_norms_orig = [] 
    l2_norms_orig = []
    l1_norms_actions = [] 
    l2_norms_actions = []
    l1_norms_perturbed = [] 
    l2_norms_perturbed = []
    num_eps_completed = 0

    # total_rewards = np.zeros((envs.num_envs))
    step_num = 0
    # curr_gamma = 1

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
        perturbed_normalized_clamp_cpu = perturbed_normalized_clamp.cpu()
    
        with torch.no_grad():
            orig_results = obj_classifier(orig_denormalized, verbose=False)
            perturbed_results = obj_classifier(perturbed_denormalized, verbose=False)

        for i, env in enumerate(envs.envs):
            if step_num == 0: 
                env.set_results(perturbed_normalized_clamp_cpu[i], orig_results[i], perturbed_results[i], False)
            else: 
                env.set_results(perturbed_normalized_clamp_cpu[i], orig_results[i], perturbed_results[i], dones[i])

        next_obs, rewards, dones, _ = envs.step(actions_npy)
        # total_rewards += curr_gamma * rewards
        # curr_gamma *= gamma

        # TODO figure out why this gives comparable values to standard normal dist, even though visually it looks different
        actions_denorm = denormalize_batch(actions)
        l1_orig = torch.norm(obs_tensor, p=1, dim=(1, 2, 3)).mean().item()
        l1_action = torch.norm(actions_denorm, p=1, dim=(1, 2, 3)).mean().item()
        l1_perturbed = torch.norm(perturbed_normalized_clamp, p=1, dim=(1, 2, 3)).mean().item()
        l2_orig = torch.norm(obs_tensor, p=2, dim=(1, 2, 3)).mean().item()
        l2_action = torch.norm(actions_denorm, p=2, dim=(1, 2, 3)).mean().item()
        l2_perturbed = torch.norm(perturbed_normalized_clamp, p=2, dim=(1, 2, 3)).mean().item()
        
        l1_norms_orig.append(l1_orig)
        l2_norms_orig.append(l2_orig)
        l1_norms_actions.append(l1_action)
        l2_norms_actions.append(l2_action)
        l1_norms_perturbed.append(l1_perturbed)
        l2_norms_perturbed.append(l2_perturbed)

        display_batch(obs_tensor)
        display_batch(perturbed_normalized_clamp)

        step_num += 1

        for idx, done in enumerate(dones):
            if done.item():
                num_eps_completed += 1
                val_envs.env_method("reset", indices=idx)

        if step_num > max_steps: 
            break
        
    return np.array(l1_norms_orig), np.array(l2_norms_orig), np.array(l1_norms_actions), np.array(l2_norms_actions), np.array(l1_norms_perturbed), np.array(l2_norms_perturbed), (step_num * envs.num_envs / num_eps_completed)

l1_orig, l2_orig, l1_action, l2_action, l1_full, l2_full, cls_num_steps = rollout(val_envs, model.policy, gamma=gamma, max_steps=num_timesteps)

# These norms are for [0,1] images
print(f"""Avg L1 norm of original: {np.mean(l1_orig)}\tAvg L2 norm of original: {np.mean(l2_orig)}
      \tAvg L1 norm of action: {np.mean(l1_action)}\tAvg L2 norm of action: {np.mean(l2_action)}
      \tAvg L1 norm of perturbed img: {np.mean(l1_full)}\tAvg L2 norm of perturbed img: {np.mean(l2_full)}
      \tAvg num steps per episode: {cls_num_steps}""")