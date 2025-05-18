import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
from ultralytics import YOLO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader, test_dataloader, val_dataloader, denormalize_batch, renormalize_batch, display_batch
from environment import DataloaderEnv

from utils import *


class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"): 
        super().__init__()
        
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
val_data = val_dataloader(batch_size=batch_size, num_workers=0)
obj_classifier = YOLO("yolo11n-cls.pt").to(device).eval()
val_envs = make_vec_env(val_data, obj_classifier, num_val_envs, latent_dim)

encoder = Encoder(latent_dim=latent_dim, device=device)


def rollout(
    envs: DummyVecEnv, 
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
        obs_tensor = torch.from_numpy(obs_batch).to(device) 

        actions = torch.clamp(torch.randn(obs_tensor.shape, device='cuda'), -1, 1)

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

        _, _, dones, info = envs.step(actions_npy)
        # total_rewards += curr_gamma * rewards
        # curr_gamma *= gamma

        actions_denorm = denormalize_batch(actions)
        l1_orig = torch.norm(orig_denormalized, p=1, dim=(1, 2, 3)).mean().item()
        l1_action = torch.norm(actions_denorm, p=1, dim=(1, 2, 3)).mean().item()
        l1_perturbed = torch.norm(perturbed_denormalized, p=1, dim=(1, 2, 3)).mean().item()
        l2_orig = torch.norm(orig_denormalized, p=2, dim=(1, 2, 3)).mean().item()
        l2_action = torch.norm(actions_denorm, p=2, dim=(1, 2, 3)).mean().item()
        l2_perturbed = torch.norm(perturbed_denormalized, p=2, dim=(1, 2, 3)).mean().item()
        
        l1_norms_orig.append(l1_orig)
        l2_norms_orig.append(l2_orig)
        l1_norms_actions.append(l1_action)
        l2_norms_actions.append(l2_action)
        l1_norms_perturbed.append(l1_perturbed)
        l2_norms_perturbed.append(l2_perturbed)

        # display_before_after(obs_tensor, perturbed_normalized_clamp, actions, info)

        # display_batch(obs_tensor)
        # display_batch(perturbed_normalized_clamp)

        step_num += 1

        for idx, done in enumerate(dones):
            if done.item():
                num_eps_completed += 1
                val_envs.env_method("reset", indices=idx)
                dones[idx] = False

        if step_num > max_steps: 
            break
        
    return np.array(l1_norms_orig), np.array(l2_norms_orig), np.array(l1_norms_actions), np.array(l2_norms_actions), np.array(l1_norms_perturbed), np.array(l2_norms_perturbed), (step_num * envs.num_envs / num_eps_completed)

l1_orig, l2_orig, l1_action, l2_action, l1_full, l2_full, cls_num_steps = rollout(val_envs, gamma=gamma, max_steps=num_timesteps)

# Result for gaussian noise: 
# Avg L1 norm: 68348.60227272728  Avg L2 norm: 186.12226680180171 Avg num steps per episode: 1.0001248907206195
# These norms are for [0,1] images
# These norms are for [0,1] images
print(f"""Avg L1 norm of original: {np.mean(l1_orig)}\tAvg L2 norm of action: {np.mean(l2_orig)}
      \tAvg L1 norm of action: {np.mean(l1_action)}\tAvg L2 norm of action: {np.mean(l2_action)}
      \tAvg L1 norm of perturbed img: {np.mean(l1_full)}\tAvg L2 norm of perturbed img: {np.mean(l2_full)}
      \tAvg num steps per episode: {cls_num_steps}""")