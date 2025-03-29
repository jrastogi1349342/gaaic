import torch
import gymnasium as gym
import numpy as np
from rewards import calc_rewards
from dataloader import denormalize_batch, renormalize_batch

class DataloaderEnv(gym.Env): 
    def __init__(self, dataloader, obj_detector, device="cuda", max_steps_per_episode=1000, batch_size=8, action_shape=(480, 480, 3)): 
        super().__init__

        self.dataloader = iter(dataloader)
        self.action_shape = action_shape
        self.batch = None
        self.batch_size = batch_size 
        self.max_steps_per_episode = max_steps_per_episode
        self.step_idx = 0 # num steps in current episode
        self.obj_detector = obj_detector
        self.device = device

        sample_state = next(self.dataloader)
        batch_size, *obs_shape = sample_state.shape

        # TODO figure this out
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(sample_state.shape))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(sample_state.shape))

    def reset(self): 
        try:
            self.batch = next(self.dataloader)  # Get next batch (episode)
        except StopIteration:
            self.dataloader = iter(self.dataloader)  # Restart DataLoader
            self.batch = next(self.dataloader)

        self.step_idx = 0
        return self.batch.half().to(self.device)
    
    # Action: batch_size x 480x480x3
    def step(self, action): 
        orig_states = self.batch.to(self.device)
        self.step_idx += 1

        # print(torch.min(orig_states), torch.max(orig_states))

        # Add action (480x480x3) noise image to each state in batch
        perturbed_normalized_to_s = orig_states + action

        orig_denormalized = denormalize_batch(orig_states)
        perturbed_denormalized = denormalize_batch(perturbed_normalized_to_s)

        # TODO implement reward function
        reward_batches, done_batches = calc_rewards(orig_denormalized, perturbed_denormalized, self.obj_detector, goal="empty", device=self.device)

        # print("Actions", action.shape, action)
        next_batch = (renormalize_batch(perturbed_denormalized)).half().to(self.device)

        done_batches = torch.tensor([True] * self.batch_size) if self.step_idx >= self.max_steps_per_episode else done_batches

        del self.batch

        self.batch = next_batch

        return next_batch, reward_batches, done_batches, {}, {}