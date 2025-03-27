import torch
import gymnasium as gym
import numpy as np
from rewards import calc_rewards

class DataloaderEnv(gym.Env): 
    def __init__(self, dataloader, object_detector, max_steps_per_episode=1000, batch_size=8, action_shape=(480, 480, 3)): 
        super().__init__

        self.dataloader = iter(dataloader)
        self.action_shape = action_shape
        self.batch = None
        self.batch_size = batch_size 
        self.max_steps_per_episode = max_steps_per_episode
        self.step_idx = 0 # num steps in current episode
        self.object_detector = object_detector

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
        return self.batch
    
    # Action: batch_size x 480x480x3
    def step(self, action): 
        orig_states = self.batch
        self.step_idx += 1

        # Add action (480x480x3) noise image to each state in batch
        next_batch = torch.clamp(self.batch + action, 0, 1)

        # TODO implement reward function
        reward_batches, done_batches = calc_rewards(orig_states, next_batch, self.obj_detector, "empty")

        done_batches = torch.tensor([True] * self.batch_size) if self.step_idx >= self.max_steps_per_episode else done_batches

        self.batch = next_batch

        return next_batch, reward_batches, done_batches, torch.tensor(), {}