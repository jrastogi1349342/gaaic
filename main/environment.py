import torch
import gymnasium as gym
import numpy as np
from rewards import calc_rewards
from dataloader import denormalize_batch, renormalize_batch

class DataloaderEnv(gym.Env): 
    def __init__(self, dataloader, obj_detector, idx, latent_dim, device="cuda", max_steps_per_episode=1000, batch_size=8, action_shape=(480, 480, 3)): 
        super().__init__

        self.dataloader = iter(dataloader)
        self.action_shape = action_shape
        self.batch = None
        self.orig_yolo = None
        self.perturbed_yolo = None
        self.next_batch = None
        self.done = None
        self.batch_size = batch_size 
        self.max_steps_per_episode = max_steps_per_episode
        self.step_idx = 0 # num steps in current episode
        self.obj_detector = obj_detector
        self.device = device
        self.info = []
        self.index = idx
        self.latent_dim = latent_dim


        sample_state = next(self.dataloader)[0]
        # print(len(sample_state))
        batch_size, *obs_shape = sample_state.shape

        # print(batch_size)
        # print(obs_shape)
        # print(sample_state.shape)

        # TODO figure this out
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_shape))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=((latent_dim, )))

    def reset(self, **kwargs): 
        try:
            self.batch = next(self.dataloader)[0]  # Get next batch (episode)
        except StopIteration:
            self.dataloader = iter(self.dataloader)  # Restart DataLoader
            self.batch = next(self.dataloader)[0]

        self.step_idx = 0
        return self.batch, {}
    
    # Action: batch_size x 480x480x3 --> not used here because already added to state in full.py
    def step(self, action): 
        orig_states = self.batch

        reward_batches, done_batches = calc_rewards(orig_states, self.next_batch, self.orig_yolo, self.perturbed_yolo, self.step_idx, done=self.done, device=self.device)

        self.step_idx += 1
        self.batch = self.next_batch

        self.orig_yolo = None
        self.perturbed_yolo = None
        self.next_batch = None
        self.done = None

        # print(next_batch.shape, type(next_batch), reward_batches.shape, type(reward_batches), done_batches.shape, type(done_batches))
        # self.info[self.step_idx].append

        return self.batch, reward_batches, done_batches, {}, {}

    def set_results(self, next_batch, orig_yolo, perturbed_yolo, done=None):
        self.next_batch = next_batch.unsqueeze(0)
        self.orig_yolo = orig_yolo
        self.perturbed_yolo = perturbed_yolo
        self.done = done
    
    # def set_result(self, reward, done):
    #     self.reward = reward
    #     self.done = done
