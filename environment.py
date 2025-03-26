import torch
import gymnasium as gym
import numpy as np


class DataloaderEnv(gym.Env): 
    def __init__(self, dataloader, max_steps_per_episode=1000, batch_size=8, action_shape=(480, 480, 3)): 
        super().__init__

        self.dataloader = iter(dataloader)
        self.action_shape = action_shape
        self.state = None
        self.batch_size = batch_size 
        self.max_steps_per_episode = max_steps_per_episode
        self.step_ct = 0 # num steps in current episode

        # TODO figure this out
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(batch_size, *dataloader.dataset[0][0].shape), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(batch_size, *action_shape), dtype=np.float32)

    def reset(self): 
        try:
            self.state = next(self.dataloader)  
        except StopIteration:
            self.dataloader = iter(self.dataloader)  # restart dataLoader
            self.state = next(self.dataloader)

        self.step_ct = 0
        return self.state
    
    def step(self, action): 
        orig_states = self.state

        # TODO implement reward function
        # reward_batches

        # TODO finish/implement episodes of > 1 step
        try:
            self.state = next(self.dataloader)  
        except StopIteration:
            self.dataloader = iter(self.dataloader)  # restart dataLoader
            self.state = next(self.dataloader)





        
        pass

    pass