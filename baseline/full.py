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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader
from environment import DataloaderEnv

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
args = parser.parse_args()

# To run: python3 baseline/full.py

class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"): 
        super().__init__()
        
        # TODO fix code for if weights are not pretrained
        # 166 MB VRAM
        model_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(model_base.children())[:-1]).half().to(device) # remove head
        self.fc = nn.Linear(512, latent_dim).half().to(device)

        for param in self.encoder.parameters(): 
            param.requires_grad = False

        for module in self.encoder.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

        torch.nn.utils.clip_grad_norm_(self.fc.parameters(), max_norm=1.0)

    def forward(self, x): 
        assert torch.min(x).positive() 
        assert not torch.isnan(x).any(), "State contains NaNs!"
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        assert not torch.isnan(x).any(), "NaN detected after encoder!"
        assert not torch.isinf(x).any(), "Inf detected after encoder!"
        print("Min of latent embedding:", torch.min(x), "Max of latent embedding:", torch.max(x), "Nan:", torch.any(torch.isnan(x)), "Inf:", torch.any(torch.isinf(x)))
        print("Encoder fc:", self.fc.weight)
        return self.fc(x)

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

        # Gaussian distribution in latent space
        self.mu = nn.Linear(latent_dim, latent_dim).half().to(device)
        self.log_std = nn.Linear(latent_dim, latent_dim).half().to(device)
        self.low_rank_factor = nn.Linear(latent_dim, latent_dim * low_rank).half().to(device)

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
        print("Mu:", self.mu.weight)
        print("Log std:", self.log_std.weight)
        print("Low rank factor:", self.low_rank_factor.weight)

        mus = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-10, max=2) # prevent exploding values
        std = torch.exp(log_std)
        # print(mus.shape, log_std.shape, std.shape)

        diag_cov = torch.diag_embed(std ** 2)

        # print(x.shape) 
        # print(self.low_rank_factor(x).shape)
        U = self.low_rank_factor(x).view(self.batch_size, self.latent_dim, self.low_rank)
        low_rank_cov = torch.matmul(U, U.transpose(-1, -2)) # PSD
        # print(diag_cov.shape, U.shape, low_rank_cov.shape)

        cov_mtxs = diag_cov + low_rank_cov + torch.eye(self.latent_dim, device=x.device) * 1e-3 # for stability

        return mus, cov_mtxs
    
    def decode(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 15, 15)
        x = self.deconv(x)

        # Alternative
        # return self.deconv(x.unsqueeze(-1).unsqueeze(-1))  # Add dimensions for ConvTranspose2d

        return x
    
    # Bound L2 norm
    def bound_l2(self, perturbation): 
        norm = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True).clamp(min=1e-6)
        factor = torch.clamp(self.l2_norm / norm, max=1.0)
        return perturbation * factor.view(-1, 1, 1, 1)


class Actor_Critic(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, encoder, latent_dim, batch_size, replay_buffer=None, curl=None, target_alpha=0.3, device="cuda"):
        super(Actor_Critic, self).__init__()
        self.feature_encoder = encoder
        self.device = device
        self.batch_size = batch_size

        self.actor = PerturbationModel(latent_dim, device=device, batch_size=batch_size)

        # Input: concatenated state and action latent vectors
        self.critic = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim // 2), 
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 2, latent_dim // 8), 
            nn.BatchNorm1d(latent_dim // 8),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim // 8, 1)
        ).half().to(self.device)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    # For actor
    def forward(self, x):
        # Latent vector for image
        x = self.feature_encoder(x)

        mus, cov_mtxs = self.actor(x)

        return x, mus, cov_mtxs

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
batch_size = 8
num_episodes = 5
encoder = Encoder(latent_dim=latent_dim, device=device)
model = Actor_Critic(encoder=encoder, latent_dim=latent_dim, device=device, batch_size=batch_size)
print(model)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

train_data = train_dataloader(batch_size=batch_size)
obj_detector = YOLO("yolo11n.pt").to(device).eval()

env = DataloaderEnv(train_data, obj_detector=obj_detector, batch_size=batch_size)

def select_action(state):
    latent_state, mus, cov_mtxs = model(state)
    print("Latent state:", latent_state)
    print("Mus:", mus)
    print("Covs:", cov_mtxs)

    dist = torch.distributions.MultivariateNormal(mus.to(torch.float32), covariance_matrix=cov_mtxs.to(torch.float32))

    actions_latent = dist.rsample().half()

    values = model.critic(torch.cat([latent_state, actions_latent], dim=1))

    # save to action buffer, MultivariateNormal automatically sums
    model.saved_actions.append(SavedAction(dist.log_prob(actions_latent), values))

    return model.actor.decode(actions_latent).squeeze(0)

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns_lst = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns_lst.insert(0, R)

    # print(type(returns_lst[0]))

    returns = torch.stack(returns_lst)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # print("Returns", returns)

    for (log_prob, value), R in zip(saved_actions, returns):
        # print("Log prob", log_prob)
        # print("Val", value)
        # print("Reward", R)
        advantage = R - value

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, R.unsqueeze(dim=1)))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print("Loss", loss)

    # perform backprop
    loss.backward()
    optimizer.step()

    for param in model.parameters():
        if param.grad is not None:
            print("Param", torch.any(torch.isnan(param.grad)), torch.any(torch.isinf(param.grad)))

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

# ------------------TODO adapt for my environment and train---------------------------

def main():
    running_reward = 0
    model.train()

    for i_episode in range(num_episodes):

        # reset environment and episode reward
        states = env.reset()
        ep_reward = 0

        print("Next episode")
        print(states.dtype)

        for t in range(1, env.max_steps_per_episode):
            print("Type:", states.dtype)

            # select action from policy
            action = select_action(states)

            # take the action
            states, rewards, dones, _, _ = env.step(action)

            model.rewards.append(rewards)
            ep_reward += rewards.sum()
            print(dones, torch.all(dones))
            if torch.all(dones):
                break

            print("Next step")


        # Alpha = 0.05
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))


if __name__ == '__main__':
    main()