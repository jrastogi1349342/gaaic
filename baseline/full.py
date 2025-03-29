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

    def forward(self, x): 
        assert torch.min(x).positive() 
        assert not torch.isnan(x).any(), "State contains NaNs in Encoder!"
        x = self.encoder(x).squeeze(-1).squeeze(-1)
        assert not torch.isnan(x).any(), "NaN detected after Encoder encoder!"
        assert not torch.isinf(x).any(), "Inf detected after Encoder encoder!"
        # print("Min of latent embedding:", torch.min(x), "Max of latent embedding:", torch.max(x), "Nan:", torch.any(torch.isnan(x)), "Inf:", torch.any(torch.isinf(x)))
        # print("Encoder fc:", self.fc.weight)
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
        # print("Mu min:", torch.min(self.mu.weight), "Max:", torch.max(self.mu.weight), "Nan:", torch.any(torch.isnan(self.mu.weight)), "Inf:", torch.any(torch.isinf(self.mu.weight)))
        # print("Log std min:", torch.min(self.log_std.weight), "Max:", torch.max(self.log_std.weight), "Nan:", torch.any(torch.isnan(self.log_std.weight)), "Inf:", torch.any(torch.isinf(self.log_std.weight)))
        # print("Low rank factor min:", torch.min(self.low_rank_factor.weight), "Max:", torch.max(self.low_rank_factor.weight), "Nan:", torch.any(torch.isnan(self.low_rank_factor.weight)), "Inf:", torch.any(torch.isinf(self.low_rank_factor.weight)))
        # print("x min:", torch.min(x), "Max:", torch.max(x), "Nan:", torch.any(torch.isnan(x)), "Inf:", torch.any(torch.isinf(x)))

        assert not torch.isnan(x).any(), "NaN detected Perturbation Model 1!"
        assert not torch.isinf(x).any(), "Inf detected Perturbation Model 1!"

        mus = self.mu(x).float()
        log_std = self.log_std(x).float()
        log_std = torch.clamp(log_std, min=-5, max=2) # prevent exploding values
        std = torch.exp(log_std)
        print(f"Std min: {std.min()}, max: {std.max()}")
        # print(mus.shape, log_std.shape, std.shape)
        assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 2!"
        assert not torch.isinf(log_std).any(), "Inf detected Perturbation Model 2!"
        assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 3!"
        assert not torch.isinf(log_std).any(), "Inf detected Perturbation Model 3!"
        assert not torch.isnan(mus).any(), "NaN detected Perturbation Model 4!"
        assert not torch.isinf(log_std).any(), "Inf detected Perturbation Model 4!"

        diag_cov = torch.diag_embed(std ** 2 + 1e-2)

        # print(x.shape) 
        # print(self.low_rank_factor(x).shape)
        U = self.low_rank_factor(x).view(self.batch_size, self.latent_dim, self.low_rank).float()
        low_rank_cov = torch.matmul((U * 15), (U * 15).transpose(-1, -2)) # PSD

        ranks = torch.linalg.matrix_rank(U)
        print(f"Ranks of U: {ranks}\tDesired rank: {self.low_rank}")
        # print(diag_cov.shape, U.shape, low_rank_cov.shape)

        print(f"Mean norm of U: {torch.norm(U, dim=[-2, -1]).mean().item()}")

        cov_mtxs = diag_cov + low_rank_cov + torch.eye(self.latent_dim, device=x.device) * 2e-1 # for stability
        cov_mtxs *= 0.1 # For stability
        print(f"Covs min: {cov_mtxs.min()}, max: {cov_mtxs.max()}, type: {cov_mtxs.dtype}")

        ranks = torch.linalg.matrix_rank(cov_mtxs)
        print(f"Ranks of Cov mtxs: {ranks}\tDesired rank: {self.latent_dim}")

        eigenvalues = torch.linalg.eigvalsh(cov_mtxs) 
        print(f"Min eigenvalues: {eigenvalues.min(dim=-1)[0]}")  # Get the minimum eigenvalue per batch
        assert torch.all(eigenvalues >= 0), "Some covariance matrices are not PSD!"

        print(f"Trace of covariance: {cov_mtxs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)}")

        logdet = torch.slogdet(cov_mtxs)
        print(f"Sign of determinant of covariance matrices: {logdet.sign}\tLog of det: {logdet.logabsdet}")

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
latent_dim = 128
batch_size = 2
num_episodes = 5
encoder = Encoder(latent_dim=latent_dim, device=device)
model = Actor_Critic(encoder=encoder, latent_dim=latent_dim, device=device, batch_size=batch_size)
# print(model)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
eps = np.finfo(np.float32).eps.item()

train_data = train_dataloader(batch_size=batch_size)
obj_detector = YOLO("yolo11n.pt").to(device).eval()

env = DataloaderEnv(train_data, obj_detector=obj_detector, batch_size=batch_size)

def select_action(state):
    latent_state, mus, cov_mtxs = model(state)

    # mus = mus.to(torch.float32)
    # cov_mtxs = cov_mtxs.to(torch.float32)
    # print("Latent state:", latent_state)
    # print("Mus:", mus)
    # print("Covs:", cov_mtxs)

    dist = torch.distributions.MultivariateNormal(mus, covariance_matrix=cov_mtxs)

    actions_latent = dist.rsample()
    actions_latent = torch.tanh(actions_latent)

    log_prob = dist.log_prob(actions_latent)
    log_prob = torch.tanh(log_prob / 100) * 100

    print(f"Latent action min: {actions_latent.min()}\tmax: {actions_latent.max()}\tLog prob: {log_prob}")
    print(latent_state.dtype, mus.dtype, cov_mtxs.dtype, actions_latent.dtype)

    diff = (actions_latent - mus).abs().mean()
    print(f"Mean absolute difference from mean: {diff}")

    half_actions_latent = actions_latent.half()
    combined = torch.cat([latent_state, half_actions_latent], dim=1)
    # print(half_actions_latent.dtype, combined.dtype)
    values = model.critic(combined)

    # save to action buffer, MultivariateNormal automatically sums
    model.saved_actions.append(SavedAction(log_prob, values))

    return model.actor.decode(half_actions_latent).squeeze(0)

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
        advantage = torch.clamp(R - value, -10, 10)
        print(f"R: {R}\tvalue: {value}\tAdv: {advantage}, log prob: {log_prob}")

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, R.unsqueeze(dim=1)))

    # reset gradients
    optimizer.zero_grad()

    print(f"Actor loss: {torch.stack(policy_losses).sum()}\tCritic loss: {torch.stack(value_losses).sum()}")

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # loss /= len(policy_losses)
    print("Loss", loss)

    # perform backprop
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            assert False, f"NaN detected in gradients of {name}"
        if param.grad is not None and torch.isinf(param.grad).any():
            assert False, f"Inf detected in gradients of {name}"
    optimizer.step()

    # for param in model.parameters():
    #     if param.grad is not None:
    #         print("Param", param, ":", torch.any(torch.isnan(param.grad)), torch.any(torch.isinf(param.grad)))

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
        # print(states.dtype)

        for t in range(1, env.max_steps_per_episode):
            # print("Type:", states.dtype)

            # select action from policy
            action = select_action(states)

            # take the action
            states, rewards, dones, _, _ = env.step(action)

            model.rewards.append(rewards)
            ep_reward += rewards.sum()
            # print(dones, torch.all(dones))
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