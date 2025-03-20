import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from encoder import Encoder

import utils

class PerturbationModel(nn.Module): 
    def __init__(self, latent_dim, l_inf_norm = 0.05, l2_norm=0.1, device="cuda"): 
        super().__init__()

        self.l_inf_norm = l_inf_norm
        self.l2_norm = l2_norm
        self.device = device

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
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 15, 15)
        x = self.deconv(x) * self.l_inf_norm

        return self.bound_l2(x)
    
    # Bound L2 norm
    def bound_l2(self, perturbation): 
        norm = torch.norm(perturbation.view(perturbation.shape[0], -1), dim=1, keepdim=True).clamp(min=1e-6)
        factor = torch.clamp(self.l2_norm / norm, max=1.0)
        return perturbation * factor.view(-1, 1, 1, 1)

class SAC(nn.Module): 
    def __init__(self, latent_dim, replay_buffer, curl, target_alpha=0.3, device="cuda"):
        super().__init__()

        self.device = device
        self.replay_buffer = replay_buffer
        self.curl = curl

        self.actor = PerturbationModel(latent_dim, device=device)
        print(f"With Perturbation Model: {utils.get_mem_used()} MB")

        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=3e-4)
        
        self.critic_one = self.create_critic(latent_dim)
        print(f"With Critic 1: {utils.get_mem_used()} MB")

        self.critic_one_opt = optim.AdamW(self.critic_one.parameters(), lr=3e-4)
        
        self.critic_two = self.create_critic(latent_dim)
        print(f"With Critic 2: {utils.get_mem_used()} MB")

        self.critic_two_opt = optim.AdamW(self.critic_two.parameters(), lr=3e-4)

        # Log of entropy: start at ln(0.0) = 1 (exploration) and reduce gradually to target 
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.target_alpha = target_alpha
        self.alpha_opt = optim.AdamW([self.log_alpha], lr=3e-4)

        self.scaler = GradScaler("cuda") # Mixed precision training

    # TODO rethink this model: use conv, pooling layers instead of linear
    # Take inspiration from encoders/unets

    # Input: s and a, output: Q(s, a)
    # s: latent dim x 1 vector; a: 480 * 480 * 3 image --> first pass through resnet encoder --> latent dim x 1 vector
    # s, a concat
    def create_critic(self, latent_dim): 
        return nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim / 2), 
            nn.ReLU(inplace=True), 
            nn.Linear(latent_dim / 2, latent_dim / 8), 
            nn.ReLU(inplace=True), 
            nn.Linear(latent_dim / 8, 1)
        ).half().to(self.device)

    def select_action(self, state): 
        return self.actor(state)

    def update(self, batch_size=64, gamma=0.99): 
        if len(self.replay_buffer) < batch_size: 
            return
        
        batch = self.replay_buffer.sample(batch_size)
        s, a, r, s_prime, dones = zip(*batch)

        s = torch.stack(s).to(self.device)
        a = torch.stack(a).to(self.device)
        r = torch.stack(r, dtype=torch.float32).to(self.device).unsqueeze(1)
        s_prime = torch.stack(s_prime).to(self.device)
        dones = torch.stack(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        with autocast(self.device): 
            # TODO ensure a is latent dim x 1 vector here, instead of full image
            sa = torch.cat([s, a], dim=1)

            # Q(s, a) for each critic network for current state
            q_one = self.critic_one(sa)
            q_two = self.critic_two(sa)

            s_prime_a = torch.cat([s_prime, self.actor(s_prime)], dim=1)

            # Q(s, a) for each critic network for next state
            q_one_prime = self.critic_one(s_prime_a).detach()
            q_two_prime = self.critic_two(s_prime_a).detach()

            fixed_pt_q = r + gamma * (1 - dones) * torch.min(q_one_prime, q_two_prime)

            priority = max((q_one - fixed_pt_q).abs().mean().item(), (q_two - fixed_pt_q).abs().mean().item())

            loss_critic_one = F.mse_loss(q_one, fixed_pt_q)
            loss_critic_two = F.mse_loss(q_two, fixed_pt_q)

            alpha = self.log_alpha.exp()

            # TODO verify this equation for actor loss
            actor_loss = -self.critic_one(torch.cat([s, self.actor(s)], dim=1)).mean() + alpha * torch.mean(a)

            alpha_loss = -self.log_alpha * (self.target_alpha + torch.mean(a).detach())

        self.critic_one_opt.zero_grad()
        self.scaler.scale(loss_critic_one).backward()
        self.scaler.step(self.critic_one_opt)

        self.critic_two_opt.zero_grad()
        self.scaler.scale(loss_critic_two).backward()
        self.scaler.step(self.critic_two_opt)

        self.actor_opt.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_opt)

        self.alpha_opt.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_opt)

        self.scaler.update()

        for i in range(batch_size): 
            self.replay_buffer.add((s[i], a[i], r[i], s_prime[i], dones[i]), priority)