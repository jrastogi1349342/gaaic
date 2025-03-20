import torch
import numpy as np

from ultralytics import YOLO

from encoder import CURL
from sac import SAC
from replay_buffer import PrioritizedReplayBuffer
from dataloader import train_dataloader
from rewards import calc_rewards
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256

replay_buffer = PrioritizedReplayBuffer()

curl = CURL(pretrained=True, latent_dim=latent_dim, device=device)

print(f"Curl: {utils.get_mem_used()} MB")

sac = SAC(latent_dim=latent_dim, replay_buffer=replay_buffer, curl=curl, device=device)

print(f"With SAC: {utils.get_mem_used()} MB")

obj_detector = YOLO("yolov11n.pt")

print(f"With YOLO: {utils.get_mem_used()} MB")

num_epochs = 1
batch_size = 8

train_data = train_dataloader()

for epoch in range(num_epochs): 
    sac.train()

    for batch_idx, images in enumerate(train_data): 
        images = images.to(device)
        
        s = curl(images)
        a = sac.select_action(s)

        perturbed_imgs = torch.clamp(images + a, 0, 1)

        s_prime = curl(perturbed_imgs) 
        r = calc_rewards(s, s_prime, "empty", obj_detector) 
        dones = [False] * len(images) 

        for i in range(len(images)): 
            sac.replay_buffer.add((s[i], a[i], r[i], s_prime[i], dones[i]), np.abs(r[i]))

        sac.update()

    print(f"Avg reward: {np.mean(r)}")

    pass