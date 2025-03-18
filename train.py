import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import random
from collections import deque
from torch.cuda.amp import autocast, GradScaler

from encoder import CURL
from sac import SAC
from replay_buffer import PrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
curl = CURL(pretrained=True, latent_dim=latent_dim)
sac = SAC(latent_dim=latent_dim, device=device)

num_epochs = 1000
batch_size = 64

# TODO data loader

