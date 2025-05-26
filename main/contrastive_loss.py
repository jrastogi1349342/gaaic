import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import random

# # Config
# PATCH_SIZE = 32
# NUM_PATCHES = 50
# K_CLUSTERS = 5
# CONFIDENCE_THRESHOLD = 1.5
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# def extract_random_patches(batch_images, patch_size=32, num_patches=50):
#     B, C, H, W = batch_images.shape
#     patches = []
#     coords = []
#     for b in range(B):
#         for _ in range(num_patches):
#             y = random.randint(0, H - patch_size)
#             x = random.randint(0, W - patch_size)
#             patch = batch_images[b, :, y:y + patch_size, x:x + patch_size]
#             patches.append(patch)
#             coords.append([x / W, y / H])  # normalized
#     patches = torch.stack(patches).to(DEVICE)
#     coords = torch.tensor(coords, dtype=torch.float32).to(DEVICE)
#     return patches, coords

# # Feature Extractor
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = models.resnet18(pretrained=True)
#         self.encoder = nn.Sequential(*list(resnet.children())[:-2])
#     def forward(self, x):
#         with torch.no_grad():
#             feat_map = self.encoder(x)
#         return F.adaptive_avg_pool2d(feat_map, 1).squeeze(-1).squeeze(-1)

# # Compactness Heuristic
# def find_compact_clusters(labels, coords, k):
#     compactness = []
#     for i in range(k):
#         cluster_coords = coords[labels == i]
#         if len(cluster_coords) < 3:
#             continue
#         spread = torch.norm(cluster_coords - cluster_coords.mean(dim=0), dim=1).mean()
#         compactness.append((i, spread.item()))
#     sorted_clusters = sorted(compactness, key=lambda x: x[1])
#     return [i for i, _ in sorted_clusters[:2]]  # Top 2 compact clusters

# # Contrastive Loss
# def simple_contrastive_loss(pos_feats, neg_feats, temperature=0.2):
#     pos_feats = F.normalize(pos_feats, dim=1)
#     neg_feats = F.normalize(neg_feats, dim=1)
#     pos_sim = torch.matmul(pos_feats, pos_feats.T)
#     neg_sim = torch.matmul(pos_feats, neg_feats.T)
#     pos_mask = torch.eye(pos_feats.size(0), device=pos_feats.device)
#     pos_sim = pos_sim * (1 - pos_mask)
#     pos_exp = torch.exp(pos_sim / temperature).sum(dim=1)
#     neg_exp = torch.exp(neg_sim / temperature).sum(dim=1)
#     loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-6))
#     return loss.mean()

# # Main Pipeline
# def patch_clustering_contrastive(
#     batch_images,
#     max_retries=3,
#     patch_size=32,
#     num_patches=50,
#     k_clusters=5,
#     confidence_threshold=1.5,
#     min_patches_required=10
# ):
#     model = FeatureExtractor().to(DEVICE).eval()

#     for attempt in range(max_retries):
#         patches, coords = extract_random_patches(batch_images, patch_size, num_patches)
#         feats = model(patches)
#         feats_with_coords = torch.cat([feats, coords], dim=1).cpu().numpy()

#         kmeans = KMeans(n_clusters=k_clusters).fit(feats_with_coords)
#         labels = kmeans.labels_
#         dists = pairwise_distances(feats_with_coords, kmeans.cluster_centers_)
#         confidence = dists.min(axis=1)
#         confident_idx = confidence < confidence_threshold

#         if confident_idx.sum() < min_patches_required:
#             print(f"[Retry {attempt+1}/{max_retries}] Too few confident patches ({confident_idx.sum()}). Retrying...")
#             continue  # Try again

#         # Filtered patches
#         feats = feats[confident_idx]
#         coords = coords[confident_idx]
#         labels = torch.tensor(labels[confident_idx], device=feats.device)

#         fg_clusters = find_compact_clusters(labels, coords, k_clusters)
#         fg_mask = torch.zeros_like(labels, dtype=torch.bool)
#         for fg in fg_clusters:
#             fg_mask |= (labels == fg)

#         if fg_mask.sum() < 5 or (~fg_mask).sum() < 5:
#             print(f"[Retry {attempt+1}/{max_retries}] Too few FG/BG patches. Retrying...")
#             continue

#         # Contrastive loss
#         fg_feats = feats[fg_mask]
#         bg_feats = feats[~fg_mask]
#         loss = simple_contrastive_loss(fg_feats, bg_feats)
#         return loss

#     # If all retries fail
#     print("All retries failed. Skipping batch.")
#     return torch.tensor(0.0, requires_grad=True, device=DEVICE)


def get_coords(H, W, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).view(-1, 2)  # [H*W, 2]
    return coords  # [H*W, 2]

def sample_indices_with_dist(prob_map, coords, k, min_dist, oversample_factor=5):
    """
    Vectorized: sample up to k indices from prob_map [H*W] with minimum distance filtering.
    """
    num_samples = k * oversample_factor
    prob_map = torch.clamp(prob_map, min=0.0)
    prob_map = prob_map / (prob_map.sum() + 1e-8)

    sampled_idx = torch.multinomial(prob_map, num_samples, replacement=False)
    sampled_coords = coords[sampled_idx]  # [num_samples, 2]

    selected = []
    for i in range(sampled_coords.size(0)):
        if len(selected) == 0:
            selected.append(i)
            continue
        dists = (sampled_coords[i] - sampled_coords[selected]).float().norm(dim=1)
        if (dists >= min_dist).all():
            selected.append(i)
        if len(selected) >= k:
            break

    return sampled_idx[selected] if len(selected) > 0 else sampled_idx[:k]

def sample_pos_neg_patches(image, gate_mask, patch_size=32, k=5, min_dist=10, temperature=0.1):
    B, C, H, W = image.shape
    pad = patch_size // 2
    padded_img = F.pad(image, (pad, pad, pad, pad), mode='reflect')  # [B, C, H+2P, W+2P]
    coords = get_coords(H, W, device=image.device)

    patches_pos, patches_neg = [], []

    for b in range(B):
        # Flatten saliency and normalize
        sal_flat = gate_mask[b].view(-1)
        prob_pos = F.softmax(sal_flat / temperature, dim=0)
        prob_neg = 1.0 - sal_flat
        prob_neg = prob_neg / (prob_neg.sum() + 1e-8)

        idx_pos = sample_indices_with_dist(prob_pos, coords, k, min_dist)
        idx_neg = sample_indices_with_dist(prob_neg, coords, k, min_dist)

        for idx in idx_pos:
            y, x = coords[idx]
            y, x = y + pad, x + pad
            patches_pos.append(padded_img[b:b+1, :, y - pad:y + pad, x - pad:x + pad])

        for idx in idx_neg:
            y, x = coords[idx]
            y, x = y + pad, x + pad
            patches_neg.append(padded_img[b:b+1, :, y - pad:y + pad, x - pad:x + pad])

    return torch.cat(patches_pos, dim=0), torch.cat(patches_neg, dim=0)

class ResNetProjectionHead(torch.nn.Module):
    def __init__(self, output_dim=64, device="cuda"):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)  # Remove fc
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        ).to(device)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.encoder(x).flatten(1)  # [B, 512]
        x = F.normalize(self.proj(x), dim=1)  # [B, output_dim]
        return x
    
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Compute NT-Xent contrastive loss between embeddings z_i and z_j.
    z_i, z_j: [N, D], normalized embeddings
    """
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
    # Mask out similarity with self
    mask = (~torch.eye(sim.shape[0], dtype=torch.bool, device=z.device)).float()

    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Positive pairs are (i, i+N) and (i+N, i)
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(pos_sim / denom.squeeze())
    return loss.mean()

# Example usage in training loop:
def contrastive_saliency_loss(image, gate_mask, resnet_proj, patch_size=32, k=5, min_dist=32, temperature=0.1):
    patches_pos, patches_neg = sample_pos_neg_patches(
        image=image,
        gate_mask=gate_mask,
        patch_size=patch_size,
        k=k,
        min_dist=min_dist,
        temperature=temperature
    )

    z_pos = resnet_proj(patches_pos)  # [B*k, D]
    z_neg = resnet_proj(patches_neg)  # [B*k, D]

    return nt_xent_loss(z_pos, z_neg, temperature)