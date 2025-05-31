import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import random

# # def get_coords(H, W, device):
# #     yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
# #     coords = torch.stack([yy, xx], dim=-1).view(-1, 2)  # [H*W, 2]
# #     return coords  # [H*W, 2]

# # def sample_indices_with_dist(prob_map, coords, k, min_dist, oversample_factor=5):
# #     """
# #     Vectorized: sample up to k indices from prob_map [H*W] with minimum distance filtering.
# #     """
# #     num_samples = k * oversample_factor
# #     prob_map = torch.clamp(prob_map, min=0.0)
# #     prob_map = prob_map / (prob_map.sum() + 1e-8)

# #     sampled_idx = torch.multinomial(prob_map, num_samples, replacement=False)
# #     sampled_coords = coords[sampled_idx]  # [num_samples, 2]

# #     selected = []
# #     for i in range(sampled_coords.size(0)):
# #         if len(selected) == 0:
# #             selected.append(i)
# #             continue
# #         dists = (sampled_coords[i] - sampled_coords[selected]).float().norm(dim=1)
# #         if (dists >= min_dist).all():
# #             selected.append(i)
# #         if len(selected) >= k:
# #             break

# #     return sampled_idx[selected] if len(selected) > 0 else sampled_idx[:k]

# # def sample_pos_neg_patches(image, gate_mask, patch_size=32, k=5, min_dist=10, temperature=0.1):
# #     B, C, H, W = image.shape
# #     pad = patch_size // 2
# #     padded_img = F.pad(image, (pad, pad, pad, pad), mode='reflect')  # [B, C, H+2P, W+2P]
# #     coords = get_coords(H, W, device=image.device)

# #     patches_pos, patches_neg = [], []

# #     for b in range(B):
# #         # Flatten saliency and normalize
# #         sal_flat = gate_mask[b].view(-1)
# #         prob_pos = F.softmax(sal_flat / temperature, dim=0)
# #         prob_neg = 1.0 - sal_flat
# #         prob_neg = prob_neg / (prob_neg.sum() + 1e-8)

# #         idx_pos = sample_indices_with_dist(prob_pos, coords, k, min_dist)
# #         idx_neg = sample_indices_with_dist(prob_neg, coords, k, min_dist)

# #         for idx in idx_pos:
# #             y, x = coords[idx]
# #             y, x = y + pad, x + pad
# #             patches_pos.append(padded_img[b:b+1, :, y - pad:y + pad, x - pad:x + pad])

# #         for idx in idx_neg:
# #             y, x = coords[idx]
# #             y, x = y + pad, x + pad
# #             patches_neg.append(padded_img[b:b+1, :, y - pad:y + pad, x - pad:x + pad])

# #     return torch.cat(patches_pos, dim=0), torch.cat(patches_neg, dim=0)

# class ResNetProjectionHead(torch.nn.Module):
#     def __init__(self, output_dim=64, device="cuda"):
#         super().__init__()
#         resnet = models.resnet18(pretrained=True)
#         self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)  # Remove fc
#         self.proj = torch.nn.Sequential(
#             torch.nn.Linear(512, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, output_dim)
#         ).to(device)

#     def forward(self, x):  # x: [B, C, H, W]
#         x = self.encoder(x).flatten(1)  # [B, 512]
#         x = F.normalize(self.proj(x), dim=1)  # [B, output_dim]
#         return x
    
# def nt_xent_loss(z_i, z_j, temperature=0.5):
#     """
#     Compute NT-Xent contrastive loss between embeddings z_i and z_j.
#     z_i, z_j: [N, D], normalized embeddings
#     """
#     N = z_i.shape[0]
#     z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

#     sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
#     # Mask out similarity with self
#     mask = (~torch.eye(sim.shape[0], dtype=torch.bool, device=z.device)).float()

#     exp_sim = torch.exp(sim) * mask
#     denom = exp_sim.sum(dim=1, keepdim=True)

#     # Positive pairs are (i, i+N) and (i+N, i)
#     pos_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)
#     pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

#     loss = -torch.log(pos_sim / denom.squeeze())
#     return loss.mean()

# # # Example usage in training loop:
# # def contrastive_saliency_loss(image, gate_mask, resnet_proj, patch_size=32, k=5, min_dist=32, temperature=0.1):
# #     patches_pos, patches_neg = sample_pos_neg_patches(
# #         image=image,
# #         gate_mask=gate_mask,
# #         patch_size=patch_size,
# #         k=k,
# #         min_dist=min_dist,
# #         temperature=temperature
# #     )

# #     z_pos = resnet_proj(patches_pos)  # [B*k, D]
# #     z_neg = resnet_proj(patches_neg)  # [B*k, D]

# #     return nt_xent_loss(z_pos, z_neg, temperature)


# def batched_sample_coords_with_min_dist(prob_map, k, min_dist, oversample_factor=5, temperature=0.1):
#     B, H, W = prob_map.shape
#     device = prob_map.device
#     total_pixels = H * W
#     num_candidates = min(k * oversample_factor, total_pixels)

#     prob_flat = prob_map.view(B, -1)
#     prob_flat = F.softmax(prob_flat / temperature, dim=1)

#     candidate_indices = torch.multinomial(prob_flat, num_candidates, replacement=False)

#     y_coords, x_coords = torch.meshgrid(
#         torch.arange(H, device=device),
#         torch.arange(W, device=device),
#         indexing="ij"
#     )
#     all_coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)

#     candidate_coords = all_coords[candidate_indices]  # [B, num_candidates, 2]
#     selected_coords = torch.zeros(B, k, 2, dtype=torch.long, device=device)

#     for b in range(B):
#         chosen = []
#         for c in candidate_coords[b]:
#             if len(chosen) == 0:
#                 chosen.append(c)
#                 continue
#             dists = torch.norm(torch.stack(chosen).float() - c.float(), dim=1)
#             if torch.all(dists >= min_dist):
#                 chosen.append(c)
#             if len(chosen) == k:
#                 break
#         while len(chosen) < k:
#             for c in candidate_coords[b]:
#                 if not any(torch.equal(c, cc) for cc in chosen):
#                     chosen.append(c)
#                     if len(chosen) == k:
#                         break
#         selected_coords[b] = torch.stack(chosen[:k])
#     return selected_coords  # [B, k, 2]

# # ---- Patch Extraction ----

# def extract_patches_at_coords(image, coords, patch_size=32):
#     B, C, H, W = image.shape
#     k = coords.shape[1]
#     pad = patch_size // 2
#     padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
#     patches = []

#     for b in range(B):
#         for i in range(k):
#             y, x = coords[b, i]
#             y += pad
#             x += pad
#             patch = padded[b:b+1, :, y-pad:y+pad, x-pad:x+pad]  # [1, C, P, P]
#             patches.append(patch)

#     return torch.cat(patches, dim=0)  # [B*k, C, P, P]

# # ---- Saliency Contrastive Loss V2 ----

# def contrastive_saliency_loss_v2(
#     image,
#     perturbed_image,
#     gate_mask,  # [B, 1, H, W]
#     resnet_proj,
#     patch_size=32,
#     k=5,
#     min_dist=16,
#     temperature=0.1
# ):
#     gate_mask = gate_mask.clamp(min=0)
#     prob_pos = gate_mask.squeeze(1)  # [B, H, W]
#     prob_neg = 1.0 - prob_pos
#     prob_neg /= prob_neg.sum(dim=(1, 2), keepdim=True) + 1e-6

#     coords_pos = batched_sample_coords_with_min_dist(prob_pos, k, min_dist, temperature=temperature)
#     coords_neg = batched_sample_coords_with_min_dist(prob_neg, k, min_dist, temperature=temperature)

#     patches_orig_pos = extract_patches_at_coords(image, coords_pos, patch_size)
#     patches_pert_pos = extract_patches_at_coords(perturbed_image, coords_pos, patch_size)

#     patches_orig_neg = extract_patches_at_coords(image, coords_neg, patch_size)
#     patches_pert_neg = extract_patches_at_coords(perturbed_image, coords_neg, patch_size)

#     # Encode patches
#     z_orig_pos = resnet_proj(patches_orig_pos)
#     z_pert_pos = resnet_proj(patches_pert_pos)

#     z_orig_neg = resnet_proj(patches_orig_neg)
#     z_pert_neg = resnet_proj(patches_pert_neg)

#     # Positives: same (x,y) from image & perturbed
#     loss_pos = nt_xent_loss(z_orig_pos, z_pert_pos, temperature)

#     # Negatives: random low-saliency patches across image & perturbed
#     loss_neg = nt_xent_loss(z_orig_neg, z_pert_neg, temperature)

#     return loss_pos + loss_neg


def attention_contrastive_loss(orig, pert, saliency_map, resnet, downsampler, patch_size=16, k=10, temperature=0.2):
    B, _, H, W = orig.shape
    device = orig.device

    # Normalize and flatten saliency
    saliency_map = saliency_map.clamp(min=0)
    saliency = saliency_map.view(B, -1)
    saliency = saliency / (saliency.sum(dim=1, keepdim=True) + 1e-8)

    # Sample k locations from high-saliency
    idx = torch.multinomial(saliency, k, replacement=False)  # [B, k]

    patches_orig = []
    patches_pert = []

    pad = patch_size // 2
    orig_pad = F.pad(orig, (pad, pad, pad, pad), mode='reflect')
    pert_pad = F.pad(pert, (pad, pad, pad, pad), mode='reflect')

    for b in range(B):
        coords = idx[b]
        for c in coords:
            y, x = c // W, c % W
            y, x = y + pad, x + pad
            patches_orig.append(orig_pad[b:b+1, :, y-pad:y+pad, x-pad:x+pad])
            patches_pert.append(pert_pad[b:b+1, :, y-pad:y+pad, x-pad:x+pad])

    patches_orig = torch.cat(patches_orig, dim=0)
    patches_pert = torch.cat(patches_pert, dim=0)

    _, z_orig = resnet(patches_orig)
    _, z_pert = resnet(patches_pert)

    z_orig = downsampler(z_orig)
    z_pert = downsampler(z_pert)

    # Use full attention-based loss (cosine sim matrix)
    z = torch.cat([z_orig, z_pert], dim=0)  # [2N, D]
    sim = torch.matmul(z, z.T) / temperature
    mask = ~torch.eye(len(z), dtype=torch.bool, device=device)
    sim = sim.masked_fill(~mask, -1e9)

    labels = torch.arange(len(z_orig), device=device)
    labels = torch.cat([labels + len(z_orig), labels], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss