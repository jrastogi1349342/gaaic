import torch
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns

def get_mem_used(): 
    free, total = torch.cuda.mem_get_info("cuda")
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

def count_nan_in_weights(model):
    nan_count = 0
    for param in model.parameters():
        if torch.isnan(param).any():  # Check if there are any NaNs in the parameter
            nan_count += torch.isnan(param).sum().item()
    
    return nan_count

def count_inf_in_weights(model):
    inf_count = 0
    for param in model.parameters():
        if torch.isinf(param).any():  # Check if there are any Infs in the parameter
            inf_count += torch.isinf(param).sum().item()
    
    return inf_count

def display_comp_graph(output, file_name): 
    dot = make_dot(output)
    dot.render(file_name, format="png")

def plot_learning_curve(avg_rewards, freq, file_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label='Deterministic Episode Rewards')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Total Reward')
    plt.title('Average Reward for Deterministic Rollouts (No Exploration)')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def plot_per_step(arr, freq, file_name, name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(arr) + 1), arr, label=name)
    plt.xlabel(f'Gradient Update (x{freq})')
    plt.ylabel('Loss')
    plt.title(f'{name} over Timesteps')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def display_before_after(clean, perturbed, noise, info, gate_mask=None, num_imgs=4):
    batch_size, _, _, _ = clean.shape
    num_rows = 4 if gate_mask != None else 3

    mean_cuda = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
    std_cuda = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")

    # Denormalize images
    noise = noise * std_cuda + mean_cuda
    noise = noise.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    noise = np.clip(noise, 0, 1)

    clean = clean * std_cuda + mean_cuda
    clean = clean.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    clean = np.clip(clean, 0, 1)

    perturbed = perturbed * std_cuda + mean_cuda
    perturbed = perturbed.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    perturbed = np.clip(perturbed, 0, 1)

    if gate_mask.all() != None: 
        gate_mask = gate_mask.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        gate_mask = np.clip(gate_mask, 0, 1)

    # Create figure
    fig, axes = plt.subplots(num_rows, num_imgs, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(num_imgs):
        if i < batch_size:
            axes[i].imshow(clean[i])
            axes[i].axis("off")  # Hide axes
            axes[i].set_title(f"Clean: {info[i]['curr_class']}")

            axes[i + num_imgs].imshow(perturbed[i])
            axes[i + num_imgs].axis("off")
            axes[i + num_imgs].set_title(f"Perturbed: {info[i]['next_class']}")

            axes[i + 2 * num_imgs].imshow(noise[i])
            axes[i + 2 * num_imgs].axis("off") 
            axes[i + 2 * num_imgs].set_title(f"Noise")

            if gate_mask.all() != None: 
                axes[i + 3 * num_imgs].imshow(gate_mask[i], cmap="hot")
                axes[i + 3 * num_imgs].axis("off") 
                axes[i + 3 * num_imgs].set_title(f"Saliency Map")

        else:
            axes[i].set_visible(False)  # Hide empty subplots
            axes[i + num_imgs].set_visible(False)  # Hide empty subplots
            axes[i + 2 * num_imgs].set_visible(False)  # Hide empty subplots

            if gate_mask.all() != None: 
                axes[i + 3 * num_imgs].set_visible(False)  # Hide empty subplots

    plt.tight_layout()
    plt.show()

def display_before_after_gate(clean, perturbed, info, gate_mask, num_imgs=4):
    batch_size, _, _, _ = clean.shape
    num_rows = 3

    mean_cuda = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
    std_cuda = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")

    clean = clean * std_cuda + mean_cuda
    clean = clean.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    clean = np.clip(clean, 0, 1)

    perturbed = perturbed * std_cuda + mean_cuda
    perturbed = perturbed.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    perturbed = np.clip(perturbed, 0, 1)

    gate_mask = gate_mask.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    gate_mask = np.clip(gate_mask, 0, 1)

    # Create figure
    fig, axes = plt.subplots(num_rows, num_imgs, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_imgs):
        if i < batch_size:
            axes[i].imshow(clean[i])
            axes[i].axis("off")  # Hide axes
            axes[i].set_title(f"Clean: {info[i]['curr_class']}")

            axes[i + num_imgs].imshow(perturbed[i])
            axes[i + num_imgs].axis("off")
            axes[i + num_imgs].set_title(f"Perturbed: {info[i]['next_class']}")

            axes[i + 2 * num_imgs].imshow(gate_mask[i], cmap="hot")
            axes[i + 2 * num_imgs].axis("off") 
            axes[i + 2 * num_imgs].set_title(f"Saliency Map")

        else:
            axes[i].set_visible(False)  # Hide empty subplots
            axes[i + num_imgs].set_visible(False)  # Hide empty subplots
            axes[i + 2 * num_imgs].set_visible(False)  # Hide empty subplots

    plt.tight_layout()
    plt.show()

def heatmap(closest_classes, file_name, k=30, save=False): 
    df = pd.DataFrame.from_dict(closest_classes, orient='index').fillna(0).astype(int)  # rows = curr_class, cols = next_class

    top_classes = df.sum(axis=1).nlargest(k).index
    df_subset = df.reindex(index=top_classes, columns=top_classes, fill_value=0)

    # df_norm = df_subset.div(df_subset.sum(axis=1), axis=0)

    # Step 3: Plot heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(df_subset, cmap="viridis", annot=False, square=True, cbar_kws={'label': 'Transition Count'})
    plt.title(f"Class Transition Heatmap (Top {k} Classes)")
    plt.xlabel("Next Class")
    plt.ylabel("Current Class")
    plt.tight_layout()
    if save: 
        plt.savefig(file_name, dpi=300)
    plt.show()