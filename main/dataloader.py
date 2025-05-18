import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

IMAGENET_DIR = "/home/jrastogi/Documents/py311/datasets/imagenet/output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Already resized to shorter side as 256 px
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mean_cuda = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
std_cuda = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")
    
def display_img(img): 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    denormalized_image = (img * std) + mean

    denormalized_image = denormalized_image.squeeze().permute(1, 2, 0).cpu().numpy()
    denormalized_image = np.clip(denormalized_image, 0, 1)

    plt.imshow(denormalized_image)
    plt.axis('off')
    plt.show()

def denormalize_batch(images): 
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")

    # Denormalize images
    images = images * std_cuda + mean_cuda

    # TODO come up with better expression for this that's still differentiable
    images = torch.sigmoid(4 * images - 2)
    # images = torch.clamp(images, 0, 1)

    return images

def renormalize_batch(images): 
    transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(images)

def display_batch(images, num_rows=2, num_cols=4):
    batch_size, C, H, W = images.shape

    # Denormalize images
    images = images * std_cuda + mean_cuda
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    images = np.clip(images, 0, 1)

    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_rows * num_cols):
        if i < batch_size:
            axes[i].imshow(images[i])
            axes[i].axis("off")  # Hide axes
        else:
            axes[i].set_visible(False)  # Hide empty subplots

    plt.tight_layout()
    plt.show()

def form_dataloader(img_name, path_name, batch_size=8, num_workers=4, pin_memory=True): 
    dataset = datasets.ImageFolder(os.path.join(IMAGENET_DIR, path_name), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def train_dataloader(batch_size=8, num_workers=4, pin_memory=True): 
    name = "train"
    return form_dataloader(name, name, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def val_dataloader(batch_size=8, num_workers=4, pin_memory=True): 
    name = "val"
    return form_dataloader(name, name, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def test_dataloader(batch_size=8, num_workers=4, pin_memory=True): 
    img_name = "test-dev"
    path_name = "test"
    return form_dataloader(img_name, path_name, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
