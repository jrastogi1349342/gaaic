import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

COCO_DIR = "/home/jrastogi/Documents/py311/datasets/coco"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class COCODataset(Dataset): 
    def __init__(self, file_path, data_dir, transform=None): 
        with open(file_path, 'r') as f:
            self.image_files = [os.path.basename(line.strip()) for line in f.readlines()]

        self.data_dir = data_dir
        
        # TODO add augmenting transforms or random resize crops
        self.transform = transform or transforms.Compose([
            transforms.Resize((480, 480)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): 
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        if img is None: 
            raise FileNotFoundError(f"Error loading image {img_path}")
        
        img = self.transform(img)

        return img
    
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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to("cuda")

    # Denormalize images
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    return images

def renormalize_batch(images): 
    transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(images)

def display_batch(images, num_rows=2, num_cols=4):
    batch_size, C, H, W = images.shape

    # ImageNet normalization stats (modify if using different dataset)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) 
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Denormalize images
    images = images * std + mean
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
    dataset = COCODataset(os.path.join(COCO_DIR, f"{img_name}2017.txt"), 
                          os.path.join(COCO_DIR, f"images/{path_name}2017"))
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

# train_loader = train_dataloader()
# if train_loader: 
#     images = next(iter(train_loader)).to(device)
#     display_img(images[0])