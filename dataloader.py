import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

COCO_DIR = "/home/jrastogi/Documents/py311/datasets/coco"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class COCODataset(Dataset): 
    def __init__(self, file_path, data_dir, transform=None): 
        with open(file_path, 'r') as f:
            self.image_files = [os.path.basename(line.strip()) for line in f.readlines()]

        self.data_dir = data_dir
        
        # TODO add augmenting transforms or random resize crops, and normalization
        self.transform = transform or transforms.Compose([
            transforms.Resize((480, 480)), 
            transforms.ToTensor()
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
    img = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def form_dataloader(name): 
    dataset = COCODataset(os.path.join(COCO_DIR, f"{name}2017.txt"), 
                          os.path.join(COCO_DIR, f"images/{name}2017"))
    return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

def train_dataloader(): 
    name = "train"
    return form_dataloader(name)

def val_dataloader(): 
    name = "val"
    return form_dataloader(name)

def test_dataloader(): 
    name = "test-dev2017"
    return form_dataloader(name)

# train_loader = train_dataloader()
# if train_loader: 
#     images = next(iter(train_loader)).to(device)
#     display_img(images[0])