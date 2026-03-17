import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGRI-VISION IMAGE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgriVisionDataset(Dataset):
    """
    Custom Dataset for satellite/drone imagery.
    Supports folder-based classification structure.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        if not self.root_dir.exists():
            print(f"⚠️ Warning: Directory {root_dir} not found.")
            return

        # 1. Detect class folders
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 2. Collect images
        # If classes exist, assume folder structure. Otherwise, collect all images in root.
        if classes:
            for cls_name in classes:
                cls_dir = self.root_dir / cls_name
                for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                    for img_path in cls_dir.glob(img_ext):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])
        else:
            # Flat structure fallback
            for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                for img_path in self.root_dir.glob(img_ext):
                    self.image_paths.append(img_path)
                    self.labels.append(0) # Default single class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def inspect_dataset(path):
    """Prints a summary of the dataset structure."""
    root = Path(path)
    if not root.exists():
        return
    
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    img_files = []
    for ext in ['*.jpg', '*.png', '*.tif']:
        img_files.extend(list(root.rglob(ext)))

    print("📊 DATASET INSPECTION")
    print(f"  • Root Path: {root}")
    print(f"  • Detected Classes: {classes if classes else 'None (Flat structure)'}")
    print(f"  • Total Images: {len(img_files)}")
    print("----------------------------------------------------------------")

def get_dataloaders(image_dir, batch_size=16, train_split=0.8):
    """
    Prepares train and validation DataLoaders.
    
    WHY RESIZING?
    Neural networks require consistent input dimensions (tensors) for batch processing.
    Downsampling to 128x128 saves memory while retaining spatial agricultural features.

    WHY NORMALIZATION?
    Standardizing pixel values to a small range (e.g., [0, 1] or mean-subtracted) 
    keeps gradients stable, prevents saturation, and leads to significantly faster 
    and more reliable model convergence.
    """
    
    # Preprocessing Pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),            # Consistent shape
        transforms.ToTensor(),                   # Scaled to [0, 1]
        transforms.Normalize(                    # Standard Distribution (ImageNet defaults)
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = AgriVisionDataset(image_dir, transform=transform)
    
    if len(full_dataset) == 0:
        print("❌ Dataset is empty. DataLoader cannot be initialized.")
        return None, None

    # Splitting logic
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"✅ Pipeline Ready!")
    print(f"  • Training Samples: {len(train_set)}")
    print(f"  • Validation Samples: {len(val_set)}")
    print(f"  • Batch Size: {batch_size}")

    return train_loader, val_loader

if __name__ == "__main__":
    IMAGE_DIR = "project/data/images/"
    
    inspect_dataset(IMAGE_DIR)
    t_loader, v_loader = get_dataloaders(IMAGE_DIR, batch_size=16)

    if t_loader:
        # Check first batch shape
        images, labels = next(iter(t_loader))
        print(f"  • Batch Tensor Shape: {images.shape} (B x C x H x W)")
