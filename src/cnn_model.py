import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGRI-VISION CNN ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SatelliteCNN(nn.Module):
    """
    CNN for agricultural feature extraction from 128x128 satellite imagery.
    
    Architecture:
    3x [Conv2d -> ReLU -> MaxPool2d] -> [Flatten] -> [Linear -> ReLU] -> [Softmax]
    """
    def __init__(self, num_classes=2):
        super(SatelliteCNN, self).__init__()
        
        # Block 1: Input 3x128x128 -> 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2: 16x64x64 -> 32x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Block 3: 32x32x32 -> 64x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully Connected layers
        # 64 channels * 16 * 16 pixels = 16384 features
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total

if __name__ == "__main__":
    # Hyperparameters
    NUM_CLASSES = 2 # e.g., Healthy vs Stressed crop
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 10
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    # Model, Loss, Optimizer
    model = SatelliteCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("📊 CNN Architecture Summary:")
    print(model)
    
    # Note: This requires the DataLoaders from image_pipeline.py
    print("\n💡 Tip: To run training, import get_dataloaders from image_pipeline.py and pass them here.")
