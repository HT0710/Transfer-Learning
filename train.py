import os, torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from pathlib import Path
from utils import train_step, test_step, save_model
from model import model_transference


# Seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Hardware
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CPU = os.cpu_count()

# Dataset
DATASET_PATH = Path('dataset')

# Data transforms
train_tfs = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
val_tfs = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# Hyperparameter
BATCH_SIZE = 8
LEARNING_RATE = 1e-2
NUM_EPOCH = 10
SAVE_FREQUENCY = 10

# Data prepare
train_data = ImageFolder(DATASET_PATH / 'train', train_tfs)
val_data = ImageFolder(DATASET_PATH / 'val', val_tfs)

train_loader = DataLoader(train_data, BATCH_SIZE, True, num_workers=NUM_CPU)
val_loader = DataLoader(val_data, BATCH_SIZE, False, num_workers=NUM_CPU)

# Get class name
CLASSES = train_data.classes

# Model transfer
model = model_transference(
    pre_trained=models.resnet18(weights='DEFAULT'),
    output_shape=2,
    device=DEVICE
)

# Define loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Training loop
for epoch in range(NUM_EPOCH):
    train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, DEVICE)

    test_loss, test_acc = test_step(model, val_loader, criterion, DEVICE)

    scheduler.step(test_loss)

    print(f"Epoch: {epoch+1} | Lr: {optimizer.param_groups[0]['lr']}                              ")
    print(f"  |- loss: {train_loss:.4f}  acc: {train_acc:.4f} | test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")

    if (epoch+1) % SAVE_FREQUENCY == 0:
        save_model(model=model, target_dir="Models", model_name=f"E{epoch+1}_L{test_loss:.4f}_A{test_acc:.2f}.pth")
