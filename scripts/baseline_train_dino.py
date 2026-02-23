# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics import f1_score
from model_utils import set_seed, EarlyStopping

# Import all config values
from config import (
    BASE_DIR,
    ddi_train_dir,
    ddi_val_dir,
    batch_size,
    num_classes,
    epochs,
    learning_rate
)

set_seed(42)  # Set seed for reproducibility

# Create transform function to transform data
# This step involves the data augmentation and normalization of images
training_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.CenterCrop((224, 224)),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(), # Ensures random orientation of images
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])

test_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load DDI datasets (train and val)
train_data = datasets.ImageFolder(ddi_train_dir, training_trans)
val_data = datasets.ImageFolder(ddi_val_dir, test_trans)

# Print dataset details
print("Train Dataset Size: ", len(train_data))
print("Classes: ", train_data.classes)
print("Class to index mapping: ", train_data.class_to_idx)
print("Malignant samples: ", sum(1 for _, label in train_data.samples if label == train_data.class_to_idx['malignant']))
print("Benign samples: ", sum(1 for _, label in train_data.samples if label == train_data.class_to_idx['benign']))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load pre-trained DINO SSL ResNet50 model
# Loading DINO model locally (for HPC node running without internet access)
try:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
except Exception as e:
        print("Warning: Failed to fetch from internet; trying local repo...")
        model = torch.hub.load('/home/ddewan/dino', 'dino_resnet50', source='local')
model = model.cpu()

with torch.no_grad():
    dummy_input = torch.zeros(1, 3, 224, 224)
    features = model(dummy_input)
    num_features = features.shape[1] # Should be 2048

print(f"Extracted feature dimension from DINO ResNet50: {num_features}")
model.fc = nn.Linear(num_features, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, verbose=True, checkpoint_path=os.path.join(f"{BASE_DIR}/dino", "best_model.pth"))

def validate(model, dataloader, loss_fn, device):
    """Validate model and return F1 score (macro average)."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            images, labels = X.to(device), y.to(device)
            pred = model(images)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1

def train(model, dataloader, optimizer, loss_fn, device):
    size = len(dataloader.dataset)

    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        # Move data to CUDA
        images, labels = X.to(device), y.to(device)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()

        optimizer.step()

        # Display loss at start and end of each epoch
        total_loss += loss.item()
        if batch % size == 0:
            print(f"Loss: {total_loss / size}")


for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_dataloader, optimizer, loss_fn, device)
    
    # Validation and early stopping
    val_f1 = validate(model, val_dataloader, loss_fn, device)
    print(f"Val F1 (macro): {val_f1:.4f}")
    
    early_stopping(val_f1, model)
    if early_stopping.early_stop:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("\nTraining complete!")

# Load best model
model.load_state_dict(torch.load(os.path.join(f"{BASE_DIR}/dino", "best_model.pth")))
print(f"Loaded best model (Best F1: {early_stopping.best_score:.4f})")