# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
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
    ham_dir,
    ddi_train_dir,
    ddi_val_dir,
    batch_size,
    num_classes,
    epochs,
    learning_rate,
    freeze_backbone
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

# ============================================================================
# PHASE 1: Train on HAM10000 dataset (90% train, 10% validation)
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: Training on HAM10000 Dataset (DINO)")
print("="*80 + "\n")

# Load HAM10000 dataset
ham_data = datasets.ImageFolder(ham_dir, training_trans)

print("HAM10000 Dataset Size: ", len(ham_data))
print("Classes: ", ham_data.classes)
print("Class to index mapping: ", ham_data.class_to_idx)
print("Malignant samples: ", sum(1 for _, label in ham_data.samples if label == ham_data.class_to_idx['malignant']))
print("Benign samples: ", sum(1 for _, label in ham_data.samples if label == ham_data.class_to_idx['benign']))

# Split HAM10000 into 90% train and 10% validation
total_samples = len(ham_data)
train_size = int(0.9 * total_samples)
val_size = total_samples - train_size
indices = list(range(total_samples))
random.seed(42)  # Ensure reproducible split across all runs
random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

ham_train_data = Subset(ham_data, train_indices)
ham_val_data = Subset(ham_data, val_indices)

print(f"HAM10000 Train size: {len(ham_train_data)} (90%)")
print(f"HAM10000 Val size: {len(ham_val_data)} (10%)\n")

ham_train_dataloader = DataLoader(ham_train_data, batch_size=batch_size, shuffle=True)
ham_val_dataloader = DataLoader(ham_val_data, batch_size=batch_size, shuffle=False)

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

# Freeze backbone layers if freeze_backbone flag is set
if freeze_backbone:
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    print("Backbone frozen: Training only layer4 and fc")

# Only optimize trainable parameters
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Initialize early stopping for HAM10000
early_stopping_ham = EarlyStopping(patience=10, verbose=True, checkpoint_path=os.path.join(f"{BASE_DIR}/dino", "ham_best_model.pth"))

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

# Train on HAM10000 for up to 100 epochs
ham_epochs = 100
for epoch in range(ham_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, ham_train_dataloader, optimizer, loss_fn, device)
    
    # Validation and early stopping
    val_f1 = validate(model, ham_val_dataloader, loss_fn, device)
    print(f"HAM10000 Val F1 (macro): {val_f1:.4f}")
    
    early_stopping_ham(val_f1, model)
    if early_stopping_ham.early_stop:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("\nHAM10000 training complete!")

# Load best HAM10000 model
model.load_state_dict(torch.load(os.path.join(f"{BASE_DIR}/dino", "ham_best_model.pth")))
print(f"Loaded best HAM10000 model (Best F1: {early_stopping_ham.best_score:.4f})")

# ============================================================================
# PHASE 2: Fine-tune on DDI dataset
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: Fine-tuning on DDI Dataset")
print("="*80 + "\n")

# Load DDI datasets (train and val)
ddi_train_data = datasets.ImageFolder(ddi_train_dir, training_trans)
ddi_val_data = datasets.ImageFolder(ddi_val_dir, test_trans)

# Print dataset details
print("DDI Train Dataset Size: ", len(ddi_train_data))
print("Classes: ", ddi_train_data.classes)
print("Class to index mapping: ", ddi_train_data.class_to_idx)
print("Malignant samples: ", sum(1 for _, label in ddi_train_data.samples if label == ddi_train_data.class_to_idx['malignant']))
print("Benign samples: ", sum(1 for _, label in ddi_train_data.samples if label == ddi_train_data.class_to_idx['benign']))

ddi_train_dataloader = DataLoader(ddi_train_data, batch_size=batch_size, shuffle=True)
ddi_val_dataloader = DataLoader(ddi_val_data, batch_size=batch_size, shuffle=False)

# Re-initialize optimizer for DDI fine-tuning
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

# Initialize early stopping for DDI
early_stopping_ddi = EarlyStopping(patience=10, verbose=True, checkpoint_path=os.path.join(f"{BASE_DIR}/dino", "best_model.pth"))

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, ddi_train_dataloader, optimizer, loss_fn, device)
    
    # Validation and early stopping
    val_f1 = validate(model, ddi_val_dataloader, loss_fn, device)
    print(f"DDI Val F1 (macro): {val_f1:.4f}")
    
    early_stopping_ddi(val_f1, model)
    if early_stopping_ddi.early_stop:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("\nDDI fine-tuning complete!")

# Load best DDI model
model.load_state_dict(torch.load(os.path.join(f"{BASE_DIR}/dino", "best_model.pth")))
print(f"Loaded best DDI model (Best F1: {early_stopping_ddi.best_score:.4f})")

print("\n" + "="*80)
print("Complete Pipeline Training Finished!")
print("="*80)
