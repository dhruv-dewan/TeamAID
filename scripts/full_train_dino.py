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

def load_dino_resnet50_weights(model, ckpt_path, use_teacher=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # pick teacher or student
    key = "teacher" if use_teacher and "teacher" in ckpt else "student"
    state_dict = ckpt[key] if key in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        # remove DDP prefix
        if k.startswith("module."):
            k = k[len("module."):]

        # keep only backbone weights
        if k.startswith("backbone."):
            k = k[len("backbone."):]

        # skip projection head / classifier
        if k.startswith("head.") or k.startswith("fc."):
            continue

        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded DINO weights.")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    return model

# Import all config values
from config import (
    BASE_DIR,
    ddi_train_dir,
    ddi_val_dir,
    batch_size,
    num_classes,
    epochs,
    learning_rate,
    freeze_backbone # For freezing backbone layers during training (experiments)
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

# Load ResNet50 architecture with no supervised ImageNet weights
model = models.resnet50(weights=None)
# Path to your DINO checkpoint
dino_ckpt_path = "/scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/data/dino_pretrained/full/checkpoint0060.pth"
model = load_dino_resnet50_weights(model, dino_ckpt_path, use_teacher=True)
# Replace classifier head for your downstream task
model.fc = nn.Linear(model.fc.in_features, num_classes)

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

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, verbose=True, checkpoint_path=os.path.join(f"{BASE_DIR}/full_dino", "best_model.pth"))

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
model.load_state_dict(torch.load(os.path.join(f"{BASE_DIR}/full_dino", "best_model.pth")))
print(f"Loaded best model (Best F1: {early_stopping.best_score:.4f})")