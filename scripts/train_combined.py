# --- File: train.py ---

# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
import pandas as pd
import numpy as np
import os, random
from sklearn.model_selection import train_test_split

# Import all config values
from config import (
    BASE_DIR,
    isic_dir,
    stanford_dir,
    metadata_path,
    label,
    batch_size,
    num_classes
)

# Transforms
training_trans = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load ISIC dataset
isic_data = datasets.ImageFolder(isic_dir, training_trans)

# Load Stanford dataset + metadata
metadata = pd.read_csv(metadata_path)
filename_to_skin = dict(zip(metadata['DDI_file'], metadata['skin_tone']))
stanford_data = datasets.ImageFolder(stanford_dir, training_trans)

# Group Stanford samples by skin tone
skin_tone_groups = {12: [], 34: [], 56: []}
for path, lbl in stanford_data.samples:
    fname = os.path.basename(path)
    if fname in filename_to_skin:
        tone = filename_to_skin[fname]
        if tone in skin_tone_groups:
            skin_tone_groups[tone].append((path, lbl))

# Sample 10% of Stanford equally by skin tone
balanced_samples = []
for tone, items in skin_tone_groups.items():
    n = int(0.1 * len(items))
    balanced_samples.extend(random.sample(items, n))

stanford_subset = torch.utils.data.Subset(
    stanford_data,
    [stanford_data.samples.index(s) for s in balanced_samples]
)

# Combine ISIC (90%) + Stanford balanced subset (10%)
train_dataset = ConcatDataset([isic_data, stanford_subset])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("ISIC samples:", len(isic_data))
print("Stanford 10% balanced samples:", len(stanford_subset))
print("Total training samples:", len(train_dataset))

# Model setup
model = models.resnet34(weights='DEFAULT')
model.fc = nn.Linear(512, num_classes)

epochs = 50
learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % (len(dataloader) - 1) == 0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Training Complete.")

# Save model and Stanford indices used for training
torch.save(model.state_dict(), os.path.join(BASE_DIR, "model.pth"))

stanford_train_indices = [stanford_data.samples.index(s) for s in balanced_samples]
torch.save(stanford_train_indices, os.path.join(BASE_DIR, "stanford_train_indices.pt"))
