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
from utils import set_seed

# Import all config values
from config import (
    BASE_DIR,
    isic_dir,
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

# Load ISIC dataset (CHANGE TO HAM10000) -> Make sure to create balanced dataloader
train_data = datasets.ImageFolder(isic_dir, training_trans)

# Print dataset details
print("Train Dataset Size: ", len(train_data))
print("Classes: ", train_data.classes)
print("Class to index mapping: ", train_data.class_to_idx)
print("Malignant samples: ", sum(1 for _, label in train_data.samples if label == train_data.class_to_idx['malignant']))
print("Benign samples: ", sum(1 for _, label in train_data.samples if label == train_data.class_to_idx['benign']))

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet50 model
model = models.resnet50(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

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

print("Training complete!")

torch.save(model.state_dict(), "baseline_model.pth")