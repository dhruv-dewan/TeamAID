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

#train_dir="/scratch/zt1/project/heng-prj/user/mnapa/data/archive/train" 
train_dir="data/archive/train"

label=["malignant","benign"]

# Create transform function to transform data
# This step involves the data augmentation and normalization of images

training_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(), # Ensures random orientation of images
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
test_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, training_trans)
#test_data = datasets.ImageFolder(test_dir, test_trans)

print("Train Dataset Size: ", len(train_data))
#print("Train Dataset Size: ", len(test_data))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# model = models.resnet50(weights="DEFAULT") # Using pre-trained DINO model
model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50') # Using pre-trained DINO model
# Some DINO backbone variants replace `model.fc` with `Identity`. Determine the
# correct feature dimension robustly and attach a classifier.
if hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
  in_features = model.fc.in_features
else:
  # Try to infer from the last conv block (works for ResNet34/50)
  in_features = None
  try:
    if hasattr(model, 'layer4'):
      last_block = model.layer4[-1]
      if hasattr(last_block, 'conv3'):
        in_features = last_block.conv3.out_channels
      elif hasattr(last_block, 'conv2'):
        in_features = last_block.conv2.out_channels
  except Exception:
    in_features = None

  # Fallbacks: ResNet50 -> 2048, ResNet34 -> 512
  if in_features is None:
    in_features = 2048

model.fc = nn.Linear(in_features, 2)
# Freeze all but last layer
"""
for param in model.parameters():
  param.requires_grad = False
for param in model.fc.parameters():
  param.requires_grad = True
"""


batch_size = 64
epochs = 50
learning_rate = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Robust device selection with diagnostics
cuda_available = torch.cuda.is_available()
cuda_count = torch.cuda.device_count() if cuda_available else 0
cuda_version = torch.version.cuda
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")
print(f"torch.cuda.is_available(): {cuda_available}")
print(f"torch.cuda.device_count(): {cuda_count}")
print(f"torch.version.cuda: {cuda_version}")
if cuda_available and cuda_count > 0:
  try:
    print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
  except Exception as e:
    print(f"Could not get CUDA device name: {e}")
else:
  print("CUDA not available or no CUDA devices found; running on CPU.")

model.to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  # Set the model to training mode - important for batch normalization and dropout layers
  # Unnecessary in this situation but added for best practices

  model.train()
  for batch, (X, y) in enumerate(dataloader):

    # Moving Data to GPU
    X = X.to(device)
    y = y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Display at start and at end
    if batch % (len(dataloader)-1) == 0:
        loss, current = loss.item(), batch * batch_size + len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train_loop(train_dataloader, model, loss_fn, optimizer)
  # test_loop(test_dataloader, model, loss_fn)
print("Training Complete.")

torch.save(model.state_dict(), 'model.pth')
