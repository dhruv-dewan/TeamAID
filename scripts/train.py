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

# torchvision.models --> ResNet34 (Try other sizes and test later)
model = models.resnet34(weights='DEFAULT') # Using pre-trained model
model.fc = nn.Linear(512,2)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
