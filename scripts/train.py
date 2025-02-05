# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import numpy as np

from google.colab import drive
import os

import random

from tqdm.auto import tqdm
from sklearn.metrics import f1_score

train_dir="/data/train" 
test_dir="/data/test"

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
test_data = datasets.ImageFolder(test_dir, test_trans)

print("Train Dataset Size: ", len(train_data))
print("Train Dataset Size: ", len(test_data))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# torchvision.models --> ResNet34 (Try other sizes and test later)
model = models.resnet34(weights='DEFAULT') # Using pre-trained model

