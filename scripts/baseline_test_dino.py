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
from model_utils import set_seed

# Import all config values
from config import (
    BASE_DIR,
    ddi_test_dir,
    batch_size,
    num_classes
)

set_seed(42)  # Set seed for reproducibility

test_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(ddi_test_dir, test_trans)

print("Test Dataset Size: ", len(test_data))
print("Classes: ", test_data.classes)
print("Class to index mapping: ", test_data.class_to_idx)
print("Malignant samples: ", sum(1 for _, label in test_data.samples if label == test_data.class_to_idx['malignant']))
print("Benign samples: ", sum(1 for _, label in test_data.samples if label == test_data.class_to_idx['benign']))

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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

model.load_state_dict(torch.load(f"{BASE_DIR}/dino/ham_ddi_split/ham_best_model.pth"))

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluate on test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
f1_weighted = f1_score(all_labels, all_preds, average='weighted')
f1_macro = f1_score(all_labels, all_preds, average='macro')
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Macro Average: {f1_macro:.4f}")
print(f"Test F1 Weighted Average: {f1_weighted:.4f}")
