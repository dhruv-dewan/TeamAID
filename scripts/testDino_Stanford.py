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
from sklearn.metrics import f1_score, recall_score
import pandas as pd

MODEL= "dino"  # Options: "dino" or "supervised"

# First test directory (diverse test set)
original_test_dir = "data/stanford"

label=["malignant","benign"]

test_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\nEvaluating on diverse test set...")
original_test_data = datasets.ImageFolder(original_test_dir, test_trans)
original_test_dataloader = DataLoader(original_test_data, batch_size=64, shuffle=False)

print("Test Dataset Size: ", len(original_test_data))

model = None

if MODEL == "dino":
    # Loading DINO model locally (for HPC node running without internet access)
    try:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    except Exception as e:
        print("Warning: Failed to fetch from internet; trying local repo...")
        model = torch.hub.load('/home/ddewan/dino', 'dino_resnet50', source='local')

    model = model.cpu()

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
    model.load_state_dict(torch.load('dino model/model.pth'))

elif MODEL == "supervised":
    model = models.resnet50(weights='DEFAULT')
    model = model.cpu()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('supervised model/model.pth'))

else:
    raise ValueError("Invalid MODEL specified. Choose 'dino' or 'supervised'.")

# model.fc = nn.Linear(model.fc.in_features, 2)
# Freeze all but last layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

print(f"Loaded {MODEL} model successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load metadata
metadata_df = pd.read_csv('data/ddi_metadata.csv')

# Create a mapping from image filename to skin tone
filename_to_skin_tone = dict(zip(metadata_df['DDI_file'], metadata_df['skin_tone']))

# Initialize dictionaries to track metrics per skin tone
skin_tone_metrics = {
    12: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    34: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    56: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
}

model.eval()
size = len(original_test_dataloader.dataset)
num_batches = len(original_test_dataloader)
test_loss = 0

loss_fn = nn.CrossEntropyLoss()

# Evaluating the model
with torch.no_grad():
    for batch_idx, (X, y) in enumerate(original_test_dataloader):
        # Get filenames for this batch and extract just the image name without the class folder
        filenames = [original_test_dataloader.dataset.samples[i][0].split('/')[-1].split('\\')[-1]
                     for i in range(batch_idx*original_test_dataloader.batch_size,
                                 min((batch_idx+1)*original_test_dataloader.batch_size, len(original_test_dataloader.dataset)))]

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        
        # Get predictions
        predictions = pred.argmax(1)
        
        # Update metrics for each image based on skin tone
        for i, filename in enumerate(filenames):
            try:
                skin_tone = filename_to_skin_tone[filename]
                correct = predictions[i] == y[i]
                
                skin_tone_metrics[skin_tone]['total'] += 1
                skin_tone_metrics[skin_tone]['correct'] += correct.item()
                skin_tone_metrics[skin_tone]['predictions'].append(predictions[i].cpu().item())
                skin_tone_metrics[skin_tone]['labels'].append(y[i].cpu().item())
            except KeyError:
                print(f"Warning: No skin tone data found for image {filename}")
                continue

test_loss /= num_batches

# Print overall metrics and per skin tone metrics
print(f"\nOverall Test Loss: {test_loss:>8f}")

for skin_tone in skin_tone_metrics:
    metrics = skin_tone_metrics[skin_tone]
    if metrics['total'] > 0:
        accuracy = 100.0 * metrics['correct'] / metrics['total']
        f1 = f1_score(metrics['labels'], metrics['predictions'], average='weighted')
        
        print(f"\nSkin Tone {skin_tone} Metrics:")
        print(f"Total samples: {metrics['total']}")
        print(f"Accuracy: {accuracy:>0.1f}%")
        print(f"F1 Score: {f1:>0.4f}")

