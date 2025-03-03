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

import pandas as pd
import argparse

# Add command line arguments
parser = argparse.ArgumentParser(description='Test skin lesion classification model')
parser.add_argument('--exclude-file', type=str, default=None, 
                    help='Path to file containing filenames to exclude from testing')
parser.add_argument('--model-path', type=str, default='model.pth',
                    help='Path to the model weights file')
args = parser.parse_args()

# First test directory (original test set)
original_test_dir = "data/archive/test"  # Match the format used in train.py
# Second test directory (diverse skin data)
diverse_test_dir = "data/stanford/"

label=["malignant","benign"]

# Load metadata
metadata_df = pd.read_csv('data/ddi_metadata.csv')

test_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test on original test set first
print("\nEvaluating on original test set...")
original_test_data = datasets.ImageFolder(original_test_dir, test_trans)
original_test_dataloader = DataLoader(original_test_data, batch_size=64, shuffle=True)

model = models.resnet34(weights='DEFAULT')
model.fc = nn.Linear(512,2)
# Freeze all but last layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
model.load_state_dict(torch.load(args.model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predictions = pred.argmax(1)
            correct += (predictions == y).type(torch.float).sum().item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = 100 * correct / size
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return test_loss, accuracy, f1

# Evaluate on original test set
loss_fn = nn.CrossEntropyLoss()
test_loss, accuracy, f1 = evaluate_model(original_test_dataloader, model, loss_fn)

print(f"\nOriginal Test Set Results:")
print(f"Test Loss: {test_loss:>8f}")
print(f"Accuracy: {accuracy:>0.1f}%")
print(f"F1 Score: {f1:>0.4f}")

# Now test on diverse skin tone dataset
print("\nEvaluating on diverse skin tone dataset...")

# Load the Stanford dataset
diverse_test_data = datasets.ImageFolder(diverse_test_dir, test_trans)
original_size = len(diverse_test_data)
print(f"Total Stanford dataset size: {original_size}")

# Check if we need to exclude files
if args.exclude_file and os.path.exists(args.exclude_file):
    # Load the list of files to exclude
    with open(args.exclude_file, 'r') as f:
        files_to_exclude = set([line.strip() for line in f.readlines()])
    
    print(f"Loaded {len(files_to_exclude)} files to exclude from {args.exclude_file}")
    
    # Filter out samples that should be excluded
    filtered_samples = []
    for file_path, class_idx in diverse_test_data.samples:
        filename = file_path.split('/')[-1].split('\\')[-1]
        if filename not in files_to_exclude:
            filtered_samples.append((file_path, class_idx))
    
    # Create a subset with only the filtered samples
    filtered_indices = [diverse_test_data.samples.index(sample) for sample in filtered_samples]
    diverse_test_data = torch.utils.data.Subset(diverse_test_data, filtered_indices)
    
    print(f"Excluded {len(files_to_exclude)} files from testing")
    print(f"Testing on {len(diverse_test_data)} Stanford samples (after exclusion)")
else:
    print(f"Testing on all {len(diverse_test_data)} Stanford samples")

diverse_test_dataloader = DataLoader(diverse_test_data, batch_size=64, shuffle=True)

# Create a mapping from image filename to skin tone
filename_to_skin_tone = dict(zip(metadata_df['DDI_file'], metadata_df['skin_tone']))

# Initialize dictionaries to track metrics per skin tone
skin_tone_metrics = {
    12: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    34: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    56: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
}

model = models.resnet34(weights='DEFAULT')
model.fc = nn.Linear(512,2)
# Freeze all but last layer
for param in model.parameters():
  param.requires_grad = False
for param in model.fc.parameters():
  param.requires_grad = True
model.load_state_dict(torch.load('model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode - important for batch normalization and dropout layers
# Unnecessary in this situation but added for best practices
model.eval()
size = len(diverse_test_dataloader.dataset)
num_batches = len(diverse_test_dataloader)
test_loss = 0

loss_fn = nn.CrossEntropyLoss()

# Evaluating the model
with torch.no_grad():
    for batch_idx, (X, y) in enumerate(diverse_test_dataloader):
        # Get filenames for this batch
        if isinstance(diverse_test_data, torch.utils.data.Subset):
            # Handle Subset case
            indices = diverse_test_data.indices[batch_idx*diverse_test_dataloader.batch_size:
                                              min((batch_idx+1)*diverse_test_dataloader.batch_size, len(diverse_test_data))]
            filenames = [diverse_test_data.dataset.samples[i][0].split('/')[-1].split('\\')[-1] for i in indices]
        else:
            # Handle regular dataset case
            filenames = [diverse_test_data.samples[i][0].split('/')[-1].split('\\')[-1]  
                        for i in range(batch_idx*diverse_test_dataloader.batch_size, 
                                     min((batch_idx+1)*diverse_test_dataloader.batch_size, len(diverse_test_data)))]
        
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

