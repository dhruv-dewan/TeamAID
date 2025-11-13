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

MODEL= "dino"  # Options: "dino" or "supervised"

# First test directory (original test set)
original_test_dir = "data/archive/test"  # Match the format used in train.py

label=["malignant","benign"]

test_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test on original test set first
print("\nEvaluating on original test set...")
original_test_data = datasets.ImageFolder(original_test_dir, test_trans)
original_test_dataloader = DataLoader(original_test_data, batch_size=64, shuffle=False)

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
    recall = recall_score(all_labels, all_preds, average='weighted')

    return test_loss, accuracy, f1, recall

# Evaluate on original test set
loss_fn = nn.CrossEntropyLoss()
test_loss, accuracy, f1, recall = evaluate_model(original_test_dataloader, model, loss_fn)

print(f"\nOriginal Test Set Results:")
print(f"Test Loss: {test_loss:>8f}")
print(f"Accuracy: {accuracy:>0.1f}%")
print(f"F1 Score: {f1:>0.4f}")
print(f"Recall: {recall:>0.4f}")
# End of testing - only ISIC/original test set is evaluated

