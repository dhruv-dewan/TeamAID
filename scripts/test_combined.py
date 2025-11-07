# Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import pandas as pd
import os, random
from sklearn.metrics import f1_score

# --- Import from config ---
from config import (
    original_test_dir,
    stanford_dir,
    metadata_path,
    model_path,
    stanford_train_indices,
    label,
    batch_size
)

# Transforms
test_trans = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Part 1: Original ISIC test evaluation ---
print("\nEvaluating on original test set...")
original_test_data = datasets.ImageFolder(original_test_dir, test_trans)
original_test_dataloader = DataLoader(original_test_data, batch_size=batch_size, shuffle=True)

model = models.resnet34(weights='DEFAULT')
model.fc = nn.Linear(512, len(label))
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
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

loss_fn = nn.CrossEntropyLoss()
test_loss, accuracy, f1 = evaluate_model(original_test_dataloader, model, loss_fn)

print(f"\nOriginal Test Set Results:")
print(f"Test Loss: {test_loss:>8f}")
print(f"Accuracy: {accuracy:>0.1f}%")
print(f"F1 Score: {f1:>0.4f}")

# --- Part 2: Stanford test evaluation (remaining 90%) with per-skin-tone breakdown ---
print("\nEvaluating on remaining 90% of Stanford dataset (per-skin-tone)...")

metadata = pd.read_csv(metadata_path)
filename_to_skin_tone = dict(zip(metadata['DDI_file'], metadata['skin_tone']))

stanford_data = datasets.ImageFolder(stanford_dir, test_trans)

# Load train indices (10% Stanford used in training)
stanford_train_indices = torch.load(stanford_train_indices)
all_indices = set(range(len(stanford_data)))
test_indices = sorted(list(all_indices - set(stanford_train_indices)))

stanford_test_dataset = torch.utils.data.Subset(stanford_data, test_indices)
stanford_test_dataloader = DataLoader(stanford_test_dataset, batch_size=batch_size, shuffle=False)

# Initialize per-skin-tone metrics
skin_tone_metrics = {
    12: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    34: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []},
    56: {'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
}

# Evaluate and collect per-skin-tone stats
model.eval()
num_batches = len(stanford_test_dataloader)
test_loss = 0

with torch.no_grad():
    for batch_idx, (X, y) in enumerate(stanford_test_dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        predictions = pred.argmax(1).cpu().numpy()
        labels = y.cpu().numpy()

        batch_start = batch_idx * stanford_test_dataloader.batch_size
        for i in range(len(labels)):
            subset_idx = batch_start + i
            if subset_idx >= len(stanford_test_dataset):
                break
            original_idx = stanford_test_dataset.indices[subset_idx]
            path, _ = stanford_data.samples[original_idx]
            filename = os.path.basename(path)

            skin = filename_to_skin_tone.get(filename)
            if skin not in skin_tone_metrics:
                continue

            m = skin_tone_metrics[skin]
            m['total'] += 1
            m['correct'] += int(predictions[i] == labels[i])
            m['predictions'].append(int(predictions[i]))
            m['labels'].append(int(labels[i]))

# Aggregate and print results
if num_batches > 0:
    test_loss /= num_batches

print(f"\nStanford Test Set (90%) Overall Test Loss: {test_loss:>8f}")

for skin_tone, metrics in skin_tone_metrics.items():
    if metrics['total'] > 0:
        accuracy = 100.0 * metrics['correct'] / metrics['total']
        f1 = f1_score(metrics['labels'], metrics['predictions'], average='weighted')
        print(f"\nSkin Tone {skin_tone} Metrics:")
        print(f"Total samples: {metrics['total']}")
        print(f"Accuracy: {accuracy:>0.1f}%")
        print(f"F1 Score: {f1:>0.4f}")
    else:
        print(f"\nSkin Tone {skin_tone} Metrics: No samples found in test set")