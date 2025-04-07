# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from sklearn.metrics import f1_score

# Original training data directory
#train_dir="/scratch/zt1/project/heng-prj/user/mnapa/data/archive/train" 
train_dir = "data/archive/train"
# Stanford data directory
stanford_dir = "data/stanford/"

# Load metadata for skin tone information
metadata_df = pd.read_csv('data/ddi_metadata.csv')

label = ["malignant", "benign"]

# Create transform function to transform data
# This step involves the data augmentation and normalization of images
training_trans = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),  # Ensures random orientation of images
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.rotate(x, random.choice([90, 180, 270])))]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization
])

# Load original training data
original_train_data = datasets.ImageFolder(train_dir, training_trans)
print("Original Train Dataset Size: ", len(original_train_data))

# Load Stanford data
stanford_data = datasets.ImageFolder(stanford_dir, training_trans)
print("Stanford Dataset Size: ", len(stanford_data))

# Create a mapping from image filename to skin tone
filename_to_skin_tone = dict(zip(metadata_df['DDI_file'], metadata_df['skin_tone']))

# Function to get skin tone for a given file path
def get_skin_tone(file_path):
    filename = file_path.split('/')[-1].split('\\')[-1]
    return filename_to_skin_tone.get(filename, None)

# Function to get class (malignant/benign) for a given file path
def get_class(file_path):
    # In ImageFolder, the class is determined by the subfolder
    return os.path.basename(os.path.dirname(file_path))

# Organize Stanford data by skin tone and class
skin_tone_class_files = {
    12: {'malignant': [], 'benign': []},
    34: {'malignant': [], 'benign': []},
    56: {'malignant': [], 'benign': []}
}

# Collect files by skin tone and class
for idx, (file_path, class_idx) in enumerate(stanford_data.samples):
    skin_tone = get_skin_tone(file_path)
    class_name = 'malignant' if class_idx == 0 else 'benign'  # Adjust based on your class mapping
    
    if skin_tone is not None:
        skin_tone_class_files[skin_tone][class_name].append((file_path, class_idx))

# Calculate how many samples to take from each category
# We want 10% of the Stanford dataset, balanced across skin tones and classes
stanford_size = len(stanford_data)
target_stanford_size = int(stanford_size * 0.1)  # 10% of Stanford data
samples_per_category = target_stanford_size // 6  # 3 skin tones × 2 classes

print(f"Stanford Dataset Size: {stanford_size}")
print(f"Target Stanford samples to add (10%): {target_stanford_size}")
print(f"Target samples per category: {samples_per_category}")

# Select balanced samples from each category
selected_stanford_samples = []
selected_filenames = []  # Store filenames for later reference

for skin_tone in [12, 34, 56]:
    for class_name in ['malignant', 'benign']:
        available_samples = skin_tone_class_files[skin_tone][class_name]
        if available_samples:
            # Take min of available samples or target per category
            num_to_take = min(samples_per_category, len(available_samples))
            selected_samples = random.sample(available_samples, num_to_take)
            selected_stanford_samples.extend(selected_samples)
            
            # Store filenames of selected samples
            for file_path, _ in selected_samples:
                filename = file_path.split('/')[-1].split('\\')[-1]
                selected_filenames.append(filename)
                
            print(f"Selected {num_to_take} {class_name} samples from skin tone {skin_tone}")

# Save the list of selected filenames to a file
with open('selected_stanford_files.txt', 'w') as f:
    for filename in selected_filenames:
        f.write(f"{filename}\n")
print(f"Saved list of {len(selected_filenames)} selected Stanford files to 'selected_stanford_files.txt'")

# ------------------------------------------------------------------------------------------------------------------

# Create directory to save transformed images
transformed_dir = "data/transformedStanford"
os.makedirs(transformed_dir, exist_ok=True)
os.makedirs(os.path.join(transformed_dir, "malignant"), exist_ok=True)
os.makedirs(os.path.join(transformed_dir, "benign"), exist_ok=True)

print(f"Created directory for transformed images: {transformed_dir}")

# Function to save transformed images
def save_transformed_image(img_tensor, filename, class_name):
    # Convert tensor to PIL image
    img = transforms.ToPILImage()(img_tensor)
    # Save to appropriate class directory
    save_path = os.path.join(transformed_dir, class_name, filename)
    img.save(save_path)
    return save_path

# Save transformed versions of the selected Stanford images
print("Saving transformed Stanford images used for training...")
saved_paths = []

for file_path, class_idx in selected_stanford_samples:
    # Get original image
    original_img = torchvision.datasets.folder.default_loader(file_path)
    
    # Apply transformations
    transformed_img = training_trans(original_img)
    
    # Get filename and class
    filename = file_path.split('/')[-1].split('\\')[-1]
    class_name = 'malignant' if class_idx == 0 else 'benign'
    
    # Save transformed image
    saved_path = save_transformed_image(transformed_img, filename, class_name)
    saved_paths.append(saved_path)

print(f"Saved {len(saved_paths)} transformed Stanford images to {transformed_dir}")

# ------------------------------------------------------------------------------------------------------------------

# Create a new dataset with the selected Stanford samples
stanford_subset = torch.utils.data.Subset(stanford_data, [stanford_data.samples.index(sample) for sample in selected_stanford_samples])

# Combine original training data with Stanford subset
combined_train_data = ConcatDataset([original_train_data, stanford_subset])
print(f"Combined Dataset Size: {len(combined_train_data)}")
print(f"Added {len(stanford_subset)} Stanford samples ({len(stanford_subset)/len(combined_train_data)*100:.1f}% of combined dataset)")

# Create data loader for combined dataset
train_dataloader = DataLoader(combined_train_data, batch_size=64, shuffle=True)

# torchvision.models --> ResNet34
model = models.resnet34(weights='DEFAULT')  # Using pre-trained model
model.fc = nn.Linear(512, 2)
# Freeze all but last layer

"""
for param in model.parameters():
  param.requires_grad = False

# Unfreeze the last two layers
for param in list(model.children())[-2:]:
    for p in param.parameters():
        p.requires_grad = True
"""

for param in model.fc.parameters():
  param.requires_grad = True


batch_size = 64
epochs = 50
learning_rate = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

# Add a list to store loss values
train_losses = []

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        # Moving Data to GPU
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Track batch loss
        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display at start and at end
        if batch % (len(dataloader)-1) == 0:
            loss_val = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    
    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_epoch_loss)
    return avg_epoch_loss

# Training loop with loss tracking
print("Starting training...")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    print(f"Epoch {t+1} Average Loss: {avg_loss:>7f}")
    # test_loop(test_dataloader, model, loss_fn)
print("Training Complete.")

# Plot the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(range(1, epochs+1, max(1, epochs//10)))  # Show epoch numbers on x-axis

# Save the plot
plt.savefig('training_loss_curve.png')
print("Loss curve saved as 'training_loss_curve.png'")

# Show the plot (optional, can be commented out for headless environments)
#plt.show()

# Save the model with a different name to distinguish from the original
torch.save(model.state_dict(), 'model.pth')
