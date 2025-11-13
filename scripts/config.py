# --- File: config.py ---

# Base directory (data)
BASE_DIR = "/scratch/zt1/project/heng-prj/user/mnapa/data"

# Paths
# isic train dir:
isic_dir = f"{BASE_DIR}/archive/train"
# isic test dir:
original_test_dir = f"{BASE_DIR}/archive/test" 
# stanford train dir:
stanford_dir = f"{BASE_DIR}/stanforddata"
# ddi_metadata path:
metadata_path = f"{BASE_DIR}/ddi_metadata.csv"
# model.pth location:
model_path = f"{BASE_DIR}/model.pth"
# stanford_train_indices location:
stanford_train_indices = f"{BASE_DIR}/stanford_train_indices.pt"

# Labels and settings
label = ["malignant", "benign"]
batch_size = 64
num_classes = 2
