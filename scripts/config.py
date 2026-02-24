# --- File: config.py ---

# Base directory (data)
#BASE_DIR = "/scratch/zt1/project/heng-prj/user/mnapa/data"
BASE_DIR = "/scratch/zt1/project/heng-prj/user/ddewan/AID/TeamAID/data"

# Paths
# isic train dir:
isic_dir = f"{BASE_DIR}/archive/train"
# isic test dir:
original_test_dir = f"{BASE_DIR}/archive/test" 
# stanford train dir:
#stanford_dir = f"{BASE_DIR}/stanforddata"
stanford_dir = f"{BASE_DIR}/stanford"
# ddi_metadata path:
metadata_path = f"{BASE_DIR}/ddi_metadata.csv"
# model.pth location:
model_path = f"{BASE_DIR}/model.pth"
# stanford_train_indices location:
stanford_train_indices = f"{BASE_DIR}/stanford_train_indices.pt"

# Stratified DDI dataset paths
ddi_train_dir = f"{BASE_DIR}/stratified_split/train"
ddi_val_dir = f"{BASE_DIR}/stratified_split/val"
ddi_test_dir = f"{BASE_DIR}/stratified_split/test"

# HAM10000 dataset paths
ham_dir = f"{BASE_DIR}/HAM100000_organized"
ham_train_dir = f"{BASE_DIR}/HAM10000_split/train"
ham_val_dir = f"{BASE_DIR}/HAM10000_split/val"

# Labels and settings
label = ["malignant", "benign"]
batch_size = 32
num_classes = 2
epochs = 50
learning_rate = 1e-4

# Freezing settings
freeze_backbone = False