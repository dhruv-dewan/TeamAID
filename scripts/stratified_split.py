"""
Stratified train/val/test split for dermatology image dataset.

This script:
1. Reads metadata from ddi_metadata.csv
2. Maps images to data/stanford/ directory
3. Performs stratified split by class AND skin tone
4. Creates train/val/test directories with ImageFolder structure
5. Copies images to respective splits
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_stratified_split(
    metadata_path,
    image_dir,
    output_dir,
    train_ratio=0.65,
    val_ratio=0.175,
    test_ratio=0.175,
    random_seed=42,
    use_symlink=False,
):
    """
    Create stratified train/val/test split preserving class and skin tone distributions.
    
    Args:
        metadata_path: Path to CSV file with image metadata
        image_dir: Path to source image directory (e.g., data/stanford/)
        output_dir: Path to create split directories
        train_ratio: Proportion for training set (default 0.65)
        val_ratio: Proportion for validation set (default 0.175)
        test_ratio: Proportion for test set (default 0.175)
        random_seed: Seed for reproducibility
        use_symlink: If True, create symlinks; if False, copy files
    """
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Read metadata
    print("Reading metadata...")
    df = pd.read_csv(metadata_path)
    
    # Map 'malignant' column to class labels
    df['class_label'] = df['malignant'].map({True: 'malignant', False: 'benign'})
    
    # Filter to images that exist in the image directory
    print(f"Filtering to images in {image_dir}...")
    benign_dir = os.path.join(image_dir, 'benign')
    malignant_dir = os.path.join(image_dir, 'malignant')
    
    existing_images = set()
    for cls_dir in [benign_dir, malignant_dir]:
        if os.path.exists(cls_dir):
            existing_images.update(os.listdir(cls_dir))
    
    df = df[df['DDI_file'].isin(existing_images)].reset_index(drop=True)
    print(f"Found {len(df)} images in metadata that exist in {image_dir}")
    
    # Create stratification column combining class and skin tone
    df['strata'] = df['class_label'] + '_' + df['skin_tone'].astype(str)
    
    print(f"\nStratification groups:")
    print(df['strata'].value_counts().sort_index())
    
    # First split: train vs temp (val+test)
    print(f"\nPerforming stratified split...")
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['strata'],
        random_state=random_seed,
    )
    
    # Second split: val vs test (from temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['strata'],
        random_state=random_seed,
    )
    
    # Print split statistics
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nClass distribution per split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        class_counts = split_df['class_label'].value_counts()
        print(f"{split_name}: {dict(class_counts)}")
    
    print(f"\nSkin tone distribution per split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        tone_counts = split_df['skin_tone'].value_counts().sort_index()
        print(f"{split_name}: {dict(tone_counts)}")
    
    # Create output directories and copy/symlink files
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
    }
    
    for split_name, split_df in splits.items():
        print(f"\nCreating {split_name} split...")
        split_dir = os.path.join(output_dir, split_name)
        
        # Create class subdirectories
        for class_label in ['benign', 'malignant']:
            class_dir = os.path.join(split_dir, class_label)
            os.makedirs(class_dir, exist_ok=True)
        
        # Copy/symlink images
        for idx, row in split_df.iterrows():
            src_subdir = row['class_label']
            src_path = os.path.join(image_dir, src_subdir, row['DDI_file'])
            dst_path = os.path.join(split_dir, src_subdir, row['DDI_file'])
            
            if os.path.exists(src_path):
                if use_symlink:
                    if not os.path.exists(dst_path):
                        os.symlink(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
            else:
                print(f"  WARNING: Source file not found: {src_path}")
        
        print(f"Created {split_name} split with {len(split_df)} images")


if __name__ == '__main__':
    # Configuration
    BASE_DIR = "/Users/dhruv/Desktop/TeamAID"
    METADATA_PATH = os.path.join(BASE_DIR, "data", "ddi_metadata.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "stanford")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "stratified_split")
    
    # Run split
    create_stratified_split(
        metadata_path=METADATA_PATH,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
        use_symlink=False,  # Set to True to use symlinks instead of copying
    )
    
    print(f"\n✓ Split complete! Output directory: {OUTPUT_DIR}")
