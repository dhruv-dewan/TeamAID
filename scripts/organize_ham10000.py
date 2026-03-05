"""
Organize HAM10000 images into malignant and benign folders.

This script:
1. Reads HAM10000_metadata.csv
2. Maps diagnoses (dx) to class labels
3. Copies images to malignant/ and benign/ directories with ImageFolder structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter


def organize_ham10000(
    metadata_path,
    source_dir,
    output_dir,
):
    """
    Organize HAM10000 images into malignant and benign folders.
    
    Args:
        metadata_path: Path to HAM10000_metadata.csv
        source_dir: Path to HAM10000 source directory with images
        output_dir: Path to create malignant/benign subdirectories
    """
    
    # Define diagnosis to class mapping
    malignant_conditions = {'mel', 'bcc', 'akiec'}
    benign_conditions = {'nv', 'bkl', 'df', 'vasc'}
    
    # Read metadata
    print("Reading metadata...")
    df = pd.read_csv(metadata_path)
    
    print(f"Total records in metadata: {len(df)}")
    print(f"\nDiagnosis distribution:")
    print(df['dx'].value_counts())
    
    # Map diagnosis to class
    def get_class(dx):
        if dx in malignant_conditions:
            return 'malignant'
        elif dx in benign_conditions:
            return 'benign'
        else:
            return None
    
    df['class'] = df['dx'].map(get_class)
    
    # Filter out unknown diagnoses
    unknown_count = df['class'].isna().sum()
    if unknown_count > 0:
        print(f"\nWarning: {unknown_count} records with unknown diagnosis types")
        print(f"Unknown diagnoses: {df[df['class'].isna()]['dx'].unique()}")
        df = df[df['class'].notna()]
    
    # Find available images in source directory
    print(f"\nScanning source directory: {source_dir}")
    available_images = set()
    if os.path.exists(source_dir):
        available_images = set(os.listdir(source_dir))
    
    print(f"Found {len(available_images)} image files in source directory")
    
    # Filter to images that exist
    # Image files are named like "ISIC_0027419.jpg" from image_id "ISIC_0027419"
    df['filename'] = df['image_id'] + '.jpg'
    df_available = df[df['filename'].isin(available_images)].copy()
    
    print(f"Found {len(df_available)} images that exist in source directory")
    print(f"\nClass distribution of available images:")
    print(df_available['class'].value_counts())
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'malignant'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
    
    # Copy images to appropriate directories
    print(f"\nCopying images to {output_dir}...")
    
    copied_count = {'malignant': 0, 'benign': 0}
    error_count = 0
    
    for idx, row in df_available.iterrows():
        src_path = os.path.join(source_dir, row['filename'])
        class_label = row['class']
        dst_path = os.path.join(output_dir, class_label, row['filename'])
        
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count[class_label] += 1
            else:
                print(f"  ERROR: Source file not found: {src_path}")
                error_count += 1
        except Exception as e:
            print(f"  ERROR copying {src_path}: {e}")
            error_count += 1
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1} / {len(df_available)} images")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Organization complete!")
    print(f"{'='*60}")
    print(f"Malignant images: {copied_count['malignant']}")
    print(f"Benign images:    {copied_count['benign']}")
    print(f"Total copied:     {sum(copied_count.values())}")
    print(f"Errors:           {error_count}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Configuration
    BASE_DIR = "/Users/dhruv/Desktop/TeamAID"
    METADATA_PATH = os.path.join(BASE_DIR, "data", "HAM10000_metadata.csv")
    SOURCE_DIR = os.path.join(BASE_DIR, "data", "HAM10000")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "HAM10000_organized")
    
    # Run organization
    organize_ham10000(
        metadata_path=METADATA_PATH,
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
    )
