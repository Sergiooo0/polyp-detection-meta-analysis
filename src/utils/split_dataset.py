import os
import pandas as pd
import shutil
import random

def copy_yolo_files(src_images_dir: str, src_labels_dir: str, dst_images_dir: str, dst_labels_dir: str, file_list: list = None):
    """
    Helper function to safely copy images and labels to a new directory.
    If a label doesn't exist, it creates an empty one (background image).
    If file_list is provided, it only processes those specific files.
    """
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # Use the provided list, or fall back to reading the entire directory
    files_to_process = file_list if file_list is not None else os.listdir(src_images_dir)
    
    for img_file in files_to_process:
        if img_file.lower().endswith(valid_ext):
            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(src_images_dir, img_file)
            src_txt = os.path.join(src_labels_dir, base_name + '.txt')
            dst_img = os.path.join(dst_images_dir, img_file)
            dst_txt = os.path.join(dst_labels_dir, base_name + '.txt')

            # Copy image
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Missing source image {src_img}")
                continue

            # Copy or create label
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            else:
                # If no bbox file exists, create an empty one (healthy image)
                open(dst_txt, 'a').close()


def split_dataset_by_sequence(csv_path, images_dir, labels_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Divide a dataset in Train and Val ensuring that frames from the same
    sequence (video) are not mixed between the two sets.
    """
    df = pd.read_csv(csv_path)
    
    # Get unique sequence IDs from the metadata CSV
    sequences = df['sequence_id'].unique()

    # Shuffle sequences randomly (using seed for reproducibility)
    random.seed(seed)
    random.shuffle(sequences)
    
    # Split sequences into Train and Val based on the specified ratio
    split_idx = int(len(sequences) * train_ratio)
    
    train_seqs = set(sequences[:split_idx])
    val_seqs = set(sequences[split_idx:])

    print(f"Total sequences: {len(sequences)}")
    print(f"Assigned to Train: {len(train_seqs)} sequences")
    print(f"Assigned to Val: {len(val_seqs)} sequences")

    # 1. Group filenames into Train and Val lists
    train_files = []
    val_files = []

    for _, row in df.iterrows():
        current_seq = row['sequence_id']
        img_filename = os.path.basename(row['png_image_path'])
        
        if current_seq in train_seqs:
            train_files.append(img_filename)
        else:
            val_files.append(img_filename)

    print("Copying files to Train folder...")
    copy_yolo_files(
        src_images_dir=images_dir,
        src_labels_dir=labels_dir,
        dst_images_dir=os.path.join(output_dir, 'images', 'train'),
        dst_labels_dir=os.path.join(output_dir, 'labels', 'train'),
        file_list=train_files
    )

    print("Copying files to Val folder...")
    copy_yolo_files(
        src_images_dir=images_dir,
        src_labels_dir=labels_dir,
        dst_images_dir=os.path.join(output_dir, 'images', 'val'),
        dst_labels_dir=os.path.join(output_dir, 'labels', 'val'),
        file_list=val_files
    )