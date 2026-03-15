import os
import pandas as pd
import shutil
import random
from tqdm import tqdm

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
    
    for img_file in tqdm(files_to_process, desc="Copying YOLO files"):
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
    Uses metadata CSV with 'sequence_id' and 'png_image_path' columns.
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

    # Group filenames into Train and Val lists
    train_files = []
    val_files = []

    for _, row in df.iterrows():
        current_seq = row['sequence_id']
        img_filename = os.path.basename(row['png_image_path'])
        if current_seq in train_seqs:
            train_files.append(img_filename)
        else:
            val_files.append(img_filename)

    print(f"Total files — Train: {len(train_files)}, Val: {len(val_files)}")
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


def split_sun_dataset_by_case(case_to_files, images_dir, labels_dir, output_dir,
                              neg_case_to_files=None, train_ratio=0.8, seed=42):
    """
    Divide a SUN-style dataset into Train and Val ensuring frames from the same
    case are not mixed between the two sets.

    Args:
        case_to_files:     dict mapping case_id (int) -> list of image filenames (positive cases)
        images_dir:        Flat directory where all images were collected
        labels_dir:        Flat directory where all YOLO labels were collected
        output_dir:        Root of the YOLO experiment (e.g. .../YOLO_Experiments/T2)
        neg_case_to_files: dict mapping neg_case_name -> list of image filenames (negative/healthy).
                           If provided, negative cases are also split by case.
        train_ratio:       Fraction of cases used for training (default 0.8)
        seed:              Random seed for reproducibility
    """
    # --- Split positive cases ---
    case_ids = sorted(case_to_files.keys())
    random.seed(seed)
    random.shuffle(case_ids)

    split_idx = int(len(case_ids) * train_ratio)
    train_cases = set(case_ids[:split_idx])
    val_cases = set(case_ids[split_idx:])

    print(f"Positive cases — Total: {len(case_ids)}, Train: {len(train_cases)}, Val: {len(val_cases)}")

    train_files = []
    val_files = []
    for cid in case_ids:
        files = case_to_files[cid]
        if cid in train_cases:
            train_files.extend(files)
        else:
            val_files.extend(files)

    # --- Split negative cases (if provided) ---
    if neg_case_to_files:
        neg_case_names = sorted(neg_case_to_files.keys())
        random.seed(seed + 1)  # different seed to avoid correlation
        random.shuffle(neg_case_names)

        neg_split_idx = int(len(neg_case_names) * train_ratio)
        neg_train = set(neg_case_names[:neg_split_idx])
        neg_val = set(neg_case_names[neg_split_idx:])

        print(f"Negative cases — Total: {len(neg_case_names)}, Train: {len(neg_train)}, Val: {len(neg_val)}")

        for nc in neg_case_names:
            files = neg_case_to_files[nc]
            if nc in neg_train:
                train_files.extend(files)
            else:
                val_files.extend(files)

    print(f"Total files — Train: {len(train_files)}, Val: {len(val_files)}")

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