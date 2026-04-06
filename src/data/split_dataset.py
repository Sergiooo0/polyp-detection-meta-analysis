import os
import pandas as pd
import shutil
import random
import cv2
import numpy as np
from typing import Optional
from tqdm import tqdm


def _deduplicate_consecutive_frames(file_list: list, images_dir: str, threshold: float = 0.02):
    """
    Remove near-duplicate consecutive frames from an ordered list.

    Returns:
        (filtered_files, stats_dict)
    """
    if threshold == 0.0 or len(file_list) == 0:
        return file_list, {"total": len(file_list), "removed": 0}

    filtered = []
    prev_frame = None

    for filename in file_list:
        img_path = os.path.join(images_dir, filename)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            continue

        frame = cv2.resize(frame, (160, 160)).astype(np.float32)

        if prev_frame is None:
            filtered.append(filename)
            prev_frame = frame
            continue

        diff = np.mean(np.abs(frame - prev_frame)) / 255.0

        if diff > threshold:
            filtered.append(filename)
            prev_frame = frame

    return filtered, {"total": len(file_list), "removed": len(file_list) - len(filtered)}


def copy_yolo_files(
    src_images_dir: str,
    src_labels_dir: str,
    dst_images_dir: str,
    dst_labels_dir: str,
    file_list: Optional[list] = None,
):
    """
    Copy images and labels to a new directory.
    Creates empty label files for images without labels (background images).
    """
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    files_to_process = file_list if file_list is not None else os.listdir(src_images_dir)

    for img_file in tqdm(files_to_process, desc="Copying files"):
        if not img_file.lower().endswith(valid_ext):
            continue

        base_name = os.path.splitext(img_file)[0]
        src_img = os.path.join(src_images_dir, img_file)
        src_txt = os.path.join(src_labels_dir, base_name + ".txt")
        dst_img = os.path.join(dst_images_dir, img_file)
        dst_txt = os.path.join(dst_labels_dir, base_name + ".txt")

        if not os.path.exists(src_img):
            continue

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)
        else:
            open(dst_txt, "a").close()


def split_by_groups(
    group_to_files: dict,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    duplicate_threshold: float = 0.02,
    group_label: str = "group",
):
    """
    Split a dataset into train/val ensuring frames from the same group stay together.

    This is the core split function. It:
    1. Deduplicates all groups first
    2. Assigns groups to train/val based on FRAME count (not group count)
    3. Copies files to the output directory

    Args:
        group_to_files: dict mapping group_id -> list of image filenames
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO labels
        output_dir: Root output directory (will create images/train, images/val, etc.)
        train_ratio: Target ratio of frames for training (default 0.8)
        seed: Random seed for reproducibility
        duplicate_threshold: Threshold for deduplication (0.0 to disable)
        group_label: Label for logging (e.g., "sequence", "case")

    Returns:
        dict with split statistics
    """
    if not group_to_files:
        print(f"Warning: No {group_label}s provided, skipping split")
        return {}

    # Deduplicate all groups
    print(f"Deduplicating {len(group_to_files)} {group_label}s...")
    deduped = {}
    total_original = 0
    total_removed = 0

    for gid in tqdm(sorted(group_to_files.keys()), desc=f"Deduplicating {group_label}s"):
        files = group_to_files[gid]
        filtered, stats = _deduplicate_consecutive_frames(files, images_dir, duplicate_threshold)
        deduped[gid] = filtered
        total_original += stats["total"]
        total_removed += stats["removed"]

    if duplicate_threshold > 0.0:
        print(f"Deduplication: {total_original} -> {total_original - total_removed} "
              f"(removed {total_removed}, {100*total_removed/total_original:.1f}%)")

    # Calculate frame counts and assign groups
    frame_counts = {gid: len(files) for gid, files in deduped.items()}
    total_frames = sum(frame_counts.values())
    target_train = int(total_frames * train_ratio)

    # Shuffle groups for random assignment
    group_ids = list(deduped.keys())
    random.seed(seed)
    random.shuffle(group_ids)

    # Greedily assign groups to train until target is reached
    train_groups = set()
    train_frames = 0

    for gid in group_ids:
        if train_frames < target_train:
            train_groups.add(gid)
            train_frames += frame_counts[gid]

    val_groups = set(group_ids) - train_groups
    val_frames = total_frames - train_frames

    print(f"Split: {len(train_groups)} train {group_label}s ({train_frames} frames, "
          f"{100*train_frames/max(1,total_frames):.1f}%), "
          f"{len(val_groups)} val {group_label}s ({val_frames} frames, "
          f"{100*val_frames/max(1,total_frames):.1f}%)")

    # Collect files for each split
    train_files = []
    val_files = []

    for gid, files in deduped.items():
        if gid in train_groups:
            train_files.extend(files)
        else:
            val_files.extend(files)

    # Copy files
    print("Copying train files...")
    copy_yolo_files(
        images_dir, labels_dir,
        os.path.join(output_dir, "images", "train"),
        os.path.join(output_dir, "labels", "train"),
        train_files,
    )

    print("Copying val files...")
    copy_yolo_files(
        images_dir, labels_dir,
        os.path.join(output_dir, "images", "val"),
        os.path.join(output_dir, "labels", "val"),
        val_files,
    )

    return {
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_frames": len(train_files),
        "val_frames": len(val_files),
    }


def split_dataset_by_sequence(
    csv_path: str,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    duplicate_threshold: float = 0.02,
):
    """
    Split a dataset using a metadata CSV with sequence information.
    Ensures frames from the same video sequence stay together.

    Args:
        csv_path: Path to CSV with 'sequence_id' and 'png_image_path' columns
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO labels
        output_dir: Root output directory
        train_ratio: Target ratio of frames for training
        seed: Random seed
        duplicate_threshold: Deduplication threshold
    """
    df = pd.read_csv(csv_path)

    # Build sequence -> files mapping from CSV
    seq_to_files = {}
    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        img_filename = os.path.basename(row["png_image_path"])
        if seq_id not in seq_to_files:
            seq_to_files[seq_id] = []
        seq_to_files[seq_id].append(img_filename)

    print(f"Loaded {len(seq_to_files)} sequences from CSV")

    return split_by_groups(
        group_to_files=seq_to_files,
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
        duplicate_threshold=duplicate_threshold,
        group_label="sequence",
    )


def split_sun_dataset_by_case(
    case_to_files: dict,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    neg_case_to_files: Optional[dict] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    duplicate_threshold: float = 0.02,
):
    """
    Split a SUN-style dataset ensuring frames from the same case stay together.

    Args:
        case_to_files: dict mapping case_id -> list of positive image filenames
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO labels
        output_dir: Root output directory
        neg_case_to_files: Optional dict mapping case_name -> list of negative image filenames
        train_ratio: Target ratio of frames for training
        seed: Random seed
        duplicate_threshold: Deduplication threshold
    """
    # Merge positive and negative cases into a single dict
    # Use string keys to unify types (positive cases are int, negative are str)
    all_cases = {f"pos_{k}": v for k, v in case_to_files.items()}

    if neg_case_to_files:
        all_cases.update(neg_case_to_files)
        print(f"Combined {len(case_to_files)} positive + {len(neg_case_to_files)} negative cases")

    return split_by_groups(
        group_to_files=all_cases,
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
        duplicate_threshold=duplicate_threshold,
        group_label="case",
    )
