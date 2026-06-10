import os
import pandas as pd
import shutil
import random
import math
from typing import Optional
from tqdm import tqdm


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

    valid_ext = (".png", ".jpg", ".jpeg")
    files_to_process = file_list if file_list is not None else sorted(os.listdir(src_images_dir))

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
            with open(dst_txt, "w"):
                pass


def split_by_groups(
    group_to_files: dict,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    group_label: str = "group",
    positive_only_count: bool = False,
):
    """
    Split a dataset into train/val ensuring frames from the same group stay together.

    This function:
    1. Assigns groups to train/val based on frames count
    2. Copies files to the output directory

    Args:
        group_to_files: dict mapping group_id -> list of image filenames
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO labels
        output_dir: Root output directory (will create images/train, images/val, etc.)
        train_ratio: Target ratio of frames for training (default 0.8)
        seed: Random seed for reproducibility
        group_label: Label for logging (e.g., "sequence", "case")

    Returns:
        dict with split statistics
    """
    if not group_to_files:
        print(f"Warning: No {group_label}s provided, skipping split")
        return {}

    # Calculate frame counts and assign groups
    if positive_only_count:
        # Count only positive examples towards the train/val target. Positive groups are expected
        # to be prefixed (e.g., 'pos_') by callers (see split_sun_dataset_by_case).
        pos_counts = {gid: len(files) if str(gid).startswith("pos_") else 0
                      for gid, files in group_to_files.items()}
        total_frames = sum(pos_counts.values())
        if total_frames == 0:
            # Fallback to counting all files if no positives were detected
            frame_counts = {gid: len(files) for gid, files in group_to_files.items()}
            total_frames = sum(frame_counts.values())
        else:
            frame_counts = pos_counts
    else:
        frame_counts = {gid: len(files) for gid, files in group_to_files.items()}
        total_frames = sum(frame_counts.values())
    target_train = int(total_frames * train_ratio)

    # Shuffle groups for random assignment
    group_ids = sorted(group_to_files.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

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

    for gid, files in group_to_files.items():
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
):
    """
    Split a dataset using a metadata CSV with sequence information.
    Ensures frames from the same video sequence stay together.

    Only includes images that have corresponding label files in labels_dir.
    This automatically filters out images that were removed during deduplication.

    Args:
        csv_path: Path to CSV with 'sequence_id' and 'png_image_path' columns
        images_dir: Directory containing all images
        labels_dir: Directory containing all YOLO labels
        output_dir: Root output directory
        train_ratio: Target ratio of frames for training
        seed: Random seed
    """
    df = pd.read_csv(csv_path)

    # Build sequence -> files mapping from CSV
    seq_to_files = {}
    skipped_count = 0

    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        img_filename = os.path.basename(row["png_image_path"])

        # Only include if label file exists (filters out deduplicated images)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(label_path):
            skipped_count += 1
            continue

        if seq_id not in seq_to_files:
            seq_to_files[seq_id] = []
        seq_to_files[seq_id].append(img_filename)

    print(f"Loaded {len(seq_to_files)} sequences from CSV")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images without labels (removed during deduplication)")

    return split_by_groups(
        group_to_files=seq_to_files,
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
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
        group_label="case",
        positive_only_count=True,
    )


def select_sun_ood_subset(
    case_to_files: dict,
    neg_case_to_files: Optional[dict] = None,
    positive_ratio: float = 0.2,
    negative_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Select a SUN OOD subset using positive cases in reverse case order.

    Positive cases are accumulated from the last case to the first until the
    requested fraction of positive images is reached. Negative cases are then
    sampled at random until the requested fraction of the selected positive
    images is reached.
    """

    case_sizes = {case_id: len(files) for case_id, files in case_to_files.items()}
    if case_sizes:
        print("SUN positive case sizes after deduplication:")
        for case_id in sorted(case_sizes.keys()):
            print(f"  case {case_id}: {case_sizes[case_id]} images")

    def _balanced_sample(case_dict: dict, target_total: int, sample_seed: int):
        if not case_dict:
            return {}, 0, 0

        case_ids = list(case_dict.keys())
        target_per_case = max(1, math.ceil(target_total / len(case_ids)))
        smallest_case_size = min(len(files) for files in case_dict.values())
        quota_per_case = min(target_per_case, smallest_case_size)

        rng = random.Random(sample_seed)
        balanced_case_dict = {}

        for case_id in case_ids:
            files = list(case_dict[case_id])
            if len(files) <= quota_per_case:
                balanced_case_dict[case_id] = files
            else:
                balanced_case_dict[case_id] = rng.sample(files, quota_per_case)

        total_files = sum(len(files) for files in balanced_case_dict.values())
        return balanced_case_dict, total_files, quota_per_case

    total_positive_images = sum(len(files) for files in case_to_files.values())
    target_positive_images = max(1, math.ceil(total_positive_images * positive_ratio))

    smallest_case_size = min(case_sizes.values()) if case_sizes else 0
    images_per_case = smallest_case_size
    factor = 1
    while images_per_case * len(case_sizes) < target_positive_images:
        factor += 1
        images_per_case = smallest_case_size * factor

    print(f"Target positive images: {target_positive_images}")
    print(f"Smallest case size: {smallest_case_size}")
    print(f"Selected images per case: {images_per_case} (factor {factor})")

    selected_positive_cases = []
    selected_positive_case_dict = {}
    remaining_positive_images = target_positive_images
    rng = random.Random(seed)

    for case_id in sorted(case_to_files.keys(), reverse=True):
        if remaining_positive_images <= 0:
            break

        files = list(case_to_files[case_id])
        if not files:
            continue

        take_count = min(images_per_case, len(files), remaining_positive_images)
        if take_count <= 0:
            continue

        if take_count == len(files):
            selected_files = files
        else:
            selected_files = rng.sample(files, take_count)

        selected_positive_cases.append(case_id)
        selected_positive_case_dict[case_id] = selected_files
        remaining_positive_images -= len(selected_files)

        if remaining_positive_images <= 0:
            break

    selected_negative_cases = []
    selected_negative_case_dict = {}

    selected_positive_images = sum(len(files) for files in selected_positive_case_dict.values())

    target_negative_images = math.ceil(selected_positive_images * negative_ratio)

    if neg_case_to_files and target_negative_images > 0:
        negative_case_sizes = {case_name: len(files) for case_name, files in neg_case_to_files.items()}
        print("SUN negative case sizes after deduplication:")
        for case_name in sorted(negative_case_sizes.keys()):
            print(f"  {case_name}: {negative_case_sizes[case_name]} images")

        negative_case_ids = sorted(neg_case_to_files.keys())
        num_negative_cases = len(negative_case_ids)
        negative_images_per_case = max(1, target_negative_images // num_negative_cases)
        remainder = target_negative_images % num_negative_cases

        print(f"Target negative images: {target_negative_images}")
        print(f"Selected negative images per case: {negative_images_per_case} (+{remainder} cases get 1 extra)")

        rng = random.Random(seed + 1)
        for index, case_name in enumerate(negative_case_ids):
            files = list(neg_case_to_files[case_name])
            if not files:
                continue

            quota = negative_images_per_case + (1 if index < remainder else 0)
            quota = min(quota, len(files))
            if quota <= 0:
                continue

            selected_negative_cases.append(case_name)
            if quota == len(files):
                selected_negative_case_dict[case_name] = files
            else:
                selected_negative_case_dict[case_name] = rng.sample(files, quota)

        selected_negative_images = sum(len(files) for files in selected_negative_case_dict.values())
    else:
        selected_negative_images = 0

    selected_positive_files = [img for files in selected_positive_case_dict.values() for img in files]
    selected_negative_files = [img for files in selected_negative_case_dict.values() for img in files]

    return selected_positive_files, selected_negative_files, {
        "positive_cases": selected_positive_cases,
        "positive_images": selected_positive_images,
        "positive_case_to_files": selected_positive_case_dict,
        "negative_cases": selected_negative_cases,
        "negative_images": selected_negative_images,
        "negative_case_to_files": selected_negative_case_dict,
        "target_positive_images": target_positive_images,
        "target_negative_images": target_negative_images,
        "available_positive_images": total_positive_images,
        "available_negative_images": sum(len(files) for files in neg_case_to_files.values()) if neg_case_to_files else 0,
    }
