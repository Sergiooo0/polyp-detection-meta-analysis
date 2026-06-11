import os
import pandas as pd
import shutil
import random
from typing import Optional
from tqdm import tqdm
import json


def copy_yolo_files(
    src_images_dir: str,
    src_labels_dir: str,
    dst_images_dir: str,
    dst_labels_dir: str,
    file_list: Optional[list] = None,
):
    """
    Copies images and YOLO labels to a new destination directory.
    Automatically creates empty label files for background images (images without direct labels).
    """
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg")
    files_to_process = file_list if file_list is not None else sorted(os.listdir(src_images_dir))

    for img_file in tqdm(files_to_process, desc="Copying files", leave=False):
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
            # Force empty file generation for YOLO background compliance
            with open(dst_txt, "w"):
                pass


def copy_image_files(
    src_images_dir: str,
    dst_images_dir: str,
    file_list: Optional[list] = None,
):
    """Copies only image assets from a source directory to a destination directory."""
    os.makedirs(dst_images_dir, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg")
    files_to_process = file_list if file_list is not None else sorted(os.listdir(src_images_dir))

    for img_file in tqdm(files_to_process, desc="Copying images", leave=False):
        if not img_file.lower().endswith(valid_ext):
            continue

        src_img = os.path.join(src_images_dir, img_file)
        dst_img = os.path.join(dst_images_dir, img_file)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)


def split_grouped_dataset(
    group_to_files: dict,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    split_ratios: dict,
    seed: int = 42,
    dataset_name: str = "dataset"
):
    """
    Allocates ENTIRE CASES to target distribution splits based on target percentage targets.
    """
    if not group_to_files:
        print(f"Warning: No groups provided, skipping split")
        return {}

    total_frames = sum(len(files) for files in group_to_files.values())
    split_names = list(split_ratios.keys())
    target_frames = {split_name: int(total_frames * ratio) for split_name, ratio in split_ratios.items()}

    group_ids = sorted(group_to_files.keys())
    random.Random(seed).shuffle(group_ids)

    split_to_files = {split_name: [] for split_name in split_names}
    split_to_groups = {split_name: [] for split_name in split_names}
    split_frames = {split_name: 0 for split_name in split_names}

    current_split_idx = 0
    
    for gid in group_ids:
        files = group_to_files[gid]
        split_name = split_names[current_split_idx]
        
        split_to_groups[split_name].append(gid)
        split_to_files[split_name].extend(files)
        split_frames[split_name] += len(files)
        
        if split_frames[split_name] >= target_frames[split_name] and current_split_idx < len(split_names) - 1:
            current_split_idx += 1

    split_metadata = {
        "dataset_name": dataset_name,
        "seed": seed,
        "strategy": "split_grouped_dataset",
        "splits": {}
    }
    
    for sn in split_names:
        # Calculate the number of images per group for this split
        images_per_group = {str(gid): len(group_to_files[gid]) for gid in split_to_groups[sn]}
        
        split_metadata["splits"][sn] = {
            "group_ids": sorted(split_to_groups[sn]),
            "total_images": split_frames[sn],
            "images_per_group": images_per_group,
            "files": sorted(split_to_files[sn])
        }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"split_metadata_{dataset_name}_seed_{seed}.json")
    with open(json_path, "w") as f:
        json.dump(split_metadata, f, indent=4)
    print(f"  [Frozen Split] Saved subset metadata to {json_path}")

    for split_name in split_names:
        percentage = (split_frames[split_name] / total_frames) * 100 if total_frames > 0 else 0
        print(
            f"Split {split_name}: {len(split_to_groups[split_name])} "
            f"({split_frames[split_name]} frames, approx {percentage:.1f}% | Target: {target_frames[split_name]} frames)"
        )

        copy_yolo_files(
            images_dir,
            labels_dir,
            os.path.join(output_dir, "images", split_name),
            os.path.join(output_dir, "labels", split_name),
            split_to_files[split_name],
        )

    return {f"{sn}_groups": len(split_to_groups[sn]) for sn in split_names} | \
           {f"{sn}_frames": len(split_to_files[sn]) for sn in split_names}


def build_sequence_group_mapping(csv_path: str, labels_dir: str):
    """Builds a sequence extraction mapping using valid labels (skips missing labels)."""
    df = pd.read_csv(csv_path)
    seq_to_files = {}
    skipped_count = 0

    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        img_filename = os.path.basename(row["png_image_path"])
        label_filename = os.path.splitext(img_filename)[0] + ".txt"

        if not os.path.exists(os.path.join(labels_dir, label_filename)):
            skipped_count += 1
            continue

        seq_to_files.setdefault(seq_id, []).append(img_filename)

    return seq_to_files, skipped_count