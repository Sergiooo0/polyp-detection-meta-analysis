import hydra
from hydra.core.config_store import ConfigStore
from config import PolypDetectionConfig
from data.process_images import masks_to_yolo, check_yolo_bboxes, sun_annotations_to_yolo, sun_copy_negative_images, deduplicate_consecutive_frames
from data.split_dataset import split_dataset_by_sequence, split_sun_dataset_by_case, select_sun_ood_subset, copy_yolo_files
from utils.connect_to_jetson import transfer_folder, ping
import csv
import os
import shutil

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)


def process_mask_dataset(dataset_name, dataset_info, base_path, duplicate_threshold=0.02):
    """
    Process a mask-based dataset (CVC-ClinicDB, CVC-ColonDB).
    Converts binary masks to YOLO bboxes into a temporary _bboxes directory.

    Args:
        dataset_name: Name of the dataset
        dataset_info: Dataset configuration
        base_path: Base path for datasets
        duplicate_threshold: Threshold for frame deduplication (0.0 to disable)

    Returns:
        (images_dir, bboxes_dir) paths for downstream use.
    """
    mask_dir = os.path.join(base_path, dataset_info.mask_dir)
    images_dir = os.path.join(base_path, dataset_info.images_dir)
    bboxes_dir = os.path.join(base_path, dataset_info.mask_dir + "_bboxes")

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask folder for '{dataset_name}' does not exist: {mask_dir}")

    print(f"Processing mask dataset: {dataset_name}")
    print(f"- Masks: {mask_dir}")
    print(f"- Images: {images_dir}")
    print(f"- Output: {bboxes_dir}")

    # Get list of all mask files
    files_to_process = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])

    if duplicate_threshold > 0.0 and files_to_process:
        dedup_report_dir = os.path.join(os.getcwd(), "dedup_debug", dataset_name)
        filtered_files, removed_files, stats = deduplicate_consecutive_frames(
            files_to_process, images_dir, duplicate_threshold, dedup_report_dir
        )
        files_to_process = filtered_files
        print(f"Deduplication: {stats['total']} -> {stats['kept']} (removed {stats['removed']}, {100*stats['removed']/max(1,stats['total']):.1f}%)")
        if stats['removed'] > 0:
            print(f"Deduplication report saved to: {dedup_report_dir}")

    # Generate YOLO labels only for non-duplicate files
    masks_to_yolo(mask_dir, bboxes_dir, class_id=0, file_list=files_to_process)

    # Generate verification images
    print(f"Generating verification images for '{dataset_name}'...")
    verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
    check_yolo_bboxes(images_dir, bboxes_dir, verification_dir, num_images=50)
    print()

    return images_dir, bboxes_dir


def process_sun_dataset(dataset_name, dataset_info, base_path, duplicate_threshold=0.02):
    """
    Process a SUN-style annotation dataset.
    Converts per-case annotation txt files to YOLO bboxes and collects all images
    into flat temporary directories.

    Args:
        dataset_name: Name of the dataset
        dataset_info: Dataset configuration
        base_path: Base path for datasets
        duplicate_threshold: Threshold for frame deduplication (0.0 to disable)

    Returns:
        (images_dir, labels_dir, case_to_files, neg_case_to_files) for downstream use.
    """
    positive_dir = os.path.join(base_path, dataset_info.positive_dir)
    negative_dir = os.path.join(base_path, dataset_info.negative_dir)
    annotation_dir = os.path.join(base_path, dataset_info.annotation_dir)

    # Temporary flat directories for all SUN images and labels
    images_dir = os.path.join(base_path, dataset_info.positive_dir + "_flat_images")
    labels_dir = os.path.join(base_path, dataset_info.positive_dir + "_flat_labels")

    # Directory to save deduplication verification reports
    dedup_report_dir = os.path.join(os.getcwd(), "dedup_debug", dataset_name)

    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation folder for '{dataset_name}' does not exist: {annotation_dir}")

    print(f"Processing SUN-style dataset: {dataset_name}")
    print(f"    Annotations: {annotation_dir}")
    print(f"    Positive images: {positive_dir}")
    print(f"    Negative images: {negative_dir}")
    print(f"    Flat images output: {images_dir}")
    print(f"    Flat labels output: {labels_dir}")

    # Convert positive annotations to YOLO and collect images
    case_to_files = sun_annotations_to_yolo(
        annotation_dir, positive_dir, images_dir, labels_dir, class_id=0,
        duplicate_threshold=duplicate_threshold,
        dedup_report_dir=dedup_report_dir
    )

    # Copy negative images (with empty labels)
    print(f"Processing negative (healthy) images...")
    neg_case_to_files = sun_copy_negative_images(
        negative_dir, images_dir, labels_dir,
        duplicate_threshold=duplicate_threshold,
        dedup_report_dir=dedup_report_dir
    )

    # Generate verification images
    print(f"Generating verification images for '{dataset_name}'...")
    verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
    check_yolo_bboxes(images_dir, labels_dir, verification_dir, num_images=50)

    # Print deduplication report summary
    if duplicate_threshold > 0.0:
        print(f"\nDeduplication report saved to: {dedup_report_dir}")
    print()

    return images_dir, labels_dir, case_to_files, neg_case_to_files


def create_data_yaml(output_dir):
    yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = (
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names: ['polyp']\n"
    )
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)
    print(f"Created YOLO config file: {yaml_path}\n")


def save_case_metadata_csv(metadata_path, positive_case_to_files, negative_case_to_files):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", newline="") as metadata_handle:
        writer = csv.DictWriter(
            metadata_handle,
            fieldnames=["case_type", "case_id", "image_count", "image_files"],
        )
        writer.writeheader()

        for case_id in sorted(positive_case_to_files.keys()):
            files = positive_case_to_files[case_id]
            writer.writerow({
                "case_type": "positive",
                "case_id": case_id,
                "image_count": len(files),
                "image_files": "|".join(files),
            })

        for case_name in sorted(negative_case_to_files.keys()):
            files = negative_case_to_files[case_name]
            writer.writerow({
                "case_type": "negative",
                "case_id": case_name,
                "image_count": len(files),
                "image_files": "|".join(files),
            })

    print(f"Saved SUN test metadata to: {metadata_path}")

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):

    seed = cfg.params.seed
    val_ratio = cfg.params.val_ratio
    base_path = cfg.files.base_path
    
    # We will track temp directories to delete them later
    temp_dirs = set()
    
    # Store processed dataset info for protocol building
    # Each entry: { "type": ..., "images_dir": ..., "labels_dir": ..., ... }
    dataset_results = {}

    print("-"*50)
    print("Processing raw datasets to YOLO bounding boxes")
    print("-"*50 + "\n")

    for dataset_name, dataset_info in cfg.files.raw_datasets.items():
        try:
            duplicate_threshold = dataset_info.duplicate_threshold
            ds_type = dataset_info.type if hasattr(dataset_info, 'type') else "mask"

            if ds_type == "mask":
                images_dir, bboxes_dir = process_mask_dataset(dataset_name, dataset_info, base_path, duplicate_threshold)
                temp_dirs.add(bboxes_dir)
                dataset_results[dataset_name] = {
                    "type": "mask",
                    "images_dir": images_dir,
                    "labels_dir": bboxes_dir,
                }

            elif ds_type == "sun_annotation":
                images_dir, labels_dir, case_to_files, neg_case_to_files = \
                    process_sun_dataset(dataset_name, dataset_info, base_path, duplicate_threshold)
                temp_dirs.add(images_dir)
                temp_dirs.add(labels_dir)
                dataset_results[dataset_name] = {
                    "type": "sun_annotation",
                    "images_dir": images_dir,
                    "labels_dir": labels_dir,
                    "case_to_files": case_to_files,
                    "neg_case_to_files": neg_case_to_files,
                }

            else:
                print(f"Unknown dataset type '{ds_type}' for '{dataset_name}'. Skipping.\n")
                continue

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}\n")
            continue

    print("\n" + "-"*50)
    print("Building YOLO dataset structure (protocols)")
    print("-"*50 + "\n")

    protocol_outputs = []

    for protocol_name, protocol_info in cfg.files.protocols.items():
        print(f"Generating Protocol: {protocol_name.upper()}...")
        print(f"Info: {protocol_info.description}")
        try:
            train_ds_name = protocol_info.train_source
            ood_ds_name = protocol_info.ood_source
            yolo_out = os.path.join(base_path, protocol_info.yolo_output_dir)

            # --- Validate that both datasets were processed ---
            if train_ds_name not in dataset_results:
                print(f"ERROR: Train source '{train_ds_name}' was not processed. Skipping protocol.\n")
                continue
            if ood_ds_name not in dataset_results:
                print(f"ERROR: OOD source '{ood_ds_name}' was not processed. Skipping protocol.\n")
                continue

            train_info = dataset_results[train_ds_name]
            ood_info = dataset_results[ood_ds_name]

            # --- Build Train/Val split based on dataset type ---
            if train_info["type"] == "mask":
                # Mask-based dataset: use sequence-level split via metadata CSV
                train_ds_config = cfg.files.raw_datasets[train_ds_name]
                csv_path = os.path.join(base_path, train_ds_config.metadata_file) if train_ds_config.metadata_file else None

                if csv_path is None:
                    print(f"ERROR: Mask dataset '{train_ds_name}' has no metadata_file for sequence split.\n")
                    continue

                print(f"Extracting train/val sequences from '{train_ds_name}'...")
                split_dataset_by_sequence(
                    csv_path,
                    train_info["images_dir"],
                    train_info["labels_dir"],
                    yolo_out,
                    train_ratio=1 - val_ratio,
                    seed=seed,
                )

            elif train_info["type"] == "sun_annotation":
                # SUN-style dataset: use case-level split
                print(f"    Extracting train/val by case from '{train_ds_name}'...")
                split_sun_dataset_by_case(
                    case_to_files=train_info["case_to_files"],
                    images_dir=train_info["images_dir"],
                    labels_dir=train_info["labels_dir"],
                    output_dir=yolo_out,
                    neg_case_to_files=train_info.get("neg_case_to_files"),
                    train_ratio=1 - val_ratio,
                    seed=seed,
                )

            # --- Build OOD test set ---
            print(f"Adding OOD samples from '{ood_ds_name}' to protocol '{protocol_name}'...")

            ood_images_dst = os.path.join(yolo_out, "images", "test")
            ood_labels_dst = os.path.join(yolo_out, "labels", "test")

            if ood_info["type"] == "sun_annotation":
                positive_ratio = getattr(protocol_info, "ood_positive_ratio", None)
                negative_ratio = getattr(protocol_info, "ood_negative_ratio", None)

                if positive_ratio is not None and negative_ratio is not None:
                    selected_positive_files, selected_negative_files, subset_stats = select_sun_ood_subset(
                        case_to_files=ood_info["case_to_files"],
                        neg_case_to_files=ood_info.get("neg_case_to_files"),
                        positive_ratio=positive_ratio,
                        negative_ratio=negative_ratio,
                        seed=seed,
                    )

                    selected_files = selected_positive_files + selected_negative_files
                    print(
                        f"SUN OOD subset: {subset_stats['positive_images']} positive images from "
                        f"{len(subset_stats['positive_cases'])} cases, "
                        f"{subset_stats['negative_images']} negative images from "
                        f"{len(subset_stats['negative_cases'])} cases"
                    )

                    save_case_metadata_csv(
                        os.path.join(yolo_out, "metadata.csv"),
                        subset_stats["positive_case_to_files"],
                        subset_stats["negative_case_to_files"],
                    )

                    copy_yolo_files(
                        ood_info["images_dir"],
                        ood_info["labels_dir"],
                        ood_images_dst,
                        ood_labels_dst,
                        selected_files,
                    )
                else:
                    copy_yolo_files(
                        ood_info["images_dir"],
                        ood_info["labels_dir"],
                        ood_images_dst,
                        ood_labels_dst,
                    )
            else:
                copy_yolo_files(
                    ood_info["images_dir"],
                    ood_info["labels_dir"],
                    ood_images_dst,
                    ood_labels_dst
                )

            #Create YOLO config file

            create_data_yaml(yolo_out)
            protocol_outputs.append((protocol_name, yolo_out))
            print()

        except Exception as e:
            print(f"Error generating protocol '{protocol_name}': {str(e)}\n")
            continue

    print("\n" + "-"*50)
    print("Sending test sets to Jetson")
    print("-"*50 + "\n")
    connection_info = cfg.connection
    if ping(connection_info.host, connection_info.port):
        for protocol_name, protocol_output_dir in protocol_outputs:
            local_test_images_dir = os.path.join(protocol_output_dir, "images", "test")
            local_test_labels_dir = os.path.join(protocol_output_dir, "labels", "test")

            temp_transfer_dir = os.path.join("/tmp", f"jetson_test_set_{protocol_name}")
            if os.path.exists(temp_transfer_dir):
                shutil.rmtree(temp_transfer_dir)
            os.makedirs(temp_transfer_dir, exist_ok=True)
            temp_dirs.add(temp_transfer_dir)

            temp_images_test_dir = os.path.join(temp_transfer_dir, "images", "test")
            temp_labels_test_dir = os.path.join(temp_transfer_dir, "labels", "test")
            os.makedirs(temp_images_test_dir, exist_ok=True)
            os.makedirs(temp_labels_test_dir, exist_ok=True)

            copy_yolo_files(
                local_test_images_dir,
                local_test_labels_dir,
                temp_images_test_dir,
                temp_labels_test_dir
            )

            temp_data_yaml = os.path.join(temp_transfer_dir, "data.yaml")
            with open(temp_data_yaml, "w") as yaml_file:
                yaml_file.write(
                    "train: images/train\n"
                    "val: images/train\n"
                    "test: images/test\n"
                    "\n"
                    "nc: 1\n"
                    "names: ['polyp']\n"
                )

            protocol_metadata_path = os.path.join(protocol_output_dir, "metadata.csv")
            if os.path.exists(protocol_metadata_path):
                shutil.copy2(protocol_metadata_path, os.path.join(temp_transfer_dir, "metadata.csv"))

            remote_test_dir = os.path.join(connection_info.test_folder_remote, protocol_name)

            # Transfer each protocol test folder to its own remote directory.
            transfer_folder(
                host=connection_info.host,
                port=connection_info.port,
                username=connection_info.username,
                password=connection_info.password,
                local_dir=temp_transfer_dir,
                remote_dir=remote_test_dir,
                tar_name=f"test_folder_{protocol_name}.tar.gz"
            )
    else:
        print("Skipping transfer: Jetson host is unreachable.")

    print("\n" + "-"*50)
    print("Cleaning up temporary files")
    print("-"*50 + "\n")

    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            print(f"Removing temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    print("\nPreprocess pipeline completed successfully.")

if __name__ == "__main__":
    main()
