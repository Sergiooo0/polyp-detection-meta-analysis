import hydra
from hydra.core.config_store import ConfigStore
from config import PolypDetectionConfig
from data.process_images import masks_to_yolo, check_yolo_bboxes, sun_annotations_to_yolo, sun_copy_negative_images
from data.split_dataset import split_dataset_by_sequence, split_sun_dataset_by_case, copy_yolo_files
import os
import shutil

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)


def process_mask_dataset(dataset_name, dataset_info, base_path):
    """
    Process a mask-based dataset (CVC-ClinicDB, CVC-ColonDB).
    Converts binary masks to YOLO bboxes into a temporary _bboxes directory.
    
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
    print(f"- Output: {bboxes_dir}")

    masks_to_yolo(mask_dir, bboxes_dir, class_id=0)

    print(f"Generating verification images for '{dataset_name}'...")
    verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
    check_yolo_bboxes(images_dir, bboxes_dir, verification_dir, num_images=10)
    print()

    return images_dir, bboxes_dir


def process_sun_dataset(dataset_name, dataset_info, base_path):
    """
    Process a SUN-style annotation dataset.
    Converts per-case annotation txt files to YOLO bboxes and collects all images
    into flat temporary directories.
    
    Returns:
        (images_dir, labels_dir, case_to_files, neg_case_to_files) for downstream use.
    """
    positive_dir = os.path.join(base_path, dataset_info.positive_dir)
    negative_dir = os.path.join(base_path, dataset_info.negative_dir)
    annotation_dir = os.path.join(base_path, dataset_info.annotation_dir)

    # Temporary flat directories for all SUN images and labels
    images_dir = os.path.join(base_path, dataset_info.positive_dir + "_flat_images")
    labels_dir = os.path.join(base_path, dataset_info.positive_dir + "_flat_labels")

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
        annotation_dir, positive_dir, images_dir, labels_dir, class_id=0
    )

    # Copy negative images (with empty labels)
    neg_case_to_files = {}
    if os.path.exists(negative_dir):
        print(f"Processing negative (healthy) images...")
        neg_case_to_files = sun_copy_negative_images(
            negative_dir, images_dir, labels_dir
        )

    # Generate verification images
    print(f"Generating verification images for '{dataset_name}'...")
    verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
    check_yolo_bboxes(images_dir, labels_dir, verification_dir, num_images=10)
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

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):

    seed = cfg.params.seed
    val_ratio = cfg.params.val_ratio
    duplicate_threshold = cfg.params.duplicate_threshold
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
            ds_type = dataset_info.type if hasattr(dataset_info, 'type') else "mask"

            if ds_type == "mask":
                images_dir, bboxes_dir = process_mask_dataset(dataset_name, dataset_info, base_path)
                temp_dirs.add(bboxes_dir)
                dataset_results[dataset_name] = {
                    "type": "mask",
                    "images_dir": images_dir,
                    "labels_dir": bboxes_dir,
                }

            elif ds_type == "sun_annotation":
                images_dir, labels_dir, case_to_files, neg_case_to_files = \
                    process_sun_dataset(dataset_name, dataset_info, base_path)
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
                    duplicate_threshold=duplicate_threshold
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
                    duplicate_threshold=duplicate_threshold
                )

            # --- Build OOD test set ---
            print(f"Adding OOD samples from '{ood_ds_name}' to protocol '{protocol_name}'...")

            ood_images_dst = os.path.join(yolo_out, "images", "test")
            ood_labels_dst = os.path.join(yolo_out, "labels", "test")

            copy_yolo_files(
                ood_info["images_dir"],
                ood_info["labels_dir"],
                ood_images_dst,
                ood_labels_dst
            )

            #Create YOLO config file

            create_data_yaml(yolo_out)
            print()

        except Exception as e:
            print(f"Error generating protocol '{protocol_name}': {str(e)}\n")
            continue

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