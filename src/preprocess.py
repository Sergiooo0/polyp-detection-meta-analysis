import hydra
from hydra.core.config_store import ConfigStore
from config import PolypDetectionConfig
from utils.process_images import masks_to_yolo, check_yolo_bboxes
from utils.split_dataset import split_dataset_by_sequence, copy_yolo_files
from omegaconf import OmegaConf
import os
import shutil

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):

    seed = cfg.params.seed
    val_ratio = cfg.params.val_ratio
    base_path = cfg.files.base_path
    
    # We will track temp directories to delete them later
    temp_bbox_dirs = set()

    print("\nProcessing masks to YOLO Bounding Boxes\n")

    for dataset_name, dataset_info in cfg.files.raw_datasets.items():
        mask_dir = os.path.join(base_path, dataset_info.mask_dir)
        images_dir = os.path.join(base_path, dataset_info.images_dir)

        # Save temporarily bboxes next to the masks
        bboxes_dir = os.path.join(base_path, dataset_info.mask_dir + "_bboxes")
        temp_bbox_dirs.add(bboxes_dir)

        if not os.path.exists(mask_dir):
            print(f"Mask folder for '{dataset_name}' does not exist: {mask_dir}")
            print("Skipping to next dataset...\n")
            continue
        
        try:
            print(f"Processing dataset: {dataset_name}")
            print(f"- Masks: {mask_dir}")
            print(f"- Output: {bboxes_dir}")
            
            masks_to_yolo(mask_dir, bboxes_dir, class_id=0)
            
            print(f"Generating verification images for '{dataset_name}'...")
            verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
            check_yolo_bboxes(images_dir, bboxes_dir, verification_dir, num_images=10)
            print()
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}\n")
            continue

    print("\nBuilding YOLO dataset structure\n")
    
    for protocol_name, protocol_info in cfg.files.protocols.items():
        print(f"Generating Protocol: {protocol_name.upper()}...")
        print(f"Info: {protocol_info.description}")
        try:

            train_ds_name = protocol_info.train_source
            ood_ds_name = protocol_info.ood_source
            train_ds_config = cfg.files.raw_datasets[train_ds_name]
            ood_ds_config = cfg.files.raw_datasets[ood_ds_name]

            # Build Train/Val split
            src_images = os.path.join(base_path, train_ds_config.images_dir)
            src_bboxes = os.path.join(base_path, train_ds_config.mask_dir + "_bboxes")
            csv_path = os.path.join(base_path, train_ds_config.metadata_file) if train_ds_config.metadata_file else None
            yolo_out = os.path.join(base_path, protocol_info.yolo_output_dir)

            print(f"    Extracting train/val sequences from '{train_ds_name}'...")
            split_dataset_by_sequence(csv_path, src_images, src_bboxes, yolo_out, train_ratio=1-val_ratio, seed=seed)

            # Build OOD test set
            print(f"    Adding OOD samples from '{ood_ds_name}' to protocol '{protocol_name}'...")
            ood_images_src = os.path.join(base_path, ood_ds_config.images_dir)
            ood_bboxes_src = os.path.join(base_path, ood_ds_config.mask_dir + "_bboxes")
            
            ood_images_dst = os.path.join(yolo_out, "images", "test")
            ood_labels_dst = os.path.join(yolo_out, "labels", "test")
            
            # Copy OOD samples to the test folder (creating empty labels if needed)
            copy_yolo_files(ood_images_src, ood_bboxes_src, ood_images_dst, ood_labels_dst)
            print()
        except Exception as e:
            print(f"Error generating protocol '{protocol_name}': {str(e)}\n")
            continue

    print("\nCleaning up temporary files\n")
    for temp_dir in temp_bbox_dirs:
        if os.path.exists(temp_dir):
            print(f"Removing temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    print("\nPipeline completed successfully. All datasets are ready for YOLO")

if __name__ == "__main__":
    main()