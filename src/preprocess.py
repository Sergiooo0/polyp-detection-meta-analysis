import os
import random
import shutil
import tempfile
import json
import hydra
from hydra.core.config_store import ConfigStore

from config import PolypDetectionConfig
from data.process_images import (
    masks_to_yolo, 
    check_yolo_bboxes, 
    sun_annotations_to_yolo, 
    sun_copy_negative_images, 
    deduplicate_consecutive_frames
)
from data.split_dataset import (
    build_sequence_group_mapping,
    copy_yolo_files,
    copy_image_files,
    split_grouped_dataset
)
from utils.connect_to_jetson import transfer_folder, ping

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)


def apply_uniform_subsampling(files, drop_every_n):
    """Removes every N-th frame sequentially to avoid selection bias."""
    if drop_every_n <= 1:
        return files
    return [file for idx, file in enumerate(sorted(files)) if (idx + 1) % drop_every_n != 0]


def print_case_counts(title, case_to_files):
    """Prints image counts per case and the accumulated total."""
    print(title)
    total_images = 0
    
    def extract_key(k):
        if isinstance(k, int):
            return k
        cleaned = str(k).replace("case", "")
        return int(cleaned) if cleaned.isdigit() else str(k)

    sorted_keys = sorted(case_to_files.keys(), key=extract_key)
    for case_id in sorted_keys:
        count = len(case_to_files[case_id])
        total_images += count
        print(f"  case {case_id}: {count} images")
    print(f"  Total images: {total_images}\n")
    return total_images


def scan_sun_structured_cache(cache_root):
    """Scans the structured case directories directly from the cache path."""
    def _scan_category(category_path):
        mapping = {}
        if not os.path.exists(category_path):
            return mapping
        for case_name in sorted(os.listdir(category_path)):
            case_dir = os.path.join(category_path, case_name)
            if not os.path.isdir(case_dir):
                continue
            files = sorted([
                f for f in os.listdir(case_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if files:
                try:
                    case_id = int(case_name.replace("case", ""))
                except ValueError:
                    case_id = case_name
                mapping[case_id] = files
        return mapping

    pos_case_to_files = _scan_category(os.path.join(cache_root, "images", "positive"))
    neg_case_to_files = _scan_category(os.path.join(cache_root, "images", "negative"))
    return pos_case_to_files, neg_case_to_files


def process_mask_dataset(dataset_name, dataset_info, base_path, duplicate_threshold, force_rebuild):
    """Processes flat mask datasets (CVC-ClinicDB / CVC-ColonDB)."""
    mask_dir = os.path.join(base_path, dataset_info.mask_dir)
    source_images_dir = os.path.join(base_path, dataset_info.images_dir)
    
    cache_dir = dataset_info.cache_dir or os.path.join("processed_datasets", dataset_name)
    cache_root = os.path.join(base_path, cache_dir)
    
    images_output_dir = os.path.join(cache_root, "images")
    labels_output_dir = os.path.join(cache_root, "labels")

    if os.path.exists(images_output_dir) and os.path.exists(labels_output_dir) and not force_rebuild:
        print(f"Using cached mask dataset for '{dataset_name}': {cache_root}")
        return images_output_dir, labels_output_dir

    print(f"Processing mask dataset: {dataset_name}")
    files_to_process = sorted([f for f in os.listdir(source_images_dir) if f.lower().endswith(".png")])

    if duplicate_threshold > 0.0 and files_to_process:
        dedup_report_dir = os.path.join(os.getcwd(), "dedup_debug", dataset_name)
        files_to_process, _, stats = deduplicate_consecutive_frames(
            files_to_process, source_images_dir, duplicate_threshold, dedup_report_dir
        )
        print(f"Deduplication: {stats['total']} -> {stats['kept']} frames kept.")

    masks_to_yolo(mask_dir, labels_output_dir, class_id=0, file_list=files_to_process)
    copy_image_files(source_images_dir, images_output_dir, files_to_process)

    verification_dir = os.path.join(os.getcwd(), "bbox_debug", dataset_name)
    check_yolo_bboxes(source_images_dir, labels_output_dir, verification_dir, num_images=50)

    return images_output_dir, labels_output_dir


def process_sun_dataset(dataset_name, dataset_info, base_path, duplicate_threshold, force_rebuild, drop_every_n=5):
    """Processes SUN dataset utilizing case structures directly without flat intermediates."""
    cache_dir = dataset_info.cache_dir or os.path.join("processed_datasets", dataset_name)
    cache_root = os.path.join(base_path, cache_dir)

    # Check if structured case directory already exists in cache
    if os.path.exists(os.path.join(cache_root, "images", "positive")) and not force_rebuild:
        print(f"Using cached structured SUN dataset from: {cache_root}")
        case_to_files, neg_case_to_files = scan_sun_structured_cache(cache_root)
        print_case_counts("SUN Positive Case Counts (From Cache):", case_to_files)
        print_case_counts("SUN Negative Case Counts (From Cache):", neg_case_to_files)
        return cache_root, case_to_files, neg_case_to_files

    print(f"Processing raw SUN dataset: {dataset_name}")
    positive_dir = os.path.join(base_path, dataset_info.positive_dir)
    negative_dir = os.path.join(base_path, dataset_info.negative_dir)
    annotation_dir = os.path.join(base_path, dataset_info.annotation_dir)
    dedup_report_dir = os.path.join(os.getcwd(), "dedup_debug", dataset_name)

    # Use a temporary directory to build standard outputs before structuring
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_images = os.path.join(temp_dir, "images")
        temp_labels = os.path.join(temp_dir, "labels")

        case_to_files = sun_annotations_to_yolo(
            annotation_dir, positive_dir, temp_images, temp_labels, class_id=0,
            duplicate_threshold=duplicate_threshold, dedup_report_dir=dedup_report_dir
        )
        neg_case_to_files = sun_copy_negative_images(
            negative_dir, temp_images, temp_labels,
            duplicate_threshold=duplicate_threshold, dedup_report_dir=dedup_report_dir
        )

        # Apply sequential uniform subsampling to clean remaining selection biases
        for c_id, f_list in case_to_files.items():
            case_to_files[c_id] = apply_uniform_subsampling(f_list, drop_every_n)
        for c_id, f_list in neg_case_to_files.items():
            neg_case_to_files[c_id] = apply_uniform_subsampling(f_list, drop_every_n)

        # Re-materialize directly into case structures inside cache root
        for case_id, files in case_to_files.items():
            case_str = f"case{case_id}"
            copy_yolo_files(
                temp_images, temp_labels,
                os.path.join(cache_root, "images", "positive", case_str),
                os.path.join(cache_root, "labels", "positive", case_str),
                files
            )
        for case_name, files in neg_case_to_files.items():
            copy_yolo_files(
                temp_images, temp_labels,
                os.path.join(cache_root, "images", "negative", case_name),
                os.path.join(cache_root, "labels", "negative", case_name),
                files
            )

    print_case_counts("SUN Positive Case Counts (Freshly Processed):", case_to_files)
    print_case_counts("SUN Negative Case Counts (Freshly Processed):", neg_case_to_files)

    return cache_root, case_to_files, neg_case_to_files


def create_data_yaml(output_dir):
    """Generates standard YOLO data configuration file."""
    yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = (
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names: ['polyp']\n"
    )
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)


def build_protocols(cfg, dataset_results, base_path):
    """Assembles final training and evaluation structures per protocol specifications."""
    print("\n" + "-"*50 + "\nBuilding YOLO dataset structure (protocols)\n" + "-"*50)
    protocol_outputs = []

    for protocol_name, protocol_info in cfg.files.protocols.items():
        print(f"Generating Protocol: {protocol_name.upper()} | {protocol_info.description}")
        yolo_out = os.path.join(base_path, protocol_info.yolo_output_dir)

        split_specs = {
            "train": (protocol_info.train_source, protocol_info.train_ratio),
            "val": (protocol_info.val_source, protocol_info.val_ratio),
            "test": (protocol_info.test_source, protocol_info.test_ratio),
        }

        source_to_splits = {}
        for split_name, (source_name, ratio) in split_specs.items():
            if source_name and ratio is not None:
                source_to_splits.setdefault(source_name, {})[split_name] = ratio

        for source_name, split_ratios in source_to_splits.items():
            if source_name not in dataset_results:
                print(f"ERROR: Source '{source_name}' not processed. Skipping.")
                continue

            res = dataset_results[source_name]
            config = cfg.files.raw_datasets[source_name]

            if res["type"] == "mask":
                print(f"  Processing mask dataset source '{source_name}' for protocol '{protocol_name}'")
                
                if "metadata_file" in config and config.metadata_file:
                    csv_path = os.path.join(base_path, config.metadata_file)
                    group_to_files, _ = build_sequence_group_mapping(csv_path, res["labels_dir"])
                else:
                    print(f"  No metadata_file found for '{source_name}'. Sending entire block directly.")
                    valid_ext = (".png", ".jpg", ".jpeg")
                    all_files = [f for f in sorted(os.listdir(res["images_dir"])) if f.lower().endswith(valid_ext)]
                    group_to_files = {"all_dataset_frames": all_files}
                
                split_grouped_dataset(
                    group_to_files=group_to_files,
                    images_dir=res["images_dir"],
                    labels_dir=res["labels_dir"],
                    output_dir=yolo_out,
                    split_ratios=split_ratios,
                    seed=cfg.params.seed,
                    dataset_name=source_name,
                )
            
            elif res["type"] == "sun_annotation":
                print(f"  Processing SUN dataset source '{source_name}' for protocol '{protocol_name}'")
                

                sorted_pos = sorted(res["case_to_files"].keys(), key=lambda k: len(res["case_to_files"][k]))
                sorted_neg = sorted(res["neg_case_to_files"].keys(), key=lambda k: len(res["neg_case_to_files"][k]))
                
                # Postives: Positions 2, 7, 12, 17... (every 5th starting from index 2)
                pos_test = [sorted_pos[i] for i in range(2, len(sorted_pos), 5)]
                
                neg_test = [sorted_neg[i] for i in [2, 8] if i < len(sorted_neg)]
                
                out_img = os.path.join(yolo_out, "images", "test")
                out_lbl = os.path.join(yolo_out, "labels", "test")
                os.makedirs(out_img, exist_ok=True)
                os.makedirs(out_lbl, exist_ok=True)
                
                test_files = []
                test_groups = []
                
                # Direct copy
                def copy_sun_cases(cases, is_positive):
                    cat = "positive" if is_positive else "negative"
                    img_root = os.path.join(res["cache_root"], "images", cat)
                    lbl_root = os.path.join(res["cache_root"], "labels", cat)
                    
                    for c in cases:
                        c_name = f"case{c}" if is_positive else str(c)
                        test_groups.append(c_name)
                        
                        src_img_dir = os.path.join(img_root, c_name)
                        src_lbl_dir = os.path.join(lbl_root, c_name)
                        
                        if not os.path.exists(src_img_dir): 
                            continue
                        
                        for f in os.listdir(src_img_dir):
                            test_files.append(f)
                            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(out_img, f))
                            
                            # Generación automática de YOLO txt
                            txt_name = os.path.splitext(f)[0] + ".txt"
                            src_txt = os.path.join(src_lbl_dir, txt_name)
                            dst_txt = os.path.join(out_lbl, txt_name)
                            
                            if os.path.exists(src_txt):
                                shutil.copy2(src_txt, dst_txt)
                            else:
                                open(dst_txt, 'w').close()
                                
                print("  Copying stratified positive cases to test...")
                copy_sun_cases(pos_test, is_positive=True)
                
                print("  Copying stratified negative cases to test...")
                copy_sun_cases(neg_test, is_positive=False)
                
                images_per_group = {}
                for c in pos_test:
                    images_per_group[f"case{c}"] = len(res["case_to_files"][c])
                for c in neg_test:
                    images_per_group[str(c)] = len(res["neg_case_to_files"][c])

                split_metadata = {
                    "dataset_name": source_name,
                    "strategy": "stratified_by_size",
                    "splits": {
                        "test": {
                            "group_ids": sorted(test_groups),
                            "total_images": len(test_files),
                            "images_per_group": images_per_group,
                            "files": sorted(test_files)
                        }
                    }
                }
                json_path = os.path.join(yolo_out, f"split_metadata_{source_name}.json")
                with open(json_path, "w") as f:
                    json.dump(split_metadata, f, indent=4)
                    
                print(f"  [Frozen Split] Saved stratified SUN metadata to {json_path}")

        create_data_yaml(yolo_out)
        protocol_outputs.append((protocol_name, yolo_out))
        
    return protocol_outputs


def transfer_to_jetson(protocol_outputs, connection_info):
    """Pings and safely registers out the processed target evaluation splits to edge hardware."""
    print("\n" + "-"*50 + "\nSending test sets to Jetson\n" + "-"*50)
    if not ping(connection_info.host, connection_info.port):
        print("Skipping transfer: Jetson host is unreachable.")
        return

    with tempfile.TemporaryDirectory() as temp_transfer_root:
        for protocol_name, protocol_output_dir in protocol_outputs:
            local_test_images_dir = os.path.join(protocol_output_dir, "images", "test")
            local_test_labels_dir = os.path.join(protocol_output_dir, "labels", "test")

            temp_transfer_dir = os.path.join(temp_transfer_root, f"jetson_test_set_{protocol_name}")
            temp_images_test_dir = os.path.join(temp_transfer_dir, "images", "test")
            temp_labels_test_dir = os.path.join(temp_transfer_dir, "labels", "test")
            
            os.makedirs(temp_images_test_dir, exist_ok=True)
            os.makedirs(temp_labels_test_dir, exist_ok=True)

            copy_yolo_files(local_test_images_dir, local_test_labels_dir, temp_images_test_dir, temp_labels_test_dir)
            
            with open(os.path.join(temp_transfer_dir, "data.yaml"), "w") as yaml_file:
                yaml_file.write("train: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames: ['polyp']\n")

            remote_test_dir = os.path.join(connection_info.test_folder_remote, protocol_name)
            transfer_folder(
                host=connection_info.host, port=connection_info.port,
                username=connection_info.username, password=connection_info.password,
                local_dir=temp_transfer_dir, remote_dir=remote_test_dir,
                tar_name=f"test_folder_{protocol_name}.tar.gz"
            )

            break


@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
    base_path = cfg.files.base_path
    force_rebuild = cfg.params.force_rebuild_cache
    dataset_results = {}

    print("-" * 50 + "\nProcessing raw datasets to YOLO bounding boxes\n" + "-" * 50)

    for dataset_name, dataset_info in cfg.files.raw_datasets.items():
        ds_type = getattr(dataset_info, 'type', 'mask')
        duplicate_threshold = dataset_info.duplicate_threshold

        try:
            if ds_type == "mask":
                img_d, lbl_d = process_mask_dataset(
                    dataset_name, dataset_info, base_path, duplicate_threshold, force_rebuild
                )
                dataset_results[dataset_name] = {
                    "type": "mask", "images_dir": img_d, "labels_dir": lbl_d
                }
            elif ds_type == "sun_annotation":
                c_root, case_files, neg_case_files = process_sun_dataset(
                    dataset_name, dataset_info, base_path, duplicate_threshold, force_rebuild, drop_every_n=1
                )
                dataset_results[dataset_name] = {
                    "type": "sun_annotation", "cache_root": c_root,
                    "case_to_files": case_files, "neg_case_to_files": neg_case_files
                }
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}\n")

    protocol_outputs = build_protocols(cfg, dataset_results, base_path)
    transfer_to_jetson(protocol_outputs, cfg.connection)
    print("\nPreprocess pipeline completed successfully.")


if __name__ == "__main__":
    main()