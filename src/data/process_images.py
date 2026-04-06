import cv2
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm

def deduplicate_consecutive_frames(file_list, images_dir, threshold=0.02, save_duplicates_dir=None):
    """
    Remove near-duplicate consecutive frames from an ordered list.

    Args:
        file_list: Ordered list of image filenames
        images_dir: Directory containing the images
        threshold: MSE difference threshold (0.0-1.0). Below this = duplicate
        save_duplicates_dir: If provided, saves side-by-side comparisons of detected duplicates

    Returns:
        (filtered_files, removed_files, stats_dict)
    """
    if threshold == 0.0 or len(file_list) == 0:
        return file_list, [], {"total": len(file_list), "kept": len(file_list), "removed": 0}

    # Sort files to ensure consistent ordering
    file_list = sorted(file_list)

    filtered = []
    removed = []
    prev_frame = None
    prev_filename = None
    duplicate_pairs = []  # For verification report

    for filename in file_list:
        img_path = os.path.join(images_dir, filename)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            continue

        # Resize for faster comparison
        frame_resized = cv2.resize(frame, (160, 160)).astype(np.float32)

        if prev_frame is None:
            filtered.append(filename)
            prev_frame = frame_resized
            prev_filename = filename
            continue

        # Calculate Mean Absolute Error normalized to [0, 1]
        diff = np.mean(np.abs(frame_resized - prev_frame)) / 255.0

        if diff > threshold:
            # Different enough - keep it
            filtered.append(filename)
            prev_frame = frame_resized
            prev_filename = filename
        else:
            # Duplicate - skip it
            removed.append(filename)
            duplicate_pairs.append((prev_filename, filename, diff))

    # Save duplicate pairs for visual verification
    if save_duplicates_dir and duplicate_pairs:
        os.makedirs(save_duplicates_dir, exist_ok=True)
        # Save up to 20 examples
        for i, (kept, removed_file, score) in enumerate(duplicate_pairs[:20]):
            _save_duplicate_comparison(
                os.path.join(images_dir, kept),
                os.path.join(images_dir, removed_file),
                score,
                os.path.join(save_duplicates_dir, f"dup_{i:03d}.png")
            )

    return filtered, removed, {
        "total": len(file_list),
        "kept": len(filtered),
        "removed": len(removed),
        "duplicate_pairs": duplicate_pairs
    }


def _save_duplicate_comparison(img1_path, img2_path, diff_score, output_path):
    """Save side-by-side comparison of two images marked as duplicates."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return

    # Resize for display
    h, w = 300, 400
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    # Create side-by-side comparison
    comparison = np.hstack([img1_resized, img2_resized])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, f"KEPT", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f"REMOVED (diff={diff_score:.4f})", (w + 10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, os.path.basename(img1_path), (10, h - 10), font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, os.path.basename(img2_path), (w + 10, h - 10), font, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, comparison)

def masks_to_yolo(input_dir, output_dir, class_id=0, file_list=None):
    """
    Transform a dir of masks into YOLO bounding box format.
    Assumes all objects belong to the same class (default 0, e.g., polyp).
    Works with flat-directory mask datasets (CVC-ClinicDB, CVC-ColonDB).

    Args:
        input_dir: Directory containing mask images
        output_dir: Directory where YOLO labels will be saved
        class_id: YOLO class id (default 0 for polyp)
        file_list: Optional list of filenames to process. If None, processes all files.

    Returns:
        List of processed filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get files to process
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    if file_list is None:
        files_to_process = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    else:
        files_to_process = file_list

    processed_files = []

    # Process each file
    for filename in files_to_process:
        # Filter to process only image files
        if not filename.lower().endswith(valid_ext):
            continue

        mask_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error loading image: {filename}")
            continue

        img_height, img_width = mask.shape

        # Binarize and find contours
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Open the .txt file to write the coordinates
        with open(txt_path, 'w') as f:
            for cnt in contours:
                if cv2.contourArea(cnt) > 50: # Filter out small noise
                    x, y, w, h = cv2.boundingRect(cnt)

                    # Calculate the normalized YOLO coordinates (from 0 to 1)
                    x_center = (x + w / 2.0) / img_width
                    y_center = (y + h / 2.0) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height

                    # Write the line in the file with 6 decimal places of precision
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        processed_files.append(filename)

    print(f"Saved YOLO files to: {output_dir}\n")

    return processed_files


def sun_annotations_to_yolo(annotation_dir, positive_dir, output_images_dir, output_labels_dir, class_id=0,
                            duplicate_threshold=0.02, dedup_report_dir=None):
    """
    Convert SUN-style annotations (per-case txt files with x_min,y_min,x_max,y_max,class)
    into YOLO bounding box format.

    Also copies the positive images into a single flat directory so downstream code
    can treat them the same way as CVC datasets.

    Args:
        annotation_dir: Path to sundatabase_positive/annotation_txt/ with caseX.txt files
        positive_dir:   Path to sundatabase_positive/ containing caseX/ image folders
        output_images_dir: Path where all images will be copied (flat)
        output_labels_dir: Path where YOLO .txt labels will be saved (flat)
        class_id: YOLO class id (default 0 for polyp)
        duplicate_threshold: Threshold for deduplication (0.0 to disable)
        dedup_report_dir: Directory to save deduplication verification images

    Returns:
        dict mapping case_id (int) -> list of image filenames belonging to that case
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    case_to_files = {}
    total_images = 0
    total_bboxes = 0
    total_original = 0
    total_removed = 0

    # Find all annotation txt files
    annotation_files = sorted(glob.glob(os.path.join(annotation_dir, "case*.txt")))

    if not annotation_files:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")

    for ann_file in tqdm(annotation_files, desc="Processing sun annotations"):
        # Extract case ID from filename (e.g., "case1.txt" -> 1)
        case_name = os.path.splitext(os.path.basename(ann_file))[0]  # "case1"
        case_id = int(case_name.replace("case", ""))
        case_image_dir = os.path.join(positive_dir, case_name)

        if not os.path.isdir(case_image_dir):
            print(f"Warning: Image folder missing for {case_name}: {case_image_dir}")
            continue

        case_files = []
        case_seen_files = set()
        labels_by_image = {}
        image_size_cache = {}

        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: "filename x_min,y_min,x_max,y_max,class_id"
                parts = line.split(' ')
                if len(parts) < 2:
                    print(f"Warning: Malformed annotation line in {case_name}: {line}")
                    continue

                img_filename = parts[0]
                bbox_str = parts[1]

                # Parse bbox: x_min,y_min,x_max,y_max,class_id
                bbox_parts = bbox_str.split(',')
                if len(bbox_parts) < 4:
                    print(f"Warning: Malformed bbox in {case_name}: {bbox_str}")
                    continue

                x_min = int(bbox_parts[0])
                y_min = int(bbox_parts[1])
                x_max = int(bbox_parts[2])
                y_max = int(bbox_parts[3])

                src_img_path = os.path.join(case_image_dir, img_filename)
                if not os.path.exists(src_img_path):
                    print(f"Warning: Image not found: {src_img_path}")
                    continue

                # Read image only to get dimensions (grayscale is cheaper than color)
                if img_filename not in image_size_cache:
                    img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image: {src_img_path}")
                        continue
                    image_size_cache[img_filename] = img.shape[:2]

                img_height, img_width = image_size_cache[img_filename]

                # Convert to YOLO normalized format
                w = x_max - x_min
                h = y_max - y_min
                x_center = (x_min + w / 2.0) / img_width
                y_center = (y_min + h / 2.0) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                # Clamp values to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))


                # Accumulate YOLO lines per image and write once at end of case
                if img_filename not in labels_by_image:
                    labels_by_image[img_filename] = []
                labels_by_image[img_filename].append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

                if img_filename not in case_seen_files:
                    case_seen_files.add(img_filename)
                    case_files.append(img_filename)
                total_bboxes += 1

        # Deduplicate case files BEFORE copying
        if duplicate_threshold > 0.0 and case_files:
            case_dedup_dir = os.path.join(dedup_report_dir, f"positive_{case_name}") if dedup_report_dir else None
            filtered_files, removed_files, stats = deduplicate_consecutive_frames(
                case_files, case_image_dir, duplicate_threshold, case_dedup_dir
            )
            total_original += stats["total"]
            total_removed += stats["removed"]
            case_files = filtered_files

        # Now copy only the filtered (non-duplicate) images
        for img_filename in case_files:
            src_img_path = os.path.join(case_image_dir, img_filename)
            dst_img_path = os.path.join(output_images_dir, img_filename)

            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)

            total_images += 1

        # Write labels only for kept images
        for img_filename in case_files:
            if img_filename in labels_by_image:
                txt_filename = os.path.splitext(img_filename)[0] + ".txt"
                txt_path = os.path.join(output_labels_dir, txt_filename)
                with open(txt_path, 'w') as lf:
                    lf.writelines(labels_by_image[img_filename])

        case_to_files[case_id] = case_files

    print(f"Processed {len(case_to_files)} cases, {total_images} images, {total_bboxes} bboxes")
    if duplicate_threshold > 0.0:
        print(f"Positive deduplication: {total_original} -> {total_images} (removed {total_removed}, {100*total_removed/max(1,total_original):.1f}%)")
    print(f"Saved images to: {output_images_dir}")
    print(f"Saved YOLO labels to: {output_labels_dir}\n")

    return case_to_files


def sun_copy_negative_images(negative_dir, output_images_dir, output_labels_dir,
                             duplicate_threshold=0.02, dedup_report_dir=None):
    """
    Copy SUN negative (no-polyp) images to the flat output directories.
    Creates empty YOLO label files for each image (background/healthy frames).

    Args:
        negative_dir: Path to sundatabase_negative/ containing caseX/ folders
        output_images_dir: Path where images will be copied (flat)
        output_labels_dir: Path where empty YOLO .txt labels will be created
        duplicate_threshold: Threshold for deduplication (0.0 to disable)
        dedup_report_dir: Directory to save deduplication verification images

    Returns:
        dict mapping case_name (str like "neg_case1") -> list of image filenames
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    neg_case_to_files = {}
    total = 0
    total_original = 0
    total_removed = 0

    for entry in tqdm(sorted(os.listdir(negative_dir)), desc="Processing negative images"):
        case_dir = os.path.join(negative_dir, entry)
        if not os.path.isdir(case_dir) or not entry.startswith("case"):
            continue

        # Collect all image files in this case
        case_files = []
        for img_file in sorted(os.listdir(case_dir)):
            if img_file.lower().endswith(valid_ext):
                case_files.append(img_file)

        # Deduplicate case files BEFORE copying
        if duplicate_threshold > 0.0 and case_files:
            case_dedup_dir = os.path.join(dedup_report_dir, f"negative_{entry}") if dedup_report_dir else None
            filtered_files, removed_files, stats = deduplicate_consecutive_frames(
                case_files, case_dir, duplicate_threshold, case_dedup_dir
            )
            total_original += stats["total"]
            total_removed += stats["removed"]
            case_files = filtered_files

        # Copy only the filtered files
        for img_file in case_files:
            src_path = os.path.join(case_dir, img_file)
            dst_img_path = os.path.join(output_images_dir, img_file)
            txt_filename = os.path.splitext(img_file)[0] + ".txt"
            dst_txt_path = os.path.join(output_labels_dir, txt_filename)

            if not os.path.exists(dst_img_path):
                shutil.copy2(src_path, dst_img_path)

            # Create empty label file (no polyp)
            if not os.path.exists(dst_txt_path):
                open(dst_txt_path, 'w').close()

            total += 1

        neg_case_to_files[f"neg_{entry}"] = case_files

    print(f"Copied {total} negative images from {len(neg_case_to_files)} cases")
    if duplicate_threshold > 0.0:
        print(f"Negative deduplication: {total_original} -> {total} (removed {total_removed}, {100*total_removed/max(1,total_original):.1f}%)")
    print(f"Saved to: {output_images_dir}\n")

    return neg_case_to_files

def check_yolo_bboxes(images_dir, labels_dir, output_dir, num_images=50):
    """
    Read YOLO .txt files and draw bounding boxes on the original images for verification.
    Only processes images that have corresponding label files.
    Use num_images=-1 to process all images, or set a specific number to limit the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get valid images that have corresponding label files
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    all_label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # Get image filenames from label filenames
    images_with_labels = []
    for label_file in all_label_files:
        base_name = os.path.splitext(label_file)[0]
        # Find corresponding image with any valid extension
        for ext in valid_ext:
            img_filename = base_name + ext
            img_path = os.path.join(images_dir, img_filename)
            if os.path.exists(img_path):
                images_with_labels.append(img_filename)
                break

    # Sort for consistent ordering
    images_with_labels = sorted(images_with_labels)

    # Limit the number of images if not -1
    if num_images != -1:
        images_to_process = images_with_labels[:num_images]
    else:
        images_to_process = images_with_labels

    for filename in images_to_process:
        img_path = os.path.join(images_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_filename)
        # Save always with .png extension
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)

        # Load original image in color
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {filename}")
            continue

        img_height, img_width = img.shape[:2]

        # Read the YOLO .txt file
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Unnormalize to pixel coordinates
                    w = int(w_norm * img_width)
                    h = int(h_norm * img_height)
                    x_min = int((x_center * img_width) - (w / 2.0))
                    y_min = int((y_center * img_height) - (h / 2.0))

                    # Draw the bounding box in green on the image
                    cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)

                    # Opcional: Put a text label above the bounding box
                    cv2.putText(img, f"Polyp", (x_min, max(y_min - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the test image
        cv2.imwrite(output_path, img)

    print(f"\nSaved {len(images_to_process)} verification images to: {output_dir}\n")