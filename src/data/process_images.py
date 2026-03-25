import cv2
import os
import glob
import shutil
from tqdm import tqdm

def masks_to_yolo(input_dir, output_dir, class_id=0):
    """
    Transform a dir of masks into YOLO bounding box format.
    Assumes all objects belong to the same class (default 0, e.g., polyp).
    Works with flat-directory mask datasets (CVC-ClinicDB, CVC-ColonDB).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Follows all files in the input directory
    for filename in os.listdir(input_dir):
        # Filter to process only image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
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
                    
    print(f"Saved YOLO files to: {output_dir}\n")


def sun_annotations_to_yolo(annotation_dir, positive_dir, output_images_dir, output_labels_dir, class_id=0):
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
    
    Returns:
        dict mapping case_id (int) -> list of image filenames belonging to that case
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    case_to_files = {}
    total_images = 0
    total_bboxes = 0
    
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

                    # Copy image to flat output directory without re-encoding
                    dst_img_path = os.path.join(output_images_dir, img_filename)
                    if not os.path.exists(dst_img_path):
                        shutil.copy2(src_img_path, dst_img_path)

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
                    total_images += 1
                total_bboxes += 1

        # Write labels once per image (overwrite to avoid accumulation on reruns)
        for img_filename, yolo_lines in labels_by_image.items():
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"
            txt_path = os.path.join(output_labels_dir, txt_filename)
            with open(txt_path, 'w') as lf:
                lf.writelines(yolo_lines)
        
        case_to_files[case_id] = case_files
    
    print(f"Processed {len(case_to_files)} cases, {total_images} images, {total_bboxes} bboxes")
    print(f"Saved images to: {output_images_dir}")
    print(f"Saved YOLO labels to: {output_labels_dir}\n")
    
    return case_to_files


def sun_copy_negative_images(negative_dir, output_images_dir, output_labels_dir):
    """
    Copy SUN negative (no-polyp) images to the flat output directories.
    Creates empty YOLO label files for each image (background/healthy frames).
    
    Args:
        negative_dir: Path to sundatabase_negative/ containing caseX/ folders
        output_images_dir: Path where images will be copied (flat)
        output_labels_dir: Path where empty YOLO .txt labels will be created
    
    Returns:
        dict mapping case_name (str like "neg_case1") -> list of image filenames
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    neg_case_to_files = {}
    total = 0
    
    for entry in tqdm(sorted(os.listdir(negative_dir)), desc="Processing negative images"):
        case_dir = os.path.join(negative_dir, entry)
        if not os.path.isdir(case_dir) or not entry.startswith("case"):
            continue
        
        case_files = []
        for img_file in os.listdir(case_dir):
            if not img_file.lower().endswith(valid_ext):
                continue
            
            src_path = os.path.join(case_dir, img_file)
            dst_img_path = os.path.join(output_images_dir, img_file)
            txt_filename = os.path.splitext(img_file)[0] + ".txt"
            dst_txt_path = os.path.join(output_labels_dir, txt_filename)
            
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_path, dst_img_path)
            
            # Create empty label file (no polyp)
            if not os.path.exists(dst_txt_path):
                open(dst_txt_path, 'w').close()
            
            case_files.append(img_file)
            total += 1
        
        neg_case_to_files[f"neg_{entry}"] = case_files
    
    print(f"Copied {total} negative images from {len(neg_case_to_files)} cases")
    print(f"Saved to: {output_images_dir}\n")
    
    return neg_case_to_files

def check_yolo_bboxes(images_dir, labels_dir, output_dir, num_images=5):
    """
    Read YOLO .txt files and draw bounding boxes on the original images for verification.
    Use num_images=-1 to process all images, or set a specific number to limit the output.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get valid images in the directory
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    
    # Limit the number of images if not -1
    if num_images != -1:
        images_to_process = all_images[:num_images]
    else:
        images_to_process = all_images
        
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
        
        # Read the YOLO .txt file if it exists
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
        else:
            print(f"Warning: The image {filename} does not have an associated .txt file.")
            
        # Save the test image
        cv2.imwrite(output_path, img)
        
    print(f"\nSaved {len(images_to_process)} verification images to: {output_dir}\n")