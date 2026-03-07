import cv2
import numpy as np
import os

def masks_to_yolo(input_dir, output_dir, class_id=0):
    """
    Transform a dir of masks into YOLO bounding box format.
    Assumes all objects belong to the same class (default 0, e.g., polyp).
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