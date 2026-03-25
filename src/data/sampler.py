import os
import random
import yaml

def create_balanced_train_split(data_yaml_path, output_yaml_path, r=1.0, seed=42):
    """
    Lee el data.yaml original, balancea los negativos/positivos del set de train,
    y crea un nuevo .yaml y un .txt para el entrenamiento de YOLO.
    """
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # A YOLO data.yaml typically has a structure like:
    # train: path/to/train/images
    # val: path/to/val/images
    # test: path/to/test/images
    train_images_dir = data_cfg['train'] 
    
    # Get the absolute path to the train images directory based on the location of the data.yaml
    base_dir = os.path.dirname(data_yaml_path)
    if not os.path.isabs(train_images_dir):
        train_images_dir = os.path.join(base_dir, train_images_dir)

    # Get the corresponding labels directory by replacing 'images' with 'labels' in the path
    train_labels_dir = train_images_dir.replace('images', 'labels')

    positives = []
    negatives = []

    # Classify images as positive (have a corresponding non-empty .txt label) or negative (no label or empty label)
    for img_name in os.listdir(train_images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(train_images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)

        # Positive if label file exists and is not empty, otherwise negative
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            positives.append(img_path)
        else:
            negatives.append(img_path)

    # Apply determenistic sampling to balance the dataset according to the specified ratio r
    random.seed(seed)
    num_negatives_to_sample = int(len(positives) * r)
    
    # Make sure we don't sample more negatives than we have available
    num_negatives_to_sample = min(num_negatives_to_sample, len(negatives))
    
    sampled_negatives = random.sample(negatives, num_negatives_to_sample)

    # Save all together in a new .txt file for training
    balanced_train_list = positives + sampled_negatives
    
    train_txt_path = os.path.join(base_dir, "train_balanced.txt")
    with open(train_txt_path, 'w') as f:
        for item in balanced_train_list:
            f.write("%s\n" % item)

    # Create a new YAML file pointing to the .txt file instead of the folder
    new_data_cfg = data_cfg.copy()
    new_data_cfg['train'] = train_txt_path
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(new_data_cfg, f)
        
    print(f"Positives: {len(positives)}, Negatives: {len(sampled_negatives)} (Ratio: {r})")
    
    return output_yaml_path