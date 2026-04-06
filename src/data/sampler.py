import os
import random
import yaml


def _classify_images(images_dir):
    """
    Classify images in a directory as positive or negative based on their labels.

    Returns:
        (positives, negatives) lists of absolute image paths
    """
    labels_dir = images_dir.replace('images', 'labels')

    positives = []
    negatives = []

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # Positive if label file exists and is not empty, otherwise negative
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            positives.append(img_path)
        else:
            negatives.append(img_path)

    return positives, negatives


def _balance_split(positives, negatives, r, seed):
    """
    Balance a split by sampling negatives to match the ratio r (negatives = positives * r).

    Returns:
        (balanced_list, num_positives, num_sampled_negatives)
    """
    random.seed(seed)
    num_negatives_to_sample = int(len(positives) * r)

    # Make sure we don't sample more negatives than we have available
    num_negatives_to_sample = min(num_negatives_to_sample, len(negatives))

    sampled_negatives = random.sample(negatives, num_negatives_to_sample)
    balanced_list = positives + sampled_negatives

    return balanced_list, len(positives), num_negatives_to_sample


def create_balanced_dataset(data_yaml_path, output_yaml_path, output_dir, r=1.0, seed=42):
    """
    Read the original data.yaml, balance BOTH train AND val sets with the same
    positive:negative ratio, and create a new .yaml with .txt files for each split.

    This ensures consistency between train and val, and corrects any ratio imbalance
    that may have occurred during preprocessing.

    Args:
        data_yaml_path: Path to the original data.yaml
        output_yaml_path: Path for the new balanced data.yaml
        output_dir: Directory to store the balanced .txt files
        r: Ratio of negatives to positives (1.0 = equal, 2.0 = 2x negatives)
        seed: Random seed for reproducibility

    Returns:
        Path to the new data.yaml
    """
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(data_yaml_path)
    os.makedirs(output_dir, exist_ok=True)

    new_data_cfg = data_cfg.copy()
    stats = {}

    # Balance train and val splits
    for split in ['train', 'val']:
        if split not in data_cfg:
            continue

        images_dir = data_cfg[split]
        if not os.path.isabs(images_dir):
            images_dir = os.path.join(base_dir, images_dir)

        # Skip if the path is already a .txt file (already balanced)
        if images_dir.endswith('.txt'):
            print(f"{split}: Already using .txt file, skipping")
            continue

        positives, negatives = _classify_images(images_dir)

        # Use different seed offsets for train/val to ensure different samples
        split_seed = seed if split == 'train' else seed + 1
        balanced_list, n_pos, n_neg = _balance_split(positives, negatives, r, split_seed)

        # Save to .txt file
        txt_path = os.path.join(output_dir, f'{split}_balanced.txt')
        with open(txt_path, 'w') as f:
            for item in balanced_list:
                f.write(f"{item}\n")

        new_data_cfg[split] = txt_path
        stats[split] = {'positives': n_pos, 'negatives': n_neg, 'total': n_pos + n_neg}

        print(f"{split.capitalize()} - Positives: {n_pos}, Negatives: {n_neg}, "
              f"Total: {n_pos + n_neg} (Ratio: {r})")

    # Keep test as absolute path (no balancing for test)
    if 'test' in new_data_cfg:
        test_path = new_data_cfg['test']
        if not os.path.isabs(test_path):
            new_data_cfg['test'] = os.path.join(base_dir, test_path)

    with open(output_yaml_path, 'w') as f:
        yaml.dump(new_data_cfg, f)

    # Print summary
    if 'train' in stats and 'val' in stats:
        train_total = stats['train']['total']
        val_total = stats['val']['total']
        total = train_total + val_total
        print(f"\nBalanced split summary:")
        print(f"  Train: {train_total} ({100*train_total/total:.1f}%)")
        print(f"  Val:   {val_total} ({100*val_total/total:.1f}%)")

    return output_yaml_path