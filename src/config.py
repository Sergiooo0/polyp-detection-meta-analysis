from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RawDataset:
    """
    Configuration for a single raw dataset.
    
    Supported types:
      - 'mask': Flat directory with binary masks (e.g. CVC-ClinicDB, CVC-ColonDB).
                Requires: images_dir, mask_dir. Optional: metadata_file.
      - 'sun_annotation': SUN-style dataset with per-case folders and annotation txt files.
                Requires: positive_dir, negative_dir, annotation_dir.
                Optional: metadata_file.
    """
    type: str # "mask" or "sun_annotation"
    
    # --- Fields for type="mask" ---
    images_dir: Optional[str] = None
    mask_dir: Optional[str] = None
    
    # --- Fields for type="sun_annotation" ---
    positive_dir: Optional[str] = None
    negative_dir: Optional[str] = None
    annotation_dir: Optional[str] = None
    
    # --- Common fields ---
    metadata_file: Optional[str] = None

@dataclass
class ProtocolConfig:
    description: str
    train_source: str
    ood_source: str
    yolo_output_dir: str
    
@dataclass
class Params:
    # Parameters for the training model process
    protocol: str
    seed : int
    val_ratio: float
    # Diff threshold for considering two consecutive frames as near-duplicates (0.0 to 1.0, representing percentage of pixel difference).
    duplicate_threshold: float
    # Force regeneration of cached intermediate artifacts (e.g., SUN flat images/labels).
    force_rebuild_cache: bool
    model: str
    pretrained_weights: str
    img_size: int
    experiment_name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str
    device: list[int]

@dataclass
class Files:
    # File paths for the training process
    base_path: str
    raw_datasets: Dict[str, RawDataset]
    protocols: Dict[str, ProtocolConfig]

@dataclass
class PolypDetectionConfig:
    # Configuration for the polyp detection process
    params: Params
    files: Files
