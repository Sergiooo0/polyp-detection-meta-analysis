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
    duplicate_threshold: float = 0.0

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
    # Force regeneration of cached intermediate artifacts (e.g., SUN flat images/labels).
    force_rebuild_cache: bool
    model: Optional[str]
    pretrained_weights: str
    model_name: Optional[str]
    img_size: int
    experiment_name: str
    epochs: int
    batch_size: int
    lr: float
    lrf: float
    momentum: float
    cos_lr: bool
    warmup_epochs: float
    patience: int
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    flipud: float
    fliplr: float
    mosaic: float
    close_mosaic: int
    scale: float
    erasing: float
    mixup: float
    final_phase_lr: float
    final_no_mosaic_epochs: int
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
class Connection:
    host: str
    port: int
    username: str
    password: str
    test_folder_remote: str

@dataclass
class MlflowConfig:
    # Configuration for MLflow tracking and MinIO artifact storage
    tracking_uri: Optional[str]
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    artifact_location: str

@dataclass
class TestConfig:
    # Evaluation specific parameters
    top_k: int              # How many top models to evaluate on Jetson
    metric: str             # Which metric to use to rank the best runs
    half_precision: bool    # Use FP16 precision
    experiment_name: str    # MLflow experiment name to query for runs

@dataclass
class PolypDetectionConfig:
    # Configuration for the polyp detection process
    params: Params
    files: Files
    connection: Connection
    mlflow: MlflowConfig
    test: TestConfig