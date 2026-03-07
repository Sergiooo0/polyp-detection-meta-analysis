from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RawDataset:
    images_dir: str
    mask_dir: str
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
    seed : int
    model: str
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str

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
