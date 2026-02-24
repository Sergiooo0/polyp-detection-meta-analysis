from dataclasses import dataclass

@dataclass
class Params:
    # Parameters for the training model process
    model: str
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float
    optimizer: str

@dataclass
class Files:
    # File paths for the training process
    train_data: str
    val_data: str
    test_data: str
    model_save_path: str

@dataclass
class PolypDetectionConfig:
    # Configuration for the polyp detection process
    params: Params
    files: Files
