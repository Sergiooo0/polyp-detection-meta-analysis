from models.registry import register_custom_modules
from ultralytics import YOLO
import numpy as np
import torch
from ultralytics import settings
import hydra
from hydra.core.config_store import ConfigStore
from config import PolypDetectionConfig

# Update a setting
settings.update({"mlflow": True})

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
    print(cfg)
    register_custom_modules()

# Reset settings to default values
#settings.reset()
if __name__ == "__main__":
    main()