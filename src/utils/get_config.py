import sys
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from config import PolypDetectionConfig

def get_config(config_name: str = "conf") -> PolypDetectionConfig:
    """
    Returns a PolypDetectionConfig instance by composing the YAML files 
    and applying command-line overrides.
    """
    
    cs = ConfigStore.instance()
    cs.store(name="polyp_detection_config", node=PolypDetectionConfig)

    overrides = sys.argv[1:]

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
        
        return cfg # type: ignore