from datetime import datetime
from ultralytics import YOLO
import os
import shutil
from ultralytics import settings
import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from config import PolypDetectionConfig

# Update a setting
settings.update({"mlflow": True})

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
    # Directory of Hydra outputs
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    # Set MLflow tracking URI to a local directory within the repository for better organization
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow_db_path = os.path.join(repo_dir, "mlruns.db")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_db_path}"

    #Group all runs under the same experiment name for better organization in MLflow UI
    experiment_name = cfg.params.experiment_name
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    mlflow.set_experiment(experiment_name)

    protocol_config = cfg.files.protocols[cfg.params.protocol]
    print(f"Using protocol: {cfg.params.protocol} - {protocol_config.description}")
    data_yaml_path = os.path.join(cfg.files.base_path, protocol_config.yolo_output_dir, "data.yaml")

    model_cfg = cfg.params.model
    pretained_weights = cfg.params.pretrained_weights

    if model_cfg and pretained_weights:
        print(f"Using custom model config: {model_cfg} with pretrained weights: {pretained_weights}")
        model = YOLO(model_cfg)
        model.load(pretained_weights)
        model_name = os.path.splitext(os.path.basename(model_cfg))[0]
    elif model_cfg:
        print(f"Using custom model config: {model_cfg} with default pretrained weights")
        model = YOLO(model_cfg)
        model_name = os.path.splitext(os.path.basename(model_cfg))[0]
    elif pretained_weights:
        print(f"Using default model config with pretrained weights: {pretained_weights}")
        model = YOLO(pretained_weights)
        model_name = os.path.splitext(os.path.basename(pretained_weights))[0]
    else:
        raise ValueError("At least one of 'model' or 'pretrained_weights' must be specified in the configuration.")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{model_name}_{timestamp}"

    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id

    # Set the MLflow run ID as an environment variable so that it can be accessed by the YOLO training process
    os.environ["MLFLOW_RUN_ID"] = run_id

    # Save Hydra parameters path as a MLflow artifact for reference
    mlflow.log_artifact(os.path.join(hydra_output_dir, ".hydra/config.yaml"), artifact_path="hydra_config")

    # Add tags to filter runs by protocol and model in MLflow UI
    mlflow.set_tags({
        "protocol": cfg.params.protocol,
        "model": model_name
    })

    results = model.train(
        data=data_yaml_path,
        seed=cfg.params.seed,
        imgsz=cfg.params.img_size,
        epochs=cfg.params.epochs,
        batch=cfg.params.batch_size,
        lr0=cfg.params.lr,
        weight_decay=cfg.params.weight_decay,
        optimizer=cfg.params.optimizer,
        device=cfg.params.device,
        project=hydra_output_dir,
        name=run_name,
        exist_ok=True,
        patience=50,
    )

    # Remove weights from Hydra to save space. The weights are already saved in the MLflow run artifacts.
    weights_dir = os.path.join(hydra_output_dir, "yolo_results", "weights")

    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)

    # mean Average Precision at IoU=0.5 (mAP50)
    metrics = model.val()
    map50 = metrics.box.map50
    print(f"mAP50: {map50}")

    mlflow.end_run()
    
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    return map50


if __name__ == "__main__":
    main()