from datetime import datetime
import gc
import json
import os
import shutil
import pandas as pd
import hydra
import mlflow
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from ultralytics import YOLO, settings

from config import PolypDetectionConfig
from data.sampler import create_balanced_dataset

# Update a setting
settings.update({"mlflow": True})

cs = ConfigStore.instance()
cs.store(name="polyp_detection_config", node=PolypDetectionConfig)

def get_best_epoch(results_csv_path: str) -> int:
    # Load the training log
    results = pd.read_csv(results_csv_path)

    # Strip spaces
    results.columns = results.columns.str.strip()

    # Calculate fitness
    results["fitness"] = results["metrics/precision(B)"] * 0.0 + results["metrics/recall(B)"] * 0.2 + results["metrics/mAP50(B)"] * 0.1 + results["metrics/mAP50-95(B)"] * 0.7

    # Find the epoch with the highest fitness
    best_epoch = results['fitness'].idxmax() + 1

    print(f"Best model was saved at epoch: {best_epoch}")

    return best_epoch


def configure_mlflow_environment(cfg: PolypDetectionConfig, repo_dir: str) -> None:
    # Set MLflow tracking URI and MinIO credentials from config
    tracking_uri = cfg.mlflow.tracking_uri
    if not tracking_uri:
        mlflow_db_path = os.path.join(repo_dir, "mlruns.db")
        tracking_uri = f"sqlite:///{mlflow_db_path}"

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg.mlflow.s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.mlflow.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.mlflow.aws_secret_access_key


def configure_mlflow_experiment(cfg: PolypDetectionConfig) -> None:
    # Group all runs under the same experiment name for better organization in MLflow UI
    experiment_name = cfg.params.experiment_name
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    # Try to get the experiment by the name, if it doesn't exist, create it with the MinIO bucket
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=cfg.mlflow.artifact_location)
    mlflow.set_experiment(experiment_name)


def resolve_data_yaml_path(cfg: PolypDetectionConfig, hydra_output_dir: str, original_data_yaml_path: str) -> str:
    # Verify if the dataset need a balance between negatives and positives examples
    # This is the case of protocol t2 with the SUN dataset, which has 2 negatives examples per positive example
    if cfg.params.protocol == "t2":
        print(f"Applying deterministic negative sampling for protocol {cfg.params.protocol} (Seed: 42).")

        # Save new files in Hydra output file, don't modify the dataset
        balanced_yaml_path = os.path.join(hydra_output_dir, f"data_balanced_seed_42.yaml")

        data_yaml_path = create_balanced_dataset(
            data_yaml_path=original_data_yaml_path,
            output_yaml_path=balanced_yaml_path,
            output_dir=hydra_output_dir,
            r=0.5,
            seed=42,
        )

        # Log the balanced dataset files created by create_balanced_dataset
        mlflow.log_artifact(data_yaml_path, artifact_path="dataset_splits")
        train_balanced_txt = os.path.join(hydra_output_dir, "train_balanced.txt")
        val_balanced_txt = os.path.join(hydra_output_dir, "val_balanced.txt")

        if os.path.exists(train_balanced_txt):
            mlflow.log_artifact(train_balanced_txt, artifact_path="dataset_splits")
        if os.path.exists(val_balanced_txt):
            mlflow.log_artifact(val_balanced_txt, artifact_path="dataset_splits")

        return data_yaml_path

    return original_data_yaml_path


def initialize_model(cfg: PolypDetectionConfig) -> tuple[YOLO, str]:
    model_cfg = cfg.params.model
    pretrained_weights = cfg.params.pretrained_weights

    if model_cfg and pretrained_weights:
        print(f"Using custom model config: {model_cfg} with pretrained weights: {pretrained_weights}")
        model = YOLO(model_cfg)
        model.load(pretrained_weights)
        model_name = os.path.splitext(os.path.basename(model_cfg))[0]
    elif model_cfg:
        print(f"Using custom model config: {model_cfg} with default pretrained weights")
        model = YOLO(model_cfg)
        model_name = os.path.splitext(os.path.basename(model_cfg))[0]
    elif pretrained_weights:
        print(f"Using default model config with pretrained weights: {pretrained_weights}")
        model = YOLO(pretrained_weights)
        model_name = os.path.splitext(os.path.basename(pretrained_weights))[0]
    else:
        raise ValueError("At least one of 'model' or 'pretrained_weights' must be specified in the configuration.")
    
    if cfg.params.model_name:
        model_name = cfg.params.model_name

    return model, model_name


def train_model(
    model: YOLO,
    cfg: PolypDetectionConfig,
    data_yaml_path: str,
    hydra_output_dir: str,
    run_name: str,
) -> str:
    train_common_params = {
        "data": data_yaml_path,
        "seed": cfg.params.seed,
        "imgsz": cfg.params.img_size,
        "batch": cfg.params.batch_size,
        "lrf": cfg.params.lrf,
        "weight_decay": cfg.params.weight_decay,
        "momentum": cfg.params.momentum,
        "optimizer": cfg.params.optimizer,
        "cos_lr": cfg.params.cos_lr,
        "hsv_s": cfg.params.hsv_s,
        "hsv_v": cfg.params.hsv_v,
        "degrees": cfg.params.degrees,
        "translate": cfg.params.translate,
        "flipud": cfg.params.flipud,
        "fliplr": cfg.params.fliplr,
        "mosaic": cfg.params.mosaic,
        "scale": cfg.params.scale,
        "erasing": cfg.params.erasing,
        "mixup": cfg.params.mixup,
        "device": cfg.params.device,
        "deterministic": False,
        "project": hydra_output_dir,
        "name": run_name,
        "exist_ok": True,
        "optimizer": "AdamW",
	    "workers": 4,
    }

    # Base training phase (with mosaic enabled and early stopping).
    # https://docs.ultralytics.com/modes/train/#musgd-optimizer
    base_train_params = {
        **train_common_params,
        "epochs": cfg.params.epochs,
        "lr0": cfg.params.lr,
        "warmup_epochs": cfg.params.warmup_epochs,
        "mosaic": cfg.params.mosaic,
        "patience": cfg.params.patience,
        "close_mosaic": cfg.params.close_mosaic if cfg.params.final_no_mosaic_epochs > 0 else 0,
    }
    model.train(**base_train_params)

    # Optional fine-tuning phase without mosaic augmentation.
    # Yolo has close_mosaic parameter which does what we want to do here, but it won't be apply in early stopping
    final_run_name = run_name
    if cfg.params.final_no_mosaic_epochs > 0:
        # Grab the weights from the end of phase 1
        last_pt_path = os.path.join(hydra_output_dir, run_name, "weights", "last.pt")
        model = YOLO(last_pt_path)

        final_run_name = f"{run_name}_finetune"
        final_train_params = {
            **train_common_params,
            "epochs": cfg.params.final_no_mosaic_epochs,
            "lr0": cfg.params.final_phase_lr,
            "warmup_epochs": 0.0,
            "mosaic": 0.0,
            "patience": cfg.params.final_no_mosaic_epochs,
            "close_mosaic": 0,
            "name": final_run_name,
        }
        model.train(**final_train_params)

    del model 
    gc.collect()

    return final_run_name


def extract_optimal_conf_threshold(metrics) -> float:
    optimal_conf = 0.0

    # metrics.box.curves_results is a list of lists: [x, y, x_label, y_label]
    for curve in metrics.box.curves_results:
        if "F1" in str(curve[3]):
            conf_thresholds = curve[0]  # X axis: The thresholds that YOLO tested

            # Y axis: The F1 scores.
            # curve[1] usually has shape (nc, 1000). We use [0] to get the first class (polyps)
            f1_scores = curve[1][0] if len(curve[1].shape) > 1 else curve[1]

            best_idx = np.argmax(f1_scores)

            # Get the confidence threshold corresponding to the best F1 score
            optimal_conf = float(conf_thresholds[best_idx])
            break

    return optimal_conf


def save_deployment_metadata(
    hydra_output_dir: str,
    model_name: str,
    optimal_conf: float,
    img_size: int,
    timestamp: str,
    seed: int,
    protocol: str,
) -> str:
    # Save relevant information about the model to use in inference.py
    deployment_metadata = {
        "model_name": model_name,
        "optimal_conf_threshold": optimal_conf,
        "imgsz": img_size,
        "training_date": timestamp,
        "seed": seed,
        "protocol": protocol,
    }

    metadata_path = os.path.join(hydra_output_dir, "deployment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(deployment_metadata, f, indent=4)

    return metadata_path


def cleanup_run_artifacts(hydra_output_dir: str, run_name: str) -> None:
    # Remove weights from Hydra to save space. The weights are already saved in the MLflow run artifacts.
    weights_dir = os.path.join(hydra_output_dir, run_name, "weights")
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)


@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
    print(cfg.params)
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configure_mlflow_environment(cfg, repo_dir)
    configure_mlflow_experiment(cfg)

    protocol_config = cfg.files.protocols[cfg.params.protocol]
    print(f"Using protocol: {cfg.params.protocol} - {protocol_config.description}")
    original_data_yaml_path = os.path.join(cfg.files.base_path, protocol_config.yolo_output_dir, "data.yaml")
    data_yaml_path = resolve_data_yaml_path(cfg, hydra_output_dir, original_data_yaml_path)

    model, model_name = initialize_model(cfg)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{model_name}_{timestamp}"
    final_run_name = run_name

    # Close any active MLflow run to avoid conflicts from previous failed executions
    mlflow.end_run()

    try:
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id

        # Set the MLflow run ID as an environment variable so that it can be accessed by the YOLO training process
        os.environ["MLFLOW_RUN_ID"] = run_id

        # Save Hydra parameters path as a MLflow artifact for reference
        mlflow.log_artifact(os.path.join(hydra_output_dir, ".hydra/config.yaml"), artifact_path="hydra_config")

        final_run_name = train_model(model, cfg, data_yaml_path, hydra_output_dir, run_name)

        results_csv_path = os.path.join(hydra_output_dir, final_run_name, "results.csv")
        best_epoch = get_best_epoch(results_csv_path)

        # Load best model weights for validation
        best_model_path = os.path.join(hydra_output_dir, final_run_name, "weights", "best.pt")
        model = YOLO(best_model_path)

        # mean Average Precision at IoU=0.5 (mAP50)
        #https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.DetMetrics
        metrics = model.val(data=data_yaml_path, workers=2, seed = cfg.params.seed, imgsz=cfg.params.img_size, batch=cfg.params.batch_size, device=cfg.params.device)
        # We only have one class, so we take AP instead of mAP
        ap50 = float(metrics.box.ap50[0])
        ap75 = float(metrics.box.map75)
        ap50_95 = float(metrics.box.ap[0])

        # This value is used to select the best wheights in YOLO training
        fitness = float(metrics.fitness)
        print(f"fitness: {fitness:.4f}, AP50: {ap50:.4f}, AP50-95: {ap50_95:.4f}, AP75: {ap75:.4f}")

        # Extract optimum confidence threshold based on F1 score
        best_f1 = float(metrics.box.f1[0])
        optimal_conf = extract_optimal_conf_threshold(metrics)
        print(f"Optimal Conf Threshold: {optimal_conf:.3f}, F1: {best_f1:.4f} \n")

        log_metrics = {
            "val_AP50": ap50,
            "val_AP50_95": ap50_95,
            "metrics/mAP75B": ap75,
            "val_fitness": fitness,
            "metrics/best_f1B": best_f1,
            "metrics/optimal_conf": optimal_conf,
            "metrics/best_epoch": best_epoch,
        }
        tags = {
            "protocol": cfg.params.protocol,
            "model": model_name,
            "AP50": ap50,
            "AP50_95": ap50_95,
            "best_f1": best_f1,
            "seed": cfg.params.seed,
            "best_epoch": best_epoch,
        }

        # Log all relevant metrics to MLflow for better tracking and comparison across runs
        mlflow.log_metrics(log_metrics)

        # Add tags to filter runs by protocol and model in MLflow UI
        mlflow.set_tags(tags)

        metadata_path = save_deployment_metadata(
            hydra_output_dir=hydra_output_dir,
            model_name=model_name,
            optimal_conf=optimal_conf,
            img_size=cfg.params.img_size,
            timestamp=timestamp,
            seed=cfg.params.seed,
            protocol=cfg.params.protocol,
        )
        mlflow.log_artifact(metadata_path, artifact_path="deployment_info")

        # Make sure to log the best model weights as an artifact in MLflow
        mlflow.log_artifact(best_model_path, artifact_path="weights")

        print(f"Summary results: {metrics.summary()[0]}")
        return ap50_95
    finally:
        mlflow.end_run()

        # Clean up the MLflow run ID environment variable to avoid confusion in subsequent runs
        if "MLFLOW_RUN_ID" in os.environ:
            del os.environ["MLFLOW_RUN_ID"]

        cleanup_run_artifacts(hydra_output_dir, final_run_name)


if __name__ == "__main__":
    main()
