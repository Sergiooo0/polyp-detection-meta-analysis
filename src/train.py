from datetime import datetime
import json
from ultralytics import YOLO
import os
import shutil
from ultralytics import settings
import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from config import PolypDetectionConfig
from data.sampler import create_balanced_train_split
import numpy as np

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
    original_data_yaml_path = os.path.join(cfg.files.base_path, protocol_config.yolo_output_dir, "data.yaml")

    # Verify if the dataset need a balance between negatives and positives examples
    balanced_txt_path = None
    # This is the case of protocol t2 with the SUN dataset, which has 2 negatives examples per positive example
    if cfg.params.protocol == "t2":
        print(f"Applying deterministic negative sampling for protocol {cfg.params.protocol} (Seed: {cfg.params.seed}).")
        
        # Save new files in Hydra output file, don't modify the dataset
        balanced_yaml_path = os.path.join(hydra_output_dir, f"data_balanced_seed_{cfg.params.seed}.yaml")
        balanced_txt_path = os.path.join(hydra_output_dir, f"train_balanced_seed_{cfg.params.seed}.txt")
        
        data_yaml_path = create_balanced_train_split(
            data_yaml_path=original_data_yaml_path,
            output_yaml_path=balanced_yaml_path,
            output_txt_path=balanced_txt_path, 
            r=1.0, 
            seed=cfg.params.seed 
        )
    else:
        data_yaml_path = original_data_yaml_path

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

    if balanced_txt_path:
        mlflow.log_artifact(data_yaml_path, artifact_path="dataset_splits")
        mlflow.log_artifact(balanced_txt_path, artifact_path="dataset_splits")

    log_metrics = {}
    tags = {}

    model.train(
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
        patience=100,
    )

    # mean Average Precision at IoU=0.5 (mAP50)
    #https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.DetMetrics
    metrics = model.val()
    # We only have one class, so we take AP instead of mAP
    ap50 = float(metrics.box.ap50[0])

    ap75 = float(metrics.box.map75)

    ap50_95 = float(metrics.box.ap[0])
    
    # This value is used to select the best wheights in YOLO training
    fitness = float(metrics.fitness)

    print(f"fitness: {fitness:.4f}, AP50: {ap50:.4f}, AP50-95: {ap50_95:.4f}, AP75: {ap75:.4f}")

    log_metrics.update({
        "val_AP50": ap50,
        "val_AP50_95": ap50_95,
        "metrics/mAP75B": ap75,
        "val_fitness": fitness
    })

    tags.update({
        "protocol": cfg.params.protocol,
        "model": model_name,
        "AP50": float(ap50),
        "AP50_95": float(ap50_95),
    })

    # Extract optimum confidence threshold based on F1 score
    best_f1 = float(metrics.box.f1[0])
    optimal_p = float(metrics.box.p[0])
    optimal_r = float(metrics.box.r[0])

    optimal_conf = 0.0
    
    # metrics.box.curves_results is a list of lists: [x, y, x_label, y_label]
    for curve in metrics.box.curves_results:
        if 'F1' in str(curve[3]):
            conf_thresholds = curve[0]  # X axis: The thresholds that YOLO tested
            
            # Y axis: The F1 scores.
            # curve[1] usually has shape (nc, 1000). We use [0] to get the first class (polyps)
            f1_scores = curve[1][0] if len(curve[1].shape) > 1 else curve[1]
            
            best_idx = np.argmax(f1_scores)
            
            # Get the confidence threshold corresponding to the best F1 score
            optimal_conf = float(conf_thresholds[best_idx])
            break

    print(f"Optimal Conf Threshold: {optimal_conf:.3f}, F1: {best_f1:.4f} (P: {optimal_p:.4f}, R: {optimal_r:.4f})\n")

    log_metrics.update({
        "metrics/best_f1B": best_f1,
        "metrics/precision": optimal_p,
        "metrics/recall": optimal_r,
        "metrics/optimal_conf": optimal_conf
    })

    tags["best_f1"] = best_f1
        
    # Log all relevant metrics to MLflow for better tracking and comparison across runs
    mlflow.log_metrics(log_metrics)

    # Add tags to filter runs by protocol and model in MLflow UI
    mlflow.set_tags(tags)

    # Save relevant information about the model to use in inference.py
    deployment_metadata = {
        "model_name": model_name,
        "optimal_conf_threshold": optimal_conf,
        "imgsz": cfg.params.img_size,
        "training_date": timestamp,
        "seed": cfg.params.seed
    }

    metadata_path = os.path.join(hydra_output_dir, "deployment_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(deployment_metadata, f, indent=4)

    mlflow.log_artifact(metadata_path, artifact_path="deployment_info")

    print(f"Summary results: {metrics.summary()[0]}")

    mlflow.end_run()
    
    # Clean up the MLflow run ID environment variable to avoid confusion in subsequent runs
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

        # Remove weights from Hydra to save space. The weights are already saved in the MLflow run artifacts.
    weights_dir = os.path.join(hydra_output_dir, run_name, "weights")

    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
        
    return ap50_95


if __name__ == "__main__":
    main()