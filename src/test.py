import mlflow
import os
import gc
from mlflow.tracking import MlflowClient
from fabric import Connection
import torch
from ultralytics import YOLO
import json
from utils.get_config import get_config
from config import PolypDetectionConfig

def main():
    cfg: PolypDetectionConfig = get_config()
    print(cfg.test)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = cfg.mlflow.s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = cfg.mlflow.aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = cfg.mlflow.aws_secret_access_key

    experiment_name = cfg.test.experiment_name
    if cfg.mlflow.tracking_uri:
        tracking_uri = cfg.mlflow.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
    else:
        raise ValueError("MLflow tracking URI (http://<SERVER_IP>:5000) must be set in the configuration.")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{cfg.test.metric} DESC"],
        max_results=cfg.test.top_k
    )

    if not runs:
        print("No runs found.")
        return

    for i, run in enumerate(runs):
        run_id = run.info.run_id
        metric_val = run.data.metrics.get(cfg.test.metric, "N/A")

        print(f"\n[{i+1}/{len(runs)}] Test for Run ID: {run_id} ({cfg.test.metric}: {metric_val})")

        weights_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path='weights'
        )
        weights_file = os.path.join(weights_dir, 'best.pt')

        deployment_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path='deployment_info'
        )
        deployment_file = os.path.join(deployment_dir, 'deployment_metadata.json')

        with open(deployment_file, 'r') as f:
            deployment_metadata = json.load(f)
        
        opt_conf = deployment_metadata["optimal_conf_threshold"]
        protocol = deployment_metadata.get("protocol", "t1")
        datasets_base_path = cfg.files.base_path
        files_info = cfg.files.protocols[protocol]
        data_yaml = os.path.join(datasets_base_path, files_info.yolo_output_dir, "data.yaml")

        model = YOLO(weights_file)
        model.info(detailed=True)

        print("Running validation on test set...")
        use_half_precision = cfg.test.precision_mode.upper() == "FP16"
        print(f"Using half precision: {use_half_precision}")
        print("Warming up GPU...")
        # Usamos una imagen vacía o de ceros para no depender de archivos externos
        # El tamaño debe ser el mismo que usarás en el test (cfg.test.img_size)
        warmup_dummy = torch.zeros((1, 3, cfg.test.img_size, cfg.test.img_size)).to(0)
        for _ in range(50):
            _ = model.predict(warmup_dummy, verbose=False, half=use_half_precision)
        print("Warmup completed. Starting evaluation...")
        results = model.val(data=data_yaml, conf=opt_conf, verbose=False, split="test", batch=1, device=0, imgsz=cfg.test.img_size, half=use_half_precision)

        ap50 = float(results.box.ap50[0])
        ap50_95 = float(results.box.ap[0])
        p, r = float(results.box.mp), float(results.box.mr)
        f1 = 2 * (p * r) / (p + r + 1e-9)

        speed_dict = results.speed
        inference_time_ms = speed_dict.get('inference', 0)

        metrics = {
            f'test_AP50_{cfg.test.precision_mode}': ap50,
            f'test_AP50_95_{cfg.test.precision_mode}': ap50_95,
            f'test_precision_at_opt_conf_{cfg.test.precision_mode}': p,
            f'test_recall_at_opt_conf_{cfg.test.precision_mode}': r,
            f'test_f1_at_opt_conf_{cfg.test.precision_mode}': f1,
            f'test_inference_ms_{cfg.test.precision_mode}': inference_time_ms,
            f'test_fps_{cfg.test.precision_mode}': 1000.0 / max(inference_time_ms, 0.1)
        }

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics)

        torch.cuda.synchronize()
        del model  # Clean up engine from GPU memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()