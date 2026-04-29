import os
import mlflow
from ultralytics import YOLO
from jtop import jtop
import json
import torch
import gc
from utils.jetsonMonitor import JetsonMonitor

def main():
    print("EXECUTING JETSON TEST SCRIPT...")
    # Grab dynamic run ID from SSH environment variables
    run_id = os.environ.get("MLFLOW_RUN_ID")
    print(f"MLFLOW_RUN_ID: {run_id}")
    if not run_id:
        raise ValueError("MLFLOW_RUN_ID environment variable not found.")

    # Set tracking URI (also injected by SSH, but fallback to config just in case)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    data_folder = "/app/test_data"

    print(f"Downloading weights for Run ID: {run_id}...")
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

    print("Exporting model to TensorRT...")
    model = YOLO(weights_file)
    half_precision = os.environ.get("HALF_PRECISION", "False").lower() == "true"
    engine_path = model.export(
        format='engine',
        half=half_precision,
        device=0,
        imgsz=int(os.environ.get("IMGSZ", 640)),
        simplify=False,
        opset=13
    )

    print("Loading TensorRT engine and warming up...")
    optimized = YOLO(engine_path, task="detect")
    for _ in range(3):
        optimized.predict(source=os.path.join(data_folder, "images", "test"), device=0, verbose=False)

    print(f"Running Testing")
    data_yaml = os.path.join(data_folder, "data.yaml")

    monitor = JetsonMonitor(delay=0.2)
    monitor.start()

    results = optimized.val(
        data=data_yaml,
        device=0,
        batch=1,    
        verbose=True,
        split='test',
        conf=opt_conf,
        imgsz=int(os.environ.get("IMGSZ", 640))
    )

    monitor.stopped = True
    monitor.join()
    hw = monitor.get_stats()

    print("Logging metrics back to MLflow (Nested Run)...")
    with mlflow.start_run(run_id=run_id, nested=True):
        ap50 = float(results.box.ap50[0])
        ap50_95 = float(results.box.ap[0])

        precision_at_opt = float(results.box.mp)
        recall_at_opt = float(results.box.mr)
        f1_at_opt = 2 * (precision_at_opt * recall_at_opt) / (precision_at_opt + recall_at_opt + 1e-9)

        print(results.box.curves_results)

        mlflow.log_metric('jetson_AP50', ap50)
        mlflow.log_metric('jetson_AP50_95', ap50_95)
        mlflow.log_metric('jetson_precision', results.box.p[0])
        mlflow.log_metric('jetson_recall', results.box.r[0])
        mlflow.log_metric('jetson_f1', results.box.f1[0])
        mlflow.log_metric('jetson_precision_at_opt_conf', precision_at_opt)
        mlflow.log_metric('jetson_recall_at_opt_conf', recall_at_opt)
        mlflow.log_metric('jetson_f1_at_opt_conf', f1_at_opt)
        mlflow.log_metric('jetson_inference_ms', results.speed.get('inference', 0))
        mlflow.log_metric('jetson_fps', 1000.0 / max(results.speed.get('inference', 1), 0.1))
        
        for key, value in hw.items():
            mlflow.log_metric(f'jetson_{key}', value)
        
        mlflow.log_param('jetson_precision_mode', 'FP16' if half_precision else 'FP32')

        mlflow.log_param('jetson_engine_path', engine_path)

    torch.cuda.synchronize()
    del optimized  # Clean up engine from GPU memory
    gc.collect()
    print(f"Deployment Evaluation Complete for Run ID: {run_id}")

if __name__ == "__main__":
    main()