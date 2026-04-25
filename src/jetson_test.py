import os
import mlflow
from ultralytics import YOLO
from jtop import jtop

def main():
    # Grab dynamic run ID from SSH environment variables
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise ValueError("MLFLOW_RUN_ID environment variable not found.")

    # 2. Set tracking URI (also injected by SSH, but fallback to config just in case)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    data_yaml = "/app/test_data/data.yaml"  # Path inside the Docker container

    print(f"Downloading weights for Run ID: {run_id}...")
    weights_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path='weights'
    )
    weights_file = os.path.join(weights_dir, 'best.pt')

    print("Exporting model to TensorRT...")
    model = YOLO(weights_file)
    half_precision = os.environ.get("HALF_PRECISION", "False").lower() == "true"
    engine_path = model.export(
        format='engine',
        half=half_precision,
        device=0,
        imgsz=os.environ.get("IMGSZ", "640"),
        simplify=False,
        opset=13
    )

    print("Loading TensorRT engine and warming up...")
    optimized = YOLO(engine_path, task="detect")
    for _ in range(3):
        optimized.predict(source=os.path.join(data_yaml, "images", "test"), device=0, verbose=False)

    print(f"Running validation")
    results = optimized.val(
        data=data_yaml,
        device=0,
        batch=1, 
        verbose=True,
    )

    # Collect hardware metrics via jtop
    try:
        with jtop() as jetson:
            hw = {
                'gpu_util_pct': jetson.stats.get('GPU', 0),
                'gpu_temp_c': jetson.temperature.get('GPU', 0),
                'power_tot_mw': jetson.power[1].get('tot', {}).get('cur', 0) if len(jetson.power) > 1 else 0,
            }
    except Exception as e:
        print(f"Warning: jtop not available ({e}).")
        hw = {'gpu_util_pct': -1, 'gpu_temp_c': -1, 'power_tot_mw': -1}

    print("Logging metrics back to MLflow (Nested Run)...")
    with mlflow.start_run(run_id=run_id, nested=True):
        ap50 = float(results.box.ap50[0]) if hasattr(results.box, 'ap50') else 0
        ap50_95 = float(results.box.ap[0]) if hasattr(results.box, 'ap') else 0
        
        mlflow.log_metric('jetson_AP50', ap50)
        mlflow.log_metric('jetson_AP50_95', ap50_95)
        mlflow.log_metric('jetson_inference_ms', results.speed.get('inference', 0))
        mlflow.log_metric('jetson_fps', 1000.0 / max(results.speed.get('inference', 1), 0.1))
        
        mlflow.log_metric('jetson_gpu_util_pct', hw['gpu_util_pct'])
        mlflow.log_metric('jetson_gpu_temp_c', hw['gpu_temp_c'])
        mlflow.log_metric('jetson_power_mw', hw['power_tot_mw'])
        
        mlflow.log_param('jetson_precision_mode', 'FP16' if half_precision else 'FP32')

        mlflow.log_param('jetson_engine_path', engine_path)

    print(f"Deployment Evaluation Complete for Run ID: {run_id}")

if __name__ == "__main__":
    main()