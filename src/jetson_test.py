import os
import mlflow
from ultralytics import YOLO
from jtop import jtop
import json
import torch
import gc
from utils.jetsonMonitor import JetsonMonitor
import time
import shutil

def cleanup_memory():
    """Clean the memory by clearing GPU cache and collecting garbage."""
    print("Cleaning memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print("Cleanup complete.")

def main():
    precision_mode = os.environ.get("PRECISION_MODE", "FP32").upper()
    protocol = os.environ.get("PROTOCOL", "t1").lower()
    print(f"EXECUTING JETSON TEST SCRIPT with precision mode: {precision_mode}")
    # Grab dynamic run ID from SSH environment variables
    run_id = os.environ.get("PARENT_RUN_ID")
    print(f"MLFLOW_RUN_ID: {run_id}")
    if not run_id:
        raise ValueError("PARENT_RUN_ID environment variable not found.")

    # Set tracking URI (also injected by SSH, but fallback to config just in case)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    data_folder = "/app/test_data"

    client = mlflow.tracking.MlflowClient()

    parent_run = client.get_run(run_id)
    experiment_id = parent_run.info.experiment_id
    
    # Extract the 'model' tag from the parent run
    model_tag = parent_run.data.tags.get("model", "YOLO_model_unknown")
    seed_tag = parent_run.data.tags.get("seed", "seed_unknown")

    print(f"Downloading weights for Run ID: {run_id}...")
    max_retries = 5
    weights_file = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to download...")
            weights_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path='weights'
            )
            weights_file = os.path.join(weights_dir, 'best.pt')
            
            # If the file is corrupt, this will raise an error and trigger the except block
            _ = torch.load(weights_file, map_location='cpu')
            
            print("Weights downloaded and validated correctly.")
            break
            
        except Exception as e:
            print(f"Error or corrupt file detected: {e}")
            if attempt < max_retries - 1:
                print("Cleaning cache and retrying in 3 seconds...")
                # Remove the potentially corrupt file
                if 'weights_dir' in locals() and os.path.exists(weights_dir):
                    shutil.rmtree(weights_dir)
                time.sleep(3)
            else:
                raise RuntimeError(f"Critical failure: Could not download weights after {max_retries} attempts.")
    if weights_file is None:
        raise RuntimeError("Critical failure: Could not obtain weights file after several attempts.")

    deployment_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path='deployment_info'
    )
    deployment_file = os.path.join(deployment_dir, 'deployment_metadata.json')

    with open(deployment_file, 'r') as f:
        deployment_metadata = json.load(f)
    
    opt_conf = deployment_metadata["optimal_conf_threshold"]

    print("Exporting model to TensorRT...")
    cleanup_memory()  # Clean memory before export to ensure maximum available resources
    model = YOLO(weights_file)
    data_yaml = os.path.join(data_folder, "data.yaml")
    image_size = int(os.environ.get("IMGSZ", 640))
    
    if precision_mode == "ONNX-FP32":
        export_args = {
            "format": "onnx",
            "device": 0, 
            "imgsz": image_size,
            "simplify": False,
            "opset": 13
        }
        device_target = "cpu"
    else:
        export_args = {
            "format": "engine",
            "device": 0,
            "imgsz": image_size,
            "simplify": False,
            "opset": 13
        }
        device_target = 0
        
        if precision_mode == "INT8":
            export_args["int8"] = True
            export_args["data"] = data_yaml      
            export_args["batch"] = 1             
            export_args["workspace"] = 4         
        elif precision_mode == "FP16":
            export_args["half"] = True
        else:
            export_args["half"] = False

    exported_model_path = model.export(**export_args)
    cleanup_memory()

    print("Loading TensorRT engine and warming up...")
    optimized = YOLO(exported_model_path, task="detect")
    test_images = os.path.join(data_folder, "images", "test")
    first_image = sorted(os.listdir(test_images))[0]
    warmup_path = os.path.join(test_images, first_image)

    for _ in range(3):
        optimized.predict(source=warmup_path, device=device_target, verbose=False)

    print(f"Running Testing on Test Subset...")

    monitor = JetsonMonitor(delay=0.2)
    monitor.start()

    results = optimized.val(
        data=data_yaml,
        device=device_target,
        batch=1,    
        verbose=True,
        split='test',
        imgsz=image_size,
        conf=opt_conf
    )

    monitor.stopped = True
    monitor.join()
    hw = monitor.get_stats()
    ap50 = float(results.box.ap50[0])
    ap50_95 = float(results.box.ap[0])
    p, r = float(results.box.mp), float(results.box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    speed_dict = results.speed
    inference_ms = speed_dict.get('inference', 0)

    metrics = {
        'AP50': ap50,
        'AP50_95': ap50_95,
        'precision': p,
        'recall': r,
        'f1': f1,
        'fps': 1000.0 / max(inference_ms, 0.1)
    }
    
    for key, value in hw.items():
        metrics[f'jetson_{key}'] = value

    # Define the updated parameters dictionary
    params = {
        'jetson_precision_mode': precision_mode,
        'jetson_image_size': image_size,
        'jetson_nvp_model': monitor.nvp_model,
        'model': model_tag,
        'imgsz': image_size,
        'precision_mode': precision_mode,
        'protocol': protocol,
        'opt_conf_threshold': opt_conf
    }
    
    # Define the corresponding tags dictionary (MLflow requires tag values to be strings)
    tags = {
        'model': model_tag,
        'imgsz': str(image_size),
        'precision_mode': precision_mode,
        'protocol': protocol,
        'jetson_nvp_model': monitor.nvp_model,
        'seed': seed_tag
    }

    print("Logging metrics back to MLflow (Nested Run)...")
    run_name = f"Test_Jetson_{precision_mode}_{image_size}_{monitor.nvp_model}"
    for attempt in range(max_retries):
        try:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                parent_run_id=run_id
            ):
                try:
                    mlflow.log_params(params)
                    mlflow.set_tags(tags)  # Log tags to the nested execution
                except Exception as e:
                    print(f"Error while logging params/tags: {e}")
                
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(exported_model_path, artifact_path="jetson_engines")
            
            if precision_mode == "FP32" and image_size == 640 and monitor.nvp_model == "25W":
                # Save this case also in the parent run for easier comparison in the dashboard
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({
                        f"jetson_ap50_{protocol}": ap50,
                        f"jetson_ap50_95_{protocol}": ap50_95,
                        f"jetson_precision_{protocol}": p,
                        f"jetson_recall_{protocol}": r,
                        f"jetson_f1_{protocol}": f1,
                        f"jetson_fps_{protocol}": 1000.0 / max(inference_ms, 0.1)
                    })
                    mlflow.log_artifact(exported_model_path, artifact_path="jetson_engines")
            print("Metrics, params and tags logged successfully.")
            break
        except Exception as e:
            print(f"Error while logging metrics: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print("Critical failure: Could not register metrics after several attempts.")

    print(f"Deployment Evaluation Complete for Run ID: {run_id}")
    os._exit(0)

if __name__ == "__main__":
    main()