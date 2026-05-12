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
    max_retries = 5
    weights_file = None
    for attempt in range(max_retries):
        try:
            print(f"Intento {attempt + 1}/{max_retries} de descarga...")
            weights_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path='weights'
            )
            weights_file = os.path.join(weights_dir, 'best.pt')
            
            # Validación: Intentar cargar los metadatos del modelo en CPU
            # Si el archivo está corrupto (cortado), esto lanzará un error (EOFError, BadZipFile, etc.)
            _ = torch.load(weights_file, map_location='cpu')
            
            print("Pesos descargados y validados correctamente.")
            break  # Salimos del bucle si todo fue bien
            
        except Exception as e:
            print(f"Error o archivo corrupto detectado: {e}")
            if attempt < max_retries - 1:
                print("Limpiando caché y reintentando en 3 segundos...")
                # Borramos la carpeta corrupta de MLflow para forzar una descarga limpia
                if 'weights_dir' in locals() and os.path.exists(weights_dir):
                    shutil.rmtree(weights_dir)
                time.sleep(3)
            else:
                raise RuntimeError(f"Fallo crítico: No se pudieron descargar los pesos tras {max_retries} intentos.")
    if weights_file is None:
        raise RuntimeError("Fallo crítico: No se pudo obtener el archivo de pesos después de varios intentos.")
    
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
    test_images = os.path.join(data_folder, "images", "test")
    first_image = sorted(os.listdir(test_images))[0]
    warmup_path = os.path.join(test_images, first_image)

    for _ in range(3):
        optimized.predict(source=warmup_path, device=0, verbose=False)

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
    ap50 = float(results.box.ap50[0])
    ap50_95 = float(results.box.ap[0])
    p, r = float(results.box.mp), float(results.box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    #print(results.box.curves_results)

    speed_dict = results.speed
    total_time_ms = speed_dict.get('preprocess', 0) + speed_dict.get('inference', 0) + speed_dict.get('postprocess', 0)

    metrics = {
        'jetson_AP50': ap50,
        'jetson_AP50_95': ap50_95,
        'jetson_precision_at_opt_conf': p,
        'jetson_recall_at_opt_conf': r,
        'jetson_f1_at_opt_conf': f1,
        'jetson_inference_ms': speed_dict.get('inference', 0),
        'jetson_total_ms': total_time_ms,
        'jetson_fps': 1000.0 / max(total_time_ms, 0.1)
    }
    # Añadir métricas de hardware
    for key, value in hw.items():
        metrics[f'jetson_{key}'] = value
    print("Logging metrics back to MLflow (Nested Run)...")
    for attempt in range(max_retries):
        try:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics(metrics)
                try:
                    mlflow.log_param('jetson_engine_path', engine_path)
                    mlflow.log_param('jetson_precision_mode', 'FP16' if half_precision else 'FP32')
                except mlflow.exceptions.MlflowException:
                    print("Los parámetros ya estaban registrados en MLflow. Saltando...")
            print("Métricas registradas correctamente en MLflow.")
            break
        except Exception as e:
            print(f"Error al registrar métricas en MLflow: {e}")
            if attempt < max_retries - 1:
                print("Reintentando en 3 segundos...")
                time.sleep(3)
            else:
                print("Fallo crítico: No se pudieron registrar las métricas tras varios intentos.")

    torch.cuda.synchronize()
    del optimized  # Clean up engine from GPU memory
    gc.collect()
    print(f"Deployment Evaluation Complete for Run ID: {run_id}")

if __name__ == "__main__":
    main()