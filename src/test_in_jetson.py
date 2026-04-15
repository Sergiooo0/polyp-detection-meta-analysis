import os
import mlflow
from ultralytics import YOLO
from jtop import jtop 

os.environ['MLFLOW_TRACKING_URI'] = "http://192.168.110.41:5000" 
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://192.168.110.41:9000" 

def evaluate_on_jetson(run_id, data_yaml):
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="Jetson_Test", nested=True):
            
            print(f"Downloading model from run {run_id}...")
            model_uri = f"runs:/{run_id}/weights/best.pt"
            local_model_path = mlflow.artifacts.download_artifacts(model_uri)
            
            model = YOLO(local_model_path)

            print("Convirtiendo a TensorRT FP32...")
            model.export(format='engine', device=0)
            
            engine_model = YOLO(local_model_path.replace('.pt', '.engine'))

            print("Starting evaluation...")
            results = engine_model.val(data=data_yaml, split='test', batch=1, device=0, verbose=True)
            try:
                with jtop() as jetson:
                hw = {
                    'gpu_util_pct': jetson.stats.get('GPU', 0),
                    'gpu_temp_c': jetson.temperature.get('GPU', 0),
                    'power_tot_mw': jetson.power[1].get('tot', {}).get('cur', 0),
                }
            except Exception:
                hw = {'gpu_util_pct': -1, 'gpu_temp_c': -1, 'power_tot_mw': -1}

            d = results.results_dict
            mlflow.log_metric('jetson_mAP50', d.get('metrics/mAP50(B)', 0))
            mlflow.log_metric('jetson_mAP50_95', d.get('metrics/mAP50-95(B)', 0))
            mlflow.log_metric('jetson_precision', d.get('metrics/precision(B)', 0))
            mlflow.log_metric('jetson_recall', d.get('metrics/recall(B)', 0))
            mlflow.log_metric('jetson_inference_ms', results.speed.get('inference', 0))
            mlflow.log_metric('jetson_fps', 1000.0 / max(results.speed.get('inference', 1), 0.1))
            mlflow.log_metric('jetson_preprocess_ms',results.speed.get('preprocess', 0))
            mlflow.log_metric('jetson_postprocess_ms',results.speed.get('postprocess', 0))
            mlflow.log_metric('jetson_gpu_util_pct', hw['gpu_util_pct'])
            mlflow.log_metric('jetson_gpu_temp_c', hw['gpu_temp_c'])
            mlflow.log_metric('jetson_power_mw', hw['power_tot_mw'])
            mlflow.log_param('jetson_precision_mode','FP16')
            mlflow.log_param('jetson_trt_engine', engine_path)
            mlflow.log_artifact('jetson_results.json') if os.path.exists('jetson_results.json') else None
            
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate_on_jetson(sys.argv[1], sys.argv[2])