import os
import json
import time
import shutil
import gc
import tempfile
from pathlib import Path

import numpy as np
import torch
import cv2
import mlflow
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

from utils.jetsonMonitor import JetsonMonitor
import random


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def get_run_config():
    """Reads and validates environment variables required for execution."""
    precision_mode = os.environ.get("PRECISION_MODE", "FP32").upper()
    protocol = os.environ.get("PROTOCOL", "t1").lower()
    image_size = int(os.environ.get("IMGSZ", 640))
    run_id = os.environ.get("PARENT_RUN_ID")

    if not run_id:
        raise ValueError("PARENT_RUN_ID environment variable not found.")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    print(f"EXECUTING JETSON TEST SCRIPT with precision mode: {precision_mode}")
    print(f"MLFLOW_RUN_ID: {run_id}")

    return {
        "precision_mode": precision_mode,
        "protocol": protocol,
        "image_size": image_size,
        "run_id": run_id,
    }


def cleanup_memory():
    """Clears system memory by freeing GPU cache and forcing garbage collection."""
    print("Cleaning memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print("Cleanup complete.")


def list_images(directory):
    """Returns a sorted list of image files found in the directory."""
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))


def download_weights(run_id, max_retries=5):
    """Downloads and validates trained model weights from MLflow."""
    print(f"Downloading weights for Run ID: {run_id}...")
    weights_dir = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to download...")
            weights_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights")
            weights_file = os.path.join(weights_dir, "best.pt")
            _ = torch.load(weights_file, map_location="cpu")  # Throws exception if file is corrupted
            print("Weights downloaded and validated correctly.")
            return weights_file
        except Exception as e:
            print(f"Error or corrupt file detected: {e}")
            if weights_dir and os.path.exists(weights_dir):
                shutil.rmtree(weights_dir)
            if attempt < max_retries - 1:
                print("Cleaning cache and retrying in 3 seconds...")
                time.sleep(3)

    raise RuntimeError(f"Critical failure: Could not download weights after {max_retries} attempts.")


def download_deployment_metadata(run_id):
    """Downloads deployment metadata JSON containing the optimal confidence threshold."""
    deployment_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="deployment_info")
    deployment_file = os.path.join(deployment_dir, "deployment_metadata.json")
    with open(deployment_file, "r") as f:
        return json.load(f)


def build_export_args(precision_mode, image_size, data_yaml):
    """Builds kwargs for ultralytics export() and specifies the target inference device."""
    base_args = {"device": 0, "imgsz": image_size, "simplify": False, "opset": 13}

    if precision_mode in ("ONNX-FP32", "ONNX-INT8"):
        return {**base_args, "format": "onnx"}, "cpu"

    if precision_mode == "ONNX-FP16":
        return {**base_args, "format": "onnx", "half": True}, "cpu"

    # TensorRT Engine: FP32 / FP16 / INT8
    export_args = {**base_args, "format": "engine"}
    if precision_mode == "INT8":
        export_args.update({"int8": True, "data": data_yaml, "batch": 1, "workspace": 4})
    elif precision_mode == "FP16":
        export_args["half"] = True
    else:
        export_args["half"] = False

    return export_args, 0


def quantize_onnx_to_int8(onnx_fp32_path):
    """Dynamically quantizes an ONNX FP32 model to INT8 and removes the original file."""
    print("Starting dynamic INT8 quantization for ONNX...")
    onnx_int8_path = onnx_fp32_path.replace(".onnx", "_int8.onnx")

    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QUInt8)
    print(f"Quantization completed successfully: {onnx_int8_path}")

    if os.path.exists(onnx_fp32_path):
        os.remove(onnx_fp32_path)

    return onnx_int8_path


def export_model(weights_file, precision_mode, image_size, data_yaml):
    """Exports the trained model to the target format and precision. Returns (exported_path, device_target)."""
    print("Exporting model...")
    cleanup_memory()

    model = YOLO(weights_file)
    export_args, device_target = build_export_args(precision_mode, image_size, data_yaml)
    exported_path = model.export(**export_args)
    cleanup_memory()

    if precision_mode == "ONNX-INT8":
        try:
            exported_path = quantize_onnx_to_int8(exported_path)
        except Exception as e:
            print(f"Critical error during ONNX-INT8 quantization: {e}")
            raise

    return exported_path, device_target


def warmup_model(model, test_images_dir, device_target, n_warmup=3):
    """Runs warmup inference rounds to prevent profiling skew on subsequent time measurements."""
    first_image = list_images(test_images_dir)[0]
    warmup_path = os.path.join(test_images_dir, first_image)
    for _ in range(n_warmup):
        model.predict(source=warmup_path, device=device_target, verbose=False)


def measure_latency_percentiles(model, test_images_dir, device_target, conf):
    """
    Executes image-by-image inference over the test set to collect latency percentiles (ms),
    since val() only provides an aggregated batch average instead of the true distribution.
    """
    image_files = list_images(test_images_dir)

    latencies_ms = []
    for image_name in image_files:
        image_path = os.path.join(test_images_dir, image_name)
        result = model.predict(source=image_path, device=device_target, conf=conf, verbose=False)[0]
        latencies_ms.append(result.speed["inference"])

    latencies_ms = np.array(latencies_ms)
    return {
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p90_ms": float(np.percentile(latencies_ms, 90)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_std_ms": float(np.std(latencies_ms)),
        "latency_n_samples": int(len(latencies_ms)),
    }


def load_yolo_labels(label_path, img_width, img_height):
    """Parses a YOLO format label file into pixel coordinates (class, x1, y1, x2, y2)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, xc, yc, w, h = parts[:5]
            cls_id = int(float(cls_id))
            xc, yc = float(xc) * img_width, float(yc) * img_height
            w, h = float(w) * img_width, float(h) * img_height
            boxes.append((cls_id, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))

    return boxes


def compute_iou(box_a, box_b):
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Greedily matches predictions with ground truth bounding boxes by class using IoU.
    Returns lists of indices for false positives and false negatives.
    """
    unmatched_gt = list(range(len(gt_boxes)))
    false_positives = []

    for p_idx, pred in enumerate(pred_boxes):
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx in unmatched_gt:
            gt = gt_boxes[gt_idx]
            if gt[0] != pred[0]:
                continue
            iou = compute_iou(pred[1:], gt[1:])
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx

        if best_iou >= iou_threshold:
            unmatched_gt.remove(best_gt_idx)
        else:
            false_positives.append(p_idx)

    false_negatives = unmatched_gt
    return false_positives, false_negatives


def draw_boxes(image, pred_boxes, gt_boxes, false_positive_idx, false_negative_idx):
    """Draws ground truth (green) and predictions (blue/red) on a copy of the input image."""
    vis = image.copy()

    for idx, (_, x1, y1, x2, y2) in enumerate(gt_boxes):
        color = (0, 200, 0)
        label = "GT (missed)" if idx in false_negative_idx else "GT"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, label, (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for idx, (_, x1, y1, x2, y2) in enumerate(pred_boxes):
        color = (0, 0, 255) if idx in false_positive_idx else (255, 140, 0)
        label = "FP" if idx in false_positive_idx else "Pred"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, label, (int(x1), int(y2) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


def calculate_iou(box1, box2):
    """Calcula el IoU de dos cajas en formato [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def generate_qualitative_examples(model, data_folder, device_target, conf, output_dir,
                                   iou_threshold=0.5, max_search=200, num_examples=3, max_fp_iou=0.1):
    """
    Scans a random subset of the test data to find and save:
      - Multiple 'clean' frames where predictions match ground truth boxes perfectly.
      - Multiple 'error' frames containing true false negatives or STRAY false positives 
        (predictions that barely/don't overlap with any ground truth).
    """
    images_dir = os.path.join(data_folder, "images", "test")
    labels_dir = os.path.join(data_folder, "labels", "test")
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {"clean_examples": [], "error_examples": []}

    # 1. Obtener imágenes y mezclarlas para que sea random en cada ejecución
    image_files = list_images(images_dir)
    random.shuffle(image_files)

    for image_name in image_files[:max_search]:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, Path(image_name).stem + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_boxes = load_yolo_labels(label_path, w, h)

        result = model.predict(source=image_path, device=device_target, conf=conf, verbose=False)[0]
        pred_boxes = [(int(c), *box) for c, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())]

        # match_detections detectará FPs estándar (IoU < iou_threshold)
        fp_idx, fn_idx = match_detections(pred_boxes, gt_boxes, iou_threshold)
        
        # 2. Filtrar los FPs: Solo nos quedamos con los que NO solapan casi nada con las cajas reales
        strict_fp_idx = []
        for idx in fp_idx:
            p_box = pred_boxes[idx][1:] # Extraer coordenadas [x1, y1, x2, y2]
            max_iou_with_any_gt = 0.0
            
            for gt in gt_boxes:
                g_box = gt[1:]
                iou = calculate_iou(p_box, g_box)
                if iou > max_iou_with_any_gt:
                    max_iou_with_any_gt = iou
            
            # Si el solapamiento máximo es menor al umbral estricto (ej. 0.1), es un FP fantasma real
            if max_iou_with_any_gt < max_fp_iou:
                strict_fp_idx.append(idx)

        # Ahora el error se basa en FPs estrictos o Falsos Negativos
        has_strict_error = bool(strict_fp_idx) or bool(fn_idx)
        
        # Le pasamos a draw_boxes los strict_fp_idx para que solo te marque en rojo los errores feos
        vis = draw_boxes(image, pred_boxes, gt_boxes, strict_fp_idx, fn_idx)

        # 3. Guardar múltiples ejemplos
        if has_strict_error and len(saved_paths["error_examples"]) < num_examples:
            path = os.path.join(output_dir, f"qualitative_error_{Path(image_name).stem}.png")
            cv2.imwrite(path, vis)
            saved_paths["error_examples"].append(path)
            
        # Para que sea "clean", exigimos que no haya NINGÚN error estándar (ni siquiera los "casi aciertos")
        elif not bool(fp_idx) and not bool(fn_idx) and gt_boxes and len(saved_paths["clean_examples"]) < num_examples:
            path = os.path.join(output_dir, f"qualitative_clean_{Path(image_name).stem}.png")
            cv2.imwrite(path, vis)
            saved_paths["clean_examples"].append(path)

        if len(saved_paths["error_examples"]) == num_examples and len(saved_paths["clean_examples"]) == num_examples:
            break

    if not saved_paths["clean_examples"] and not saved_paths["error_examples"]:
        print("Warning: Could not generate qualitative examples. Check image/label directories.")
    else:
        if len(saved_paths["error_examples"]) < num_examples:
            print(f"Warning: Only found {len(saved_paths['error_examples'])} error frames within {max_search} scanned frames.")
        if len(saved_paths["clean_examples"]) < num_examples:
            print(f"Warning: Only found {len(saved_paths['clean_examples'])} clean frames within {max_search} scanned frames.")

    return saved_paths


def compute_detection_metrics(results, hw_stats):
    """Builds the metrics dictionary using evaluation results and hardware monitor statistics."""
    ap50 = float(results.box.ap50[0])
    ap50_95 = float(results.box.ap[0])
    p, r = float(results.box.mp), float(results.box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    inference_ms = results.speed.get("inference", 0)

    metrics = {
        "AP50": ap50,
        "AP50_95": ap50_95,
        "precision": p,
        "recall": r,
        "f1": f1,
        "fps": 1000.0 / max(inference_ms, 0.1),
    }
    for key, value in hw_stats.items():
         metrics[f"jetson_{key}"] = value

    return metrics


def build_params_and_tags(config, model_tag, seed_tag, nvp_model):
    """Prepares configuration parameter and tag dictionaries for tracking in MLflow."""
    params = {
        "jetson_precision_mode": config["precision_mode"],
        "jetson_image_size": config["image_size"],
        "jetson_nvp_model": nvp_model,
        "model": model_tag,
        "imgsz": config["image_size"],
        "precision_mode": config["precision_mode"],
        "protocol": config["protocol"],
        "opt_conf_threshold": config["opt_conf"],
    }
    if config["precision_mode"] in ("INT8", "FP16"):
        params["v2"] = True

    tags = {
        "model": model_tag,
        "imgsz": str(config["image_size"]),
        "precision_mode": config["precision_mode"],
        "protocol": config["protocol"],
        "jetson_nvp_model": nvp_model,
        "seed": seed_tag,
    }
    return params, tags


def save_metrics_summary(output_dir, params, metrics):
    """Saves a local JSON summary containing all compiled run details and evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=2)
    return summary_path


def log_results_to_mlflow(config, experiment_id, run_name, params, tags, metrics,
                           exported_model_path, qualitative_paths, summary_path,
                           nvp_model, max_retries=5):
    """Logs parameters, tags, metrics, and generated file artifacts into MLflow."""
    for attempt in range(max_retries):
        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, parent_run_id=config["run_id"]):
                try:
                    mlflow.log_params(params)
                    mlflow.set_tags(tags)
                except Exception as e:
                    print(f"Error while logging params/tags: {e}")

                mlflow.log_metrics(metrics)
                mlflow.log_artifact(exported_model_path, artifact_path="jetson_engines")
                mlflow.log_artifact(summary_path, artifact_path="summary")
                for paths in qualitative_paths.values():
                    for path in paths:
                        mlflow.log_artifact(path, artifact_path="qualitative_examples")

            if config["precision_mode"] == "FP32" and config["image_size"] == 640 and nvp_model == "25W":
                protocol = config["protocol"]
                with mlflow.start_run(run_id=config["run_id"]):
                    mlflow.log_metrics({
                        f"jetson_ap50_{protocol}": metrics["AP50"],
                        f"jetson_ap50_95_{protocol}": metrics["AP50_95"],
                        f"jetson_precision_{protocol}": metrics["precision"],
                        f"jetson_recall_{protocol}": metrics["recall"],
                        f"jetson_f1_{protocol}": metrics["f1"],
                        f"jetson_fps_{protocol}": metrics["fps"],
                    })
                    mlflow.log_artifact(exported_model_path, artifact_path="jetson_engines")
                    for paths in qualitative_paths.values():
                        for path in paths:
                            mlflow.log_artifact(path, artifact_path="qualitative_examples")

            print("Metrics, params and tags logged successfully.")
            return
        except Exception as e:
            print(f"Error while logging metrics: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)

    print("Critical failure: Could not register metrics after several attempts.")


def main():
    config = get_run_config()
    data_folder = "/app/test_data"
    data_yaml = os.path.join(data_folder, "data.yaml")
    test_images_dir = os.path.join(data_folder, "images", "test")

    client = mlflow.tracking.MlflowClient()
    parent_run = client.get_run(config["run_id"])
    experiment_id = parent_run.info.experiment_id
    model_tag = parent_run.data.tags.get("model", "YOLO_model_unknown")
    seed_tag = parent_run.data.tags.get("seed", "seed_unknown")

    weights_file = download_weights(config["run_id"])
    deployment_metadata = download_deployment_metadata(config["run_id"])
    config["opt_conf"] = deployment_metadata["optimal_conf_threshold"]

    exported_model_path, device_target = export_model(
        weights_file, config["precision_mode"], config["image_size"], data_yaml
    )

    print("Loading exported model and warming up...")
    optimized = YOLO(exported_model_path, task="detect")
    warmup_model(optimized, test_images_dir, device_target)

    print("Running Testing on Test Subset...")
    monitor = JetsonMonitor(delay=0.2)
    monitor.start()

    results = optimized.val(
        data=data_yaml,
        device=device_target,
        batch=1,
        verbose=True,
        split="test",
        imgsz=config["image_size"],
        conf=config["opt_conf"],
    )

    monitor.stopped = True
    monitor.join()
    hw_stats = monitor.get_stats()

    metrics = compute_detection_metrics(results, hw_stats)

    print("Measuring per-image latency distribution (p50/p90/p95/p99)...")
    metrics.update(measure_latency_percentiles(optimized, test_images_dir, device_target, config["opt_conf"]))

    print("Generating qualitative detection examples (predicted vs ground truth)...")
    work_dir = tempfile.mkdtemp(prefix="jetson_eval_")
    qualitative_paths = generate_qualitative_examples(
        optimized, data_folder, device_target, config["opt_conf"], os.path.join(work_dir, "qualitative")
    )

    params, tags = build_params_and_tags(config, model_tag, seed_tag, monitor.nvp_model)
    summary_path = save_metrics_summary(work_dir, params, metrics)
    print(f"Local metrics summary saved to: {summary_path}")

    run_name = f"Test_Jetson_{config['precision_mode']}_{config['image_size']}_{monitor.nvp_model}"
    print("Logging metrics back to MLflow (Nested Run)...")
    log_results_to_mlflow(
        config, experiment_id, run_name, params, tags, metrics,
        exported_model_path, qualitative_paths, summary_path, monitor.nvp_model
    )

    print(f"Deployment Evaluation Complete for Run ID: {config['run_id']}")
    os._exit(0)


if __name__ == "__main__":
    main()