import hydra
import mlflow
import os
import gc
import tempfile
import shutil
from mlflow.tracking import MlflowClient
import torch
from ultralytics import YOLO
import json
from config import PolypDetectionConfig

def split_positive_negative_populations(test_images_dir, test_labels_dir):
    """Classify test images into positive and negative"""
    pos_images = []
    neg_images = []

    if os.path.exists(test_images_dir):
        for img_name in os.listdir(test_images_dir):
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(test_labels_dir, base_name + ".txt")
            
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                pos_images.append(img_name)
            else:
                neg_images.append(img_name)
                
    return pos_images, neg_images

def evaluate_positive_metrics(model, data_yaml_path, conf_threshold, img_size, use_half_precision):
    """Evaluate metrics on positive population"""
    print(f"\n Evaluating AP metrics on POSITIVE frames...")
    
    results_pos = model.val(
        data=data_yaml_path, conf=conf_threshold, verbose=False, 
        split="test", batch=1, device=0, imgsz=img_size, half=use_half_precision,
        save_json=False, save=False
    )
    print(results_pos.image_metrics)
    df = results_pos.to_df()
    print(df.columns)
    print(df.head())
    ap50 = float(results_pos.box.ap50[0]) if len(results_pos.box.ap50) > 0 else 0.0
    ap50_95 = float(results_pos.box.ap[0]) if len(results_pos.box.ap) > 0 else 0.0
    p = float(results_pos.box.mp)
    r = float(results_pos.box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-9)
    inference_time_ms = results_pos.speed.get('inference', 0)

    return ap50, ap50_95, p, r, f1, inference_time_ms

def evaluate_negative_metrics(model, neg_yaml_path, n_neg_images, conf_threshold, img_size, use_half_precision):
    if n_neg_images == 0:
        return 0.0
    print(f"\n Evaluating False Positives on NEGATIVE frames...")
    results = model.val(
        data=neg_yaml_path, conf=conf_threshold, verbose=False,
        split="test", batch=32, device=0, imgsz=img_size, half=use_half_precision, save_json=False, save=False
    )
    total_fps = int(results.confusion_matrix.matrix[0, -1])
    fp_per_frame = total_fps / n_neg_images
    print(f"    Total FPs: {total_fps} | FP/frame: {fp_per_frame:.4f}")
    return fp_per_frame

@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
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

    protocol = cfg.test.protocol
    datasets_base_path = cfg.files.base_path
    files_info = cfg.files.protocols[protocol]
    test_images_dir = os.path.join(datasets_base_path, files_info.yolo_output_dir, "images", "test")
    test_labels_dir = os.path.join(datasets_base_path, files_info.yolo_output_dir, "labels", "test")

    pos_images, neg_images = split_positive_negative_populations(test_images_dir, test_labels_dir)
    print(f"Dataset split found: {len(pos_images)} Positives | {len(neg_images)} Negatives.")

    with tempfile.TemporaryDirectory(dir=datasets_base_path) as shared_pos_dir, \
         tempfile.TemporaryDirectory(dir=datasets_base_path) as shared_neg_dir:
        pos_img_dir = os.path.join(shared_pos_dir, "images", "test")
        pos_lbl_dir = os.path.join(shared_pos_dir, "labels", "test")
        os.makedirs(pos_img_dir, exist_ok=True)
        os.makedirs(pos_lbl_dir, exist_ok=True)

        for img in pos_images:
            base_name = os.path.splitext(img)[0]
            # os.symlink creates a symbolic link to the original file, so it doesn't consume additional disk space
            os.symlink(os.path.join(test_images_dir, img), os.path.join(pos_img_dir, img))
            os.symlink(os.path.join(test_labels_dir, base_name + ".txt"), os.path.join(pos_lbl_dir, base_name + ".txt"))

        temp_yaml_path = os.path.join(shared_pos_dir, "data.yaml")
        with open(temp_yaml_path, "w") as f:
            f.write("train: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames: ['polyp']\n")

        neg_img_dir = os.path.join(shared_neg_dir, "images", "test")
        neg_lbl_dir = os.path.join(shared_neg_dir, "labels", "test")
        os.makedirs(neg_img_dir, exist_ok=True)
        os.makedirs(neg_lbl_dir, exist_ok=True)
        for img in neg_images:
            base_name = os.path.splitext(img)[0]
            os.symlink(os.path.join(test_images_dir, img), os.path.join(neg_img_dir, img))
            open(os.path.join(neg_lbl_dir, base_name + ".txt"), "w").close()  # label vacío
        neg_yaml_path = os.path.join(shared_neg_dir, "data.yaml")
        with open(neg_yaml_path, "w") as f:
            f.write("train: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames: ['polyp']\n")

        print("Shared positive dataset linked successfully.")

        for i, run in enumerate(runs):
            if i < cfg.test.start_k:
                print(f"\n[{i+1}/{len(runs)}] Skipping Run ID: {run.info.run_id} as it is below start_k ({cfg.test.start_k})")
                continue
            run_id = run.info.run_id

            client = mlflow.tracking.MlflowClient()

            parent_run = client.get_run(run_id)
            model_tag = parent_run.data.tags.get("model", "YOLO_model_unknown")
            seed_tag = parent_run.data.tags.get("seed", "seed_unknown")

            metric_val = run.data.metrics.get(cfg.test.metric, "N/A")
            if metric_val == "N/A" or metric_val < 0.01:
                print(f"\n[{i+1}/{len(runs)}] Skipping Run ID: {run_id} due to low metric value ({cfg.test.metric}: {metric_val})")
                continue

            print(f"\n[{i+1}/{len(runs)}] Test for Run ID: {run_id} ({cfg.test.metric}: {metric_val})")

            weights_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path='weights'
            )
            weights_file = os.path.join(weights_dir, 'best.pt')

            deployment_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path='deployment_info'
            )
            deployment_file = os.path.join(deployment_dir, 'deployment_metadata.json')

            print(f"Model weights downloaded to: {weights_file}")
            print(f"Deployment metadata downloaded to: {deployment_file}")

            with open(deployment_file, 'r') as f:
                deployment_metadata = json.load(f)
            
            opt_conf = deployment_metadata["optimal_conf_threshold"]
            model = YOLO(weights_file)
            model.info()

            print("Running validation on test set...")
            use_half_precision = cfg.test.precision_mode.upper() == "FP16"
            print(f"Using half precision: {use_half_precision}")
            print("Warming up GPU...")
            # Use a dummy input to warm up the GPU and load the model into memory before timing
            warmup_dummy = torch.zeros((1, 3, cfg.test.img_size, cfg.test.img_size)).to(0)
            for _ in range(50):
                _ = model.predict(warmup_dummy, verbose=False, half=use_half_precision)
            print("Warmup completed. Starting evaluation...")

            if len(pos_images) > 0:
                ap50, ap50_95, p, r, f1, inference_time_ms = evaluate_positive_metrics(
                    model, temp_yaml_path, opt_conf, cfg.test.img_size, use_half_precision
                )
            else:
                ap50, ap50_95, p, r, f1, inference_time_ms = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            fp_per_frame = evaluate_negative_metrics(
                model, neg_yaml_path, len(neg_images), opt_conf, cfg.test.img_size, use_half_precision
            )

            precision_mode = cfg.test.precision_mode.upper()
            metrics = {
                'AP50': ap50,
                'AP50_95': ap50_95,
                'precision': p,
                'recall': r,
                'f1': f1,
                "FP_per_frame": fp_per_frame,
                'fps': 1000.0 / max(inference_time_ms, 0.1)
            }

            params = {
                'precision_mode': precision_mode,
                'imgsz': cfg.test.img_size,
                'protocol': protocol,
                'device': "server",
                'model': model_tag,
            }

            tags = {
                'model': model_tag,
                'seed': seed_tag
            }

            run_name = f"LocalTest_{precision_mode}_{cfg.test.img_size}_{protocol}"

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name,
                parent_run_id =run_id
            ):  
                try:
                    mlflow.log_params(params)
                except Exception as e:
                    print(f"Error logging parameters: {e}")
                
                try:
                    mlflow.set_tags(tags)
                except Exception as e:
                    print(f"Error logging tags: {e}")
                mlflow.log_metrics(metrics)

                
                print(f"Logged successfully under nested run: {run_name}")

            if precision_mode == "FP32" and cfg.test.img_size == 640:
                    # save results to the parent run for easier comparison
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({
                        f'test_AP50_{protocol}': ap50,
                        f'test_AP50_95_{protocol}': ap50_95,
                        f'test_precision_{protocol}': p,
                        f'test_recall_{protocol}': r,
                        f'test_f1_{protocol}': f1,
                        f'test_fps_{protocol}': 1000.0 / max(inference_time_ms, 0.1),
                        f'test_FP_per_frame_{protocol}': fp_per_frame
                    })
                    print(f"Logged summary metrics to parent run {run_id} for easier comparison.")
            torch.cuda.synchronize()
            del model  # Clean up engine from GPU memory
            gc.collect()
            torch.cuda.empty_cache()

            try:
                shutil.rmtree(weights_dir)
                shutil.rmtree(deployment_dir)
                print("Cleaned up temporary artifact directories.")
            except Exception as e:
                print(f"Error during cleanup of temporary directories: {e}")

if __name__ == "__main__":
    main()