import mlflow
from mlflow.tracking import MlflowClient
from fabric import Connection
from utils.get_config import get_config
from config import PolypDetectionConfig

def main():
    cfg: PolypDetectionConfig = get_config()
    print(cfg.test)
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

    # Prepare Fabric connection kwargs (handle password if provided)
    connect_kwargs = {}
    if cfg.connection.password:
        connect_kwargs["password"] = str(cfg.connection.password)

    remote_src = "/home/jetuser/polyp-detection-meta-analysis/src"
    for i, run in enumerate(runs):
        run_id = run.info.run_id
        metric_val = run.data.metrics.get(cfg.test.metric, "N/A")
        print(f"\n[{i+1}/{cfg.test.top_k}] Triggering Jetson for Run ID: {run_id} ({cfg.test.metric}: {metric_val})")
        
        with Connection(
            host=cfg.connection.host,
            user=cfg.connection.username,
            port=cfg.connection.port,
            connect_kwargs=connect_kwargs
        ) as c:
            
            cmd = (
                "docker run --rm --runtime nvidia " # Use NVIDIA runtime for GPU access
                f"-v {cfg.connection.test_folder_remote}:/app/test_data "  # Mount remote test folder
                "-e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
                f"-v {remote_src}:/app/src "  # Mount remote source folder (avoid rebuilding) (remove at the end)
                "-v /run/jtop.sock:/run/jtop.sock " # Mount jtop socket for hardware monitoring
                f"-e MLFLOW_TRACKING_URI={tracking_uri} "
                f"-e MLFLOW_S3_ENDPOINT_URL={cfg.mlflow.s3_endpoint_url} "
                f"-e AWS_ACCESS_KEY_ID={cfg.mlflow.aws_access_key_id} "
                f"-e AWS_SECRET_ACCESS_KEY={cfg.mlflow.aws_secret_access_key} "
                f"-e MLFLOW_RUN_ID={run_id} "
                f"-e HALF_PRECISION={str(cfg.test.half_precision)} "
                f"-e IMGSZ={cfg.test.img_size} "
                "yolo-jetson-test" 
            )
            
            print(f"Executing over SSH on {cfg.connection.host}...")
            result = c.run(cmd, hide=False, warn=True, pty=True)

            if result is not None and result.failed:
                print(f"Evaluation of Run ID {run_id} failed with exit code {result.return_code}.")
                print(result.stderr)
                if int(result.return_code) == 139:
                    print("Segmentation fault detected. This may indicate an out-of-memory error on the Jetson device.")
                    print(f"Retrying the evaluation for Run ID {run_id}...")
                    result_retry = c.run(cmd, hide=False, warn=True, pty=True)
                    if result_retry is not None and result_retry.failed:
                        print(f"Retry also failed with exit code {result_retry.return_code}.")
                        print(result_retry.stderr)


if __name__ == "__main__":
    main()