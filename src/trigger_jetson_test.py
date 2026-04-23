import os
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
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    else:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlflow_db_path = os.path.join(repo_dir, "mlruns.db")
        tracking_uri = f"sqlite:///{mlflow_db_path}"
        mlflow.set_tracking_uri(tracking_uri)

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
            
            # 1. CD into the remote test folder
            # 2. Export MLflow credentials from the Hydra config
            # 3. Export the dynamic MLFLOW_RUN_ID
            # 4. Run the Jetson Hydra script
            cmd = (
                "docker run --rm --network host --runtime nvidia "
                f"-v {cfg.connection.test_folder_remote}:/app/test_data "  # Mount remote test folder
                "-e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
                f"-e MLFLOW_TRACKING_URI={cfg.mlflow.tracking_uri} "
                f"-e MLFLOW_S3_ENDPOINT_URL={cfg.mlflow.s3_endpoint_url} "
                f"-e AWS_ACCESS_KEY_ID={cfg.mlflow.aws_access_key_id} "
                f"-e AWS_SECRET_ACCESS_KEY={cfg.mlflow.aws_secret_access_key} "
                f"-e MLFLOW_RUN_ID={run_id} "
                f"-e HALF_PRECISION={str(cfg.test.half_precision)} "
                f"-e IMGSZ={cfg.params.img_size} "
                "yolo-jetson-test" 
            )
            
            print(f"Executing over SSH on {cfg.connection.host}...")
            result = c.run(cmd, hide=False, warn=True)

            if result is not None and result.failed:
                print(f"Evaluation of Run ID {run_id} failed with exit code {result.return_code}.")
                print(result.stderr)


if __name__ == "__main__":
    main()