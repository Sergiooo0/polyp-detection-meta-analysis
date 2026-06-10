import csv
import os

import hydra
import mlflow
from mlflow.tracking import MlflowClient
from hydra.utils import get_original_cwd

from config import PolypDetectionConfig


def _get_parent_run_name(client, parent_run_id, parent_name_cache):
    if not parent_run_id:
        return ""

    if parent_run_id in parent_name_cache:
        return parent_name_cache[parent_run_id]

    try:
        parent_run = client.get_run(parent_run_id)
        parent_name = parent_run.data.tags.get("mlflow.runName", "")
    except Exception:
        parent_name = ""

    parent_name_cache[parent_run_id] = parent_name
    return parent_name


def _flatten_run(run, experiment_name, client, parent_name_cache):
    parent_run_id = run.data.tags.get("mlflow.parentRunId", "")
    row = {
        "experiment_name": experiment_name,
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "parent_run_id": parent_run_id,
        "parent_run_name": _get_parent_run_name(client, parent_run_id, parent_name_cache),
        "start_time": run.info.start_time
    }

    for key, value in run.data.params.items():
        row[f"param_{key}"] = value

    for key, value in run.data.metrics.items():
        row[f"metric_{key}"] = value

    for key, value in run.data.tags.items():
        row[f"tag_{key}"] = value

    return row


def _get_parent_run_ids(runs):
    return {
        run.data.tags.get("mlflow.parentRunId", "")
        for run in runs
        if run.data.tags.get("mlflow.parentRunId", "")
    }


def _is_empty_value(value):
    return value is None or value == ""


def _filter_sparse_columns(rows, missing_threshold=0.1):
    if not rows:
        return []

    columns = sorted({column for row in rows for column in row.keys()})
    kept_columns = []

    for column in columns:
        missing_count = sum(1 for row in rows if _is_empty_value(row.get(column)))
        missing_ratio = missing_count / len(rows)
        if missing_ratio < missing_threshold:
            kept_columns.append(column)

    return kept_columns


@hydra.main(version_base=None, config_path="configs", config_name="conf.yaml")
def main(cfg: PolypDetectionConfig):
    experiment_name = "T1_2"

    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    else:
        raise ValueError("MLflow tracking URI must be set in the configuration.")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10000,
    )

    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return

    parent_run_ids = _get_parent_run_ids(runs)
    parent_name_cache = {}
    rows = [_flatten_run(run, experiment_name, client, parent_name_cache) for run in runs]
    parent_rows = [
        row for run, row in zip(runs, rows)
        if run.info.run_id in parent_run_ids
    ]

    output_dir = os.path.join(get_original_cwd(), "mlflow_exports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_name}_runs.csv")
    parent_output_path = os.path.join(output_dir, f"{experiment_name}_parent_runs.csv")

    all_columns = sorted({column for row in rows for column in row.keys()})
    parent_columns = _filter_sparse_columns(parent_rows, missing_threshold=0.1)

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} runs from experiment '{experiment_name}' to: {output_path}")

    with open(parent_output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=parent_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(parent_rows)

    print(
        f"Exported {len(parent_rows)} parent runs from experiment '{experiment_name}' to: "
        f"{parent_output_path}"
    )


if __name__ == "__main__":
    main()