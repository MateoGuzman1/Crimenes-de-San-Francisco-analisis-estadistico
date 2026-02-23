from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sfcrime_model.config.core import PROJECT_ROOT, read_yaml_config

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
BEST_MODEL_PKL = MODELS_DIR / "best_model.pkl"


def _find_best_run(client: mlflow.tracking.MlflowClient, experiment_id: str):
    # Try f1_macro first, fall back to accuracy
    for metric in ("f1_macro", "accuracy"):
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"metrics.{metric} > 0",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if runs:
            return runs[0], metric
    raise RuntimeError("No finished runs with metrics found in this experiment.")


def main() -> None:
    cfg = read_yaml_config()
    tracking_uri = cfg.mlflow.get("tracking_uri", "file:./mlruns")
    exp_name = cfg.mlflow.get("experiment_name", "sf-crime-classification")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{exp_name}' not found. Run training first.")

    best_run, sort_metric = _find_best_run(client, experiment.experiment_id)
    run_id = best_run.info.run_id
    score = best_run.data.metrics.get(sort_metric)
    model_key = best_run.data.params.get("model_key", "unknown")

    print(f"Best run : {run_id}")
    print(f"  model  : {model_key}")
    print(f"  {sort_metric}: {score:.4f}")

    # Load artifact from MLflow
    model_uri = f"runs:/{run_id}/model"
    pipe = mlflow.sklearn.load_model(model_uri)

    # Save as MLflow model directory (overwrite if exists)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    mlflow.sklearn.save_model(pipe, path=str(BEST_MODEL_DIR))
    print(f"Saved MLflow model  -> {BEST_MODEL_DIR}/")

    # Save as joblib pickle
    joblib.dump(pipe, BEST_MODEL_PKL)
    print(f"Saved joblib pickle -> {BEST_MODEL_PKL}")


if __name__ == "__main__":
    main()
