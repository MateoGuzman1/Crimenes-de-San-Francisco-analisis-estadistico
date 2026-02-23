from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import mlflow
import mlflow.sklearn

from sfcrime_model.config.core import PROJECT_ROOT, read_yaml_config
from sfcrime_model.processing.data_manager import train_and_eval


TFIDF_MAX_FEATURES = 10_000
VAL_SIZE = 0.2


def _log_params(params: Dict[str, object]) -> None:
    for k, v in params.items():
        mlflow.log_param(k, v)


def main() -> None:
    cfg = read_yaml_config()

    # --- Config (from config.yml) ---
    tracking_uri = cfg.mlflow.get("tracking_uri", "file:./mlruns")
    exp_name = cfg.mlflow.get("experiment_name", "sf-crime-classification")
    seed = int(cfg.project.get("random_seed", 42))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")

    model_keys: List[str] = ["logreg_tfidf", "linear_svc", "mnb"]
    results: List[Dict[str, object]] = []

    for model_key in model_keys:
        run_name = model_key
        with mlflow.start_run(run_name=run_name):
            t0 = time.time()

            _log_params(
                {
                    "model_key": model_key,
                    "random_seed": seed,
                    "tfidf_max_features": TFIDF_MAX_FEATURES,
                    "val_size": VAL_SIZE,
                }
            )

            pipe, metrics = train_and_eval(
                train_df, None,
                model_key=model_key,
                seed=seed,
                tfidf_max_features=TFIDF_MAX_FEATURES,
                val_size=VAL_SIZE,
            )

            # Log metrics (if test has labels)
            for mk, mv in metrics.items():
                mlflow.log_metric(mk, mv)

            # Log the model artifact
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            mlflow.log_metric("train_seconds", time.time() - t0)

            print(f"[OK] {model_key} -> {metrics}")
            results.append({"model_key": model_key, **metrics})

    print(f"\n{'model_key':<15} {'accuracy':>10} {'f1_macro':>10}")
    print("-" * 38)
    for r in results:
        print(f"{r['model_key']:<15} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")
    print("\nDone. Check MLflow UI for runs.")


if __name__ == "__main__":
    main()
