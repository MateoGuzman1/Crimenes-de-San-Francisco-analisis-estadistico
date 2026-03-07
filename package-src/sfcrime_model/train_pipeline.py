# Uso: python -m sfcrime_model.train_pipeline
# Requiere data/train.csv (dvc pull si no existe)
from __future__ import annotations

import time

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sfcrime_model.config.core import DATASET_DIR, config
from sfcrime_model.pipeline import sfcrime_pipe
from sfcrime_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    mlflow.set_experiment(config.app_config.mlflow_experiment_name)

    print("Cargando datos...")
    data = load_dataset(file_name=config.app_config.train_data_file)

    le = LabelEncoder()
    y  = le.fit_transform(data[config.ml_config.target])

    X  = data.drop(
        columns=[config.ml_config.target] + config.ml_config.drop_cols,
        errors="ignore",
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.ml_config.test_size,
        stratify=y,
        random_state=config.ml_config.random_state,
    )

    with mlflow.start_run(run_name="xgboost_multiclass_pipeline"):
        mlflow.log_params({
            "model":        "XGBoost",
            "n_estimators": config.ml_config.n_estimators,
            "max_depth":    config.ml_config.max_depth,
            "learning_rate":config.ml_config.learning_rate,
            "num_classes":  config.ml_config.num_classes,
            "target":       "multiclass (39 categorias)",
            "random_state": config.ml_config.random_state,
        })

        t0 = time.time()
        print("Entrenando pipeline...")
        sfcrime_pipe.fit(X_train, y_train)

        y_pred = sfcrime_pipe.predict(X_val)
        metrics = {
            "accuracy":      float(accuracy_score(y_val, y_pred)),
            "f1_macro":      float(f1_score(y_val, y_pred, average="macro")),
            "train_seconds": time.time() - t0,
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sfcrime_pipe, artifact_path="model")

        save_pipeline(pipeline_to_persist=sfcrime_pipe, label_classes=le.classes_)

        print("\nPipeline guardado en trained/")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    run_training()
