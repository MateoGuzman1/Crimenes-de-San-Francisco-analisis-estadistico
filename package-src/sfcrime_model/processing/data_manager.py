from __future__ import annotations

import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import sfcrime_model
from sfcrime_model.config.core import (
    DATASET_DIR,
    TRAINED_MODEL_DIR,
    config,
)

_version = sfcrime_model.__version__


def load_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / file_name)


def save_pipeline(*, pipeline_to_persist: Pipeline, label_classes: np.ndarray) -> None:
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    _remove_old_pipelines(files_to_keep=[save_file_name, "label_classes.npy", "__init__.py"])

    joblib.dump(pipeline_to_persist, save_path)
    np.save(str(TRAINED_MODEL_DIR / "label_classes.npy"), label_classes)


def load_pipeline(*, file_name: str) -> Pipeline:
    return joblib.load(TRAINED_MODEL_DIR / file_name)


def load_label_classes() -> np.ndarray:
    classes_path = TRAINED_MODEL_DIR / "label_classes.npy"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"label_classes.npy no encontrado en {TRAINED_MODEL_DIR}\n"
            "Ejecute: python -m sfcrime_model.train_pipeline"
        )
    return np.load(str(classes_path), allow_pickle=True)


def _remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in files_to_keep:
            model_file.unlink()
