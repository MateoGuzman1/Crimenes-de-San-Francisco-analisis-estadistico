from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel

import sfcrime_model

PACKAGE_ROOT = Path(sfcrime_model.__file__).resolve().parent
PROJECT_ROOT  = PACKAGE_ROOT.parents[1]          # raiz del repo
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR      = PROJECT_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


class AppConfig(BaseModel):
    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class ModelConfig(BaseModel):
    target: str
    drop_cols: List[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    num_classes: int


class Config(BaseModel):
    app_config: AppConfig
    ml_config: ModelConfig


def _load_raw_config(path: Optional[Path] = None) -> dict:
    cfg_path = path or CONFIG_FILE_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config no encontrada en: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_and_validate_config(raw: Optional[dict] = None) -> Config:
    if raw is None:
        raw = _load_raw_config()
    return Config(
        app_config=AppConfig(**raw),
        ml_config=ModelConfig(**raw),
    )


config: Config = create_and_validate_config()
