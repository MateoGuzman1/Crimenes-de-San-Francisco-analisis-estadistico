from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from sfcrime_model.config.core import PROJECT_ROOT, read_yaml_config

_default_model: Any = None


def load_default_model() -> Any:
    """Load the LightGBM binary model from the MLflow backup artifact directory.

    The model path is resolved relative to the project root so the package
    works regardless of the current working directory.
    """
    global _default_model
    if _default_model is None:
        cfg = read_yaml_config()
        model_cfg = cfg.model
        artifact_path = PROJECT_ROOT / model_cfg["path"]
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Default model artifact not found at: {artifact_path}\n"
                "Ensure MLFLOW_BACKUP/SF-Crimes-Binary/models/LGBM_bin/artifacts "
                "is present in the project root."
            )
        _default_model = mlflow.pyfunc.load_model(str(artifact_path))
    return _default_model


def predict(input_df: pd.DataFrame) -> pd.Series:
    """Run inference with the default LightGBM binary model.

    Parameters
    ----------
    input_df:
        DataFrame with the same feature columns the model was trained on.

    Returns
    -------
    pd.Series
        Binary predictions (0 / 1) for each row.
    """
    model = load_default_model()
    preds = model.predict(input_df)
    return pd.Series(preds, index=input_df.index, name="prediction")
