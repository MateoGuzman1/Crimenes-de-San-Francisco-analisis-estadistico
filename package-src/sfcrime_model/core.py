from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from sfcrime_model.config.core import PROJECT_ROOT, read_yaml_config

_default_model: Any = None


def load_default_model() -> Any:
    """Carga el modelo binario LightGBM desde el directorio de artefactos del backup de MLflow.

    La ruta se resuelve relativa a la raíz del proyecto para que el paquete
    funcione independientemente del directorio de trabajo actual.
    """
    global _default_model
    if _default_model is None:
        cfg = read_yaml_config()
        model_cfg = cfg.model
        artifact_path = PROJECT_ROOT / model_cfg["path"]
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artefacto del modelo por defecto no encontrado en: {artifact_path}\n"
                "Asegúrese de que MLFLOW_BACKUP/SF-Crimes-Binary/models/LGBM_bin/artifacts "
                "exista en la raíz del proyecto."
            )
        _default_model = mlflow.pyfunc.load_model(str(artifact_path))
    return _default_model


def predict(input_df: pd.DataFrame) -> pd.Series:
    """Ejecuta inferencia con el modelo binario LightGBM por defecto.

    Parámetros
    ----------
    input_df:
        DataFrame con las mismas columnas de características usadas en el entrenamiento.

    Retorna
    -------
    pd.Series
        Predicciones binarias (0 / 1) para cada fila.
    """
    model = load_default_model()
    preds = model.predict(input_df)
    return pd.Series(preds, index=input_df.index, name="prediction")
