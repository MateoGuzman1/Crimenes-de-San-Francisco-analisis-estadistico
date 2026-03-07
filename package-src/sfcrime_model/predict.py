from __future__ import annotations

import typing as t
from datetime import datetime

import numpy as np
import pandas as pd

import sfcrime_model
from sfcrime_model.config.core import config
from sfcrime_model.processing.data_manager import (
    load_label_classes,
    load_pipeline,
)
from sfcrime_model.processing.validation import validate_inputs

_version = sfcrime_model.__version__

_pipeline_file = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_sfcrime_pipe  = load_pipeline(file_name=_pipeline_file)
_label_classes = load_label_classes()

_DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _build_raw_df(input_data: pd.DataFrame) -> pd.DataFrame:
    """Convierte payload de la API al formato crudo del pipeline."""
    rows = []
    for _, row in input_data.iterrows():
        date_obj    = datetime.strptime(str(row["fecha"]), "%Y-%m-%d")
        day_of_week = _DAY_NAMES[date_obj.weekday()]
        dates_str   = f"{row['fecha']} {int(row['hora']):02d}:00:00"
        rows.append({
            "Dates":      dates_str,
            "DayOfWeek":  day_of_week,
            "PdDistrict": row["distrito"],
            "Address":    "",
            "X":          float(row["longitud"]),
            "Y":          float(row["latitud"]),
        })
    return pd.DataFrame(rows)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Retorna {"predictions": [...], "version": str, "errors": str|None}."""
    data = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
    validated_data, errors = validate_inputs(input_data=data)

    results: dict = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        raw_df       = _build_raw_df(validated_data)
        pred_encoded = _sfcrime_pipe.predict(raw_df).astype(int)
        proba_matrix = _sfcrime_pipe.predict_proba(raw_df)
        max_probs    = proba_matrix[np.arange(len(pred_encoded)), pred_encoded]
        categories   = _label_classes[pred_encoded]

        results["predictions"] = [
            {"category": cat, "probability": float(prob)}
            for cat, prob in zip(categories, max_probs)
        ]

    return results
