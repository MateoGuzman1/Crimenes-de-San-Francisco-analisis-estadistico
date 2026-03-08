import joblib
import os
import time
import pandas as pd

from typing import List
from pydantic import BaseModel


# =========================
# SCHEMAS (LOS MANTENEMOS IGUAL)
# =========================

class CrimeInput(BaseModel):
    fecha: str
    hora: int
    latitud: float
    longitud: float
    distrito: str


class MultipleDataInputs(BaseModel):
    inputs: List[CrimeInput]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "fecha": "2026-03-07",
                        "hora": 12,
                        "latitud": 37.777635,
                        "longitud": -122.434720,
                        "distrito": "NORTHERN"
                    }
                ]
            }
        }


class PredictionResults(BaseModel):
    crimen_predicho: str
    probabilidad: float
    latency_ms: float
    note: str

    class Config:
        schema_extra = {
            "example": {
                "crimen_predicho": "LARCENY/THEFT",
                "probabilidad": 0.2485,
                "latency_ms": 23.4,
                "note": "Resultado generado por API XGBoost"
            }
        }


# =========================
# CARGA DEL MODELO PKL ⭐
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../model-pkg/model.pkl"
)

model = joblib.load(MODEL_PATH)


# =========================
# PREDICCIÓN
# =========================

def make_prediction(input_data: pd.DataFrame):

    start_time = time.time()

    try:

        predictions = model.predict(input_data)

        # Probabilidades si el modelo las soporta
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data).max(axis=1)
        else:
            probabilities = [0.0] * len(predictions)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "latency_ms": (time.time() - start_time) * 1000,
            "errors": None
        }

    except Exception as e:

        return {
            "predictions": None,
            "probabilities": None,
            "latency_ms": (time.time() - start_time) * 1000,
            "errors": str(e)
        }