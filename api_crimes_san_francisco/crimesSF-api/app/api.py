import json
import time
from typing import Any

import joblib
import numpy as np
import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model-pkg/model.pkl")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "../model-pkg/classes.json")

model = joblib.load(MODEL_PATH)

with open(CLASSES_PATH, encoding="utf-8") as f:
    classes = json.load(f)

model_version = getattr(model, "__version__", "pkl-model")

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["DayOfWeek"] = pd.to_datetime(df["fecha"]).dt.day_name()
    out["Hour"] = df["hora"].astype(int)
    out["PdDistrict"] = df["distrito"].str.upper()
    out["X"] = df["longitud"].astype(float)
    out["Y"] = df["latitud"].astype(float)
    return out


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    h = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )
    return h.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    start = time.time()
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info(f"Received inputs: {input_data.inputs}")

    try:
        features = _transform(input_df)
        preds = model.predict(features)
        probas = model.predict_proba(features).max(axis=1)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    crime = classes[int(preds[0])]
    prob = float(probas[0])
    latency = (time.time() - start) * 1000

    logger.info(f"Predicted: {crime} ({prob:.4f})")
    return {
        "crimen_predicho": crime,
        "probabilidad": prob,
        "latency_ms": latency,
        "note": "Resultado generado por API XGBoost",
    }
