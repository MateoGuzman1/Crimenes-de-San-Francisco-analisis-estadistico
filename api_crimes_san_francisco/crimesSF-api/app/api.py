import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

# Cargar el pickle del modelo al iniciar la API
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model-pkg/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

model_version = getattr(model, "__version__", "pkl-model")  # mantiene compatibilidad

from sfcrime_model.predict import make_prediction
from app import __version__, schemas
from app.config import settings

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Prediccion usando el modelo de bankchurn
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results