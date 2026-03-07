import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_local_model():
    repo_root = Path(__file__).resolve().parents[3]
    model_path = repo_root / "MLFLOW_BACKUP" / "SF-Crimes-MultiModel-FE" / "models" / "Xgboost" / "artifacts" / "model.pkl"
    classes_path = repo_root / "MLFLOW_BACKUP" / "SF-Crimes-MultiModel-FE" / "models" / "Xgboost" / "artifacts" / "classes.json"

    model = joblib.load(model_path)

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    return model, classes


def predict_local(payload: dict) -> dict:
    model, classes = load_local_model()

    fecha = pd.to_datetime(payload["fecha"])

    X = pd.DataFrame([{
        "DayOfWeek": fecha.day_name(),
        "Hour": int(payload["hora"]),
        "PdDistrict": str(payload["distrito"]),
        "X": float(payload["longitud"]),
        "Y": float(payload["latitud"]),
    }])

    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    prob = float(probs[pred_idx])

    return {
        "crimen_predicho": classes[pred_idx],
        "probabilidad": prob,
        "latency_ms": 0.0,
        "note": "Resultado generado con modelo local XGBoost"
    }