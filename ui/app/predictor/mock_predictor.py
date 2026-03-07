from __future__ import annotations

import random
import time
from datetime import datetime

# Categorías típicas (puedes ajustar según tu dataset)
CATEGORIES = [
    "LARCENY/THEFT",
    "ASSAULT",
    "NON-CRIMINAL",
    "ROBBERY",
    "BURGLARY",
    "VEHICLE THEFT",
    "VANDALISM",
    "DRUG/NARCOTIC",
    "WARRANTS",
    "SUSPICIOUS OCC"
]

# Sesgos simples para que se sienta “real”
DISTRICT_BIAS = {
    "SOUTHERN": ["LARCENY/THEFT", "DRUG/NARCOTIC", "ASSAULT", "VEHICLE THEFT"],
    "TENDERLOIN": ["DRUG/NARCOTIC", "ASSAULT", "WARRANTS", "SUSPICIOUS OCC"],
    "MISSION": ["ASSAULT", "LARCENY/THEFT", "ROBBERY"],
    "NORTHERN": ["LARCENY/THEFT", "VANDALISM", "BURGLARY"],
}

NIGHT_BIAS = ["ROBBERY", "ASSAULT", "VEHICLE THEFT", "BURGLARY"]


def _weighted_choice(options: list[str], weights: list[float]) -> str:
    # random.choices está ok para mock
    return random.choices(options, weights=weights, k=1)[0]


def mock_predict(payload: dict) -> dict:
    """
    Genera una predicción simulada basada en distrito y hora.
    Retorna:
      - crimen_predicho
      - probabilidad (0-1)
      - latency_ms
      - note
    """
    t0 = time.time()

    distrito = payload.get("distrito", "SOUTHERN")
    hora = int(payload.get("hora", 12))

    base = CATEGORIES.copy()

    # Pesos base uniformes
    weights = [1.0] * len(base)

    # Sesgo por distrito
    bias = DISTRICT_BIAS.get(distrito, [])
    for i, cat in enumerate(base):
        if cat in bias:
            weights[i] += 1.5

    # Sesgo por hora (noche)
    if hora >= 20 or hora <= 4:
        for i, cat in enumerate(base):
            if cat in NIGHT_BIAS:
                weights[i] += 1.2

    pred = _weighted_choice(base, weights)

    # Probabilidad “creíble” (más alta si cae en una categoría sesgada)
    prob = 0.55 + random.random() * 0.35
    if pred in bias:
        prob += 0.05
    prob = min(prob, 0.95)

    latency_ms = (time.time() - t0) * 1000 + random.randint(10, 60)

    # Nota para la UI
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "crimen_predicho": pred,
        "probabilidad": float(prob),
        "latency_ms": float(latency_ms),
        "note": f"Resultados basados en datos históricos (mock) • {now}"
    }
