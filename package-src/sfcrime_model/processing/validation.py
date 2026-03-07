from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

VALID_DISTRICTS = [
    "BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN",
    "PARK", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN",
]


class CrimeInputSchema(BaseModel):
    fecha:    str
    hora:     int
    latitud:  float
    longitud: float
    distrito: str


class MultipleCrimeInputs(BaseModel):
    inputs: List[CrimeInputSchema]


def validate_inputs(
    *, input_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[str]]:
    errors = None
    try:
        MultipleCrimeInputs(
            inputs=input_data.replace({float("nan"): None}).to_dict(orient="records")
        )
    except ValidationError as exc:
        errors = exc.json()

    return input_data, errors
