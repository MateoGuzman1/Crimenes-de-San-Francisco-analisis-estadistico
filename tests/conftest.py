import pytest

from sfcrime_model.config.core import config
from sfcrime_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    # Carga el test set y lo convierte al formato de input de la API
    df = load_dataset(file_name=config.app_config.test_data_file)
    # El test set no tiene Category, solo las columnas de features crudas
    # Construimos el input en el formato que espera make_prediction
    import pandas as pd
    rows = []
    for _, row in df.head(5).iterrows():
        import pandas as _pd
        dates = _pd.to_datetime(row["Dates"])
        rows.append({
            "fecha":    dates.strftime("%Y-%m-%d"),
            "hora":     int(dates.hour),
            "latitud":  float(row["Y"]),
            "longitud": float(row["X"]),
            "distrito": str(row["PdDistrict"]),
        })
    return rows
