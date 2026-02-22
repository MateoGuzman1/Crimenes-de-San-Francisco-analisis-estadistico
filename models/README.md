# Modelos

Artefactos entrenados para el clasificador de crímenes de San Francisco.

## Flujo de trabajo

### 1 — Entrenar (3 corridas de MLflow)

```bash
cd <raíz-del-proyecto>
python3 -m sfcrime_model.train
```

Entrena `logreg_tfidf`, `linear_svc` y `mnb` con una división 80/20 estratificada
de `data/train.csv` y registra parámetros y métricas en MLflow.

### 2 — Exportar el mejor modelo

```bash
python3 -m sfcrime_model.export_best_model
```

Selecciona la corrida con el mayor `f1_macro` (alternativa: `accuracy`), descarga
el artefacto y genera:
- `models/best_model/`  — directorio de modelo MLflow sklearn
- `models/best_model.pkl` — pickle joblib (ignorado por git)

### 3 — Interfaz de MLflow

```bash
.venv/bin/mlflow ui --backend-store-uri file:./mlruns
```

---

## Uso del modelo por defecto desde el backup de MLflow

El paquete incluye un **clasificador binario LightGBM** preentrenado (mejor corrida
de Jesus) almacenado en:

```
MLFLOW_BACKUP/SF-Crimes-Binary/models/LGBM_bin/artifacts/
```

Este artefacto está registrado en `config.yml` bajo la clave `model` y se carga
automáticamente mediante `sfcrime_model.core.load_default_model()`.

### Inferencia rápida

```python
import pandas as pd
from sfcrime_model.core import predict

# El DataFrame debe contener las mismas columnas de características usadas en el entrenamiento.
df = pd.read_csv("data/test.csv")
predicciones = predict(df)
print(predicciones.value_counts())
```

### Cómo funciona

1. `load_default_model()` lee `model.path` desde `config.yml`.
2. La ruta se resuelve relativa a la **raíz del proyecto** mediante `PROJECT_ROOT`,
   por lo que funciona independientemente del directorio de trabajo actual.
3. `mlflow.pyfunc.load_model()` deserializa el artefacto — no se requiere servidor
   de seguimiento de MLflow (carga local pura).
4. El modelo cargado queda en caché en una variable a nivel de módulo; las llamadas
   posteriores a `predict()` lo reutilizan sin volver a cargarlo desde disco.
