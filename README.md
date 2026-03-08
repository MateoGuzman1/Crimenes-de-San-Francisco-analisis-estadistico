# Clasificación de Crímenes en San Francisco

Pipeline de MLOps para clasificación multiclase de crímenes urbanos usando datos del SFPD.
Incluye un modelo XGBoost empaquetado, una API de inferencia con FastAPI, un tablero interactivo con Streamlit, experimentos con MLflow, versionado de datos con DVC y despliegue con Docker.

---

## Estructura del repositorio

```
.
├── api_crimes_san_francisco/          # API de inferencia (Jesus)
│   ├── Dockerfile
│   └── crimesSF-api/
│       ├── app/
│       │   ├── main.py                # Entrypoint FastAPI
│       │   ├── api.py                 # Rutas y lógica de predicción
│       │   └── schemas/               # Schemas Pydantic
│       ├── model-pkg/                 # model.pkl y classes.json (no en git)
│       ├── requirements.txt
│       └── run.sh
├── ui/                                # Tablero interactivo (Sergio)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── Home.py                    # App principal Streamlit
│       ├── settings.py                # MOCK_MODE, API_URL
│       ├── components/                # Formulario, mapa, panel de resultados
│       └── predictor/                 # api_client, mock_predictor
├── package-src/
│   └── sfcrime_model/                 # Paquete Python instalable
│       ├── train_xgboost_pipeline.py  # Entrenamiento XGBoost multiclase (Sergio)
│       ├── config.yml                 # Hiperparámetros del modelo
│       └── processing/
│           └── feature_engineering.py
├── data/                              # CSVs de train/test (versionados con DVC)
│   ├── train.csv.dvc
│   └── test.csv.dvc
├── MLFLOW_BACKUP/
│   └── SF-Crimes-MultiModel-FE/
│       └── models/Xgboost/artifacts/  # classes.json (model.pkl no en git)
├── manuales/
│   ├── manual-de-instalacion.md
│   └── manual-de-usuario-dashboard.md
├── docker-compose.yml                 # Orquestación API + UI
├── pyproject.toml
└── .dvc/config                        # Remoto DVC: S3 (s3://crimen-sf-dvcstore)
```

---

## Modelo

XGBoost multiclase entrenado con datos históricos del SFPD (39 categorías de crimen).

**Features:** `DayOfWeek`, `Hour`, `PdDistrict`, `X` (longitud), `Y` (latitud)

**Pipeline:** `OneHotEncoder` + `StandardScaler` → `XGBClassifier`

**Split:** temporal (corte en 2015-04-01)

El `model.pkl` no está en git por su tamaño (~110 MB). Se genera localmente con:

```bash
PYTHONPATH=package-src python3 -m sfcrime_model.train_xgboost_pipeline \
  --out-pkl MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/model.pkl \
  --out-classes MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/classes.json
```

---

## Despliegue con Docker

La forma recomendada de correr el sistema completo es con Docker Compose.
Ver las instrucciones detalladas en [`manuales/manual-de-instalacion.md`](manuales/manual-de-instalacion.md).

```bash
docker-compose up --build -d
```

- Tablero: `http://<IP>:8501`
- API docs: `http://<IP>:8001/docs`

---

## Desarrollo local

### Requisitos previos

- Python ≥ 3.12
- AWS CLI configurado con acceso al bucket `s3://crimen-sf-dvcstore`

### Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
pip install -e .
```

### Descargar datos con DVC

```bash
dvc pull
```

### Ver experimentos en MLflow UI

```bash
.venv/bin/mlflow ui --backend-store-uri file:./mlruns
```

Disponible en `http://localhost:5000`

---

## Tests

```bash
pip install -r requirements/test_requirements.txt
pytest tests/
```

---

## Autores

- Camilo Durango
- Mateo Guzmán
- Jesus Vilardi
- Sergio Angarita
