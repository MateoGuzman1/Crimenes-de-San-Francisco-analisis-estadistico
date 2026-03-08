# Manual de instalación: Clasificación de Crímenes en San Francisco

A continuación se presenta la guía de instalación del clasificador desarrollado para el Dataset de Crímenes de San Francisco.

## Definición

Pipeline de MLOps para clasificación binaria de crímenes urbanos usando datos del SFPD.
Incluye experimentos con MLflow, versionado de datos con DVC y un paquete Python instalable.

---

## Estructura del repositorio

```
.
├── data/                          # CSVs de train/test (versionados con DVC)
│   ├── train.csv.dvc
│   └── test.csv.dvc
├── dev/
│   └── notebooks/                 # Notebooks de exploración
├── MLFLOW_BACKUP/
│   └── SF-Crimes-Binary/
│       └── models/LGBM_bin/
│           └── artifacts/         # Modelo LightGBM binario preentrenado (modelo oficial)
├── mlruns/                        # Runs MLflow (backend de archivos, ignorado por git)
├── mlflow.db                      # Backend SQLite alternativo
├── models/                        # Artefactos exportados del mejor modelo
│   └── README.md
├── package-src/
│   └── sfcrime_model/             # Paquete Python instalable
│       ├── config.yml             # Configuración central
│       ├── core.py                # Inferencia con el modelo oficial (LGBM)
│       ├── train.py               # Pipeline de entrenamiento (experimental)
│       ├── export_best_model.py   # Exportación del mejor run MLflow (experimental)
│       ├── config/core.py         # Carga de configuración (AppConfig, PROJECT_ROOT)
│       └── processing/
│           └── data_manager.py    # Preprocesamiento y construcción de pipelines
├── requirements/
│   ├── requirements.txt           # Dependencias de runtime
│   └── test_requirements.txt      # Dependencias de testing
├── tests/                         # Pruebas con pytest
├── pyproject.toml                 # Definición del paquete (setuptools)
└── .dvc/config                    # Remoto DVC: S3 (s3://crimen-sf-dvcstore)
```

---

## Requisitos previos

- Python ≥ 3.10
- AWS CLI configurado con acceso al bucket `s3://crimen-sf-dvcstore`

---

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
pip install -e .
```

---

## Cómo correr el proyecto en una VM (EC2) con DVC

### A) Clonar el repositorio

```bash
git clone https://github.com/MateoGuzman1/Crimenes-de-San-Francisco-analisis-estadistico.git
cd Crimenes-de-San-Francisco-analisis-estadistico
```

### B) Crear entorno virtual e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
pip install -e .
```

### C) Configurar credenciales AWS

El remoto DVC usa S3 (`s3://crimen-sf-dvcstore`). Elegir una de estas opciones:

**Opción 1 — IAM Role adjunto a la instancia EC2**

No requiere configuración adicional. Verificar con:

```bash
aws sts get-caller-identity
```

**Opción 2 — `aws configure`**

```bash
aws configure
# Ingresar: AWS Access Key ID, Secret Access Key, región (us-east-1)
```

**Opción 3 — Variables de entorno**

```bash
export AWS_ACCESS_KEY_ID=tu_key
export AWS_SECRET_ACCESS_KEY=tu_secret
export AWS_DEFAULT_REGION=us-east-1
```

### D) Descargar datos con DVC

```bash
dvc pull
ls -lh data/train.csv data/test.csv   
```

### E) Ejecutar inferencia con el modelo oficial

El modelo XGBoost está incluido en el repositorio en
`MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts`.
No se requiere entrenamiento ni servidor MLflow activo.

```python
import pandas as pd
from sfcrime_model.core import predict

df = pd.read_csv("data/test.csv")
predicciones = predict(df)
print(predicciones.value_counts())
```

### F) Ver experimentos en MLflow UI

```bash
# Backend de archivos
.venv/bin/mlflow ui --backend-store-uri file:./mlruns

# Backend SQLite
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Disponible en `http://<ip-ec2>:5000`
(abrir el puerto 5000 en el Security Group de la instancia EC2).

---

## Entrenamiento 

Entrena tres modelos (`logreg_tfidf`, `linear_svc`, `mnb`) y registra cada run en MLflow.
**No es necesario para reproducir el modelo default.**

```bash
python -m sfcrime_model.train
```

## Exportar el mejor modelo 

Selecciona el run con mayor `f1_macro` del experimento MLflow activo y lo serializa en `models/`.

```bash
python -m sfcrime_model.export_best_model
```

---

## Tests

```bash
pip install -r requirements/test_requirements.txt
pytest tests/
```

---

## Despliegue con Docker en EC2

El sistema se puede desplegar completo (API + tablero) usando Docker Compose. Esta sección describe el proceso paso a paso.

### Prerequisitos en la VM

- Ubuntu 22.04, t2.medium, 30 GB de almacenamiento
- Puertos abiertos en el Security Group: 22 (SSH), 8001 (API), 8501 (UI)
- Docker y docker-compose instalados:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
newgrp docker
```

### A) Entrenar el modelo localmente (Mac/Linux)

El modelo no está versionado en git por su tamaño (≈110 MB). Se genera localmente con:

```bash
cd <raiz-del-repo>
PYTHONPATH=package-src python3 -m sfcrime_model.train_xgboost_pipeline \
  --out-pkl MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/model.pkl \
  --out-classes MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/classes.json
```

Luego copiar los artefactos al directorio de la API:

```bash
cp MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/model.pkl \
   api_crimes_san_francisco/crimesSF-api/model-pkg/model.pkl

cp MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts/classes.json \
   api_crimes_san_francisco/crimesSF-api/model-pkg/classes.json
```

### B) Clonar el repositorio en la VM

```bash
git clone -b docker-despliegue https://github.com/MateoGuzman1/Crimenes-de-San-Francisco-analisis-estadistico.git
cd Crimenes-de-San-Francisco-analisis-estadistico
```

### C) Crear carpetas para los artefactos

```bash
mkdir -p api_crimes_san_francisco/crimesSF-api/model-pkg
mkdir -p MLFLOW_BACKUP/SF-Crimes-MultiModel-FE/models/Xgboost/artifacts
```

### D) Enviar los artefactos a la VM por SCP

Desde la máquina local (reemplazar `<pem>` e `<IP>`):

```bash
scp -i <pem> api_crimes_san_francisco/crimesSF-api/model-pkg/model.pkl \
  ubuntu@<IP>:~/Crimenes-de-San-Francisco-analisis-estadistico/api_crimes_san_francisco/crimesSF-api/model-pkg/model.pkl

scp -i <pem> api_crimes_san_francisco/crimesSF-api/model-pkg/classes.json \
  ubuntu@<IP>:~/Crimenes-de-San-Francisco-analisis-estadistico/api_crimes_san_francisco/crimesSF-api/model-pkg/classes.json
```

### E) Levantar los servicios

```bash
docker-compose up --build -d
docker-compose ps
```

Ambos servicios deben aparecer en estado `Up`.

### F) Acceder al tablero

```
http://<IP-publica-de-la-VM>:8501
```

La API está disponible en:

```
http://<IP-publica-de-la-VM>:8001/docs
```

### Nota sobre AWS Academy

Las instancias de AWS Academy asignan una IP pública nueva cada vez que se reinicia la instancia. Al volver a usarla:

1. Verificar la nueva IP en EC2 → Instances → Public IPv4 address
2. Conectarse por SSH con esa IP
3. Levantar los contenedores si están parados: `docker-compose up -d`
4. Compartir la nueva IP a los usuarios

