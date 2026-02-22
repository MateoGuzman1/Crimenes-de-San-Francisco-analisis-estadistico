# Models

Trained artifacts for the SF Crime classifier.

## Workflow

### 1 — Train (3 MLflow runs)

```bash
cd <project-root>
python3 -m sfcrime_model.train
```

Trains `logreg_tfidf`, `linear_svc`, and `mnb` on an 80/20 stratified split
of `data/train.csv` and logs params + metrics to MLflow.

### 2 — Export best model

```bash
python3 -m sfcrime_model.export_best_model
```

Picks the run with the highest `f1_macro` (fallback: `accuracy`), downloads
its artifact, and writes:
- `models/best_model/`  — MLflow sklearn model directory
- `models/best_model.pkl` — joblib pickle (gitignored)

### 3 — MLflow UI

```bash
.venv/bin/mlflow ui --backend-store-uri file:./mlruns
```

---

## Using default model from MLflow backup

The package ships with a pre-trained **LightGBM binary classifier** (Jesus's
best run) stored under:

```
MLFLOW_BACKUP/SF-Crimes-Binary/models/LGBM_bin/artifacts/
```

This artifact is registered in `config.yml` under the `model` key and loaded
automatically by `sfcrime_model.core.load_default_model()`.

### Quick inference

```python
import pandas as pd
from sfcrime_model.core import predict

# DataFrame must contain the same feature columns used at training time.
df = pd.read_csv("data/test.csv")
predictions = predict(df)
print(predictions.value_counts())
```

### How it works

1. `load_default_model()` reads `model.path` from `config.yml`.
2. The path is resolved relative to the **project root** via `PROJECT_ROOT`,
   so it works regardless of the current working directory.
3. `mlflow.pyfunc.load_model()` deserialises the artifact — no MLflow
   tracking server required (pure local file load).
4. The loaded model is cached in a module-level variable; subsequent calls to
   `predict()` reuse it without re-loading from disk.
