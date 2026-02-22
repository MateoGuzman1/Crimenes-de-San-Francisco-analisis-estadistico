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
