from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Validaciones mínimas
    required_cols = {"Dates", "PdDistrict", "X", "Y", "Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    # Derivar variables temporales
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    df["DayOfWeek"] = df["Dates"].dt.day_name()
    df["Hour"] = df["Dates"].dt.hour

    # Mantener solo columnas necesarias
    df = df[["DayOfWeek", "Hour", "PdDistrict", "X", "Y", "Category"]].copy()

    # Limpiar nulos
    df = df.dropna()

    return df


def temporal_split(df: pd.DataFrame, cutoff: str = "2015-04-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Releer Dates para split temporal
    # Como ya lo quitamos arriba, volvemos a cargarlo desde índice auxiliar si se necesitara.
    # Más simple: este split lo hacemos antes de recortar columnas.
    raise NotImplementedError("Usa load_data_with_dates() en main.")


def build_pipeline(random_seed: int = 42) -> Pipeline:
    categorical_features = ["DayOfWeek", "PdDistrict"]
    numeric_features = ["Hour", "X", "Y"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_seed,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return pipeline


def load_data_with_dates(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    required_cols = {"Dates", "PdDistrict", "X", "Y", "Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    df["DayOfWeek"] = df["Dates"].dt.day_name()
    df["Hour"] = df["Dates"].dt.hour

    df = df[["Dates", "DayOfWeek", "Hour", "PdDistrict", "X", "Y", "Category"]].copy()
    df = df.dropna()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Ruta relativa al repo del CSV de entrenamiento",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2015-04-01",
        help="Fecha de corte para split temporal",
    )
    parser.add_argument(
        "--out-pkl",
        type=str,
        default="models/best_model_xgb.pkl",
        help="Ruta relativa al repo para guardar el pipeline pickle",
    )
    parser.add_argument(
        "--out-classes",
        type=str,
        default="models/best_model_xgb_classes.json",
        help="Ruta relativa al repo para guardar las clases",
    )
    parser.add_argument(
        "--out-mlflow",
        type=str,
        default="models/best_model_xgb",
        help="Ruta relativa al repo para guardar el modelo MLflow",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="Tracking URI de MLflow",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sf-crime-xgboost-multiclass",
        help="Nombre del experimento MLflow",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    data_path = repo_root / args.data
    out_pkl = repo_root / args.out_pkl
    out_classes = repo_root / args.out_classes
    out_mlflow = repo_root / args.out_mlflow

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_classes.parent.mkdir(parents=True, exist_ok=True)
    out_mlflow.parent.mkdir(parents=True, exist_ok=True)

    df = load_data_with_dates(data_path)

    train_df = df[df["Dates"] < args.cutoff].copy()
    test_df = df[df["Dates"] >= args.cutoff].copy()

    feature_cols = ["DayOfWeek", "Hour", "PdDistrict", "X", "Y"]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_raw = train_df["Category"]
    y_test_raw = test_df["Category"]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    pipeline = build_pipeline(random_seed=42)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="xgboost_multiclass_pipeline"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model_type", "xgboost_multiclass_pipeline")
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")

    # Guardar pipeline completo
    joblib.dump(pipeline, out_pkl)

    # Guardar nombres reales de clases
    with open(out_classes, "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

    # Guardar también en formato MLflow local
    if out_mlflow.exists():
        # limpiar contenido previo si existe
        import shutil
        shutil.rmtree(out_mlflow)

    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=str(out_mlflow),
    )

    print(f"\nPipeline guardado en: {out_pkl}")
    print(f"Clases guardadas en: {out_classes}")
    print(f"Modelo MLflow guardado en: {out_mlflow}")


if __name__ == "__main__":
    main()