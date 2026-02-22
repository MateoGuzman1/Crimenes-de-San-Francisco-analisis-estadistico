from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def _ravel(X):
    return X.ravel()


TARGET_COL = "Category"

# Columns from your dataset
TEXT_COL = "Address"
CAT_COLS = ["DayOfWeek", "PdDistrict"]          # keep simple, high-signal
NUM_COLS = ["X", "Y"]                           # coordinates are strong signal
DROP_COLS = ["Descript", "Resolution", "Dates"] # prevent leakage / noise


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def make_preprocessor(tfidf_max_features: int = 10_000) -> ColumnTransformer:
    # Text pipeline
    text_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", FunctionTransformer(_ravel)),
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=tfidf_max_features,
            )),
        ]
    )

    # Categorical pipeline
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Numeric pipeline
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, [TEXT_COL]),
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre


def build_model(model_key: str, seed: int = 42) -> Any:
    if model_key == "logreg_tfidf":
        return LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            solver="saga",
        )

    if model_key == "linear_svc":
        return LinearSVC(max_iter=200, random_state=seed)

    if model_key == "mnb":
        return MultinomialNB()

    raise ValueError(f"Unknown model_key: {model_key}")


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    model_key: str,
    seed: int = 42,
    tfidf_max_features: int = 10_000,
    val_size: float = 0.2,
) -> Tuple[Pipeline, Dict[str, float]]:
    # Drop leakage/noise columns if present
    train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=seed,
        stratify=y,
    )

    pre = make_preprocessor(tfidf_max_features=tfidf_max_features)
    model = build_model(model_key=model_key, seed=seed)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    metrics: Dict[str, float] = {}
    preds_val = pipe.predict(X_val)
    metrics["accuracy"] = float(accuracy_score(y_val, preds_val))
    metrics["f1_macro"] = float(f1_score(y_val, preds_val, average="macro"))
    return pipe, metrics
