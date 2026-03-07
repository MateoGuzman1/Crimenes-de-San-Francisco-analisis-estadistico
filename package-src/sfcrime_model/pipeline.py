import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sfcrime_model.config.core import config
from sfcrime_model.processing.feature_engineering import (
    CAT_COLS,
    NUM_COLS,
    FeatureEngineerTransformer,
)

col_transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ],
    remainder="drop",
)

sfcrime_pipe = Pipeline([
    ("feature_eng", FeatureEngineerTransformer()),
    ("preprocess",  col_transformer),
    ("clf", xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=config.ml_config.n_estimators,
        max_depth=config.ml_config.max_depth,
        learning_rate=config.ml_config.learning_rate,
        subsample=config.ml_config.subsample,
        colsample_bytree=config.ml_config.colsample_bytree,
        random_state=config.ml_config.random_state,
        n_jobs=-1,
    )),
])
