#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

RANDOM_STATE = 42

# =========================================
# FEATURE ENGINEERING
# =========================================
def feature_engineer(df):
    out = df.copy()

    dt = pd.to_datetime(out['Dates'])

    out['Year'] = dt.dt.year.astype(np.int16)
    out['Month'] = dt.dt.month.astype(np.int8)
    out['Day'] = dt.dt.day.astype(np.int8)
    out['Hour'] = dt.dt.hour.astype(np.int8)
    out['Minute'] = dt.dt.minute.astype(np.int8)
    out['IsWeekend'] = (dt.dt.dayofweek >= 5).astype(np.int8)

    out['sin_hour'] = np.sin(2*np.pi*out['Hour']/24.0).astype(np.float32)
    out['cos_hour'] = np.cos(2*np.pi*out['Hour']/24.0).astype(np.float32)

    dow_map = {
        'Monday':0,'Tuesday':1,'Wednesday':2,
        'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6
    }

    out['DOW_idx'] = out['DayOfWeek'].map(dow_map).fillna(0).astype(np.int8)
    out['sin_dow'] = np.sin(2*np.pi*out['DOW_idx']/7.0).astype(np.float32)
    out['cos_dow'] = np.cos(2*np.pi*out['DOW_idx']/7.0).astype(np.float32)

    addr = out['Address'].fillna("")
    out['IsIntersection'] = addr.str.contains("/", regex=False).astype(np.int8)
    out['HasBlockWord'] = addr.str.contains("Block", case=False, regex=False).astype(np.int8)

    dx = (out['X'] + 122.4194) * (111.0*np.cos(np.deg2rad(37.77)))
    dy = (out['Y'] - 37.7749) * 111.0
    out['dist_km_center'] = np.sqrt(dx*dx + dy*dy).astype(np.float32)

    out['X_round2'] = out['X'].round(2).astype(np.float32)
    out['Y_round2'] = out['Y'].round(2).astype(np.float32)

    return out


# =========================================
# MLFLOW CONFIG
# =========================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.set_experiment("SF-Crimes-MultiModel-FE")
experiment_id = experiment.experiment_id


# =========================================
# LOAD DATA
# =========================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_fe = feature_engineer(train_df)
test_fe = feature_engineer(test_df)


# =========================================
# TARGET ENCODING
# =========================================
le = LabelEncoder()
y = le.fit_transform(train_fe['Category'])


# =========================================
# FEATURES
# =========================================
categorical_cols = ['DayOfWeek','PdDistrict','IsWeekend','IsIntersection','HasBlockWord']

numeric_cols = [
    'Year','Month','Day','Hour','Minute',
    'sin_hour','cos_hour','DOW_idx','sin_dow','cos_dow',
    'X','Y','dist_km_center','X_round2','Y_round2'
]

X_df = train_fe[categorical_cols + numeric_cols]


# =========================================
# SPLIT
# =========================================
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_df, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

X_test_df = test_fe[categorical_cols + numeric_cols]


# =========================================
# PREPROCESSING
# =========================================
ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

ohe.fit(X_train_df[categorical_cols])
scaler.fit(X_train_df[numeric_cols])

def transform_data(df):
    X_cat = ohe.transform(df[categorical_cols]).toarray()
    X_num = scaler.transform(df[numeric_cols])
    return np.hstack([X_cat, X_num]).astype(np.float32)

X_train = transform_data(X_train_df)
X_val = transform_data(X_val_df)
X_test = transform_data(X_test_df)


# =========================================
# MODELS
# =========================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10),
    "DecisionTree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier()
}


# =========================================
# TRAINING + TRACKING
# =========================================
best_score = 0
best_model_name = None
best_model_object = None

for model_name, model in models.items():

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=model_name
    ):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_prob, multi_class="ovr")
        else:
            auc = 0

        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        # Log model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Best model tracking
        if accuracy > best_score:
            best_score = accuracy
            best_model_name = model_name
            best_model_object = model

        print(f"{model_name} → accuracy {accuracy:.4f}")


print(f"Best model = {best_model_name} | Score = {best_score}")


# =========================================
# FINAL PREDICTION (BEST MODEL)
# =========================================
best_model_object.fit(X_train, y_train)

test_predictions = best_model_object.predict(X_test)
test_predictions = le.inverse_transform(test_predictions)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "PredictedCategory": test_predictions
})

submission.to_csv("submission_out_of_sample.csv", index=False)

print("Predicción generada.")
