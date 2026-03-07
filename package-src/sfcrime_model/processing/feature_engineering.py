from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CAT_COLS = ['DayOfWeek', 'PdDistrict', 'IsWeekend', 'IsIntersection', 'HasBlockWord']
NUM_COLS = [
    'Year', 'Month', 'Day', 'Hour', 'Minute',
    'sin_hour', 'cos_hour', 'DOW_idx', 'sin_dow', 'cos_dow',
    'X', 'Y', 'dist_km_center', 'X_round2', 'Y_round2',
]


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering sobre el DataFrame crudo del SFPD.

    Columnas requeridas: Dates, DayOfWeek, Address, X, Y
    Produce 20 features (5 categoricas + 15 numericas) usadas por el modelo.
    """
    out = df.copy()

    dt = pd.to_datetime(out['Dates'])
    out['Year']      = dt.dt.year.astype(np.int16)
    out['Month']     = dt.dt.month.astype(np.int8)
    out['Day']       = dt.dt.day.astype(np.int8)
    out['Hour']      = dt.dt.hour.astype(np.int8)
    out['Minute']    = dt.dt.minute.astype(np.int8)
    out['IsWeekend'] = (dt.dt.dayofweek >= 5).astype(np.int8)

    out['sin_hour'] = np.sin(2 * np.pi * out['Hour'] / 24.0).astype(np.float32)
    out['cos_hour'] = np.cos(2 * np.pi * out['Hour'] / 24.0).astype(np.float32)

    dow_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6,
    }
    out['DOW_idx'] = out['DayOfWeek'].map(dow_map).fillna(0).astype(np.int8)
    out['sin_dow'] = np.sin(2 * np.pi * out['DOW_idx'] / 7.0).astype(np.float32)
    out['cos_dow'] = np.cos(2 * np.pi * out['DOW_idx'] / 7.0).astype(np.float32)

    addr = out['Address'].fillna("")
    out['IsIntersection'] = addr.str.contains("/", regex=False).astype(np.int8)
    out['HasBlockWord']   = addr.str.contains("Block", case=False, regex=False).astype(np.int8)

    dx = (out['X'] + 122.4194) * (111.0 * np.cos(np.deg2rad(37.77)))
    dy = (out['Y'] - 37.7749) * 111.0
    out['dist_km_center'] = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    out['X_round2'] = out['X'].round(2).astype(np.float32)
    out['Y_round2'] = out['Y'].round(2).astype(np.float32)

    return out


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Transformer sklearn stateless que aplica feature_engineer()."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_engineer(X)
