import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from ml_features import build_monthly_df, get_db


FEATURE_COLS = [
    # time encodings
    "month", "year", "month_sin", "month_cos",
    # lags/rollings
    "y_event_count_lag_1", "y_event_count_lag_2", "y_event_count_lag_3",
    "y_event_count_roll3_mean", "y_event_count_roll6_mean",
    "y_event_count_same_month_ly",
    "y_host_demand_lag_1", "y_host_demand_lag_2", "y_host_demand_lag_3",
    "y_host_demand_roll3_mean", "y_host_demand_roll6_mean",
    # supply as covariate
    "y_host_avail",
]


def time_backtest_indices(df: pd.DataFrame, min_train: int = 12, horizon: int = 3):
    yms = sorted(df["ym"].unique())
    for cut in range(min_train, len(yms) - horizon):
        train_months = set(yms[:cut])
        val_months = set(yms[cut:cut + horizon])
        train_idx = df["ym"].isin(train_months)
        val_idx = df["ym"].isin(val_months)
        yield train_idx, val_idx


def train_models(scope: str = "industry", out_dir: str = "models") -> Dict[int, Pipeline]:
    os.makedirs(out_dir, exist_ok=True)
    db = get_db()
    df = build_monthly_df(db=db, scope=scope)
    # Fill missing lag features conservatively and keep rows; only require horizon target per model
    for col in [
        "y_event_count_lag_1", "y_event_count_lag_2", "y_event_count_lag_3",
        "y_event_count_roll3_mean", "y_event_count_roll6_mean",
        "y_host_demand_lag_1", "y_host_demand_lag_2", "y_host_demand_lag_3",
        "y_host_demand_roll3_mean", "y_host_demand_roll6_mean",
        "y_host_avail",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    models = {}
    metrics = {}
    calib = {}
    for horizon in [1, 2, 3]:
        target = f"y_t+{horizon}"
        dff = df.dropna(subset=[target]).copy()
        if dff.empty:
            continue
        y = dff[target].values
        X = dff[FEATURE_COLS].values

        # Use recent window (last 12 months) for training/validation
        yms = sorted(dff["ym"].unique())
        recent_yms = yms[-12:] if len(yms) > 12 else yms
        dff = dff[dff["ym"].isin(recent_yms)]
        y = dff[target].values
        X = dff[FEATURE_COLS].values

        # last split: train on first 9, validate on last 3 months
        split = max(3, len(recent_yms) - 3)
        train_idx = dff["ym"].isin(recent_yms[:split]).values
        val_idx = dff["ym"].isin(recent_yms[split:]).values

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Tune alpha using simple rolling-origin CV across a few candidates
        alphas = [0.5, 1.0, 5.0, 10.0, 20.0]
        best_mae = float("inf")
        best_pipe = None
        best_pred = None
        for alpha in alphas:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", PoissonRegressor(alpha=alpha, max_iter=1000)),
            ])
            if X_train.shape[0] > 0 and X_val.shape[0] > 0:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
            else:
                pipe.fit(X, y)
                y_pred = pipe.predict(X)
                mae = mean_absolute_error(y, y_pred)
            if mae < best_mae:
                best_mae = mae
                best_pipe = pipe
                best_pred = y_pred
        metrics[horizon] = best_mae
        models[horizon] = best_pipe
        # Calibration factor on validation window: clip to avoid extreme scaling
        if best_pred is not None and best_pred.sum() > 0:
            factor = float(y_val.sum() / best_pred.sum()) if X_val.shape[0] > 0 else 1.0
            factor = max(0.5, min(1.5, factor))
        else:
            factor = 1.0
        calib[horizon] = factor
        joblib.dump(best_pipe, os.path.join(out_dir, f"poisson_h{horizon}.joblib"))

    # Save metadata
    joblib.dump({"scope": scope, "features": FEATURE_COLS, "metrics_mae": metrics, "calibration": calib}, os.path.join(out_dir, "meta.joblib"))
    return models


if __name__ == "__main__":
    train_models(scope="industry", out_dir="models_industry")

