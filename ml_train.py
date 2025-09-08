import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

from ml_features import build_monthly_df, get_db


FEATURE_COLS = [
    # time encodings
    "month", "year", "month_sin", "month_cos",
    # lags/rollings
    "y_event_count_lag_1", "y_event_count_lag_2", "y_event_count_lag_3",
    "y_event_count_roll3_mean", "y_event_count_roll6_mean",
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


def train_models(scope: str = "industry", out_dir: str = "models") -> Dict[int, PoissonRegressor]:
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
    for horizon in [1, 2, 3]:
        target = f"y_t+{horizon}"
        dff = df.dropna(subset=[target]).copy()
        if dff.empty:
            continue
        y = dff[target].values
        X = dff[FEATURE_COLS].values

        # Simple last-split validation for brevity
        yms = sorted(dff["ym"].unique())
        split = max(12, len(yms) - (horizon + 6))
        train_idx = dff["ym"].isin(yms[:split]).values
        val_idx = dff["ym"].isin(yms[split:]).values

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = PoissonRegressor(alpha=0.5, max_iter=500)
        if X_train.shape[0] > 0 and X_val.shape[0] > 0:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            metrics[horizon] = mae
        else:
            # Fallback: fit on all available rows if we cannot split
            model.fit(X, y)
            metrics[horizon] = None
        models[horizon] = model
        joblib.dump(model, os.path.join(out_dir, f"poisson_h{horizon}.joblib"))

    # Save metadata
    joblib.dump({"scope": scope, "features": FEATURE_COLS, "metrics_mae": metrics}, os.path.join(out_dir, "meta.joblib"))
    return models


if __name__ == "__main__":
    train_models(scope="industry", out_dir="models_industry")

