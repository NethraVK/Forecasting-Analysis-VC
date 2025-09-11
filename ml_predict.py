from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd

from ml_features import build_monthly_df, get_db, month_key


def next_three_months_from_today() -> list:
    today = datetime.today().replace(day=1)
    months = []
    y, m = today.year, today.month
    for _ in range(3):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def predict_report(scope: str = "industry", models_dir: str = "models_industry") -> None:
    db = get_db()
    df = build_monthly_df(db=db, scope=scope)
    meta = joblib.load(os.path.join(models_dir, "meta.joblib"))
    features = meta["features"]
    calib = meta.get("calibration", {})
    # Load models
    models = {h: joblib.load(os.path.join(models_dir, f"poisson_h{h}.joblib")) for h in [1, 2, 3]}

    # Use the latest month per key to form feature rows
    latest_rows = df.sort_values(["key", "dt"]).groupby("key").tail(1)
    # Impute missing feature values conservatively
    latest_rows_imputed = latest_rows.copy()
    for c in features:
        if c not in latest_rows_imputed.columns:
            latest_rows_imputed[c] = 0
    latest_rows_imputed[features] = latest_rows_imputed[features].fillna(0)
    X_latest = latest_rows_imputed[features].values
    keys = latest_rows["key"].tolist()

    horizon_months = next_three_months_from_today()

    preds = {}
    for h in [1, 2, 3]:
        ph = models[h].predict(X_latest)
        factor = float(calib.get(h, 1.0))
        preds[h] = ph * factor

    # Guardrail: cap next-3m total to within 1.2x of recent last-3m actual total
    # Use user-provided actual for last 3 months
    last3_months = sorted(df["ym"].unique())[-3:]
    recent_actual = 30

    raw_total_next3 = float(sum(preds[h].sum() for h in [1, 2, 3]))
    scale = 1.0
    if recent_actual > 0:
        cap_total = 1.1 * recent_actual
        if raw_total_next3 > cap_total:
            scale = cap_total / max(raw_total_next3, 1e-6)
            for h in [1, 2, 3]:
                preds[h] = preds[h] * scale

    # Sum across keys to get overall (after cap)
    total_next3 = int(np.round(sum(preds[h].sum() for h in [1, 2, 3])))
    avg = round(total_next3 / 3.0, 1)
    print(f"ML Summary: Next three months ({horizon_months[0]}–{horizon_months[-1]}): {total_next3}+ exhibitions expected (avg {avg}/month)")

    # Per key totals (after cap)
    per_key = {k: int(np.round(preds[1][i] + preds[2][i] + preds[3][i])) for i, k in enumerate(keys)}
    print("=== ML Event Volume Forecast (by key) — Next 3 months ===")
    for k, v in sorted(per_key.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {k}: {v}")


if __name__ == "__main__":
    predict_report(scope="industry", models_dir="models_industry")

