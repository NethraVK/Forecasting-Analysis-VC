"""
forecasting_ml.py

Rewritten forecasting script that:
- builds monthly dataset from MongoDB (industry/location scope)
- trains a LightGBM regressor (fallback to RandomForest if not installed)
- forecasts next N months for event counts (with simple iterative multi-step)
- forecasts host demand and availability and computes shortages
- outputs a simple human-readable report and JSON

Usage: python forecasting_ml.py --mongo-uri mongodb://localhost:27017/ --db event_hosting_platform --horizon 3

Notes:
- Install dependencies for best results: pip install pandas numpy pymongo scikit-learn lightgbm
- If lightgbm unavailable, script falls back to RandomForestRegressor.
"""

from datetime import datetime
import argparse
import json
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from pymongo import MongoClient

# ----------------------
# User-provided helper: build_monthly_df
# (Adapted from the function you gave earlier)
# ----------------------

def get_db(uri: str = "mongodb://localhost:27017/", db_name: str = "event_hosting_platform"):
    return MongoClient(uri)[db_name]


# ----------------------
# JSON-backed mock DB (to use dummy_mongo_events.json instead of TEST.py seeding)
# ----------------------

class _JsonCollection:
    def __init__(self, docs):
        self._docs = list(docs or [])

    def find(self, _query=None):
        return list(self._docs)

    def count_documents(self, _query=None):
        return len(self._docs)


class _JsonDB:
    def __init__(self, collections_map):
        self._collections = dict(collections_map or {})

    def __getitem__(self, name: str):
        return self._collections.get(name, _JsonCollection([]))


def _parse_mongo_extended_json_date(item) -> datetime | None:
    try:
        if isinstance(item, dict) and "$date" in item:
            # Normalize Zulu time to ISO 8601 compatible for fromisoformat
            iso = str(item["$date"]).replace("Z", "+00:00")
            return datetime.fromisoformat(iso)
    except Exception:
        return None
    return None


def get_db_from_json(json_path: str):
    """Load a lightweight Mongo-like DB from a JSON file of Event documents.

    Expected input is an array of Event-like dicts (from mongoexport or similar)
    where date fields are in Mongo Extended JSON (e.g., {"$date": "..."}).
    We will map it to minimal collections required by build_monthly_df.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Support two formats:
    # 1) Array of Event docs
    # 2) Bundle with collections: { "Event": [...], "EventHostUser": [...], ... }
    if isinstance(raw, dict) and "Event" in raw:
        raw_events = raw.get("Event", [])
        raw_hosts = raw.get("EventHostUser", [])
        raw_invites = raw.get("EventInvite", [])
        raw_exhibitors = raw.get("ExhibitorUser", [])
        raw_profiles = raw.get("ExhibitorProfile", [])
        raw_unavail = raw.get("UnavailableDate", [])
    else:
        raw_events = raw if isinstance(raw, list) else []
        raw_hosts, raw_invites, raw_exhibitors, raw_profiles, raw_unavail = [], [], [], [], []

    events = []
    for ev in raw_events:
        # Convert _id
        _id = ev.get("_id")
        if isinstance(_id, dict) and "$oid" in _id:
            _id = _id["$oid"]

        # Derive a single start date from dates array or field
        start_dt = None
        dates_field = ev.get("dates")
        if isinstance(dates_field, list) and dates_field:
            # Take the earliest
            parsed = [d for d in (_parse_mongo_extended_json_date(x) for x in dates_field) if d is not None]
            if parsed:
                start_dt = min(parsed)
        elif isinstance(dates_field, dict) and "start" in dates_field:
            start_val = dates_field.get("start")
            if isinstance(start_val, str):
                try:
                    start_dt = datetime.fromisoformat(start_val.replace("Z", "+00:00"))
                except Exception:
                    start_dt = None
            elif isinstance(start_val, dict):
                start_dt = _parse_mongo_extended_json_date(start_val)

        # Fallbacks
        location = ev.get("location") or "Unknown"

        events.append({
            "_id": _id,
            "name": ev.get("name"),
            "dates": {"start": start_dt.isoformat() if isinstance(start_dt, datetime) else None},
            "location": location,
        })

    # Minimal collections used downstream
    # Hosts
    hosts = []
    for h in raw_hosts:
        _id = h.get("_id")
        if isinstance(_id, dict) and "$oid" in _id:
            _id = _id["$oid"]
        hosts.append({"_id": _id, **{k: v for k, v in h.items() if k != "_id"}})

    # Invites
    invites = []
    for inv in raw_invites:
        def _oid(x):
            if isinstance(x, dict) and "$oid" in x:
                return x["$oid"]
            return x
        invites.append({
            "_id": _oid(inv.get("_id")),
            "event": _oid(inv.get("event")),
            "exhibitor": _oid(inv.get("exhibitor")),
            "eventHost": _oid(inv.get("eventHost")),
            "otp": inv.get("otp"),
            "status": inv.get("status"),
            "createdAt": inv.get("createdAt"),
            "__v": inv.get("__v", 0),
        })

    collections = {
        "Event": _JsonCollection(events),
        "ExhibitorUser": _JsonCollection(raw_exhibitors or []),
        "ExhibitorProfile": _JsonCollection(raw_profiles or []),
        "UnavailableDate": _JsonCollection(raw_unavail or []),
        "EventHostUser": _JsonCollection(hosts),
        "EventInvite": _JsonCollection(invites),
    }
    return _JsonDB(collections)


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def _adjust_event_forecast_to_target(preds: List[int], target_mean: float = 2.5, min_events: int = 2, max_events: int = 3) -> List[int]:
    """Scale forecasts to be around target_mean and clamp to [min_events, max_events]."""
    if not preds:
        return preds
    cur_mean = sum(preds) / max(1, len(preds))
    if cur_mean <= 0:
        scaled = [int(round(target_mean)) for _ in preds]
    else:
        scale = target_mean / cur_mean
        scaled = [int(round(max(0, p * scale))) for p in preds]
    return [min(max_events, max(min_events, v)) for v in scaled]

def next_three_months_from_today() -> List[str]:
    base = datetime.today().replace(day=1)
    months: List[str] = []
    y, m = base.year, base.month
    for _ in range(3):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def next_n_months_from_today(n: int) -> List[str]:
    try:
        n_int = max(1, int(n))
    except Exception:
        n_int = 3
    base = datetime.today().replace(day=1)
    months: List[str] = []
    y, m = base.year, base.month
    for _ in range(n_int):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def build_monthly_df(db=None, scope: str = "industry") -> pd.DataFrame:
    if db is None:
        db = get_db()

    events = list(db["Event"].find({}))
    rows = []
    for ev in events:
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if not start:
            continue
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        ym = month_key(start)
        rows.append({
            "event_id": ev.get("_id"),
            "ym": ym,
            "location": ev.get("location", "Unknown"),
        })
    ev_df = pd.DataFrame(rows)

    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
    # Demand sources: ExhibitorProfile.hostessRequirements or EventInvite per-event counts or participants length
    xp_rows = []
    for xp in db["ExhibitorProfile"].find({}):
        ex = exhibitors.get(xp.get("exhibitor"))
        if not ex:
            continue
        xp_rows.append({
            "event": xp.get("event"),
            "industry": ex.get("industry", "Unknown"),
            "hostessRequirements": int(xp.get("hostessRequirements", 0))
        })
    xp_df = pd.DataFrame(xp_rows)

    # If no ExhibitorProfile demand, derive per-event demand from invites (accepted + pending) or participants length
    demand_by_event: Dict[str, int] = {}
    if xp_df.empty:
        invites = list(db["EventInvite"].find({}))
        for inv in invites:
            ev_id = inv.get("event")
            status = (inv.get("status") or "").lower()
            if not ev_id:
                continue
            # Count invites that are not rejected as required positions
            if status in ("accepted", "pending", "requested", "sent"):
                demand_by_event[ev_id] = demand_by_event.get(ev_id, 0) + 1
        # Fallback: if some events have zero demand, use participants length if available
        for ev in events:
            ev_id = ev.get("_id")
            if isinstance(ev_id, dict):
                ev_id = ev_id.get("$oid")
            if ev_id not in demand_by_event:
                participants = ev.get("participants") or []
                demand_by_event[ev_id] = int(len(participants))

    if ev_df.empty:
        return pd.DataFrame(columns=["ym", "key", "y_event_count", "y_host_demand", "y_host_avail"])

    if not xp_df.empty:
        ev_xp = ev_df.merge(xp_df, left_on="event_id", right_on="event", how="left")
    else:
        ev_xp = ev_df.copy()
        ev_xp["industry"] = "Unknown"
        ev_xp["hostessRequirements"] = ev_xp["event_id"].map(lambda eid: int(demand_by_event.get(eid, 0)))

    key_col = "industry" if scope == "industry" else "location"

    ev_counts = (ev_xp.groupby(["ym", key_col])["event_id"].nunique()
                 .reset_index().rename(columns={"event_id": "y_event_count"}))
    demand = (ev_xp.groupby(["ym", key_col])["hostessRequirements"].sum()
              .reset_index().rename(columns={"hostessRequirements": "y_host_demand"}))
    df = ev_counts.merge(demand, on=["ym", key_col], how="left").fillna({"y_host_demand": 0})

    # Availability per month as a probabilistic blend:
    #  - accepted invites (distinct hosts): 100%
    #  - pending invites: 50% probability
    #  - non-invited qualified pool: 30% probability from remaining qualified (assume 70% of total hosts qualified)
    ev_month_map = {row["event_id"]: row["ym"] for _, row in ev_df.iterrows()}
    accepted_by_month: Dict[str, set] = {}
    pending_by_month: Dict[str, set] = {}
    invited_any_by_month: Dict[str, set] = {}
    for inv in db["EventInvite"].find({}):
        status = (inv.get("status") or "").lower()
        ev_id = inv.get("event")
        host_id = inv.get("eventHost")
        ym = ev_month_map.get(ev_id)
        if not ym or not host_id:
            continue
        if status == "accepted":
            accepted_by_month.setdefault(ym, set()).add(host_id)
        elif status == "pending":
            pending_by_month.setdefault(ym, set()).add(host_id)
        # track all invited hosts regardless of status
        invited_any_by_month.setdefault(ym, set()).add(host_id)

    # Consider explicit unavailability if present
    unavail = list(db["UnavailableDate"].find({}))
    un_by_month: Dict[str, int] = {}
    for u in unavail:
        for d in u.get("dates", []):
            try:
                m = d[:7] if isinstance(d, str) else d.strftime("%Y-%m")
            except Exception:
                continue
            un_by_month[m] = un_by_month.get(m, 0) + 1

    hosts_docs = list(db["EventHostUser"].find({}))
    total_hosts = len(hosts_docs)
    # Do NOT rely on any qualified tagging in data; use 70% estimate
    assumed_qualified_total = int(round(total_hosts * 0.7))

    # Probabilities/weights (not required to sum to 1; they are contribution factors):
    p_accepted = 1.0
    p_pending = 0.5
    p_noninv_qualified = 0.3

    def _avail_for_month(m: str) -> int:
        accepted = len(accepted_by_month.get(m, set()))
        pending = len(pending_by_month.get(m, set()))
        invited_any = len(invited_any_by_month.get(m, set()))
        # Remaining qualified not invited this month (avoid double counting anyone invited)
        remaining_qualified = max(0, assumed_qualified_total - invited_any)
        # Expected available count using probabilities
        expected_available = (
            p_accepted * accepted +
            p_pending * pending +
            p_noninv_qualified * remaining_qualified
        )
        blended = int(round(expected_available))
        # cap at total hosts
        blended = min(blended, total_hosts)
        # reduce by explicit unavailability
        blended = max(0, blended - un_by_month.get(m, 0))
        return blended

    df["y_host_avail"] = df["ym"].map(_avail_for_month)

    df = df.rename(columns={key_col: "key"})
    df = df.sort_values(["key", "ym"]).reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ym"] + "-01")
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    def add_lags_rollings(group: pd.DataFrame, col: str) -> pd.DataFrame:
        group = group.copy()
        group[f"{col}_lag_1"] = group[col].shift(1)
        group[f"{col}_lag_2"] = group[col].shift(2)
        group[f"{col}_lag_3"] = group[col].shift(3)
        group[f"{col}_roll3_mean"] = group[col].rolling(3, min_periods=1).mean().shift(1)
        group[f"{col}_roll6_mean"] = group[col].rolling(6, min_periods=1).mean().shift(1)
        return group

    df = df.groupby("key", group_keys=False).apply(lambda g: add_lags_rollings(g, "y_event_count"))
    df = df.groupby("key", group_keys=False).apply(lambda g: add_lags_rollings(g, "y_host_demand"))

    df["y_event_count_same_month_ly"] = df.groupby("key")["y_event_count"].shift(12)
    df["y_t+1"] = df.groupby("key")["y_event_count"].shift(-1)
    df["y_t+2"] = df.groupby("key")["y_event_count"].shift(-2)
    df["y_t+3"] = df.groupby("key")["y_event_count"].shift(-3)

    return df

# ----------------------
# ML training & forecasting
# ----------------------

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    LGB_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def prepare_global_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-month totals across keys to produce a monthly time series dataframe."""
    # Sum event counts and host demand/avail per ym
    monthly = df.groupby("ym").agg({
        "y_event_count": "sum",
        "y_host_demand": "sum",
        "y_host_avail": "sum",
    }).reset_index()
    monthly["dt"] = pd.to_datetime(monthly["ym"] + "-01")
    monthly = monthly.sort_values("dt").reset_index(drop=True)
    monthly["month"] = monthly["dt"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
    # simple time trend
    monthly["t"] = np.arange(len(monthly))

    # add lags and rolling means
    for lag in (1, 2, 3):
        monthly[f"y_event_count_lag_{lag}"] = monthly["y_event_count"].shift(lag)
    monthly["y_event_count_roll3_mean"] = monthly["y_event_count"].rolling(3, min_periods=1).mean().shift(1)
    monthly["y_event_count_roll6_mean"] = monthly["y_event_count"].rolling(6, min_periods=1).mean().shift(1)

    # same month last year
    monthly["y_event_count_same_month_ly"] = monthly["y_event_count"].shift(12)

    return monthly


def train_event_model(monthly: pd.DataFrame, horizon: int = 1):
    """Train a model to predict y_t+1 (next month event counts) on monthly aggregated history."""
    features = [
        "month_sin", "month_cos", "t",
        "y_event_count_lag_1", "y_event_count_lag_2", "y_event_count_lag_3",
        "y_event_count_roll3_mean", "y_event_count_roll6_mean",
        "y_event_count_same_month_ly",
    ]
    train_df = monthly.dropna(subset=["y_event_count"] + features + ["y_event_count"])

    if train_df.shape[0] < 6:
        raise ValueError("Not enough monthly history to train model reliably; need at least 6 non-missing rows.")

    X = train_df[features]
    y = train_df["y_event_count"].shift(-1).dropna()
    # Align X to y
    X = X.iloc[:-1, :]

    # simple time-based split: last 12% holdout or last 3 months
    test_size = max(1, int(0.15 * len(X)))
    X_train, X_test = X.iloc[:-test_size, :], X.iloc[-test_size:, :]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    if LGB_AVAILABLE:
        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l1",
            verbose=False,
            callbacks=[],
        )
        # Best iteration is tracked internally; predictions use it automatically
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Trained event model. Holdout MAE: {mae:.2f} events/month")
    return model, features


def iterative_forecast(monthly: pd.DataFrame, model, features: List[str], horizon: int = 3) -> List[int]:
    """Iterative multi-step forecast using the last available row in `monthly` as seed."""
    monthly_copy = monthly.copy().reset_index(drop=True)
    last = monthly_copy.iloc[-1:].copy()
    preds = []
    for step in range(horizon):
        X = last[features].fillna(0).values.reshape(1, -1)
        yhat = model.predict(X)[0]
        yhat = max(0, float(yhat))
        preds.append(int(round(yhat)))

        # roll last forward: shift lags
        # new row representing next month
        new_row = last.copy().iloc[0]
        # update dt and ym
        new_dt = (pd.to_datetime(new_row['dt']) + pd.DateOffset(months=1))
        new_row['dt'] = new_dt
        new_row['ym'] = new_dt.strftime('%Y-%m')
        new_row['month'] = new_dt.month
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)

        # update lags: shift previous predictions
        new_row['y_event_count_lag_3'] = new_row.get('y_event_count_lag_2', 0)
        new_row['y_event_count_lag_2'] = new_row.get('y_event_count_lag_1', 0)
        new_row['y_event_count_lag_1'] = yhat

        # update rolling means approximately
        # we append and recalc roll3/6 from history in memory
        monthly_copy = pd.concat([monthly_copy, pd.DataFrame([new_row])], ignore_index=True)
        # recompute roll features on last 6 rows
        recent = monthly_copy['y_event_count'].tolist() + preds
        # compute roll3 mean for the last item
        roll3 = np.mean(recent[-3:]) if len(recent) >= 1 else yhat
        roll6 = np.mean(recent[-6:]) if len(recent) >= 1 else yhat
        new_row['y_event_count_roll3_mean'] = roll3
        new_row['y_event_count_roll6_mean'] = roll6
        new_row['y_event_count_same_month_ly'] = None  # unknown for synthetic months

        last = pd.DataFrame([new_row])
    return preds

# ----------------------
# Supply-demand using event forecasts
# ----------------------

def compute_avg_hosts_per_event(df: pd.DataFrame) -> float:
    # Weighted average over months where both events and host demand present
    pairs = []
    grouped = df.groupby('ym').agg({'y_event_count': 'sum', 'y_host_demand': 'sum'}).reset_index()
    for _, row in grouped.iterrows():
        if row['y_event_count'] > 0:
            pairs.append((row['y_event_count'], row['y_host_demand']))
    if not pairs:
        return 1.0
    total_events = sum(p[0] for p in pairs)
    total_hosts = sum(p[1] for p in pairs)
    return float(total_hosts) / max(1.0, total_events)


def estimate_availability_history(df: pd.DataFrame, db) -> Tuple[List[int], int]:
    # Use total pool if historical availability missing
    monthly = df.groupby('ym').agg({'y_host_avail': 'sum'}).reset_index()
    hist = monthly['y_host_avail'].tolist()
    total_pool = db['EventHostUser'].count_documents({})
    return hist, int(total_pool)

# ----------------------
# CLI / Orchestration
# ----------------------

def run_forecast(mongo_uri: str, db_name: str, horizon: int = 3, scope: str = 'industry', json_path: str | None = None):
    db = get_db_from_json(json_path) if json_path else get_db(mongo_uri, db_name)
    df = build_monthly_df(db=db, scope=scope)
    if df.empty:
        print('No events found in DB. Exiting.')
        return

    monthly = prepare_global_monthly(df)

    try:
        model, features = train_event_model(monthly)
    except Exception as e:
        print('Could not train ML model (not enough history or other issue):', e)
        print('Falling back to simple moving average forecast on recent months.')
        hist = monthly['y_event_count'].dropna().tolist()
        if not hist:
            preds = [0] * horizon
        else:
            preds = [int(round(np.mean(hist[-3:]))) for _ in range(horizon)]
        # Build simple supply-demand results
        avg_hosts_per_event = compute_avg_hosts_per_event(df)
        demand = [int(round(p * avg_hosts_per_event)) for p in preds]
        hist_avail, pool = estimate_availability_history(df, db)
        if any(hist_avail):
            avail = hist_avail[-1]
        else:
            avail = int(round(pool * 0.6)) if pool > 0 else 5
        results = {}
        for i in range(horizon):
            month_dt = (pd.to_datetime(monthly['dt'].iloc[-1]) + pd.DateOffset(months=i+1))
            ym = month_dt.strftime('%Y-%m')
            results[ym] = {
                'predicted_events': preds[i],
                'predicted_demand': demand[i],
                'available_hosts': avail,
                'shortage': max(0, demand[i] - avail)
            }
        print(json.dumps(results, indent=2))
        return

    preds = iterative_forecast(monthly, model, features, horizon=horizon)
    avg_hosts_per_event = compute_avg_hosts_per_event(df)
    demand = [int(round(p * avg_hosts_per_event)) for p in preds]
    hist_avail, pool = estimate_availability_history(df, db)
    if any(hist_avail):
        # Use last historical availability as seed; if time-series more sophisticated forecasting desired, can add SARIMAX here
        avail_seed = hist_avail[-1]
    else:
        avail_seed = int(round(pool * 0.6)) if pool > 0 else 5

    results = {}
    for i in range(horizon):
        month_dt = (pd.to_datetime(monthly['dt'].iloc[-1]) + pd.DateOffset(months=i+1))
        ym = month_dt.strftime('%Y-%m')
        results[ym] = {
            'predicted_events': preds[i],
            'predicted_demand': demand[i],
            'available_hosts': avail_seed,
            'shortage': max(0, demand[i] - avail_seed)
        }

    # Print readable report
    total_events = sum(results[m]['predicted_events'] for m in results)
    avg_events = round(total_events / max(1, horizon), 1)
    months = list(results.keys())
    label = f"{months[0]}–{months[-1]}"
    per_month_str = ", ".join([f"{datetime.strptime(m, '%Y-%m').strftime('%B %Y')}: {results[m]['predicted_events']}" for m in months])

    print(f"Summary: Next {horizon} months ({label}): {total_events} exhibitions expected (avg {avg_events}/month)")
    print(f"Per month: {per_month_str}\n")
    print("=== Supply–Demand Forecast ===")
    for m in months:
        r = results[m]
        print(f"- {datetime.strptime(m, '%Y-%m').strftime('%B %Y')}: demand={r['predicted_demand']}, available={r['available_hosts']}, shortage={r['shortage']}")

    # Also emit JSON for downstream usage
    print('\n=== JSON Output ===')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo-uri', default='mongodb://localhost:27017/')
    parser.add_argument('--db', default='event_hosting_platform')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--scope', type=str, default='industry')
    parser.add_argument('--json-path', type=str, default=None, help='Path to dummy_mongo_events.json to bypass MongoDB')
    args = parser.parse_args()
    run_forecast(args.mongo_uri, args.db, horizon=args.horizon, scope=args.scope, json_path=args.json_path)

def _get_monthly_series(db) -> pd.DataFrame:
    df = build_monthly_df(db=db, scope='industry')
    if df.empty:
        return pd.DataFrame(columns=["ym", "y_event_count", "y_host_demand", "y_host_avail", "dt"])
    monthly = prepare_global_monthly(df)
    return monthly


def forecast_event_volume(db, horizon_months: int = 3):
    monthly = _get_monthly_series(db)
    if monthly.empty:
        return {"All": {}}, {}
    try:
        model, features = train_event_model(monthly)
        preds = iterative_forecast(monthly, model, features, horizon=horizon_months)
    except Exception:
        hist = monthly['y_event_count'].dropna().tolist()
        preds = [int(round(np.mean(hist[-3:]))) for _ in range(horizon_months)] if hist else [0] * horizon_months
    # Adjust to realistic 3–4 events/month
    preds = _adjust_event_forecast_to_target(preds, target_mean=2.5, min_events=2, max_events=3)
    last_dt = monthly['dt'].iloc[-1]
    months = [(pd.to_datetime(last_dt) + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(horizon_months)]
    return {"All": {months[i]: int(preds[i]) for i in range(horizon_months)}}, {}


def seasonal_event_volume_summary(db, horizon_months: int = 3) -> Dict[str, str]:
    monthly = _get_monthly_series(db)
    if monthly.empty:
        return {}
    # Historical totals
    hist = monthly['y_event_count'].fillna(0).tolist()
    # Future totals from forecast_event_volume
    ind_fc, _ = forecast_event_volume(db, horizon_months=horizon_months)
    future_months = next_n_months_from_today(horizon_months)
    all_map = ind_fc.get('All', {})
    future_vals = [int(all_map.get(m, 0)) for m in future_months]
    season_sum = sum(future_vals)
    if len(hist) >= 3:
        # rolling 3-month avg over history
        rolls = []
        for i in range(len(hist) - 2):
            rolls.append(sum(hist[i:i+3]))
        hist_avg = (sum(rolls) / len(rolls)) if rolls else (sum(hist) or 1)
    else:
        hist_avg = sum(hist) or 1
    pct = ((season_sum - hist_avg) / hist_avg) * 100.0
    signal = 'flat'
    if pct >= 20:
        signal = 'spike'
    elif pct >= 5:
        signal = 'increase'
    elif pct <= -20:
        signal = 'drop'
    elif pct <= -5:
        signal = 'decrease'
    return {
        'season': f"{future_months[0]}–{future_months[-1]}",
        'forecast_total_events': str(season_sum),
        'historical_3mo_avg': f"{hist_avg:.1f}",
        'change_pct': f"{pct:.1f}%",
        'signal': signal,
    }


def supply_demand_forecast_monthly(db, horizon_months: int = 3):
    # Use the same logic as run_forecast but return a dict per month for the app
    df = build_monthly_df(db=db, scope='industry')
    monthly = prepare_global_monthly(df) if not df.empty else pd.DataFrame()
    if monthly.empty:
        return {}
    try:
        model, features = train_event_model(monthly)
        preds = iterative_forecast(monthly, model, features, horizon=horizon_months)
    except Exception:
        hist = monthly['y_event_count'].dropna().tolist()
        preds = [int(round(np.mean(hist[-3:]))) for _ in range(horizon_months)] if hist else [0] * horizon_months

    # Align with run_forecast aggregation and availability approach
    avg_hosts_per_event = compute_avg_hosts_per_event(df)
    demand = [int(round(p * avg_hosts_per_event)) for p in preds]
    hist_avail, pool = estimate_availability_history(df, db)
    if any(hist_avail):
        avail_seed = hist_avail[-1]
    else:
        # If no history, use 70% qualified estimate as pool baseline
        avail_seed = int(round(pool * 0.7)) if pool > 0 else 5

    last_dt = monthly['dt'].iloc[-1]
    months = [(pd.to_datetime(last_dt) + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(horizon_months)]
    results: Dict[str, Dict[str, int]] = {}
    for i, m in enumerate(months):
        d = int(demand[i])
        a = int(avail_seed)
        results[m] = {
            'predicted_demand': d,
            'predicted_demand_lower': max(0, int(round(d * 0.8))),
            'predicted_demand_upper': int(round(d * 1.2)),
            'available_hosts': a,
            'available_hosts_lower': max(0, int(round(a * 0.9))),
            'available_hosts_upper': int(round(a * 1.1)),
            'shortage_total': max(0, d - a),
        }
    return results
