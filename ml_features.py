from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient


def get_db(uri: str = "mongodb://localhost:27017/", db_name: str = "event_hosting_platform"):
    return MongoClient(uri)[db_name]


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def build_monthly_df(db=None, scope: str = "industry") -> pd.DataFrame:
    """
    Build monthly dataset with targets:
    - y_event_count (events per month)
    - y_host_demand (sum hostessRequirements per month)
    - y_host_avail (hosts available per month)

    scope: "industry" or "location"
    Returns df with columns: ym, key (industry or location), y_event_count, y_host_demand, y_host_avail
    """
    if db is None:
        db = get_db()

    # Map event -> month, location
    events = list(db["Event"].find({}))
    rows = []
    for ev in events:
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if not start:
            continue
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        ym = month_key(start)
        rows.append({
            "event_id": ev.get("_id"),
            "ym": ym,
            "location": ev.get("location", "Unknown"),
        })
    ev_df = pd.DataFrame(rows)

    # Exhibitor industry per event + demand (hostessRequirements)
    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
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

    if ev_df.empty:
        return pd.DataFrame(columns=["ym", "key", "y_event_count", "y_host_demand", "y_host_avail"])

    # Join to months and aggregate
    if not xp_df.empty:
        ev_xp = ev_df.merge(xp_df, left_on="event_id", right_on="event", how="left")
    else:
        ev_xp = ev_df.copy()
        ev_xp["industry"] = "Unknown"
        ev_xp["hostessRequirements"] = 0

    if scope == "industry":
        key_col = "industry"
    else:
        key_col = "location"

    # Unique events per (ym, key)
    ev_counts = (ev_xp.groupby(["ym", key_col])["event_id"].nunique()
                 .reset_index().rename(columns={"event_id": "y_event_count"}))
    demand = (ev_xp.groupby(["ym", key_col])["hostessRequirements"].sum()
              .reset_index().rename(columns={"hostessRequirements": "y_host_demand"}))
    df = ev_counts.merge(demand, on=["ym", key_col], how="left").fillna({"y_host_demand": 0})

    # Supply (hosts available) — approximate by total hosts minus UnavailableDate in month
    unavail = list(db["UnavailableDate"].find({}))
    un_by_month = {}
    for u in unavail:
        for d in u.get("dates", []):
            m = d[:7] if isinstance(d, str) else d.strftime("%Y-%m")
            un_by_month[m] = un_by_month.get(m, 0) + 1
    total_hosts = db["EventHostUser"].count_documents({})
    df["y_host_avail"] = df["ym"].map(lambda m: max(0, total_hosts - un_by_month.get(m, 0)))

    # Rename key
    df = df.rename(columns={key_col: "key"})

    # Sort and add time features
    df = df.sort_values(["key", "ym"]).reset_index(drop=True)
    # month ordinal
    df["dt"] = pd.to_datetime(df["ym"] + "-01")
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lags and rolling per key for event count and demand
    def add_lags_rollings(group: pd.DataFrame, col: str) -> pd.DataFrame:
        group[f"{col}_lag_1"] = group[col].shift(1)
        group[f"{col}_lag_2"] = group[col].shift(2)
        group[f"{col}_lag_3"] = group[col].shift(3)
        group[f"{col}_roll3_mean"] = group[col].rolling(3, min_periods=1).mean().shift(1)
        group[f"{col}_roll6_mean"] = group[col].rolling(6, min_periods=1).mean().shift(1)
        return group

    df = df.groupby("key", group_keys=False).apply(lambda g: add_lags_rollings(g, "y_event_count"))
    df = df.groupby("key", group_keys=False).apply(lambda g: add_lags_rollings(g, "y_host_demand"))

    # Multi-horizon labels for event count
    df["y_t+1"] = df.groupby("key")["y_event_count"].shift(-1)
    df["y_t+2"] = df.groupby("key")["y_event_count"].shift(-2)
    df["y_t+3"] = df.groupby("key")["y_event_count"].shift(-3)

    # Drop rows with missing lags/labels for training convenience
    return df




