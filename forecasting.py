# forecasting.py  (REPLACEMENT / UPDATED)
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from pymongo import MongoClient

# ---------------------------
# Forecast helpers
# ---------------------------
def _sarimax_forecast(history: List[int], horizon: int, seasonal_period: int = 12) -> List[int]:
    """Best-effort SARIMAX forecast with graceful fallback to moving average."""
    try:
        if len(history) < 4 or all(v == 0 for v in history):
            raise ValueError("history too short or zero")
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, seasonal_period) if len(history) >= seasonal_period else (0, 0, 0, 0)
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        preds = res.forecast(steps=horizon)
        out = [max(0, int(round(float(x)))) for x in preds]
        return out
    except Exception:
        # fallback: moving average using recent window
        return moving_average_forecast(history, horizon=horizon, window=min(6, max(1, len(history))))

def _sarimax_forecast_with_ci(history: List[int], horizon: int, seasonal_period: int = 12, alpha: float = 0.05):
    """Return (preds, lower, upper). Fallback to moving average +/- heuristic bands."""
    try:
        if len(history) < 4 or all(v == 0 for v in history):
            raise ValueError("history too short or zero")
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, seasonal_period) if len(history) >= seasonal_period else (0, 0, 0, 0)
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci_df = fc.conf_int(alpha=alpha)
        # Pandas DataFrame-like indexable
        lower = [float(ci_df.iloc[i, 0]) for i in range(len(mean))]
        upper = [float(ci_df.iloc[i, 1]) for i in range(len(mean))]
        preds = [max(0, int(round(float(x)))) for x in mean]
        lower_i = [max(0, int(round(x))) for x in lower]
        upper_i = [max(0, int(round(x))) for x in upper]
        return preds, lower_i, upper_i
    except Exception:
        base = moving_average_forecast(history, horizon=horizon, window=min(6, max(1, len(history))))
        # +/- 20% heuristic band
        lower = [max(0, int(round(b * 0.8))) for b in base]
        upper = [max(0, int(round(b * 1.2))) for b in base]
        return base, lower, upper

def moving_average_forecast(history: List[int], horizon: int = 3, window: int = 3) -> List[int]:
    if not history:
        return [0] * horizon
    values = history[-window:] if len(history) >= window else history
    avg = sum(values) / len(values)
    return [max(0, int(round(avg))) for _ in range(horizon)]

# ---------------------------
# DB helpers
# ---------------------------
def get_db(uri: str = "mongodb://localhost:27017/", db_name: str = "event_hosting_platform"):
    client = MongoClient(uri)
    return client[db_name]

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def upcoming_months(start: datetime, n_months: int = 3) -> List[str]:
    months = []
    year = start.year
    month = start.month
    for _ in range(n_months):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months

def next_three_months_from_today() -> List[str]:
    today = datetime.today().replace(day=1)
    return upcoming_months(today, 3)

def next_n_months_from_today(n: int) -> List[str]:
    today = datetime.today().replace(day=1)
    return upcoming_months(today, max(1, int(n)))

# ---------------------------
# Aggregation functions
# ---------------------------
def aggregate_event_volume(db) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    industry_counts_by_month: Dict[str, Counter] = defaultdict(Counter)
    location_counts_by_month: Dict[str, Counter] = defaultdict(Counter)
    for ev in db["Event"].find({}):
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if not start:
            continue
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        m = month_key(start)
        industry_counts_by_month[m]["All"] += 1
        location_counts_by_month[m][ev.get("location", "Unknown")] += 1
    return industry_counts_by_month, location_counts_by_month

def map_event_to_primary_industry(db) -> Dict[str, str]:
    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
    votes: Dict[str, Counter] = defaultdict(Counter)
    for xp in db["ExhibitorProfile"].find({}):
        ev_id = xp.get("event")
        ex = exhibitors.get(xp.get("exhibitor"))
        if not ev_id or not ex:
            continue
        ind = ex.get("industry", "Unknown")
        votes[ev_id][ind] += 1
    primary: Dict[str, str] = {}
    for ev_id, ctr in votes.items():
        if ctr:
            primary[ev_id] = ctr.most_common(1)[0][0]
    return primary

def compute_historical_shares(db) -> Tuple[Dict[str, float], Dict[str, float]]:
    primary_by_event = map_event_to_primary_industry(db)
    events = list(db["Event"].find({}))
    by_month_events: Dict[str, List[dict]] = defaultdict(list)
    for ev in events:
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        if not start:
            continue
        by_month_events[month_key(start)].append(ev)
    months_sorted = sorted(by_month_events.keys())
    use_months = months_sorted[-6:] if len(months_sorted) > 6 else months_sorted

    industry_counts: Counter = Counter()
    location_counts: Counter = Counter()
    total_events = 0
    for m in use_months:
        for ev in by_month_events[m]:
            ev_id = ev.get("_id")
            ind = primary_by_event.get(ev_id, "Unknown")
            industry_counts[ind] += 1
            location_counts[ev.get("location", "Unknown")] += 1
            total_events += 1
    if total_events == 0:
        return {}, {}
    industry_shares = {k: v / total_events for k, v in industry_counts.items()}
    location_shares = {k: v / total_events for k, v in location_counts.items()}
    return industry_shares, location_shares

def allocate_by_shares(total: int, shares: Dict[str, float]) -> Dict[str, int]:
    if total <= 0 or not shares:
        return {k: 0 for k in shares}
    s = sum(shares.values())
    base = {k: (shares[k] / s) * total if s > 0 else 0 for k in shares}
    floored = {k: int(v) for k, v in base.items()}
    remainder = total - sum(floored.values())
    if remainder > 0:
        fracs = sorted(((base[k] - floored[k], k) for k in base), reverse=True)
        for i in range(remainder):
            _, kk = fracs[i % len(fracs)]
            floored[kk] += 1
    return floored

# ---------------------------
# Event volume forecasting
# ---------------------------
def forecast_event_volume(db, horizon_months: int = 3):
    industry_counts_by_month, location_counts_by_month = aggregate_event_volume(db)
    if not industry_counts_by_month:
        return {}, {}
    months_sorted = sorted(industry_counts_by_month.keys())
    hist_totals = [sum(industry_counts_by_month[m].values()) for m in months_sorted]
    future_months = next_n_months_from_today(horizon_months)

    total_preds = _sarimax_forecast(hist_totals, horizon=horizon_months)
    all_forecast = {future_months[i]: total_preds[i] for i in range(horizon_months)}

    industry_shares, location_shares = compute_historical_shares(db)

    industry_forecast: Dict[str, Dict[str, int]] = defaultdict(dict)
    location_forecast: Dict[str, Dict[str, int]] = defaultdict(dict)
    for m, total_m in all_forecast.items():
        ind_alloc = allocate_by_shares(int(total_m), industry_shares)
        loc_alloc = allocate_by_shares(int(total_m), location_shares)
        for k, v in ind_alloc.items():
            industry_forecast[k][m] = v
        for k, v in loc_alloc.items():
            location_forecast[k][m] = v

    industry_forecast["All"] = all_forecast
    return industry_forecast, location_forecast

def month_range_label(months: List[str]) -> str:
    if not months:
        return ""
    if len(months) == 1:
        return months[0]
    return f"{months[0]}–{months[-1]}"

def seasonal_event_volume_summary(db, horizon_months: int = 3) -> Dict[str, str]:
    ind_by_month, _ = aggregate_event_volume(db)
    if not ind_by_month:
        return {}
    months_sorted = sorted(ind_by_month.keys())
    future_months = next_n_months_from_today(horizon_months)

    hist_month_totals = [sum(ind_by_month[m].values()) for m in months_sorted]
    ind_fc, _ = forecast_event_volume(db, horizon_months=horizon_months)
    future_totals = []
    all_map = ind_fc.get("All", {})
    for m in future_months:
        future_totals.append(int(all_map.get(m, 0)))

    season_sum = sum(future_totals)
    if len(hist_month_totals) >= 3:
        rolling = []
        for i in range(len(hist_month_totals) - 2):
            rolling.append(sum(hist_month_totals[i:i+3]))
        hist_avg = sum(rolling) / len(rolling)
    else:
        hist_avg = sum(hist_month_totals) or 1

    pct = ((season_sum - hist_avg) / hist_avg) * 100.0
    label = "flat"
    if pct >= 20:
        label = "spike"
    elif pct >= 5:
        label = "increase"
    elif pct <= -20:
        label = "drop"
    elif pct <= -5:
        label = "decrease"

    return {
        "season": month_range_label(future_months),
        "forecast_total_events": str(season_sum),
        "historical_3mo_avg": f"{hist_avg:.1f}",
        "change_pct": f"{pct:.1f}%",
        "signal": label,
    }

def month_pretty(month_str: str) -> str:
    dt = datetime.strptime(month_str, "%Y-%m")
    return dt.strftime("%B %Y")

# ---------------------------
# Demand & availability aggregation + forecasting
# ---------------------------
def aggregate_invite_demand(db) -> Dict[str, Counter]:
    demand_by_month_industry: Dict[str, Counter] = defaultdict(Counter)
    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
    events = {e["_id"]: e for e in db["Event"].find({})}
    for xp in db["ExhibitorProfile"].find({}):
        exhibitor = exhibitors.get(xp.get("exhibitor"))
        event = events.get(xp.get("event"))
        if not exhibitor or not event:
            continue
        dates = event.get("dates") or {}
        start = dates.get("start")
        if not start:
            continue
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        m = month_key(start)
        industry = exhibitor.get("industry", "Unknown")
        demand_by_month_industry[m][industry] += int(xp.get("hostessRequirements", 0))
    return demand_by_month_industry

def available_hosts_summary(db) -> Dict[str, Dict[str, int]]:
    """Compute availability buckets by month. If historical availability is sparse, we still estimate
    availability using the host pool size and a fallback active fraction so future months aren't zero."""
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Build unavailable month sets per host
    unavailable_by_host: Dict[str, set] = defaultdict(set)
    for u in db["UnavailableDate"].find({}):
        hid = u.get("eventHost")
        for d in u.get("dates", []):
            if isinstance(d, datetime):
                m = d.strftime("%Y-%m")
            else:
                # try to parse / slice strings
                try:
                    m = str(d)[:7]
                except Exception:
                    continue
            unavailable_by_host[hid].add(m)

    event_months = set()
    for ev in db["Event"].find({}):
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        if not start:
            continue
        event_months.add(month_key(start))

    hosts = list(db["EventHostUser"].find({}))
    total_host_pool = len(hosts)
    for host in hosts:
        is_bilingual = len(host.get("languages", [])) >= 2
        is_exp3 = int(host.get("yearsOfExperience", 0)) >= 3
        industries = host.get("industries", []) or ["Unknown"]
        for m in event_months:
            if m in unavailable_by_host.get(host["_id"], set()):
                continue
            summary[m]["total"] += 1
            if is_bilingual:
                summary[m]["bilingual"] += 1
            if is_exp3:
                summary[m]["exp3plus"] += 1
            for ind in industries:
                summary[(m, ind)]["total"] += 1
                if is_bilingual:
                    summary[(m, ind)]["bilingual"] += 1
                if is_exp3:
                    summary[(m, ind)]["exp3plus"] += 1

    # Attach meta for fallback usage
    summary["_meta_total_pool"] = {"total_host_pool": total_host_pool}
    return summary

def supply_demand_forecast_monthly(db, horizon_months: int = 3):
    ind_by_month, _ = aggregate_event_volume(db)
    if not ind_by_month:
        return {}
    months_sorted = sorted(ind_by_month.keys())
    future_months = next_n_months_from_today(horizon_months)

    demand_hist_by_month_ind = aggregate_invite_demand(db)
    # avg hosts per event (weighted)
    pairs: List[Tuple[int, int]] = []
    for m in months_sorted:
        total_events_m = sum(ind_by_month[m].values())
        total_hosts_req_m = sum(int(v) for v in demand_hist_by_month_ind.get(m, {}).values())
        if total_events_m > 0 and total_hosts_req_m >= 0:
            pairs.append((total_events_m, total_hosts_req_m))
    if pairs:
        total_events_hist = sum(p[0] for p in pairs)
        total_hosts_hist = sum(p[1] for p in pairs)
        avg_hosts_per_event = total_hosts_hist / max(1, total_events_hist)
    else:
        avg_hosts_per_event = 1.0

    # Forecast events (with CI)
    hist_events = [sum(ind_by_month[m].values()) for m in months_sorted]
    ev_mean, ev_low, ev_up = _sarimax_forecast_with_ci(hist_events, horizon=horizon_months)
    all_map = {future_months[i]: ev_mean[i] for i in range(horizon_months)}

    # Build demand forecast (mean, low, up) using hosts/event
    demand_fc: Dict[str, Dict[str, int]] = {}
    for i, m in enumerate(future_months):
        evs = int(all_map.get(m, 0))
        evl = int(ev_low[i])
        evu = int(ev_up[i])
        d_mean = int(round(evs * avg_hosts_per_event))
        d_low = int(round(evl * avg_hosts_per_event))
        d_up = int(round(evu * avg_hosts_per_event))
        demand_fc[m] = {"mean": max(0, d_mean), "low": max(0, d_low), "up": max(0, d_up)}

    # Forecast availability: use historical per-month totals if available, otherwise estimate from pool
    host_availability = available_hosts_summary(db)
    # remove meta for historical series
    total_host_pool = host_availability.get("_meta_total_pool", {}).get("total_host_pool", 0)
    # historical availability series (aligned to months_sorted)
    hist_avail = [int(host_availability.get(m, {}).get("total", 0)) for m in months_sorted]
    if any(hist_avail):
        avail_mean, avail_low, avail_up = _sarimax_forecast_with_ci(hist_avail, horizon=horizon_months)
    else:
        # fallback: use a reasonable active ratio of the total host pool: assume 60% active if unknown
        fallback_ratio = 0.6
        if total_host_pool > 0:
            est = int(round(total_host_pool * fallback_ratio))
        else:
            # ultimate fallback: 5 hosts
            est = 5
        avail_mean = [est] * horizon_months
        avail_low = [max(0, int(round(est * 0.8)))] * horizon_months
        avail_up = [int(round(est * 1.2))] * horizon_months

    # For experienced hosts
    hist_exp3 = [int(host_availability.get(m, {}).get("exp3plus", 0)) for m in months_sorted]
    if any(hist_exp3):
        exp3_mean, exp3_low, exp3_up = _sarimax_forecast_with_ci(hist_exp3, horizon=horizon_months)
    else:
        # estimate that ~20% of pool are exp3 if total_host_pool known, fallback to small number
        if total_host_pool > 0:
            est_exp3 = int(round(total_host_pool * 0.2))
        else:
            est_exp3 = 1
        exp3_mean = [est_exp3] * horizon_months

    results: Dict[str, Dict[str, int]] = {}
    for i, m in enumerate(future_months):
        d = demand_fc[m]
        total_avail = int(avail_mean[i])
        results[m] = {
            "predicted_demand": d["mean"],
            "predicted_demand_lower": d["low"],
            "predicted_demand_upper": d["up"],
            "available_hosts": total_avail,
            "available_hosts_lower": int(avail_low[i]),
            "available_hosts_upper": int(avail_up[i]),
            "shortage_total": max(0, d["mean"] - total_avail),
            "available_exp3plus": int(exp3_mean[i]),
        }
    return results

def host_onboarding_needs_from_forecast_monthly(supply_demand_monthly: Dict[str, Dict[str, int]], months: List[str]) -> List[str]:
    recommendations: List[str] = []
    months_set = set(months or [])
    for m, stats in supply_demand_monthly.items():
        if months_set and m not in months_set:
            continue
        demand = int(stats.get("predicted_demand", 0))
        total_avail = int(stats.get("available_hosts", 0))
        shortage = max(0, demand - total_avail)
        if shortage > 0:
            recommendations.append(
                f"Forecasted shortage in {month_pretty(m)}: need {shortage} more hosts.")
    # Deduplicate
    out: List[str] = []
    seen = set()
    for r in recommendations:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out

# ---------------------------
# Reporting
# ---------------------------
def print_report(db):
    ind_fc, _ = forecast_event_volume(db)
    sd_m = supply_demand_forecast_monthly(db)
    recs = host_onboarding_needs_from_forecast_monthly(sd_m, next_three_months_from_today())
    season = seasonal_event_volume_summary(db)

    all_map = ind_fc.get("All", {})
    months = next_three_months_from_today()
    total_events = sum(int(all_map.get(m, 0)) for m in months)
    avg_events = round(total_events / max(1, len(months)), 1)
    label = month_range_label(months)
    per_month_str = ", ".join([f"{month_pretty(m)}: {int(all_map.get(m,0))}" for m in months])
    print(f"Summary: Next three months ({label}): {total_events}+ exhibitions expected (avg {avg_events}/month)")
    print(f"Per month: {per_month_str}")

    print("\n=== Event Volume Forecast ===")
    all_map = ind_fc.get("All", {})
    if all_map:
        line = ", ".join([f"{m}: {v}" for m, v in all_map.items()])
        print(f"- All: {line}")
    else:
        print("- All: (no forecastable history)")

    if season:
        print("\n=== Seasonal Event Volume (Next 3 months) ===")
        print(
            f"- {season['season']}: total={season['forecast_total_events']} vs hist-3mo-avg={season['historical_3mo_avg']} "
            f"({season['change_pct']}) → {season['signal']}"
        )

    print("\n=== Supply–Demand Forecast (Monthly) — Next 3 months ===")
    for m in months:
        stats = sd_m.get(m, {})
        print(
            f"- {month_pretty(m)}: demand={int(stats.get('predicted_demand', 0))}, available={int(stats.get('available_hosts', 0))}, "
            f"shortage={int(stats.get('shortage_total', 0))}"
        )

    print("\n=== Host Onboarding Needs (Recommendations) ===")
    if not recs:
        print("- No critical gaps detected based on historical data.")
    else:
        for r in recs:
            print(f"- {r}")

if __name__ == "__main__":
    db = get_db()
    print_report(db)
