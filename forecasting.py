from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from pymongo import MongoClient


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


def aggregate_event_volume(db) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    industry_counts_by_month: Dict[str, Counter] = defaultdict(Counter)
    location_counts_by_month: Dict[str, Counter] = defaultdict(Counter)

    for ev in db["Event"].find({}):
        dates = ev.get("dates") or {}
        start: datetime = dates.get("start")
        if not start:
            continue
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        m = month_key(start)
        # industry now inferred from linked exhibitors via ExhibitorProfile, but for
        # volume we only need counts, independent of industry. We'll still keep a
        # placeholder "All" industry for total volume by month.
        industry_counts_by_month[m]["All"] += 1
        location_counts_by_month[m][ev.get("location", "Unknown")] += 1

    return industry_counts_by_month, location_counts_by_month


def moving_average_forecast(history: List[int], horizon: int = 3, window: int = 3) -> List[int]:
    if not history:
        return [0] * horizon
    values = history[-window:] if len(history) >= window else history
    avg = sum(values) / len(values)
    return [max(0, round(avg)) for _ in range(horizon)]


def forecast_event_volume(db, horizon_months: int = 3):
    industry_counts_by_month, location_counts_by_month = aggregate_event_volume(db)

    # Build consistent month ordering from history
    if not industry_counts_by_month:
        return {}, {}

    months_sorted = sorted(industry_counts_by_month.keys())
    last_hist_month = datetime.strptime(months_sorted[-1], "%Y-%m")
    future_months = upcoming_months(last_hist_month + timedelta(days=1), horizon_months)

    # per industry
    industry_totals: Dict[str, List[int]] = defaultdict(list)
    for m in months_sorted:
        for industry, cnt in industry_counts_by_month[m].items():
            industry_totals[industry].append(cnt)
    industry_forecast: Dict[str, Dict[str, int]] = {}
    for industry, hist in industry_totals.items():
        preds = moving_average_forecast(hist, horizon=horizon_months)
        industry_forecast[industry] = {future_months[i]: preds[i] for i in range(horizon_months)}

    # per location
    location_totals: Dict[str, List[int]] = defaultdict(list)
    for m in months_sorted:
        for location, cnt in location_counts_by_month[m].items():
            location_totals[location].append(cnt)
    location_forecast: Dict[str, Dict[str, int]] = {}
    for location, hist in location_totals.items():
        preds = moving_average_forecast(hist, horizon=horizon_months)
        location_forecast[location] = {future_months[i]: preds[i] for i in range(horizon_months)}

    return industry_forecast, location_forecast


def month_range_label(months: List[str]) -> str:
    if not months:
        return ""
    if len(months) == 1:
        return months[0]
    return f"{months[0]}–{months[-1]}"


def seasonal_event_volume_summary(db, horizon_months: int = 3) -> Dict[str, str]:
    # Summarize next N months as a season, compare to historical rolling 3-month averages
    ind_by_month, _ = aggregate_event_volume(db)
    if not ind_by_month:
        return {}
    months_sorted = sorted(ind_by_month.keys())
    last_hist_month = datetime.strptime(months_sorted[-1], "%Y-%m")
    future_months = upcoming_months(last_hist_month + timedelta(days=1), horizon_months)

    # Historical total per month (all industries)
    hist_month_totals = [sum(ind_by_month[m].values()) for m in months_sorted]
    # Future forecast total per month from our event forecast
    ind_fc, _ = forecast_event_volume(db, horizon_months=horizon_months)
    future_totals = []
    # ind_fc has only key "All"; fall back to zeros if missing
    all_map = ind_fc.get("All", {})
    for m in future_months:
        future_totals.append(int(all_map.get(m, 0)))

    # Compare seasonal sum vs historical 3-month rolling average
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


def aggregate_invite_demand(db) -> Dict[str, Counter]:
    # Demand per month and industry: from ExhibitorProfile.hostessRequirements and ExhibitorUser.industry
    demand_by_month_industry: Dict[str, Counter] = defaultdict(Counter)
    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
    events = {e["_id"]: e for e in db["Event"].find({})}
    for xp in db["ExhibitorProfile"].find({}):
        exhibitor = exhibitors.get(xp.get("exhibitor"))
        event = events.get(xp.get("event"))
        if not exhibitor or not event:
            continue
        dates = event.get("dates") or {}
        start: datetime = dates.get("start")
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
    # Compute availability by subtracting UnavailableDate from the entire horizon months
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Build unavailable month sets per host
    unavailable_by_host: Dict[str, set] = defaultdict(set)
    for u in db["UnavailableDate"].find({}):
        hid = u.get("eventHost")
        for d in u.get("dates", []):
            if isinstance(d, datetime):
                m = d.strftime("%Y-%m")
            else:
                m = str(d)[:7]
            unavailable_by_host[hid].add(m)

    # Define the year months horizon we care about (based on events present)
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
    return summary


def supply_demand_forecast(db, horizon_months: int = 3):
    demand_hist = aggregate_invite_demand(db)
    if not demand_hist:
        return {}
    months_sorted = sorted(demand_hist.keys())
    last_hist_month = datetime.strptime(months_sorted[-1], "%Y-%m")
    future_months = upcoming_months(last_hist_month + timedelta(days=1), horizon_months)

    # Build history per industry
    industry_demand_series: Dict[str, List[int]] = defaultdict(list)
    for m in months_sorted:
        for industry, cnt in demand_hist[m].items():
            industry_demand_series[industry].append(cnt)

    industry_demand_forecast: Dict[str, Dict[str, int]] = {}
    for industry, hist in industry_demand_series.items():
        preds = moving_average_forecast(hist, horizon=horizon_months)
        industry_demand_forecast[industry] = {future_months[i]: preds[i] for i in range(horizon_months)}

    host_availability = available_hosts_summary(db)
    results: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for industry, month_to_demand in industry_demand_forecast.items():
        for m, demand in month_to_demand.items():
            avail = host_availability.get((m, industry), {})
            total_avail = int(avail.get("total", 0))
            bilingual_avail = int(avail.get("bilingual", 0))
            exp3_avail = int(avail.get("exp3plus", 0))
            results[industry][m] = {
                "predicted_demand": demand,
                "available_hosts": total_avail,
                "shortage_total": max(0, demand - total_avail),
                "available_bilingual": bilingual_avail,
                "available_exp3plus": exp3_avail,
            }
    return results


def host_onboarding_needs(db) -> List[str]:
    # Heuristics based on historical shortages and composition
    demand_hist = aggregate_invite_demand(db)
    host_availability = available_hosts_summary(db)
    recommendations: List[str] = []

    for m in sorted(demand_hist.keys()):
        for industry, demand in demand_hist[m].items():
            avail = host_availability.get((m, industry), {})
            total_avail = int(avail.get("total", 0))
            bilingual_avail = int(avail.get("bilingual", 0))
            exp3_avail = int(avail.get("exp3plus", 0))

            if demand > total_avail:
                recommendations.append(
                    f"Shortage observed in {m} for {industry}: need {demand - total_avail} more hosts.")
            # ratios
            if demand > 0 and bilingual_avail < max(1, demand // 3):
                recommendations.append(
                    f"Bilingual gap in {m} for {industry}: onboard ~{max(0, demand // 3 - bilingual_avail)} bilingual hosts.")
            if demand > 0 and exp3_avail < max(1, demand // 4):
                recommendations.append(
                    f"Experienced host gap in {m} for {industry}: onboard ~{max(0, demand // 4 - exp3_avail)} hosts with 3+ years.")
    return recommendations


def print_report(db):
    ind_fc, loc_fc = forecast_event_volume(db)
    sd = supply_demand_forecast(db)
    recs = host_onboarding_needs(db)
    season = seasonal_event_volume_summary(db)

    # Summary line with total and average for next three months (kept in addition to detailed sections)
    all_map = ind_fc.get("All", {})
    months = []
    if season and season.get("season"):
        start, end = season["season"].split("–") if "–" in season["season"] else (season["season"], season["season"])
        start_dt = datetime.strptime(start, "%Y-%m")
        end_dt = datetime.strptime(end, "%Y-%m")
        cur = start_dt
        while cur <= end_dt:
            months.append(cur.strftime("%Y-%m"))
            y = cur.year + (1 if cur.month == 12 else 0)
            m = 1 if cur.month == 12 else cur.month + 1
            cur = datetime(y, m, 1)
    else:
        months = sorted(all_map.keys())
    total_events = sum(int(all_map.get(m, 0)) for m in months)
    avg_events = round(total_events / max(1, len(months)), 1)
    label = month_range_label(months)
    print(f"Summary: Next three months ({label}): {total_events}+ exhibitions expected (avg {avg_events}/month)")

    print("\n=== Event Volume Forecast (All Industries) ===")
    for industry, m_to_v in sorted(ind_fc.items()):
        line = ", ".join([f"{m}: {v}" for m, v in m_to_v.items()])
        print(f"- {industry}: {line}")

    if season:
        print("\n=== Seasonal Event Volume (Next 3 months) ===")
        print(
            f"- {season['season']}: total={season['forecast_total_events']} vs hist-3mo-avg={season['historical_3mo_avg']} "
            f"({season['change_pct']}) → {season['signal']}"
        )

    print("\n=== Event Volume Forecast (Location) ===")
    for location, m_to_v in sorted(loc_fc.items()):
        line = ", ".join([f"{m}: {v}" for m, v in m_to_v.items()])
        print(f"- {location}: {line}")

    print("\n=== Supply–Demand Forecast (by Industry) ===")
    for industry, mstats in sorted(sd.items()):
        for m, stats in sorted(mstats.items()):
            print(
                f"- {industry} {m}: demand={stats['predicted_demand']}, available={stats['available_hosts']} "
                f"(bilingual={stats['available_bilingual']}, 3+yr={stats['available_exp3plus']}), "
                f"shortage={stats['shortage_total']}"
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