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


def next_three_months_from_today() -> List[str]:
    today = datetime.today().replace(day=1)
    return upcoming_months(today, 3)


def next_n_months_from_today(n: int) -> List[str]:
    today = datetime.today().replace(day=1)
    return upcoming_months(today, max(1, int(n)))


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
        industry_counts_by_month[m]["All"] += 1
        location_counts_by_month[m][ev.get("location", "Unknown")] += 1
    return industry_counts_by_month, location_counts_by_month
def map_event_to_primary_industry(db) -> Dict[str, str]:
    # Choose a single primary industry per event based on most frequent exhibitor industry
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
    # Industry shares based on unique events by primary industry; location shares based on event locations
    primary_by_event = map_event_to_primary_industry(db)
    events = list(db["Event"].find({}))
    # Use recent 6 months where events exist
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
    # Avoid division by zero
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


def moving_average_forecast(history: List[int], horizon: int = 3, window: int = 3) -> List[int]:
    if not history:
        return [0] * horizon
    values = history[-window:] if len(history) >= window else history
    avg = sum(values) / len(values)
    return [max(0, round(avg)) for _ in range(horizon)]


def forecast_event_volume(db, horizon_months: int = 3):
    # Step 1: forecast total events per month using historical totals
    industry_counts_by_month, location_counts_by_month = aggregate_event_volume(db)
    if not industry_counts_by_month:
        return {}, {}
    months_sorted = sorted(industry_counts_by_month.keys())
    hist_totals = [sum(industry_counts_by_month[m].values()) for m in months_sorted]
    future_months = next_three_months_from_today()
    total_preds = moving_average_forecast(hist_totals, horizon=horizon_months)
    all_forecast = {future_months[i]: total_preds[i] for i in range(horizon_months)}

    # Step 2: compute historical shares for industry and location
    industry_shares, location_shares = compute_historical_shares(db)

    # Step 3: allocate totals by shares, per month
    industry_forecast: Dict[str, Dict[str, int]] = defaultdict(dict)
    location_forecast: Dict[str, Dict[str, int]] = defaultdict(dict)
    for m, total_m in all_forecast.items():
        ind_alloc = allocate_by_shares(int(total_m), industry_shares)
        loc_alloc = allocate_by_shares(int(total_m), location_shares)
        for k, v in ind_alloc.items():
            industry_forecast[k][m] = v
        for k, v in loc_alloc.items():
            location_forecast[k][m] = v

    # Ensure 'All' is available in industry_forecast
    industry_forecast["All"] = all_forecast
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
    future_months = next_three_months_from_today()

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


def month_pretty(month_str: str) -> str:
    dt = datetime.strptime(month_str, "%Y-%m")
    return dt.strftime("%B %Y")


def forecast_event_volume_by_industry_grouped(db, horizon_months: int = 3) -> Dict[str, int]:
    by_month_industry_events: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    exhibitors = {e["_id"]: e for e in db["ExhibitorUser"].find({})}
    events = {e["_id"]: e for e in db["Event"].find({})}
    for xp in db["ExhibitorProfile"].find({}):
        exhibitor = exhibitors.get(xp.get("exhibitor"))
        ev = events.get(xp.get("event"))
        if not exhibitor or not ev:
            continue
        dates = ev.get("dates") or {}
        start = dates.get("start")
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                continue
        if not start:
            continue
        m = month_key(start)
        ind = exhibitor.get("industry", "Unknown")
        by_month_industry_events[m][ind].add(ev["_id"])

    if not by_month_industry_events:
        return {}

    months_sorted = sorted(by_month_industry_events.keys())
    future_months = next_three_months_from_today()

    # Compute historical industry shares over the full history (or last 6 months if long)
    recent_months = months_sorted[-6:] if len(months_sorted) > 6 else months_sorted
    total_events_hist = 0
    industry_event_counts: Dict[str, int] = defaultdict(int)
    for m in recent_months:
        for ind, evset in by_month_industry_events[m].items():
            c = len(evset)
            industry_event_counts[ind] += c
            total_events_hist += c
    if total_events_hist == 0:
        return {}
    shares = {ind: industry_event_counts[ind] / total_events_hist for ind in industry_event_counts}

    # Get overall forecast per future month and allocate by shares so sums match overall
    ind_fc, _ = forecast_event_volume(db, horizon_months)
    all_map = ind_fc.get("All", {})

    def allocate(total: int, weights: Dict[str, float]) -> Dict[str, int]:
        # Largest remainder method to preserve sum
        if total <= 0 or not weights:
            return {k: 0 for k in weights}
        # normalize
        s = sum(weights.values())
        if s == 0:
            base = {k: 0 for k in weights}
        else:
            base = {k: (weights[k] / s) * total for k in weights}
        floored = {k: int(v) for k, v in base.items()}
        remainder = total - sum(floored.values())
        if remainder > 0:
            # assign by largest fractional parts
            fracs = sorted(((base[k] - floored[k], k) for k in base), reverse=True)
            for i in range(remainder):
                _, kk = fracs[i % len(fracs)]
                floored[kk] += 1
        return floored

    grouped_totals: Dict[str, int] = defaultdict(int)
    for m in future_months:
        total_m = int(all_map.get(m, 0))
        alloc = allocate(total_m, shares)
        for ind, v in alloc.items():
            grouped_totals[ind] += v

    return dict(grouped_totals)


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
    # Align to next three months from today
    months_sorted = sorted(demand_hist.keys())
    future_months = next_three_months_from_today()

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
                    f"Shortage observed in {month_pretty(m)} for {industry}: need {demand - total_avail} more hosts.")
            # ratios
            if demand > 0 and bilingual_avail < max(1, demand // 3):
                recommendations.append(
                    f"Bilingual gap in {month_pretty(m)} for {industry}: onboard ~{max(0, demand // 3 - bilingual_avail)} bilingual hosts.")
            if demand > 0 and exp3_avail < max(1, demand // 4):
                recommendations.append(
                    f"Experienced host gap in {month_pretty(m)} for {industry}: onboard ~{max(0, demand // 4 - exp3_avail)} hosts with 3+ years.")
    return recommendations


def host_onboarding_needs_from_forecast(supply_demand: Dict[str, Dict[str, Dict[str, int]]], months: List[str]) -> List[str]:
    # Generate onboarding recommendations constrained to the provided future months using forecasted shortages
    recommendations: List[str] = []
    months_set = set(months or [])
    for industry, month_stats in supply_demand.items():
        for m, stats in month_stats.items():
            if months_set and m not in months_set:
                continue
            demand = int(stats.get("predicted_demand", 0))
            total_avail = int(stats.get("available_hosts", 0))
            bilingual_avail = int(stats.get("available_bilingual", 0))
            exp3_avail = int(stats.get("available_exp3plus", 0))

            shortage = max(0, demand - total_avail)
            if shortage > 0:
                recommendations.append(
                    f"Forecasted shortage in {month_pretty(m)} for {industry}: need {shortage} more hosts.")

            if demand > 0:
                required_bilingual = max(1, demand // 3)
                if bilingual_avail < required_bilingual:
                    recommendations.append(
                        f"Bilingual gap in {month_pretty(m)} for {industry}: onboard ~{required_bilingual - bilingual_avail} bilingual hosts.")

                required_exp3 = max(1, demand // 4)
                if exp3_avail < required_exp3:
                    recommendations.append(
                        f"Experienced host gap in {month_pretty(m)} for {industry}: onboard ~{required_exp3 - exp3_avail} hosts with 3+ years.")

    # Optional: deduplicate messages
    dedup = []
    seen = set()
    for r in recommendations:
        if r not in seen:
            dedup.append(r)
            seen.add(r)
    return dedup


def print_report(db):
    ind_fc, loc_fc = forecast_event_volume(db)
    sd = supply_demand_forecast(db)
    recs = host_onboarding_needs(db)
    season = seasonal_event_volume_summary(db)

    # Summary line with total and average for next three months (kept in addition to detailed sections)
    all_map = ind_fc.get("All", {})
    months = next_three_months_from_today()
    total_events = sum(int(all_map.get(m, 0)) for m in months)
    avg_events = round(total_events / max(1, len(months)), 1)
    label = month_range_label(months)
    per_month_str = ", ".join([f"{month_pretty(m)}: {int(all_map.get(m,0))}" for m in months])
    print(f"Summary: Next three months ({label}): {total_events}+ exhibitions expected (avg {avg_events}/month)")
    print(f"Per month: {per_month_str}")

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

    # Event Volume Forecast by Industries grouped over next three months
    grouped_industry = forecast_event_volume_by_industry_grouped(db)
    if grouped_industry:
        print("\n=== Event Volume Forecast (by Industries) — Next 3 months ===")
        for ind, total in sorted(grouped_industry.items()):
            print(f"- {ind}: {total}")

    print("\n=== Event Volume Forecast (Location) ===")
    for location, m_to_v in sorted(loc_fc.items()):
        line = ", ".join([f"{m}: {v}" for m, v in m_to_v.items()])
        print(f"- {location}: {line}")

    print("\n=== Supply–Demand Forecast (by Industry) — Next 3 months ===")
    # Aggregate over next three months
    agg_sd: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for industry, mstats in sd.items():
        for m in months:
            stats = mstats.get(m)
            if not stats:
                continue
            for key in ["predicted_demand", "available_hosts", "available_bilingual", "available_exp3plus", "shortage_total"]:
                agg_sd[industry][key] += int(stats.get(key, 0))
    for industry, agg in sorted(agg_sd.items()):
        print(
            f"- {industry} {month_range_label(months)}: demand={agg['predicted_demand']}, available={agg['available_hosts']} "
            f"(bilingual={agg['available_bilingual']}, 3+yr={agg['available_exp3plus']}), shortage={agg['shortage_total']}"
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