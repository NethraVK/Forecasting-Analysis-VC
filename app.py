import streamlit as st
import pandas as pd
from pymongo import MongoClient
import forecasting

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["event_hosting_platform"]

st.title("Event Forecasting Dashboard")

# Fixed horizon (model/code supports up to 3)
horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)
months = forecasting.next_n_months_from_today(horizon)

# Compute forecasts
ind_fc, _ = forecasting.forecast_event_volume(db, horizon_months=horizon)
sd = forecasting.supply_demand_forecast(db, horizon_months=horizon)
grouped_industry = forecasting.forecast_event_volume_by_industry_grouped(db, horizon_months=horizon)
season = forecasting.seasonal_event_volume_summary(db, horizon_months=horizon)

tab1, tab2, tab3, tab4 = st.tabs([
    "Summary", "Events by Industry", "Supply–Demand", "Onboarding"
])

with tab1:
    all_map = ind_fc.get("All", {})
    total_events = sum(int(all_map.get(m, 0)) for m in months)
    avg_events = round(total_events / max(1, len(months)), 1)
    per_month = {m: int(all_map.get(m, 0)) for m in months}
    c1, c2 = st.columns(2)
    c1.metric(label=f"Next {len(months)} months total", value=total_events)
    c2.metric(label="Average per month", value=avg_events)
    st.bar_chart(pd.DataFrame.from_dict(per_month, orient="index", columns=["events"])) 
    if season:
        st.caption(f"Season {season['season']}: total={season['forecast_total_events']} vs hist-3mo-avg={season['historical_3mo_avg']} ({season['change_pct']}) → {season['signal']}")

with tab2:
    # Industry forecast (grouped over next N months)
    ind_df = pd.DataFrame.from_dict(grouped_industry, orient="index", columns=["events_next_window"]).sort_values("events_next_window", ascending=False)
    st.subheader(f"Events by Industry (next {horizon} months)")
    st.bar_chart(ind_df)
    # Removed per-month breakdown table per request

with tab3:
    # Aggregate next N months per industry for supply–demand
    agg_sd = {}
    for industry, mstats in sd.items():
        agg = {"predicted_demand": 0, "available_hosts": 0, "available_bilingual": 0, "available_exp3plus": 0, "shortage_total": 0}
        for m in months:
            s = mstats.get(m)
            if not s:
                continue
            for k in agg.keys():
                agg[k] += int(s.get(k, 0))
        agg_sd[industry] = agg
    if agg_sd:
        sd_df = pd.DataFrame.from_dict(agg_sd, orient="index")
        st.subheader(f"Supply–Demand by Industry (next {horizon} months)")
        st.bar_chart(sd_df[["predicted_demand", "available_hosts"]])

with tab4:
    st.subheader(f"Onboarding Recommendations (next {horizon} months)")
    # Build tabular view of gaps by industry and month
    gap_rows = []
    for industry, month_stats in sd.items():
        for m in months:
            stats = month_stats.get(m, {})
            demand = int(stats.get("predicted_demand", 0))
            avail = int(stats.get("available_hosts", 0))
            exp3_avail = int(stats.get("available_exp3plus", 0))
            shortage = max(0, demand - avail)
            required_exp3 = max(1, demand // 4) if demand > 0 else 0
            exp3_gap = max(0, required_exp3 - exp3_avail)
            gap_rows.append({
                "industry": industry,
                "month": m,
                "predicted_demand": demand,
                "available_hosts": avail,
                "shortage": shortage,
                "required_exp3plus": required_exp3,
                "available_exp3plus": exp3_avail,
                "exp3plus_gap": exp3_gap,
            })
    if not gap_rows:
        st.write("No critical gaps detected.")
    else:
        gap_df = pd.DataFrame(gap_rows)
        # Charts only (no tables)
        st.markdown("**Shortage by industry (stacked by month)**")
        shortage_pivot = gap_df.pivot(index="industry", columns="month", values="shortage").fillna(0)
        st.bar_chart(shortage_pivot)

        st.markdown("**Experienced (3+ years) gap by industry**")
        skill_gaps = (
            gap_df.groupby("industry")[ ["exp3plus_gap"] ].sum().sort_values("exp3plus_gap", ascending=False)
        )
        st.bar_chart(skill_gaps)
