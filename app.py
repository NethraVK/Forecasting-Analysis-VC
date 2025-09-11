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
months = forecasting.next_three_months_from_today()

# Compute forecasts
ind_fc, loc_fc = forecasting.forecast_event_volume(db, horizon_months=horizon)
sd = forecasting.supply_demand_forecast(db, horizon_months=horizon)
grouped_industry = forecasting.forecast_event_volume_by_industry_grouped(db, horizon_months=horizon)
season = forecasting.seasonal_event_volume_summary(db, horizon_months=horizon)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Summary", "Events by Industry", "Events by Location", "Supply–Demand", "Onboarding"
])

with tab1:
    all_map = ind_fc.get("All", {})
    total_events = sum(int(all_map.get(m, 0)) for m in months)
    avg_events = round(total_events / max(1, len(months)), 1)
    per_month = {m: int(all_map.get(m, 0)) for m in months}
    st.metric(label=f"Next {len(months)} months total", value=total_events)
    st.metric(label="Average per month", value=avg_events)
    st.bar_chart(pd.DataFrame.from_dict(per_month, orient="index", columns=["events"])) 
    if season:
        st.caption(f"Season {season['season']}: total={season['forecast_total_events']} vs hist-3mo-avg={season['historical_3mo_avg']} ({season['change_pct']}) → {season['signal']}")

with tab2:
    # Industry forecast (grouped over next N months)
    ind_df = pd.DataFrame.from_dict(grouped_industry, orient="index", columns=["events_next_window"]).sort_values("events_next_window", ascending=False)
    st.subheader(f"Events by Industry (next {horizon} months)")
    st.bar_chart(ind_df)
    # Per-month industry breakdown table
    st.caption("Per-month breakdown")
    pm_rows = []
    for industry, m_to_v in ind_fc.items():
        if industry == "All":
            continue
        for m in months:
            pm_rows.append({"industry": industry, "month": m, "events": int(m_to_v.get(m, 0))})
    if pm_rows:
        st.dataframe(pd.DataFrame(pm_rows))

with tab3:
    # Location per-month forecast
    loc_rows = []
    for loc, m_to_v in loc_fc.items():
        for m in months:
            loc_rows.append({"location": loc, "month": m, "events": int(m_to_v.get(m, 0))})
    if loc_rows:
        loc_df = pd.DataFrame(loc_rows)
        st.subheader(f"Events by Location (next {horizon} months)")
        st.bar_chart(loc_df.pivot(index="month", columns="location", values="events").fillna(0))

with tab4:
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
        st.dataframe(sd_df)
        st.bar_chart(sd_df[["predicted_demand", "available_hosts"]])

with tab5:
    recs = forecasting.host_onboarding_needs_from_forecast(sd, months)
    st.subheader("Onboarding Recommendations (next 3 months)")
    if not recs:
        st.write("No critical gaps detected.")
    else:
        for r in recs:
            st.write(f"- {r}")
