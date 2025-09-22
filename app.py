import streamlit as st
import pandas as pd
import altair as alt
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
sd_monthly = forecasting.supply_demand_forecast_monthly(db, horizon_months=horizon)
season = forecasting.seasonal_event_volume_summary(db, horizon_months=horizon)

tab1, tab3, tab4 = st.tabs([
    "Summary", "Supply–Demand", "Onboarding"
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

with tab3:
    # Monthly supply–demand view (no industry)
    st.subheader(f"Supply–Demand (monthly) — next {horizon} months")
    if sd_monthly:
        sd_rows = []
        for m in months:
            stats = sd_monthly.get(m, {})
            sd_rows.append({
                "month": m,
                "predicted_demand": int(stats.get("predicted_demand", 0)),
                "predicted_demand_lower": int(stats.get("predicted_demand_lower", 0)),
                "predicted_demand_upper": int(stats.get("predicted_demand_upper", 0)),
                "available_hosts": int(stats.get("available_hosts", 0)),
                "available_hosts_lower": int(stats.get("available_hosts_lower", 0)),
                "available_hosts_upper": int(stats.get("available_hosts_upper", 0)),
                "shortage_total": int(stats.get("shortage_total", 0)),
            })
        sd_df_m = pd.DataFrame(sd_rows)

        # Demand with confidence band (Altair)
        st.markdown("**Demand forecast with confidence band**")
        demand_chart = alt.Chart(sd_df_m).transform_calculate(
            month_str='datum.month'
        ).encode(
            x=alt.X('month_str:N', title='Month')
        )
        band = demand_chart.mark_area(opacity=0.2, color='#1f77b4').encode(
            y=alt.Y('predicted_demand_lower:Q', title='Hosts'),
            y2='predicted_demand_upper:Q'
        )
        mean_line = demand_chart.mark_line(color='#1f77b4').encode(
            y='predicted_demand:Q'
        )
        avail_line = alt.Chart(sd_df_m).mark_line(color='#ff7f0e').encode(
            x=alt.X('month:N', title='Month'),
            y=alt.Y('available_hosts:Q', title='Hosts')
        )
        st.altair_chart(band + mean_line + avail_line, use_container_width=True)

        # Shortage chart
        st.markdown("**Shortage (total) by month**")
        st.bar_chart(sd_df_m.set_index("month")[ ["shortage_total"] ])

with tab4:
    st.subheader(f"Onboarding Recommendations (next {horizon} months)")
    # Monthly-only gaps and charts
    if not sd_monthly:
        st.write("No critical gaps detected.")
    else:
        # Build monthly gaps
        gap_rows = []
        for m in months:
            stats = sd_monthly.get(m, {})
            demand = int(stats.get("predicted_demand", 0))
            avail = int(stats.get("available_hosts", 0))
            shortage = max(0, demand - avail)
            gap_rows.append({
                "month": m,
                "shortage": shortage,
            })
        gap_df = pd.DataFrame(gap_rows).set_index("month")
        st.markdown("**Shortage by month**")
        st.bar_chart(gap_df[["shortage"]])
