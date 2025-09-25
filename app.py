import streamlit as st
import pandas as pd
import altair as alt
from pymongo import MongoClient
import forecasting

st.sidebar.header("Data Source")
json_path = st.sidebar.text_input(
    "Optional JSON path (use generated dummy JSON to bypass Mongo)",
    value="",
    help="If provided, the app will read from this JSON via the same logic as the CLI."
)
if json_path:
    db = forecasting.get_db_from_json(json_path)
    st.sidebar.success("Using JSON data source")
else:
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["event_hosting_platform"]
    st.sidebar.info("Using MongoDB at mongodb://localhost:27017/")

st.title("Event Forecasting Dashboard")

# Fixed horizon (model/code supports up to 3)
horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)
months = forecasting.next_n_months_from_today(horizon)

# What-if controls
st.sidebar.header("What-if Scenarios")
extra_hosts = st.sidebar.number_input("Additional hosts available per month", min_value=-500, max_value=5000, value=0, step=10)
event_growth_pct = st.sidebar.slider("Event growth %", -50, 100, 0, help="Applies to forecast demand")

# Skill filter (e.g., language)
st.sidebar.header("Skill Filter")
try:
    hosts_docs = list(db["EventHostUser"].find({}))
    language_counts = {}
    for h in hosts_docs:
        for lang in (h.get("languages") or []):
            language_counts[lang] = language_counts.get(lang, 0) + 1
    languages = sorted(language_counts.keys())
except Exception:
    hosts_docs = []
    language_counts = {}
    languages = []

skill_selected = st.sidebar.selectbox("Language (optional)", options=["All"] + languages, index=0)

# Compute forecasts
ind_fc, _ = forecasting.forecast_event_volume(db, horizon_months=horizon)
sd_monthly = forecasting.supply_demand_forecast_monthly(db, horizon_months=horizon)
season = forecasting.seasonal_event_volume_summary(db, horizon_months=horizon)

# Align displayed months with supply–demand results if available
months_display = list(sd_monthly.keys()) if sd_monthly else months

tab1, tab3, tab4 = st.tabs([
    "Summary", "Hostess Demand", "Onboarding"
])

with tab1:
    all_map = ind_fc.get("All", {})
    total_events = sum(int(all_map.get(m, 0)) for m in months_display)
    avg_events = round(total_events / max(1, len(months_display)), 1)
    per_month = {m: int(all_map.get(m, 0)) for m in months_display}
    c1, c2 = st.columns(2)
    c1.metric(label=f"Next {len(months_display)} months total", value=total_events)
    c2.metric(label="Average per month", value=avg_events)
    st.bar_chart(pd.DataFrame.from_dict(per_month, orient="index", columns=["events"])) 
    if season:
        st.caption(f"Season {season['season']}: total={season['forecast_total_events']} vs hist-3mo-avg={season['historical_3mo_avg']} ({season['change_pct']}) → {season['signal']}")

with tab3:
    # Monthly supply–demand view (no industry)
    st.subheader(f"Hostess Demand (monthly) — next {horizon} months")
    if sd_monthly:
        sd_rows = []
        for m in months_display:
            stats = sd_monthly.get(m, {})
            d = int(stats.get("predicted_demand", 0))
            a = int(stats.get("available_hosts", 0))
            d_adj = int(round(d * (1 + event_growth_pct / 100.0)))
            # Apply skill filter capacity reduction if selected
            if skill_selected != "All" and language_counts:
                # capacity share proportional to number of hosts with that language
                lang_share = min(1.0, max(0.0, language_counts.get(skill_selected, 0) / max(1, len(hosts_docs))))
                a = int(round(a * lang_share))
            a_adj = max(0, a + int(extra_hosts))
            shortage_adj = max(0, d_adj - a_adj)
            sd_rows.append({
                "month": m,
                "predicted_demand": d_adj,
                "available_hosts": a_adj,
                "shortage_total": shortage_adj,
            })
        sd_df_m = pd.DataFrame(sd_rows)

        # Removed utilization calculation (no longer displayed)

        # Demand vs Availability with Shortage highlighted (stacked bars)
        st.markdown("**Demand vs Availability (shortage highlighted in red)**")
        sd_df_plot = sd_df_m.copy()
        sd_df_plot["available_segment"] = sd_df_plot[["available_hosts", "predicted_demand"]].min(axis=1)
        sd_df_plot["shortage_segment"] = (sd_df_plot["predicted_demand"] - sd_df_plot["available_hosts"]).clip(lower=0)
        base = alt.Chart(sd_df_plot).encode(x=alt.X('month:N', title='Month'))
        bars_avail = base.mark_bar(color='#ff7f0e').encode(
            y=alt.Y('available_segment:Q', title='Hosts'),
            tooltip=['month', 'available_hosts']
        )
        bars_short = base.mark_bar(color='#d62728').encode(
            y='shortage_segment:Q',
            tooltip=['month', 'shortage_total']
        )
        st.altair_chart(bars_avail + bars_short, use_container_width=True)

        # Shortage chart
        st.markdown("**Shortage (total) by month**")
        st.bar_chart(sd_df_m.set_index("month")[ ["shortage_total"] ])

        # Narrative summary
        total_d = int(sd_df_m["predicted_demand"].sum())
        total_a = int(sd_df_m["available_hosts"].sum())
        total_short = max(0, total_d - total_a)
        st.markdown(
            f"We expect {total_d} host requests across the next {len(sd_df_m)} months, with {total_a} hosts available, leaving a shortfall of {total_short}."
        )

        # Removed utilization trend chart

        # Removed "% months fully staffed" metric and related calculations

with tab4:
    st.subheader(f"Onboarding Recommendations (next {horizon} months)")
    # Monthly-only gaps and charts
    if not sd_monthly:
        st.write("No critical gaps detected.")
    else:
        # Removed duplicate "Shortage by month" chart (already shown in Hostess Demand tab)

        # Historical vs Forecast events (simple trend)
        df_hist = forecasting.build_monthly_df(db=db, scope='industry')
        monthly_hist = forecasting.prepare_global_monthly(df_hist) if not df_hist.empty else pd.DataFrame()
        if not monthly_hist.empty:
            hist_tail = monthly_hist.tail(24)[["ym", "y_event_count"]].rename(columns={"ym": "month", "y_event_count": "events"})
            future_df = pd.DataFrame({
                "month": months_display,
                "events": [per_month.get(m, 0) for m in months_display]
            })
            hist_tail["type"] = "historical"
            future_df["type"] = "forecast"
            trend_df = pd.concat([hist_tail, future_df], ignore_index=True)
            chart = alt.Chart(trend_df).mark_line().encode(
                x=alt.X('month:N', title='Month'),
                y=alt.Y('events:Q', title='Events'),
                color='type:N'
            )
            st.markdown("**Historical vs Forecast events**")
            st.altair_chart(chart, use_container_width=True)

        # Busiest months heatmap by demand
        st.markdown("**Busiest months heatmap (by demand)**")
        heat_df = sd_df_m.copy()
        heat_df["year"] = heat_df["month"].str.slice(0, 4)
        heat_df["mon"] = heat_df["month"].str.slice(5, 7)
        heat = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X('mon:N', title='Month'),
            y=alt.Y('year:N', title='Year'),
            color=alt.Color('predicted_demand:Q', title='Demand', scale=alt.Scale(scheme='reds')),
            tooltip=['month', 'predicted_demand']
        )
        st.altair_chart(heat, use_container_width=True)
