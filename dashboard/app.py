"""
EVolve: Intelligent EV Route Charging Optimizer
Streamlit Interactive Dashboard
"""

import os
import sys

# Ensure project root directory is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import streamlit as st
from streamlit_folium import st_folium

from config import (
    ORIGIN_NAME,
    DESTINATION_NAME,
    TOTAL_ROUTE_DISTANCE_KM,
    BATTERY_BUFFER_PERCENT,
    DESTINATION_RESERVE_PERCENT,
    DEFAULT_CONSUMPTION_KWH_PER_100KM,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_CURRENT_SOC_PERCENT,
    DEFAULT_MAX_CHARGING_POWER_KW,
    DEFAULT_CONNECTOR_TYPE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    GOOGLE_MAPS_API_KEY,
)

from core.ev_model import EV
from core.station_model import Station
from data.route_engine import get_route, get_route_waypoints, get_route_distance
from data.station_finder import get_stations_along_route
from data.telemetry_sim import simulate_telemetry, update_single_station
from experiments.race_runner import run_race, run_experiments


# ==============================================================================
# STREAMLIT PAGE & THEME CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="EVolve - Intelligent EV Route Charging Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern glassmorphic aesthetics
st.markdown("""
<style>
    /* Dark Theme Accent Adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }

    .badge-feasible {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-infeasible {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# INITIALIZE SESSION STATE & SIMULATION DATA
# ==============================================================================

if "stations" not in st.session_state:
    raw_stations = get_stations_along_route()
    st.session_state["stations"] = simulate_telemetry(raw_stations, current_hour=14.0, seed=42)

if "race_results" not in st.session_state:
    st.session_state["race_results"] = None

if "offline_station_id" not in st.session_state:
    st.session_state["offline_station_id"] = "None"


# ==============================================================================
# HEADER SECTION
# ==============================================================================

st.markdown('<div class="main-title">⚡ EVolve — Intelligent EV Route Charging Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Real-time corridor optimization racing <b>Classical Dynamic Programming</b> against <b>Quantum-Inspired QUBO (D-Wave Annealing)</b> along the <b>{ORIGIN_NAME} ➔ {DESTINATION_NAME}</b> highway corridor ({TOTAL_ROUTE_DISTANCE_KM:.0f} km).</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 1: SIDEBAR INPUT PANEL
# ==============================================================================

st.sidebar.markdown("### 🚗 EV Parameters & Configuration")

with st.sidebar.expander("📍 Route & Google Maps Settings", expanded=True):
    st.text_input("Start Location", value=ORIGIN_NAME, disabled=True)
    st.text_input("Destination", value=DESTINATION_NAME, disabled=True)
    st.text_input("Corridor Distance", value=f"{TOTAL_ROUTE_DISTANCE_KM:.1f} km (NH48 / NH44 / NH544)", disabled=True)
    gmaps_key = st.text_input(
        "🔑 Google Maps API Key",
        value=GOOGLE_MAPS_API_KEY,
        type="password",
        help="Enter your Google Maps Platform API key to fetch real-time live traffic routes and Google Maps tiles.",
    )
    map_tile_provider = st.selectbox("Map Style", options=["Google Maps", "CartoDB Dark", "OpenStreetMap"], index=0)

with st.sidebar.expander("🔋 EV Battery & Hardware", expanded=True):
    battery_cap = st.slider("Battery Capacity (kWh)", min_value=40.0, max_value=100.0, value=DEFAULT_BATTERY_CAPACITY_KWH, step=5.0)
    current_soc = st.slider("Current State of Charge (%)", min_value=10.0, max_value=90.0, value=DEFAULT_CURRENT_SOC_PERCENT, step=5.0)
    consumption_rate = st.slider("Consumption Rate (kWh/100km)", min_value=4.0, max_value=25.0, value=DEFAULT_CONSUMPTION_KWH_PER_100KM, step=0.5)
    max_charge_power = st.select_slider("Max Charging Power (kW)", options=[30.0, 50.0, 60.0, 120.0, 150.0, 240.0], value=DEFAULT_MAX_CHARGING_POWER_KW)
    connector_type = st.selectbox("Connector Type", options=["CCS2", "Type2", "CHAdeMO"], index=0)

with st.sidebar.expander("🎯 Multi-Objective Trade-off Weights", expanded=True):
    alpha_weight = st.slider("Alpha (Weight for Journey Time)", min_value=0.0, max_value=1.0, value=DEFAULT_ALPHA, step=0.05)
    beta_weight = round(1.0 - alpha_weight, 2)
    st.write(f"**Beta (Weight for Charging Cost):** `{beta_weight}`")

# Construct EV instance
user_ev = EV(
    id="User-EV",
    battery_capacity_kwh=battery_cap,
    current_soc_percent=current_soc,
    consumption_kwh_per_100km=consumption_rate,
    max_charging_power_kw=max_charge_power,
    connector_type=connector_type,
)

st.sidebar.markdown("---")
btn_run_race = st.sidebar.button("🏎️ Run Optimization Race", type="primary", use_container_width=True)
btn_resimulate = st.sidebar.button("🔄 Re-simulate Station Telemetry", use_container_width=True)

if btn_resimulate:
    raw_st = get_stations_along_route()
    st.session_state["stations"] = simulate_telemetry(raw_st, current_hour=np.random.uniform(7.0, 22.0))
    st.session_state["race_results"] = None
    st.sidebar.success("Updated telemetry!")

# Run Race logic when clicked or auto-initialized
if btn_run_race or st.session_state["race_results"] is None:
    with st.spinner("Racing Classical DP vs Quantum QUBO solvers in parallel..."):
        st.session_state["race_results"] = run_race(
            user_ev, st.session_state["stations"], alpha=alpha_weight, beta=beta_weight
        )


# Extract active race results
race_data = st.session_state["race_results"]
dp_res = race_data["classical"]
qubo_res = race_data["quantum"]
winners = race_data["winners"]


# ==============================================================================
# SECTION 2: INTERACTIVE ROUTE MAP (Folium & Google Maps Platform Tiles)
# ==============================================================================

st.markdown("### 🗺️ Route Corridor Map & Station Telemetry")

route_points = get_route(api_key=gmaps_key)
map_center = [12.05, 78.50]  # Centered along Tamil Nadu highway spine

if map_tile_provider == "Google Maps":
    google_tile_url = f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={gmaps_key}" if gmaps_key else "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
    m = folium.Map(location=map_center, zoom_start=8, tiles=google_tile_url, attr="Google Maps")
elif map_tile_provider == "OpenStreetMap":
    m = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")
else:
    m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB dark_matter")

# Add highway route line string
polyline_coords = [(pt[0], pt[1]) for pt in route_points]
folium.PolyLine(
    polyline_coords, color="#1E88E5", weight=5, opacity=0.85, tooltip=f"NH Highway Corridor ({TOTAL_ROUTE_DISTANCE_KM:.0f} km)"
).add_to(m)

# Origin and Destination Markers
folium.Marker(
    [route_points[0][0], route_points[0][1]],
    popup=f"<b>Origin:</b> {ORIGIN_NAME}",
    icon=folium.Icon(color="green", icon="play"),
).add_to(m)

folium.Marker(
    [route_points[-1][0], route_points[-1][1]],
    popup=f"<b>Destination:</b> {DESTINATION_NAME}",
    icon=folium.Icon(color="red", icon="flag"),
).add_to(m)

# Identify stops chosen by algorithms
dp_stops = set(dp_res.get("stops", []))
qubo_stops = set(qubo_res.get("stops", []))

# Add Station Markers
for st_obj in st.session_state["stations"]:
    is_dp = st_obj.id in dp_stops
    is_qubo = st_obj.id in qubo_stops

    # Marker color rules
    if not st_obj.is_operational:
        marker_color = "red"
        status_str = "<span style='color:red;'><b>OFFLINE</b></span>"
    elif st_obj.available_slots == 0:
        marker_color = "orange"
        status_str = "<span style='color:orange;'><b>BUSY (0 Slots)</b></span>"
    else:
        marker_color = "green"
        status_str = "<span style='color:green;'><b>AVAILABLE</b></span>"

    # Icon highlight for race selections
    if is_dp and is_qubo:
        icon_type = "star"
        icon_color = "purple"
        tag = " 🌟 Chosen by Both Solvers!"
    elif is_dp:
        icon_type = "flash"
        icon_color = "orange"
        tag = " 🟧 Chosen by Classical DP"
    elif is_qubo:
        icon_type = "bolt"
        icon_color = "purple"
        tag = " 🟪 Chosen by Quantum QUBO"
    else:
        icon_type = "info-sign"
        icon_color = marker_color
        tag = ""

    popup_html = f"""
    <div style='font-family: sans-serif; min-width: 220px;'>
        <h4 style='margin-bottom:4px; color:#1E88E5;'>{st_obj.name}{tag}</h4>
        <b>Status:</b> {status_str}<br/>
        <b>Distance:</b> {st_obj.distance_from_start_km:.1f} km from start<br/>
        <b>Charger:</b> {st_obj.charger_type} ({st_obj.max_power_kw:.0f} kW)<br/>
        <b>Bays:</b> {st_obj.available_slots} / {st_obj.num_slots} available<br/>
        <b>Queue Wait:</b> {st_obj.wait_time_minutes:.0f} mins<br/>
        <b>Dynamic Price:</b> ₹{st_obj.price_per_kwh:.2f} / kWh<br/>
        <b>Connectors:</b> {', '.join(st_obj.connector_types)}<br/>
    </div>
    """

    folium.Marker(
        [st_obj.lat, st_obj.lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{st_obj.name} ({st_obj.distance_from_start_km:.0f} km)",
        icon=folium.Icon(color=icon_color, icon=icon_type, prefix="fa" if "fa" in icon_type else "glyphicon"),
    ).add_to(m)

# Render Folium Map in Streamlit
st_folium(m, width=1300, height=450)


# ==============================================================================
# SECTION 3: LIVE ALGORITHM RACE PANEL
# ==============================================================================

st.markdown("### 🏎️ Live Algorithm Race Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🟧 Classical Dynamic Programming (DP)</div>', unsafe_allow_html=True)
    
    dp_obj = dp_res["obj_details"]
    feas_badge = '<span class="badge-feasible">✓ FEASIBLE</span>' if dp_res["is_feasible"] else '<span class="badge-infeasible">⚠ INFEASIBLE</span>'
    st.markdown(f"**Feasibility:** {feas_badge}", unsafe_allow_html=True)
    
    st.markdown(f"- **Total Journey Time:** `{dp_obj['total_time_minutes']:.1f} mins` ({dp_obj['total_time_minutes']/60.0:.2f} hrs)")
    st.markdown(f"- **Total Charging Cost:** `₹{dp_obj['total_cost_inr']:.2f}`")
    st.markdown(f"- **Driving / Wait / Plug Time:** `{dp_obj['travel_time']}m` / `{dp_obj['waiting_time']}m` / `{dp_obj['charging_time']}m`")
    st.markdown(f"- **Combined Score ($\alpha T + \beta C$):** `{dp_res['objective_score']:.4f}`")
    st.markdown(f"- **Solver Runtime:** `{dp_res['execution_time_sec']:.4f} sec` (Nodes explored: `{dp_res['nodes_explored']}`)")
    
    st.markdown("**Stops Chosen:**")
    if dp_res["stops"]:
        for sid in dp_res["stops"]:
            st_name = next((s.name for s in st.session_state["stations"] if s.id == sid), sid)
            charge_kwh = dp_res["charges"].get(sid, 0.0)
            st.markdown(f"  - 📍 **{st_name}**: Charged `+{charge_kwh:.1f} kWh`")
    else:
        st.write("  - *No intermediate stops required.*")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🟪 Quantum-Inspired QUBO (D-Wave Annealing)</div>', unsafe_allow_html=True)
    
    qubo_obj = qubo_res["obj_details"]
    q_feas_badge = '<span class="badge-feasible">✓ FEASIBLE</span>' if qubo_res["is_feasible"] else '<span class="badge-infeasible">⚠ INFEASIBLE</span>'
    st.markdown(f"**Feasibility:** {q_feas_badge} (Annealing Feasibility Rate: `{qubo_res['feasibility_rate_percent']:.1f}%`)", unsafe_allow_html=True)
    
    st.markdown(f"- **Total Journey Time:** `{qubo_obj['total_time_minutes']:.1f} mins` ({qubo_obj['total_time_minutes']/60.0:.2f} hrs)")
    st.markdown(f"- **Total Charging Cost:** `₹{qubo_obj['total_cost_inr']:.2f}`")
    st.markdown(f"- **Driving / Wait / Plug Time:** `{qubo_obj['travel_time']}m` / `{qubo_obj['waiting_time']}m` / `{qubo_obj['charging_time']}m`")
    st.markdown(f"- **Combined Score ($\alpha T + \beta C$):** `{qubo_res['objective_score']:.4f}`")
    st.markdown(f"- **Solver Runtime:** `{qubo_res['execution_time_sec']:.4f} sec` (Reads sampled: `{qubo_res['num_reads']}`)")
    
    st.markdown("**Stops Chosen:**")
    if qubo_res["stops"]:
        for sid in qubo_res["stops"]:
            st_name = next((s.name for s in st.session_state["stations"] if s.id == sid), sid)
            charge_kwh = qubo_res["charges"].get(sid, 0.0)
            st.markdown(f"  - 📍 **{st_name}**: Charged `+{charge_kwh:.1f} kWh`")
    else:
        st.write("  - *No intermediate stops required.*")

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: COMPARISON TABLE
# ==============================================================================

st.markdown("### 📊 Head-to-Head Performance Comparison")

comp_df = pd.DataFrame([
    {
        "Performance Metric": "Total Cost (₹)",
        "Classical DP": f"₹{dp_obj['total_cost_inr']:.2f}",
        "Quantum-Inspired QUBO": f"₹{qubo_obj['total_cost_inr']:.2f}",
        "Winner": f"🏆 {winners['cost']}",
    },
    {
        "Performance Metric": "Total Journey Duration",
        "Classical DP": f"{dp_obj['total_time_minutes']:.1f} mins",
        "Quantum-Inspired QUBO": f"{qubo_obj['total_time_minutes']:.1f} mins",
        "Winner": f"🏆 {winners['time']}",
    },
    {
        "Performance Metric": "Number of Stops",
        "Classical DP": len(dp_res["stops"]),
        "Quantum-Inspired QUBO": len(qubo_res["stops"]),
        "Winner": "Tie" if len(dp_res["stops"]) == len(qubo_res["stops"]) else f"🏆 {winners['overall']}",
    },
    {
        "Performance Metric": "Algorithm Runtime",
        "Classical DP": f"{dp_res['execution_time_sec']:.4f} s",
        "Quantum-Inspired QUBO": f"{qubo_res['execution_time_sec']:.4f} s",
        "Winner": f"⚡ {winners['runtime_speed']}",
    },
    {
        "Performance Metric": "Feasible Solution?",
        "Classical DP": "Yes" if dp_res["is_feasible"] else "No",
        "Quantum-Inspired QUBO": "Yes" if qubo_res["is_feasible"] else "No",
        "Winner": "Both Feasible" if (dp_res["is_feasible"] and qubo_res["is_feasible"]) else "Classical DP",
    },
    {
        "Performance Metric": "Constraint Violations",
        "Classical DP": len(dp_res["violations"]),
        "Quantum-Inspired QUBO": len(qubo_res["violations"]),
        "Winner": "Zero Violations",
    },
    {
        "Performance Metric": "Combined Objective Score",
        "Classical DP": f"{dp_res['objective_score']:.4f}",
        "Quantum-Inspired QUBO": f"{qubo_res['objective_score']:.4f}",
        "Winner": f"⭐ {winners['overall']}",
    },
])

st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 5: ANALYTICAL PLOTLY CHARTS
# ==============================================================================

st.markdown("### 📈 Analytics & Diagnostic Profiles")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 🔋 Battery State of Charge (SoC %) Profile Along Route")
    
    fig_soc = go.Figure()

    # Classical DP Profile
    dp_prof = dp_res["battery_profile"]
    dp_kms = [p["km"] for p in dp_prof]
    dp_socs = [p["soc"] for p in dp_prof]
    fig_soc.add_trace(go.Scatter(
        x=dp_kms, y=dp_socs, mode="lines+markers", name="Classical DP",
        line=dict(color="#FF9800", width=3), marker=dict(size=7)
    ))

    # Quantum QUBO Profile
    qubo_prof = qubo_res["battery_profile"]
    qubo_kms = [p["km"] for p in qubo_prof]
    qubo_socs = [p["soc"] for p in qubo_prof]
    fig_soc.add_trace(go.Scatter(
        x=qubo_kms, y=qubo_socs, mode="lines+markers", name="Quantum QUBO",
        line=dict(color="#9C27B0", width=3, dash="dash"), marker=dict(size=7)
    ))

    # Add 10% Buffer and 15% Reserve horizontal threshold lines
    fig_soc.add_hline(y=BATTERY_BUFFER_PERCENT, line_dash="dot", line_color="red", annotation_text="10% Buffer Threshold")
    fig_soc.add_hline(y=DESTINATION_RESERVE_PERCENT, line_dash="dot", line_color="yellow", annotation_text="15% Destination Reserve")

    fig_soc.update_layout(
        xaxis_title="Distance from Chennai (km)",
        yaxis_title="State of Charge (SoC %)",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_soc, use_container_width=True)

with chart_col2:
    st.markdown("#### ⏳ Journey Duration Breakdown (Minutes)")
    
    labels = ["Driving Time", "Queue Wait Time", "Plug Charge Time"]
    dp_vals = [dp_obj["travel_time"], dp_obj["waiting_time"], dp_obj["charging_time"]]
    qubo_vals = [qubo_obj["travel_time"], qubo_obj["waiting_time"], qubo_obj["charging_time"]]

    fig_pie = go.Figure()
    fig_pie.add_trace(go.Bar(x=labels, y=dp_vals, name="Classical DP", marker_color="#FF9800"))
    fig_pie.add_trace(go.Bar(x=labels, y=qubo_vals, name="Quantum QUBO", marker_color="#9C27B0"))

    fig_pie.update_layout(
        barmode="group",
        yaxis_title="Minutes",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("#### 🔌 Station Charger Power & Dynamic Pricing")
    
    st_names = [s.name.split(" - ")[-1] for s in st.session_state["stations"]]
    st_powers = [s.max_power_kw for s in st.session_state["stations"]]
    st_prices = [s.price_per_kwh for s in st.session_state["stations"]]

    fig_st = go.Figure()
    fig_st.add_trace(go.Bar(x=st_names, y=st_powers, name="Max Charger Power (kW)", marker_color="#38bdf8"))
    fig_st.add_trace(go.Scatter(x=st_names, y=st_prices, name="Price (₹/kWh)", yaxis="y2", line=dict(color="#f43f5e", width=3)))

    fig_st.update_layout(
        yaxis=dict(title="Max Power (kW)"),
        yaxis2=dict(title="Price (₹/kWh)", overlaying="y", side="right"),
        template="plotly_dark",
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=30, b=80),
    )
    st.plotly_chart(fig_st, use_container_width=True)

with chart_col4:
    st.markdown("#### ⚡ Benchmark Scalability: Problem Size vs Runtime")
    
    if "exp_df" not in st.session_state:
        st.session_state["exp_df"] = run_experiments(user_ev, alpha=alpha_weight, beta=beta_weight, runs_per_size=3)

    exp_df = st.session_state["exp_df"]
    
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(
        x=exp_df["Problem Size (Stations)"], y=exp_df["Classical Mean Time (s)"],
        mode="lines+markers", name="Classical DP", line=dict(color="#FF9800", width=3)
    ))
    fig_exp.add_trace(go.Scatter(
        x=exp_df["Problem Size (Stations)"], y=exp_df["Quantum Mean Time (s)"],
        mode="lines+markers", name="Quantum QUBO", line=dict(color="#9C27B0", width=3)
    ))

    fig_exp.update_layout(
        xaxis_title="Number of Stations Along Corridor",
        yaxis_title="Execution Time (seconds)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_exp, use_container_width=True)


# ==============================================================================
# SECTION 6: DYNAMIC EVENT RE-OPTIMIZATION SIMULATOR
# ==============================================================================

st.markdown("### 🚨 Dynamic Event Re-Optimization Simulator")

st.info("Simulate live real-world disruption by taking a station offline en route (e.g. power grid trip or hardware failure). Watch the optimization algorithms instantly re-route and recalculate!")

event_col1, event_col2 = st.columns([3, 1])

with event_col1:
    st_options = ["None"] + [f"{s.id}: {s.name}" for s in st.session_state["stations"]]
    selected_offline_str = st.selectbox("Select Station to Suffer Emergency Downtime Outage:", options=st_options, index=0)

with event_col2:
    btn_trigger_event = st.button("🚨 Trigger Outage Event", type="secondary", use_container_width=True)

if btn_trigger_event and selected_offline_str != "None":
    offline_id = selected_offline_str.split(":")[0]
    st.session_state["offline_station_id"] = offline_id
    
    # Update target station in session state
    st.session_state["stations"] = [
        update_single_station(s, is_operational=False) if s.id == offline_id else s
        for s in st.session_state["stations"]
    ]

    # Re-run race
    st.session_state["race_results"] = run_race(
        user_ev, st.session_state["stations"], alpha=alpha_weight, beta=beta_weight
    )
    st.success(f"Station {selected_offline_str} marked OFFLINE! Re-optimization complete!")
    st.rerun()


# ==============================================================================
# SECTION 7: AUTOMATED REPORT GENERATOR
# ==============================================================================

st.markdown("### 📄 Automated Optimization Summary Report")

report_text = f"""================================================================================
EVOLVE: INTELLIGENT EV ROUTE CHARGING OPTIMIZER REPORT
================================================================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Corridor: {ORIGIN_NAME} -> {DESTINATION_NAME} ({TOTAL_ROUTE_DISTANCE_KM:.1f} km)

1. EV SPECIFICATIONS & STATE
--------------------------------------------------------------------------------
- Vehicle Model ID: {user_ev.id}
- Battery Capacity: {user_ev.battery_capacity_kwh:.1f} kWh
- Starting SoC: {user_ev.current_soc_percent:.1f}% ({user_ev.get_current_kwh():.1f} kWh)
- Consumption Rate: {user_ev.consumption_kwh_per_100km:.1f} kWh/100km
- Max Charging Power: {user_ev.max_charging_power_kw:.0f} kW
- Connector Type: {user_ev.connector_type}
- Multi-Objective Weights: Alpha (Time) = {alpha_weight}, Beta (Cost) = {beta_weight}

2. CLASSICAL DYNAMIC PROGRAMMING SOLVER RESULTS
--------------------------------------------------------------------------------
- Status: {'FEASIBLE' if dp_res['is_feasible'] else 'INFEASIBLE'}
- Selected Stops ({len(dp_res['stops'])}): {', '.join(dp_res['stops']) if dp_res['stops'] else 'None'}
- Total Journey Time: {dp_obj['total_time_minutes']:.1f} minutes ({dp_obj['total_time_minutes']/60.0:.2f} hrs)
- Driving Time: {dp_obj['travel_time']:.1f} mins | Wait Time: {dp_obj['waiting_time']:.1f} mins | Plug Time: {dp_obj['charging_time']:.1f} mins
- Total Charging Cost: INR ₹{dp_obj['total_cost_inr']:.2f}
- Objective Score: {dp_res['objective_score']:.4f}
- Execution Runtime: {dp_res['execution_time_sec']:.4f} seconds (Nodes Explored: {dp_res['nodes_explored']})

3. QUANTUM-INSPIRED QUBO SOLVER RESULTS
--------------------------------------------------------------------------------
- Status: {'FEASIBLE' if qubo_res['is_feasible'] else 'INFEASIBLE'}
- Selected Stops ({len(qubo_res['stops'])}): {', '.join(qubo_res['stops']) if qubo_res['stops'] else 'None'}
- Total Journey Time: {qubo_obj['total_time_minutes']:.1f} minutes ({qubo_obj['total_time_minutes']/60.0:.2f} hrs)
- Driving Time: {qubo_obj['travel_time']:.1f} mins | Wait Time: {qubo_obj['waiting_time']:.1f} mins | Plug Time: {qubo_obj['charging_time']:.1f} mins
- Total Charging Cost: INR ₹{qubo_obj['total_cost_inr']:.2f}
- Objective Score: {qubo_res['objective_score']:.4f}
- Execution Runtime: {qubo_res['execution_time_sec']:.4f} seconds (Reads: {qubo_res['num_reads']}, Feasibility: {qubo_res['feasibility_rate_percent']}%)

4. HEAD-TO-HEAD WINNER SUMMARY
--------------------------------------------------------------------------------
- Monetary Cost Winner: {winners['cost']}
- Journey Time Winner: {winners['time']}
- Overall Utility Winner: {winners['overall']}
- Solver Speed Winner: {winners['runtime_speed']}
================================================================================
"""

st.text_area("Generated Summary Report", value=report_text, height=220)

st.download_button(
    label="📥 Download Full Optimization Report (.txt)",
    data=report_text,
    file_name=f"EVolve_Report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
)
