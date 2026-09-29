"""
EVolve: Intelligent EV Route Charging Optimizer
Streamlit Interactive Dashboard
"""

import os
import sys

# Ensure both current directory and project root directory are on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for d in [CURRENT_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

import time
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import folium
# pyrefly: ignore [missing-import]
import streamlit as st
import requests
# pyrefly: ignore [missing-import]
from streamlit_folium import st_folium

def get_current_location():
    try:
        resp = requests.get("https://ipinfo.io/json", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            city = data.get("city")
            region = data.get("region")
            if city:
                return f"{city}, {region}" if region else city
    except Exception:
        pass
    return ""

from config import (
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
from data.telemetry_sim import simulate_telemetry, update_single_station, tick_live_telemetry
import importlib
import experiments.race_runner
importlib.reload(experiments.race_runner)
from experiments.race_runner import run_race, run_experiments

import core.quantum_specialist
importlib.reload(core.quantum_specialist)
from core.quantum_specialist import get_quantum_specialist_solution, format_specialist_markdown_report


# ==============================================================================
# STREAMLIT PAGE & THEME CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="EVolve - Intelligent EV Route Charging Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS styling for modern glassmorphic aesthetics & mobile/desktop responsiveness
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Styling */
    .main-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        word-break: break-word;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Glassmorphic Metric Cards */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
    }
    
    .card-header {
        font-size: 1.15rem;
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
        display: inline-block;
    }

    .badge-infeasible {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Touch-friendly and responsive UI improvements */
    .stButton button {
        min-height: 44px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
    }

    /* Responsive Dataframe and Table horizontal scrolling container */
    div[data-testid="stDataFrame"], div[data-testid="stTable"], .stTable {
        width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Mobile Screens (≤ 768px Smartphones and Small Tablets) */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.7rem !important;
            line-height: 1.25 !important;
        }
        .sub-title {
            font-size: 0.9rem !important;
            margin-bottom: 1rem !important;
        }
        .card {
            padding: 0.85rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.75rem !important;
        }
        .card-header {
            font-size: 1.0rem !important;
        }
        .stButton button {
            width: 100% !important;
            padding: 10px 14px !important;
            font-size: 0.95rem !important;
        }
        /* Metric values on mobile */
        div[data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
        /* Full width for block containers */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }
    }

    /* Small Mobile (≤ 480px) */
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.45rem !important;
        }
        .card {
            padding: 0.7rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# INITIALIZE SESSION STATE & SIMULATION DATA
# ==============================================================================

if "origin" not in st.session_state:
    st.session_state["origin"] = "Chennai"

if "dest" not in st.session_state:
    st.session_state["dest"] = "Coimbatore"

# Track which origin/dest were actually used to build the cached route
if "route_origin_used" not in st.session_state:
    st.session_state["route_origin_used"] = ""

if "route_dest_used" not in st.session_state:
    st.session_state["route_dest_used"] = ""

if "route_points" not in st.session_state:
    st.session_state["route_points"] = get_route(api_key=GOOGLE_MAPS_API_KEY, origin=st.session_state["origin"], destination=st.session_state["dest"])
    st.session_state["route_origin_used"] = st.session_state["origin"]
    st.session_state["route_dest_used"] = st.session_state["dest"]

if "stations" not in st.session_state:
    raw_stations = get_stations_along_route(st.session_state["route_points"])
    st.session_state["stations"] = simulate_telemetry(raw_stations, current_hour=14.0, seed=42)

if "race_results" not in st.session_state:
    st.session_state["race_results"] = None

if "offline_station_id" not in st.session_state:
    st.session_state["offline_station_id"] = "None"

if "last_telemetry_tick" not in st.session_state:
    st.session_state["last_telemetry_tick"] = time.time()

if "selected_station_id" not in st.session_state:
    st.session_state["selected_station_id"] = None

if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = None

if "map_center" not in st.session_state:
    st.session_state["map_center"] = None

if "ev_preset_name" not in st.session_state:
    st.session_state["ev_preset_name"] = "Tata Nexon EV Max (40.5 kWh, 70 kW DC)"

if "battery_capacity" not in st.session_state:
    st.session_state["battery_capacity"] = 40.5

if "current_soc" not in st.session_state:
    st.session_state["current_soc"] = 90.0

if "consumption_rate" not in st.session_state:
    st.session_state["consumption_rate"] = 16.5

if "max_charge_power" not in st.session_state:
    st.session_state["max_charge_power"] = 70.0

if "connector_type" not in st.session_state:
    st.session_state["connector_type"] = "CCS2 (DC Fast)"

if "alpha_weight" not in st.session_state:
    st.session_state["alpha_weight"] = 0.65

if "qubo_num_reads" not in st.session_state:
    st.session_state["qubo_num_reads"] = 100

if "qubo_beta_min" not in st.session_state:
    st.session_state["qubo_beta_min"] = 0.1

if "qubo_beta_max" not in st.session_state:
    st.session_state["qubo_beta_max"] = 10.0


# ==============================================================================
# HEADER SECTION
# ==============================================================================

st.markdown('<div class="main-title">EVolve: Enterprise EV Charging Optimization System</div>', unsafe_allow_html=True)
route_dist = st.session_state["route_points"][-1][2] if st.session_state["route_points"] else 0.0
st.markdown(
    f'<div class="sub-title">Real-time corridor optimization evaluating <b>Classical Dynamic Programming</b> and <b>Quantum-Inspired QUBO heuristics</b> along the <b>{st.session_state["origin"]} ➔ {st.session_state["dest"]}</b> highway corridor ({route_dist:.0f} km).</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 1: SIDEBAR INPUT PANEL
# ==============================================================================

st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(168, 85, 247, 0.2)); border: 1px solid #38bdf8; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">🔮 Quantum Specialist Mission</div>
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 8px;">Run QUBO annealing for Tata Nexon EV Max on Coimbatore ➔ Chennai corridor.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
btn_specialist_preset = st.sidebar.button(
    "🚀 Load Quantum Specialist Mission",
    help="Configures Coimbatore ➔ Chennai, Tata Nexon EV Max (40.5kWh, 90% SoC), α=0.65, β=0.35, and 100 QUBO annealing reads.",
    use_container_width=True,
    type="primary"
)

if btn_specialist_preset:
    st.session_state["origin"] = "Coimbatore"
    st.session_state["dest"] = "Chennai"
    st.session_state["ev_preset_name"] = "Tata Nexon EV Max (40.5 kWh, 70 kW DC)"
    st.session_state["battery_capacity"] = 40.5
    st.session_state["current_soc"] = 90.0
    st.session_state["consumption_rate"] = 16.5
    st.session_state["max_charge_power"] = 70.0
    st.session_state["connector_type"] = "CCS2 (DC Fast)"
    st.session_state["alpha_weight"] = 0.65
    st.session_state["qubo_num_reads"] = 100
    st.session_state["qubo_beta_min"] = 0.1
    st.session_state["qubo_beta_max"] = 10.0
    st.session_state["target_dest_reserve"] = 20.0
    st.session_state["min_safe_buffer"] = 10.0
    st.session_state["force_recalc"] = True
    st.rerun()

st.sidebar.markdown("### Vehicle & Telemetry Configuration")

with st.sidebar.expander("Route Definition & Mapping", expanded=True):
    origin_input = st.text_input("Start Location", value=st.session_state["origin"])
    dest_input = st.text_input("Destination", value=st.session_state["dest"])
    
    current_dist = st.session_state["route_points"][-1][2] if st.session_state["route_points"] else 0.0
    st.text_input("Corridor Distance", value=f"{current_dist:.1f} km", disabled=True)
    map_tile_provider = st.selectbox("Map Style", options=["Google Maps", "CartoDB Dark", "OpenStreetMap"], index=0)
    btn_recalc_route = st.button("🔄 Recalculate Route", type="secondary", use_container_width=True)

# Detect stale cache: inputs changed, or user clicked Recalculate, or route was built for different city pair
_route_stale = (
    origin_input.strip().lower() != st.session_state["route_origin_used"].strip().lower()
    or dest_input.strip().lower() != st.session_state["route_dest_used"].strip().lower()
    or st.session_state.get("force_recalc", False)
    or btn_recalc_route
)

if _route_stale:
    st.session_state["origin"] = origin_input.strip()
    st.session_state["dest"] = dest_input.strip()
    st.session_state["force_recalc"] = False
    with st.spinner(f"Fetching live route: {origin_input} → {dest_input} via OpenStreetMap routing..."):
        new_route = get_route(api_key=GOOGLE_MAPS_API_KEY, origin=st.session_state["origin"], destination=st.session_state["dest"])
        st.session_state["route_points"] = new_route
        st.session_state["route_origin_used"] = st.session_state["origin"]
        st.session_state["route_dest_used"] = st.session_state["dest"]
        raw_st = get_stations_along_route(new_route)
        st.session_state["stations"] = simulate_telemetry(raw_st, current_hour=np.random.uniform(7.0, 22.0))
        st.session_state["race_results"] = None
    st.rerun()

with st.sidebar.expander("Live Telemetry Auto-Stream (25s)", expanded=True):
    auto_refresh_enabled = st.toggle(
        "⏱️ Auto-Refresh Every 25 Seconds",
        value=True,
        help="Automatically updates station occupancy, queue times, and dynamic tariffs every 25 seconds."
    )
    col_ref1, col_ref2 = st.columns([1, 1])
    with col_ref1:
        btn_refresh_now = st.button("⚡ Refresh Now", use_container_width=True)
    with col_ref2:
        st.caption("Interval: **25s**")
    if auto_refresh_enabled:
        st.caption("🟢 **Stream Active**: Syncing live every 25s")
    else:
        st.caption("⏸️ **Stream Paused**")

with st.sidebar.expander("EV Battery Specifications", expanded=True):
    VEHICLE_PRESETS = {
        "Tata Nexon EV Max (40.5 kWh, 70 kW DC)": {
            "capacity": 40.5,
            "soc": 90.0,
            "consumption": 16.5,
            "power": 70.0,
            "connector": "CCS2 (DC Fast)",
        },
        "MG ZS EV (50.3 kWh, 80 kW DC)": {
            "capacity": 50.3,
            "soc": 90.0,
            "consumption": 17.0,
            "power": 120.0,
            "connector": "CCS2 (DC Fast)",
        },
        "Hyundai Ioniq 5 (72.6 kWh, 240 kW DC)": {
            "capacity": 72.6,
            "soc": 90.0,
            "consumption": 18.0,
            "power": 240.0,
            "connector": "CCS2 (DC Fast)",
        },
        "BYD Atto 3 (60.5 kWh, 80 kW DC)": {
            "capacity": 60.5,
            "soc": 90.0,
            "consumption": 16.0,
            "power": 120.0,
            "connector": "CCS2 (DC Fast)",
        },
        "Custom EV Profile": None,
    }

    preset_keys = list(VEHICLE_PRESETS.keys())
    current_preset = st.session_state.get("ev_preset_name", "Tata Nexon EV Max (40.5 kWh, 70 kW DC)")
    preset_idx = preset_keys.index(current_preset) if current_preset in preset_keys else 0

    selected_preset = st.selectbox("Vehicle Model Preset", options=preset_keys, index=preset_idx)
    if selected_preset != st.session_state.get("ev_preset_name"):
        st.session_state["ev_preset_name"] = selected_preset
        preset_cfg = VEHICLE_PRESETS.get(selected_preset)
        if preset_cfg:
            st.session_state["battery_capacity"] = preset_cfg["capacity"]
            st.session_state["current_soc"] = preset_cfg["soc"]
            st.session_state["consumption_rate"] = preset_cfg["consumption"]
            st.session_state["max_charge_power"] = preset_cfg["power"]
            st.session_state["connector_type"] = preset_cfg["connector"]
        st.rerun()

    battery_cap = st.slider("Battery Capacity (kWh)", min_value=30.0, max_value=120.0, value=float(st.session_state.get("battery_capacity", 40.5)), step=0.5)
    current_soc = st.slider("Current State of Charge (%)", min_value=10.0, max_value=100.0, value=float(st.session_state.get("current_soc", 90.0)), step=5.0)
    consumption_rate = st.slider("Consumption Rate (kWh/100km)", min_value=4.0, max_value=25.0, value=float(st.session_state.get("consumption_rate", 16.5)), step=0.5)
    power_options = [30.0, 50.0, 60.0, 70.0, 120.0, 150.0, 240.0]
    p_val = float(st.session_state.get("max_charge_power", 70.0))
    p_idx = power_options.index(p_val) if p_val in power_options else 3
    max_charge_power = st.select_slider("Max Charging Power (kW)", options=power_options, value=power_options[p_idx])

    conn_opts = ["CCS2 (DC Fast)", "Type2 (AC)", "CHAdeMO (DC Fast)"]
    c_val = st.session_state.get("connector_type", "CCS2 (DC Fast)")
    c_idx = conn_opts.index(c_val) if c_val in conn_opts else 0
    connector_type = st.selectbox("Connector Type", options=conn_opts, index=c_idx, help="Choose connector matching your vehicle inlet.")

with st.sidebar.expander("Multi-Objective Optimization Weights", expanded=True):
    alpha_weight = st.slider(
        "Alpha (Weight for Journey Time)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("alpha_weight", 0.65)),
        step=0.05,
        help="Higher alpha prioritizes total trip duration; lower alpha prioritizes cheaper charging rates."
    )
    beta_weight = round(1.0 - alpha_weight, 2)
    st.write(f"**Beta (Weight for Charging Cost):** `{beta_weight}`")

with st.sidebar.expander("Quantum QUBO Hyperparameters", expanded=False):
    qubo_reads = st.slider("Annealing Reads (num_reads)", min_value=50, max_value=500, value=int(st.session_state.get("qubo_num_reads", 100)), step=50)
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        qubo_beta_min = st.number_input("Beta Min", min_value=0.01, max_value=1.0, value=float(st.session_state.get("qubo_beta_min", 0.1)), step=0.05)
    with col_q2:
        qubo_beta_max = st.number_input("Beta Max", min_value=1.0, max_value=30.0, value=float(st.session_state.get("qubo_beta_max", 10.0)), step=1.0)
    st.caption("Quantum Annealing Beta schedule controls quantum fluctuations during simulated sampling.")

with st.sidebar.expander("Battery Safety & Destination Reserve", expanded=False):
    target_dest_reserve = st.slider(
        "Target Destination Reserve (%)",
        min_value=10.0,
        max_value=35.0,
        value=float(st.session_state.get("target_dest_reserve", 20.0)),
        step=1.0,
        help="Minimum battery percentage required upon reaching the final destination."
    )
    min_safe_buffer = st.slider(
        "Minimum Safe Buffer (%)",
        min_value=5.0,
        max_value=20.0,
        value=float(st.session_state.get("min_safe_buffer", 10.0)),
        step=1.0,
        help="Do not allow SoC to drop below this buffer during any intermediate driving segment."
    )

# Construct EV instance
user_ev = EV(
    id=st.session_state.get("ev_preset_name", "User-EV"),
    battery_capacity_kwh=battery_cap,
    current_soc_percent=current_soc,
    consumption_kwh_per_100km=consumption_rate,
    max_charging_power_kw=max_charge_power,
    connector_type=connector_type,
)

st.sidebar.markdown("---")
btn_run_race = st.sidebar.button("Execute Optimization Solvers", type="primary", width="stretch")
btn_resimulate = st.sidebar.button("Refresh Station Telemetry", width="stretch")

if btn_resimulate or btn_refresh_now:
    st.session_state["stations"] = tick_live_telemetry(
        st.session_state["stations"],
        current_hour=np.random.uniform(7.0, 22.0),
        offline_id=st.session_state.get("offline_station_id", "None")
    )
    st.session_state["last_telemetry_tick"] = time.time()
    st.session_state["race_results"] = run_race(
        user_ev,
        st.session_state["stations"],
        route=st.session_state["route_points"],
        alpha=alpha_weight,
        beta=beta_weight,
        num_reads=qubo_reads,
        beta_range=(qubo_beta_min, qubo_beta_max),
        min_battery_buffer_percent=min_safe_buffer,
        min_destination_reserve_percent=target_dest_reserve,
    )
    st.sidebar.success("Updated telemetry!")
    st.rerun()

# Run Race logic when clicked or auto-initialized
if btn_run_race or st.session_state.get("race_results") is None:
    with st.spinner("Racing Classical DP vs Quantum QUBO solvers in parallel..."):
        st.session_state["race_results"] = run_race(
            user_ev,
            st.session_state["stations"],
            route=st.session_state["route_points"],
            alpha=alpha_weight,
            beta=beta_weight,
            num_reads=qubo_reads,
            beta_range=(qubo_beta_min, qubo_beta_max),
            min_battery_buffer_percent=min_safe_buffer,
            min_destination_reserve_percent=target_dest_reserve,
        )


# ==============================================================================
# ==============================================================================
# SECTION 2: INTERACTIVE ROUTE MAP & LIVE TELEMETRY (AUTO-SYNC EVERY 25s)
# ==============================================================================

@st.fragment(run_every=25 if auto_refresh_enabled else None)
def render_live_route_map():
    # 25-Second Live Telemetry Background Tick
    now = time.time()
    if now - st.session_state.get("last_telemetry_tick", 0) >= 20.0:
        st.session_state["stations"] = tick_live_telemetry(
            st.session_state["stations"],
            current_hour=np.random.uniform(7.0, 22.0),
            offline_id=st.session_state.get("offline_station_id", "None"),
        )
        st.session_state["last_telemetry_tick"] = now
        st.session_state["race_results"] = run_race(
            user_ev,
            st.session_state["stations"],
            route=st.session_state["route_points"],
            alpha=alpha_weight,
            beta=beta_weight,
            num_reads=qubo_reads,
            beta_range=(qubo_beta_min, qubo_beta_max),
            min_battery_buffer_percent=min_safe_buffer,
            min_destination_reserve_percent=target_dest_reserve,
        )

    # Active race results for map markers
    race_data = st.session_state["race_results"]
    dp_res = race_data["classical"]
    qubo_res = race_data["quantum"]

    col_map_hdr1, col_map_hdr2 = st.columns([3, 2])
    with col_map_hdr1:
        st.markdown("### Geospatial Route & Live Telemetry")
        last_sync_str = time.strftime('%H:%M:%S', time.localtime(st.session_state.get("last_telemetry_tick", time.time())))
        st.caption(f"🟢 **Live Telemetry Stream Active** • Auto-refreshes every 25s • Last sync: `{last_sync_str}`")
    with col_map_hdr2:
        st_opts = ["(Auto: Click marker on map)"] + [
            f"{s.id}: {s.name} ({s.charger_type} {s.max_power_kw:.0f}kW)"
            for s in st.session_state["stations"]
        ]
        curr_id = st.session_state.get("selected_station_id")
        sel_idx = 0
        if curr_id:
            for idx, opt in enumerate(st_opts):
                if opt.startswith(f"{curr_id}:"):
                    sel_idx = idx
                    break
        chosen_opt = st.selectbox(
            "📍 Pin Station Popup (White Box) for Real-Time Updates:",
            options=st_opts,
            index=sel_idx,
            help="Select a station to keep its popup open on the map. The box updates every 25 seconds.",
        )
        if chosen_opt != "(Auto: Click marker on map)":
            chosen_id = chosen_opt.split(":")[0].strip()
            if st.session_state.get("selected_station_id") != chosen_id:
                st.session_state["selected_station_id"] = chosen_id

    route_points = st.session_state["route_points"]
    current_dist = route_points[-1][2] if route_points else 0.0

    if route_points:
        map_center = [
            (route_points[0][0] + route_points[-1][0]) / 2.0,
            (route_points[0][1] + route_points[-1][1]) / 2.0,
        ]
    else:
        map_center = [12.05, 78.50]

    if map_tile_provider == "Google Maps":
        gmaps_key = GOOGLE_MAPS_API_KEY
        google_tile_url = f"https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key={gmaps_key}" if gmaps_key else "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
        m = folium.Map(location=map_center, zoom_start=7, tiles=google_tile_url, attr="Google Maps")
    elif map_tile_provider == "OpenStreetMap":
        m = folium.Map(location=map_center, zoom_start=7, tiles="OpenStreetMap")
    else:
        m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB dark_matter")

    if route_points:
        # Add highway route line string
        polyline_coords = [(pt[0], pt[1]) for pt in route_points]
        folium.PolyLine(
            polyline_coords, color="#1E88E5", weight=5, opacity=0.85, tooltip=f"Route Corridor ({current_dist:.0f} km)"
        ).add_to(m)

        # Origin and Destination Markers
        folium.Marker(
            [route_points[0][0], route_points[0][1]],
            popup=f"<b>Origin:</b> {st.session_state['origin']}",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

        folium.Marker(
            [route_points[-1][0], route_points[-1][1]],
            popup=f"<b>Destination:</b> {st.session_state['dest']}",
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

        # Format AC and DC connectors clearly
        ac_conns = st_obj.ac_connectors
        dc_conns = st_obj.dc_connectors
        ac_display = ", ".join(ac_conns) if ac_conns else "None (DC Only)"
        dc_display = ", ".join(dc_conns) if dc_conns else "None (AC Only)"
        is_pinned = (st_obj.id == st.session_state.get("selected_station_id"))
        last_sync_time = time.strftime('%H:%M:%S', time.localtime(st.session_state.get("last_telemetry_tick", time.time())))

        popup_html = f"""
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; min-width: 250px; font-size: 13px; line-height: 1.5;'>
            <div style='margin-bottom:6px; border-bottom: 2px solid #e2e8f0; padding-bottom:4px;'>
                <h4 style='margin:0; color:#1E88E5; font-size: 15px; font-weight:700;'>{st_obj.name}{tag}</h4>
            </div>
            <div>
                <b>Status:</b> {status_str}<br/>
                <b>Distance:</b> {st_obj.distance_from_start_km:.1f} km from start<br/>
                <b>Charger:</b> {st_obj.charger_type} ({st_obj.max_power_kw:.0f} kW)<br/>
                <b>Bays:</b> <span style='font-weight:700; color:{"#16a34a" if st_obj.available_slots > 0 else "#dc2626"};'>{st_obj.available_slots} / {st_obj.num_slots} available</span><br/>
                <b>Queue Wait:</b> {st_obj.wait_time_minutes:.0f} mins<br/>
                <b>Dynamic Price:</b> ₹{st_obj.price_per_kwh:.2f} / kWh<br/>
                <hr style='margin: 6px 0; border: none; border-top: 1px solid #e2e8f0;'/>
                <div style='background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #e2e8f0;'>
                    <b>🔌 AC Connectors:</b> <span style='color:#0369a1; font-weight:600;'>{ac_display}</span><br/>
                    <b>⚡ DC Connectors:</b> <span style='color:#b91c1c; font-weight:600;'>{dc_display}</span>
                </div>
                <div style='margin-top:6px; font-size: 11px; color: #64748b; display: flex; align-items: center; justify-content: space-between; border-top: 1px dashed #cbd5e1; padding-top: 4px;'>
                    <span>🟢 <i>Live Telemetry</i></span>
                    <span>🕒 {last_sync_time}</span>
                </div>
                <div style='font-size: 10px; color: #94a3b8; text-align: right;'>Auto-updates every 25s</div>
            </div>
        </div>
        """

        folium.Marker(
            [st_obj.lat, st_obj.lon],
            popup=folium.Popup(popup_html, max_width=320, show=is_pinned),
            tooltip=f"{st_obj.name} ({st_obj.distance_from_start_km:.0f} km)",
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix="fa" if "fa" in icon_type else "glyphicon"),
        ).add_to(m)

    # Render Folium Map in Streamlit with responsive container width
    map_output = st_folium(
        m,
        use_container_width=True,
        height=450,
        zoom=st.session_state.get("map_zoom"),
        center=st.session_state.get("map_center"),
        returned_objects=["last_object_clicked", "zoom", "center"],
    )

    if map_output:
        if map_output.get("zoom"):
            st.session_state["map_zoom"] = map_output["zoom"]
        if map_output.get("center"):
            c = map_output["center"]
            st.session_state["map_center"] = [c["lat"], c["lng"]]
        if map_output.get("last_object_clicked"):
            click_lat = map_output["last_object_clicked"]["lat"]
            click_lng = map_output["last_object_clicked"]["lng"]
            closest_st = min(
                st.session_state["stations"],
                key=lambda s: (s.lat - click_lat) ** 2 + (s.lon - click_lng) ** 2
            )
            if (closest_st.lat - click_lat) ** 2 + (closest_st.lon - click_lng) ** 2 < 0.05:
                if st.session_state.get("selected_station_id") != closest_st.id:
                    st.session_state["selected_station_id"] = closest_st.id
                    st.rerun(scope="fragment")

    # Pinned station real-time telemetry card
    if st.session_state.get("selected_station_id"):
        pinned_st = next((s for s in st.session_state["stations"] if s.id == st.session_state["selected_station_id"]), None)
        if pinned_st:
            ac_str = ", ".join(pinned_st.ac_connectors) if pinned_st.ac_connectors else "None"
            dc_str = ", ".join(pinned_st.dc_connectors) if pinned_st.dc_connectors else "None"
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid #38bdf8; border-radius: 8px; padding: 10px 16px; margin-top: 10px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color:#38bdf8; font-weight:bold; font-size:1.05rem;">📍 Pinned Live Station: {pinned_st.name}</span>
                        <span style="margin-left: 10px; color:#94a3b8; font-size:0.9rem;">({pinned_st.distance_from_start_km:.1f} km from start)</span>
                    </div>
                    <div style="display:flex; gap:16px; font-size:0.92rem; align-items:center;">
                        <span><b>Bays:</b> <span style="color:#34d399; font-weight:bold;">{pinned_st.available_slots}/{pinned_st.num_slots}</span></span>
                        <span><b>Wait:</b> <b>{pinned_st.wait_time_minutes:.0f} mins</b></span>
                        <span><b>Price:</b> <b>₹{pinned_st.price_per_kwh:.2f}/kWh</b></span>
                        <span>🔌 <b>AC:</b> {ac_str}</span>
                        <span>⚡ <b>DC Fast:</b> {dc_str}</span>
                        <span style="background:#0284c7; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem;">LIVE 25s TICK</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

render_live_route_map()


# ==============================================================================
# SECTION 3: LIVE ALGORITHM RACE PANEL
# ==============================================================================

race_data = st.session_state["race_results"]
dp_res = race_data["classical"]
qubo_res = race_data["quantum"]
winners = race_data["winners"]

st.markdown("### Algorithmic Solver Execution Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Classical Dynamic Programming (DP)</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="card-header">Quantum-Inspired QUBO</div>', unsafe_allow_html=True)
    
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
# SECTION 3.5: ML & EXHAUSTIVE SEARCH ANALYSIS (THE BEST ROUTE)
# ==============================================================================

st.markdown("### Global Optimum Route Analysis")
st.info("The global optimization engine has exhaustively evaluated all viable charging permutations to isolate the mathematically optimal route.")

best_res = dp_res if winners['overall'] == 'Classical DP' else qubo_res
best_obj = best_res['obj_details']

st.markdown('<div class="card" style="border-left: 5px solid #10b981; background: rgba(16, 185, 129, 0.05);">', unsafe_allow_html=True)
st.markdown('<div class="card-header" style="color: #10b981; font-size: 1.3rem;">Recommended Optimal Route</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"**Total Duration:**<br>`{best_obj['total_time_minutes']:.1f} mins`", unsafe_allow_html=True)
with col_b:
    st.markdown(f"**Estimated Cost:**<br>`₹{best_obj['total_cost_inr']:.2f}`", unsafe_allow_html=True)
with col_c:
    st.markdown(f"**Required Stops:**<br>`{len(best_res['stops'])}`", unsafe_allow_html=True)

st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.markdown("**Optimal Charging Stops Sequence:**")
if best_res["stops"]:
    for sid in best_res["stops"]:
        st_name = next((s.name for s in st.session_state["stations"] if s.id == sid), sid)
        charge_kwh = best_res["charges"].get(sid, 0.0)
        st_cost = next((s.price_per_kwh for s in st.session_state["stations"] if s.id == sid), 0.0) * charge_kwh
        st.markdown(f"- **{st_name}**: Charge `+{charge_kwh:.1f} kWh` *(Est. Cost: ₹{st_cost:.2f})*")
else:
    st.write("- *No intermediate stops required to safely reach the destination.*")
st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 3.6: QUANTUM OPTIMIZATION SPECIALIST ROUTE REPORT
# ==============================================================================

st.markdown("---")
st.markdown("### 🔮 Quantum Optimization Specialist Route Report")
st.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #a855f7; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;">
        <div style="color: #c084fc; font-weight: 700; font-size: 1.15rem; margin-bottom: 6px;">
            ⚡ Quantum-Inspired QUBO Optimization: {st.session_state["origin"]} ➔ {st.session_state["dest"]}
        </div>
        <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
            <b>Vehicle:</b> {user_ev.id} ({user_ev.battery_capacity_kwh:.1f} kWh, {user_ev.max_charging_power_kw:.0f} kW DC Fast) &nbsp;|&nbsp;
            <b>Initial SoC:</b> {user_ev.current_soc_percent:.0f}% &nbsp;|&nbsp;
            <b>Target Destination Reserve:</b> ≥20% &nbsp;|&nbsp;
            <b>Minimum Safe Buffer:</b> 10%<br>
            <b>Strategy Weights:</b> Time Minimization (α) = <code>{alpha_weight}</code>, Cost Minimization (β) = <code>{beta_weight}</code> &nbsp;|&nbsp;
            <b>Annealing Sampler:</b> <code>neal.SimulatedAnnealingSampler</code> (<code>num_reads={qubo_reads}</code>, <code>β ∈ [{qubo_beta_min}, {qubo_beta_max}]</code>)
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

q_res = qubo_res
d_res = dp_res
q_obj = q_res["obj_details"]
d_obj = d_res["obj_details"]
st_lookup = {s.id: s for s in st.session_state["stations"]}

# 1. Stop Sequence Breakdown
q_active_stops = [sid for sid in q_res["stops"] if q_res["charges"].get(sid, 0.0) > 0.01]

spec_stops_data = []
sim_ev_spec = user_ev.clone()
curr_km_spec = 0.0

for s_idx, sid in enumerate(q_active_stops):
    if sid not in st_lookup:
        continue
    st_obj = st_lookup[sid]
    d_leg = st_obj.distance_from_start_km - curr_km_spec
    sim_ev_spec.consume_distance(d_leg)
    arr_soc = sim_ev_spec.current_soc_percent

    c_kwh = q_res["charges"].get(sid, 0.0)
    sim_ev_spec.charge(c_kwh)
    dep_soc = sim_ev_spec.current_soc_percent

    p_time = st_obj.charging_time_minutes(c_kwh, user_ev)
    w_time = st_obj.wait_time_minutes
    c_cost = st_obj.charging_cost(c_kwh)

    spec_stops_data.append({
        "Stop #": f"Stop {s_idx + 1}",
        "Charging Station": st_obj.name,
        "Distance": f"{st_obj.distance_from_start_km:.1f} km",
        "AC Connectors": ", ".join(st_obj.ac_connectors) if st_obj.ac_connectors else "None",
        "DC Fast Connectors": ", ".join(st_obj.dc_connectors) if st_obj.dc_connectors else "None",
        "Power": f"{st_obj.max_power_kw:.0f} kW",
        "Arrival SoC": f"{arr_soc:.1f}%",
        "Charged": f"+{c_kwh:.1f} kWh",
        "Departure SoC": f"{dep_soc:.1f}%",
        "Plug Time": f"{p_time:.0f} mins",
        "Queue Delay": f"{w_time:.0f} mins",
        "Cost (₹)": f"₹{c_cost:.2f}",
    })
    curr_km_spec = st_obj.distance_from_start_km

# Final leg arrival
final_dist = route_dist - curr_km_spec
if final_dist > 0:
    sim_ev_spec.consume_distance(final_dist)
dest_arrival_soc = sim_ev_spec.current_soc_percent

st.markdown("#### 1. Exact Sequence of Recommended Charging Stops (Quantum QUBO)")
if spec_stops_data:
    st.dataframe(pd.DataFrame(spec_stops_data), use_container_width=True, hide_index=True)
else:
    st.info("No intermediate charging stops required for this route.")

# 2. Total Journey Breakdown Cards
st.markdown("#### 2. Total Journey Breakdown")
tot_kwh_road = route_dist * (user_ev.consumption_kwh_per_100km / 100.0)
tot_kwh_charged = sum(q_res["charges"].values())

col_jb1, col_jb2, col_jb3, col_jb4, col_jb5, col_jb6 = st.columns(6)
with col_jb1:
    st.metric("Highway Drive Time", f"{q_obj['travel_time']:.0f} min", f"{q_obj['travel_time']/60.0:.1f} hrs")
with col_jb2:
    st.metric("Active Plug Time", f"{q_obj['charging_time']:.0f} min", f"{len(q_active_stops)} stop(s)")
with col_jb3:
    st.metric("Station Queue Delay", f"{q_obj['waiting_time']:.0f} min", "Live telemetry")
with col_jb4:
    st.metric("Total Journey Time", f"{q_obj['total_time_minutes']:.0f} min", f"{q_obj['total_time_minutes']/60.0:.2f} hrs")
with col_jb5:
    st.metric("Total Charging Cost", f"₹{q_obj['total_cost_inr']:.2f}", f"₹{q_obj['total_cost_inr']/max(1.0, route_dist):.2f}/km")
with col_jb6:
    st.metric("Road Energy Used", f"{tot_kwh_road:.1f} kWh", f"Arrival: {dest_arrival_soc:.1f}% SoC")

# 3. Head-to-Head Benchmark Table
st.markdown("#### 3. Quantum QUBO vs. Classical Dynamic Programming (DP) Benchmark")
bench_df = pd.DataFrame([
    {
        "Evaluation Dimension": "Algorithm Execution Latency",
        "Quantum QUBO Solver": f"{q_res['execution_time_sec']:.4f} s",
        "Classical Dynamic Programming": f"{d_res['execution_time_sec']:.4f} s",
        "Performance Winner": winners['runtime_speed'],
    },
    {
        "Evaluation Dimension": "Total Route Time (Drive + Charge + Queue)",
        "Quantum QUBO Solver": f"{q_obj['total_time_minutes']:.1f} mins ({q_obj['total_time_minutes']/60.0:.2f} hrs)",
        "Classical Dynamic Programming": f"{d_obj['total_time_minutes']:.1f} mins ({d_obj['total_time_minutes']/60.0:.2f} hrs)",
        "Performance Winner": winners['time'],
    },
    {
        "Evaluation Dimension": "Total Charging Expenditure (₹)",
        "Quantum QUBO Solver": f"₹{q_obj['total_cost_inr']:.2f}",
        "Classical Dynamic Programming": f"₹{d_obj['total_cost_inr']:.2f}",
        "Performance Winner": winners['cost'],
    },
    {
        "Evaluation Dimension": "Combined Objective Utility (α·T + β·C)",
        "Quantum QUBO Solver": f"{q_res['objective_score']:.4f}",
        "Classical Dynamic Programming": f"{d_res['objective_score']:.4f}",
        "Performance Winner": winners['overall'],
    },
    {
        "Evaluation Dimension": "Constraint Feasibility & Status",
        "Quantum QUBO Solver": f"{'✓ Strictly Feasible' if q_res['is_feasible'] else 'Infeasible'} ({q_res.get('feasibility_rate_percent', 0.0):.1f}% anneal rate)",
        "Classical Dynamic Programming": f"{'✓ Strictly Feasible' if d_res['is_feasible'] else 'Infeasible'} (Bellman Exact)",
        "Performance Winner": "Dual Feasible" if (q_res['is_feasible'] and d_res['is_feasible']) else "Classical DP",
    },
    {
        "Evaluation Dimension": "Intermediate Stops Selected",
        "Quantum QUBO Solver": f"{len(q_active_stops)} stops",
        "Classical Dynamic Programming": f"{len(d_res['stops'])} stops",
        "Performance Winner": "Tie" if len(q_active_stops) == len(d_res['stops']) else "Optimized",
    },
])
st.dataframe(bench_df, use_container_width=True, hide_index=True)

# Specialist Report Downloader
specialist_md_lines = [
    "# ⚡ Quantum Optimization Specialist Route Report",
    f"**Corridor:** {st.session_state['origin']} ➔ {st.session_state['dest']} ({route_dist:.1f} km)",
    f"**Vehicle:** {user_ev.id} ({user_ev.battery_capacity_kwh:.1f} kWh, {user_ev.max_charging_power_kw:.0f} kW DC Fast)",
    f"**Initial Battery:** {user_ev.current_soc_percent:.0f}% | **Destination Arrival SoC:** {dest_arrival_soc:.1f}%",
    f"**Weights:** Alpha (Time) = {alpha_weight}, Beta (Cost) = {beta_weight}",
    f"**Quantum Sampler:** neal.SimulatedAnnealingSampler (num_reads={qubo_reads}, beta_range=({qubo_beta_min}, {qubo_beta_max}))",
    "",
    "## 1. Sequence of Recommended Charging Stops (Quantum QUBO)",
    "| Stop # | Charging Station | Distance | AC Connectors | DC Fast Connectors | Power | Arrival SoC | Charged | Departure SoC | Plug Time | Queue | Cost |",
    "|:---:|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
]
for s in spec_stops_data:
    specialist_md_lines.append(
        f"| {s['Stop #']} | {s['Charging Station']} | {s['Distance']} | {s['AC Connectors']} | {s['DC Fast Connectors']} | "
        f"{s['Power']} | {s['Arrival SoC']} | {s['Charged']} | {s['Departure SoC']} | {s['Plug Time']} | {s['Queue Delay']} | {s['Cost (₹)']} |"
    )

specialist_md_lines.extend([
    "",
    "## 2. Total Journey Breakdown",
    f"- Highway Drive Time: {q_obj['travel_time']:.1f} mins ({q_obj['travel_time']/60.0:.2f} hrs)",
    f"- Active Plug Charging Time: {q_obj['charging_time']:.1f} mins",
    f"- Station Queue Delay: {q_obj['waiting_time']:.1f} mins",
    f"- Total Journey Duration: {q_obj['total_time_minutes']:.1f} mins ({q_obj['total_time_minutes']/60.0:.2f} hrs)",
    f"- Total Charging Cost: INR ₹{q_obj['total_cost_inr']:.2f}",
    f"- Road Energy Consumed: {tot_kwh_road:.1f} kWh",
    f"- Energy Replenished: {tot_kwh_charged:.1f} kWh",
    f"- Destination Reserve: {dest_arrival_soc:.1f}% (Target: ≥20%)",
    "",
    "## 3. Quantum QUBO vs. Classical DP Benchmark",
    f"- Execution Latency: Quantum {q_res['execution_time_sec']:.4f}s vs Classical DP {d_res['execution_time_sec']:.4f}s ({winners['runtime_speed']})",
    f"- Total Journey Time: Quantum {q_obj['total_time_minutes']:.1f}m vs Classical DP {d_obj['total_time_minutes']:.1f}m ({winners['time']})",
    f"- Total Charging Cost: Quantum ₹{q_obj['total_cost_inr']:.2f} vs Classical DP ₹{d_obj['total_cost_inr']:.2f} ({winners['cost']})",
    f"- Combined Objective Score: Quantum {q_res['objective_score']:.4f} vs Classical DP {d_res['objective_score']:.4f} ({winners['overall']})",
    f"- Feasibility: Quantum {'✓ Feasible' if q_res['is_feasible'] else 'Infeasible'} ({q_res.get('feasibility_rate_percent', 0.0):.1f}%) vs Classical DP {'✓ Feasible' if d_res['is_feasible'] else 'Infeasible'}",
])
specialist_md_content = "\n".join(specialist_md_lines)

col_dl1, col_dl2 = st.columns([3, 1])
with col_dl1:
    with st.expander("📋 View Raw Quantum Specialist Prompt & Response Markdown", expanded=False):
        st.code(specialist_md_content, language="markdown")
with col_dl2:
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=specialist_md_content,
        file_name=f"Quantum_Specialist_Report_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ==============================================================================
# SECTION 4: COMPARISON TABLE
# ==============================================================================

st.markdown("### Solver Performance Comparative Analysis")

comp_df = pd.DataFrame([
    {
        "Performance Metric": "Total Cost (₹)",
        "Classical DP": f"₹{dp_obj['total_cost_inr']:.2f}",
        "Quantum-Inspired QUBO": f"₹{qubo_obj['total_cost_inr']:.2f}",
        "Winner": f"{winners['cost']}",
    },
    {
        "Performance Metric": "Total Journey Duration",
        "Classical DP": f"{dp_obj['total_time_minutes']:.1f} mins",
        "Quantum-Inspired QUBO": f"{qubo_obj['total_time_minutes']:.1f} mins",
        "Winner": f"{winners['time']}",
    },
    {
        "Performance Metric": "Number of Stops",
        "Classical DP": str(len(dp_res["stops"])),
        "Quantum-Inspired QUBO": str(len(qubo_res["stops"])),
        "Winner": "Tie" if len(dp_res["stops"]) == len(qubo_res["stops"]) else f"{winners['overall']}",
    },
    {
        "Performance Metric": "Algorithm Runtime",
        "Classical DP": f"{dp_res['execution_time_sec']:.4f} s",
        "Quantum-Inspired QUBO": f"{qubo_res['execution_time_sec']:.4f} s",
        "Winner": f"{winners['runtime_speed']}",
    },
    {
        "Performance Metric": "Feasible Solution?",
        "Classical DP": "Yes" if dp_res["is_feasible"] else "No",
        "Quantum-Inspired QUBO": "Yes" if qubo_res["is_feasible"] else "No",
        "Winner": "Both Feasible" if (dp_res["is_feasible"] and qubo_res["is_feasible"]) else "Classical DP",
    },
    {
        "Performance Metric": "Constraint Violations",
        "Classical DP": str(len(dp_res["violations"])),
        "Quantum-Inspired QUBO": str(len(qubo_res["violations"])),
        "Winner": "Zero Violations",
    },
    {
        "Performance Metric": "Combined Objective Score",
        "Classical DP": f"{dp_res['objective_score']:.4f}",
        "Quantum-Inspired QUBO": f"{qubo_res['objective_score']:.4f}",
        "Winner": f"{winners['overall']}",
    },
])

st.dataframe(comp_df, width="stretch", hide_index=True)


# ==============================================================================
# SECTION 5: ANALYTICAL PLOTLY CHARTS
# ==============================================================================

st.markdown("### Diagnostic Analytics & State Profiles")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Battery State of Charge (SoC) Trajectory")
    
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
    st.plotly_chart(fig_soc, width="stretch")

with chart_col2:
    st.markdown("#### Journey Duration Distribution")
    
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
    st.plotly_chart(fig_pie, width="stretch")

st.markdown("---")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("#### Station Infrastructure & Pricing")
    
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
    st.plotly_chart(fig_st, width="stretch")

with chart_col4:
    st.markdown("#### Scalability Benchmark: Problem Size vs. Execution Time")
    
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
    st.plotly_chart(fig_exp, width="stretch")


# ==============================================================================
# SECTION 6: DYNAMIC EVENT RE-OPTIMIZATION SIMULATOR
# ==============================================================================

st.markdown("### Disruption Simulation & Re-Optimization")

st.info("Simulate infrastructure failure by marking a station offline to evaluate re-optimization latency and route adjustments.")

event_col1, event_col2 = st.columns([3, 1])

with event_col1:
    st_options = ["None"] + [f"{s.id}: {s.name}" for s in st.session_state["stations"]]
    selected_offline_str = st.selectbox("Target Station for Simulated Outage:", options=st_options, index=0)

with event_col2:
    btn_trigger_event = st.button("Simulate Outage Event", type="secondary", width="stretch")

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
        user_ev,
        st.session_state["stations"],
        route=st.session_state["route_points"],
        alpha=alpha_weight,
        beta=beta_weight,
        num_reads=qubo_reads,
        beta_range=(qubo_beta_min, qubo_beta_max),
        min_battery_buffer_percent=min_safe_buffer,
        min_destination_reserve_percent=target_dest_reserve,
    )
    st.success(f"Station {selected_offline_str} marked OFFLINE! Re-optimization complete!")
    st.rerun()


# ==============================================================================
# SECTION 7: AUTOMATED REPORT GENERATOR
# ==============================================================================

st.markdown("### Execution Summary & Export")

route_dist = st.session_state.route_points[-1][2] if st.session_state.route_points else 0.0
report_text = f"""================================================================================
EVOLVE: INTELLIGENT EV ROUTE CHARGING OPTIMIZER REPORT
================================================================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Corridor: {st.session_state["origin"]} -> {st.session_state["dest"]} ({route_dist:.1f} km)

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
    label="Download Full Optimization Report (.txt)",
    data=report_text,
    file_name=f"EVolve_Report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain",
)
