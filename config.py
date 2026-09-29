"""
EVolve: Intelligent EV Route Charging Optimizer
Configuration and Hyperparameters
"""

import os
from typing import List, Tuple, Dict, Any

# ==============================================================================
# GOOGLE MAPS PLATFORM API INTEGRATION (OPTIONAL)
# ==============================================================================
GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
USE_GOOGLE_MAPS_API: bool = bool(GOOGLE_MAPS_API_KEY)

# ==============================================================================
# ROUTE CONFIGURATION: Chennai (Palakarai) -> Coimbatore via NH48 / NH44 / NH544
# ==============================================================================

ORIGIN_NAME: str = "Chennai (Palakarai)"
DESTINATION_NAME: str = "Coimbatore"

# Real GPS waypoints along the NH48 -> NH44 -> NH544 highway corridor
# Each tuple represents: (latitude, longitude, cumulative_km_from_start, location_name)
ROUTE_WAYPOINTS: List[Tuple[float, float, float, str]] = [
    (13.0827, 80.2707, 0.0, "Chennai (Palakarai Central)"),
    (13.0489, 80.0910, 22.0, "Poonamallee Bypass"),
    (12.9675, 79.9439, 41.0, "Sriperumbudur Industrial Hub"),
    (12.8980, 79.8055, 60.0, "Sunguvarchatram"),
    (12.8342, 79.7036, 75.0, "Kanchipuram Highway Intersection"),
    (12.9284, 79.3333, 112.0, "Walajapet Toll Plaza"),
    (12.9246, 79.3005, 118.0, "Ranipet / Arcot Junction"),
    (12.9165, 79.1325, 140.0, "Vellore Bypass"),
    (12.9038, 78.9385, 162.0, "Pallikonda Toll Plaza"),
    (12.8395, 78.8105, 178.0, "Madhanur"),
    (12.7904, 78.7166, 192.0, "Ambur Leather Valley"),
    (12.6825, 78.6200, 210.0, "Vaniyambadi Bypass"),
    (12.6186, 78.5204, 226.0, "Natrampalli"),
    (12.5458, 78.3619, 248.0, "Bargur Link Road"),
    (12.5186, 78.2138, 266.0, "Krishnagiri (NH44 North-South Spine)"),
    (12.4285, 78.2144, 282.0, "Kaveripattinam"),
    (12.3082, 78.2046, 300.0, "Karimangalam"),
    (12.1211, 78.1582, 322.0, "Dharmapuri Central NH44"),
    (11.9432, 78.0705, 348.0, "Thoppur Ghats Section"),
    (11.7410, 78.0435, 372.0, "Omalur Junction"),
    (11.6643, 78.1460, 390.0, "Salem (NH544 Expressway Flyover)"),
    (11.4842, 77.8687, 430.0, "Sankari Highway Strip"),
    (11.4468, 77.6834, 450.0, "Komarapalayam / Bhavani River Bridge"),
    (11.3965, 77.6747, 458.0, "Chithode (Erode Bypass)"),
    (11.2778, 77.5843, 476.0, "Perundurai SIPCOT"),
    (11.2335, 77.4812, 488.0, "Vijayamangalam Toll Plaza"),
    (11.2050, 77.3712, 502.0, "Chengapalli Highway Node"),
    (11.1931, 77.2694, 514.0, "Avinashi Junction"),
    (11.1098, 77.1812, 528.0, "Karumathampatti Bypass"),
    (11.0634, 77.0858, 540.0, "Neelambur (Coimbatore East Gateway)"),
    (11.0168, 76.9558, 552.0, "Coimbatore (Gandhipuram Destination)")
]

TOTAL_ROUTE_DISTANCE_KM: float = ROUTE_WAYPOINTS[-1][2]

# ==============================================================================
# EV PHYSICAL & OPERATIONAL CONSTANTS
# ==============================================================================

# Minimum buffer to prevent battery degradation and emergency strand (10%)
BATTERY_BUFFER_PERCENT: float = 10.0

# Minimum reserve required when arriving at the final destination (15%)
DESTINATION_RESERVE_PERCENT: float = 15.0

# Maximum corridor detour distance allowed to reach a charging station (km)
CORRIDOR_WIDTH_KM: float = 25.0

# Default energy consumption specification (kWh per 100km)
DEFAULT_CONSUMPTION_KWH_PER_100KM: float = 6.5

# Default baseline EV specifications
DEFAULT_BATTERY_CAPACITY_KWH: float = 60.0
DEFAULT_CURRENT_SOC_PERCENT: float = 35.0
DEFAULT_MAX_CHARGING_POWER_KW: float = 120.0
DEFAULT_CONNECTOR_TYPE: str = "CCS2"
DEFAULT_AVERAGE_SPEED_KMH: float = 75.0

# ==============================================================================
# MULTI-OBJECTIVE BALANCING WEIGHTS
# ==============================================================================
# alpha: weight for total travel & charging time (normalized minutes)
# beta: weight for monetary expenditure (normalized INR)
DEFAULT_ALPHA: float = 0.6
DEFAULT_BETA: float = 0.4

# ==============================================================================
# QUANTUM QUBO PENALTY WEIGHTS & SAMPLER SETTINGS
# ==============================================================================
# P1: Penalty for violating minimum battery buffer (< 10% SOC)
QUBO_P1: float = 1000.0

# P2: Penalty for station inaccessibility, slot unavailability or downtime
QUBO_P2: float = 800.0

# P3: Penalty for failing to reach destination with required 15% reserve
QUBO_P3: float = 1200.0


QUBO_NUM_READS: int = 1000
QUBO_ANNEALING_STEPS: int = 1000
QUBO_CHAIN_STRENGTH: float = 2.0

DP_SOC_DISCRETIZATION_LEVELS: int = 20

MIN_PRICE_PER_KWH: float = 8.0
MAX_PRICE_PER_KWH: float = 18.0
PEAK_HOURS_SURCHARGE_FACTOR: float = 1.35
