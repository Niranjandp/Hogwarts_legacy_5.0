"""
EVolve: Intelligent EV Route Charging Optimizer
Multi-Objective Cost & Time Utility Function
"""

from typing import List, Dict, Any
from core.ev_model import EV
from core.station_model import Station
from config import DEFAULT_ALPHA, DEFAULT_BETA, TOTAL_ROUTE_DISTANCE_KM


def calculate_objective(
    ev: EV,
    stop_sequence: List[str],
    charge_amounts: Dict[str, float],
    stations: List[Station],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    average_speed_kmh: float = 75.0,
) -> Dict[str, float]:
    """
    Evaluates the multi-objective utility metric for a charging plan sequence:
    

    Metrics evaluated:
    - travel_time (minutes): Pure driving time along the 552 km highway
    - waiting_time (minutes): Queuing delay at selected stations
    - charging_time (minutes): Active plug charging duration
    - total_time_minutes: travel_time + waiting_time + charging_time
    - charging_cost (INR ₹): Monetary expenditure for energy charged
    - combined_score: alpha * (normalized time) + beta * (normalized cost)
    """
    station_map: Dict[str, Station] = {st.id: st for st in stations}

    # 1. Travel Time (highway driving time in minutes)
    travel_time_hours = TOTAL_ROUTE_DISTANCE_KM / max(10.0, average_speed_kmh)
    travel_time_minutes = travel_time_hours * 60.0

    waiting_time_minutes = 0.0
    charging_time_minutes = 0.0
    charging_cost_inr = 0.0

    # 2. Iterate through chosen charging stops
    for sid in stop_sequence:
        if sid not in station_map:
            continue
        
        st = station_map[sid]
        kwh = charge_amounts.get(sid, 0.0)

        if kwh > 0:
            # Wait time (queue delay at station)
            waiting_time_minutes += st.wait_time_minutes
            
            # Active plug charging time
            c_time = st.charging_time_minutes(kwh, ev)
            charging_time_minutes += c_time

            # Cost for energy charged
            charging_cost_inr += st.charging_cost(kwh)

    total_time_minutes = travel_time_minutes + waiting_time_minutes + charging_time_minutes
    total_cost_inr = charging_cost_inr

    # 3. Normalized Multi-Objective Combined Score
    # Baseline normalization constants (e.g. 500 mins total time, ₹1000 total cost)
    # Allows alpha & beta weights to balance time and money on comparable unitless scales
    norm_time = total_time_minutes / 450.0
    norm_cost = total_cost_inr / 800.0 if total_cost_inr > 0 else 0.1

    combined_score = (alpha * norm_time) + (beta * norm_cost)

    return {
        "total_cost_inr": round(total_cost_inr, 2),
        "total_time_minutes": round(total_time_minutes, 1),
        "charging_cost": round(charging_cost_inr, 2),
        "travel_time": round(travel_time_minutes, 1),
        "waiting_time": round(waiting_time_minutes, 1),
        "charging_time": round(charging_time_minutes, 1),
        "combined_score": round(combined_score, 4),
    }
