"""
EVolve: Intelligent EV Route Charging Optimizer
Route Plan Feasibility & Constraint Validation Engine
"""

from typing import List, Dict, Any, Tuple, Optional
from core.ev_model import EV
from core.station_model import Station
from config import (
    BATTERY_BUFFER_PERCENT,
    DESTINATION_RESERVE_PERCENT,
    TOTAL_ROUTE_DISTANCE_KM,
)


def validate_solution(
    ev: EV,
    stations: List[Station],
    stop_sequence: List[str],
    charge_amounts: Dict[str, float],
    route: Optional[List[Tuple[float, float, float]]] = None,
    start_time_hour: float = 8.0,
    average_speed_kmh: float = 75.0,
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Validates an EV charging route solution against all 6 operational & physical constraints:
    1. Battery SoC never drops below 10% between any two stops / waypoints
    2. Battery SoC never exceeds 100%
    3. Destination (Coimbatore) reached with at least 15% SoC remaining
    4. Connector compatibility at every chosen charging station
    5. Station is operational and has available slot (or queue capacity)
    6. Arrival time falls within station operating hours

    Returns:
    - is_feasible (bool): True if zero violations detected
    - violations (List[str]): Explanatory messages for all broken constraints
    - battery_profile (List[Dict]): Detailed timeline profile of (km, soc, location, action, time_hour)
    """
    violations: List[str] = []
    battery_profile: List[Dict[str, Any]] = []

    # Map station ID -> Station object
    station_map: Dict[str, Station] = {st.id: st for st in stations}

    # Clone vehicle state to simulate traversal
    sim_ev = ev.clone()

    # Sort stop sequence by distance along route to ensure valid spatial order
    stop_stations: List[Station] = []
    for sid in stop_sequence:
        if sid in station_map:
            stop_stations.append(station_map[sid])

    stop_stations.sort(key=lambda s: s.distance_from_start_km)

    current_km = 0.0
    current_soc = sim_ev.current_soc_percent
    current_time_hour = start_time_hour

    # Record starting waypoint state
    battery_profile.append({
        "km": 0.0,
        "soc": round(current_soc, 2),
        "location": "Chennai (Palakarai)",
        "action": "Departure",
        "time_hour": round(current_time_hour, 2),
    })

    # Traverse stops along the route
    for st in stop_stations:
        dist_leg = st.distance_from_start_km - current_km

        if dist_leg < 0:
            violations.append(f"Invalid sequence: Station {st.name} is behind current position.")
            continue

        # Travel leg to station
        travel_time_hours = dist_leg / max(10.0, average_speed_kmh)
        current_time_hour += travel_time_hours

        # Energy consumption
        energy_used_kwh = sim_ev.energy_needed(dist_leg)
        soc_drop = (energy_used_kwh / sim_ev.battery_capacity_kwh) * 100.0
        arrival_soc = current_soc - soc_drop

        # Constraint 1: Battery never below 10%
        if arrival_soc < BATTERY_BUFFER_PERCENT - 1e-3:
            violations.append(
                f"Constraint 1 Broken: SoC dropped to {arrival_soc:.1f}% (< {BATTERY_BUFFER_PERCENT}% buffer) "
                f"en route to {st.name} at {st.distance_from_start_km:.1f} km."
            )

        # Constraint 2: Battery never below 0%
        if arrival_soc < 0.0:
            violations.append(
                f"Constraint 2 Broken: Battery completely depleted ({arrival_soc:.1f}%) before reaching {st.name}."
            )

        sim_ev.current_soc_percent = max(0.0, arrival_soc)

        # Record arrival at station
        battery_profile.append({
            "km": round(st.distance_from_start_km, 1),
            "soc": round(sim_ev.current_soc_percent, 2),
            "location": st.name,
            "action": "Arrival",
            "time_hour": round(current_time_hour, 2),
        })

        # Constraint 4: Connector compatibility
        if not st.supports_connector(sim_ev.connector_type):
            violations.append(
                f"Constraint 4 Broken: Station '{st.name}' does not support EV connector type '{sim_ev.connector_type}'."
            )

        # Constraint 5: Operational status and bay availability
        if not st.is_operational:
            violations.append(
                f"Constraint 5 Broken: Station '{st.name}' is currently offline/under maintenance."
            )
        elif st.available_slots <= 0 and st.wait_time_minutes > 60:
            violations.append(
                f"Constraint 5 Broken: Station '{st.name}' has no available bays and queue exceeds 60 mins."
            )

        # Constraint 6: Operating hours
        if not st.is_open_at_hour(current_time_hour):
            violations.append(
                f"Constraint 6 Broken: Arrived at '{st.name}' at hour {current_time_hour:.1f}, outside operating hours ({st.operating_hours})."
            )

        # Perform Charging
        charge_kwh = charge_amounts.get(st.id, 0.0)
        
        # Check if charge amount overfills battery (> 100%)
        curr_kwh = sim_ev.get_current_kwh()
        if curr_kwh + charge_kwh > sim_ev.battery_capacity_kwh + 0.1:
            violations.append(
                f"Constraint 2 Broken: Attempting to overcharge battery past 100% capacity at {st.name} "
                f"({curr_kwh + charge_kwh:.1f} kWh / {sim_ev.battery_capacity_kwh} kWh)."
            )

        actual_kwh = sim_ev.charge(charge_kwh)
        stay_time_mins = st.total_stay_time_minutes(actual_kwh, sim_ev)
        current_time_hour += (stay_time_mins / 60.0)

        current_km = st.distance_from_start_km
        current_soc = sim_ev.current_soc_percent

        # Record departure from station after charging
        battery_profile.append({
            "km": round(st.distance_from_start_km, 1),
            "soc": round(current_soc, 2),
            "location": st.name,
            "action": f"Charged +{actual_kwh:.1f} kWh ({stay_time_mins:.0f} min)",
            "time_hour": round(current_time_hour, 2),
        })

    # Final Leg to Destination (Coimbatore)
    dist_final = TOTAL_ROUTE_DISTANCE_KM - current_km
    if dist_final > 0:
        travel_time_hours = dist_final / max(10.0, average_speed_kmh)
        current_time_hour += travel_time_hours

        energy_used_kwh = sim_ev.energy_needed(dist_final)
        soc_drop = (energy_used_kwh / sim_ev.battery_capacity_kwh) * 100.0
        final_soc = current_soc - soc_drop

        if final_soc < BATTERY_BUFFER_PERCENT - 1e-3:
            violations.append(
                f"Constraint 1 Broken: Battery dropped to {final_soc:.1f}% en route to Coimbatore."
            )

        # Constraint 3: Destination reached with 15% remaining
        if final_soc < DESTINATION_RESERVE_PERCENT - 1e-3:
            violations.append(
                f"Constraint 3 Broken: Arrived at Coimbatore with {final_soc:.1f}% SoC (required reserve is {DESTINATION_RESERVE_PERCENT}%)."
            )

        current_soc = max(0.0, final_soc)

    battery_profile.append({
        "km": round(TOTAL_ROUTE_DISTANCE_KM, 1),
        "soc": round(current_soc, 2),
        "location": "Coimbatore (Destination)",
        "action": "Arrival",
        "time_hour": round(current_time_hour, 2),
    })

    is_feasible = (len(violations) == 0)
    return is_feasible, violations, battery_profile
