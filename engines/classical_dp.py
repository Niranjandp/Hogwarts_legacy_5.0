"""
EVolve: Intelligent EV Route Charging Optimizer
Classical Dynamic Programming Engine
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
from core.ev_model import EV
from core.station_model import Station
from core.validator import validate_solution
from core.objective import calculate_objective
from config import (
    BATTERY_BUFFER_PERCENT,
    DESTINATION_RESERVE_PERCENT,
    DP_SOC_DISCRETIZATION_LEVELS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)


def solve(
    ev: EV,
    stations: List[Station],
    route: Optional[List[Tuple[float, float, float]]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    soc_levels: int = DP_SOC_DISCRETIZATION_LEVELS,
    average_speed_kmh: float = 75.0,
) -> Tuple[List[str], Dict[str, float], float, float, int]:
    """
    Solves the EV route charging stop selection and energy allocation problem using
    exact Bellman Dynamic Programming over discretized battery state space.

    Returns:
    - best_stops (List[str]): Selected station IDs in route order
    - charge_amounts (Dict[str, float]): Energy charged at each station in kWh
    - objective_value (float): Minimum multi-objective combined score
    - execution_time_seconds (float): Runtime in seconds
    - nodes_explored (int): Total state space transitions evaluated
    """
    start_time = time.perf_counter()
    nodes_explored = 0
    route_dist = route[-1][2] if route else 552.0

    # Sort stations strictly by distance along route corridor
    sorted_stations = sorted(stations, key=lambda s: s.distance_from_start_km)

    # Discretize SoC levels between BATTERY_BUFFER_PERCENT (10%) and 100%
    soc_grid: List[float] = [
        BATTERY_BUFFER_PERCENT + i * ((100.0 - BATTERY_BUFFER_PERCENT) / (soc_levels - 1))
        for i in range(soc_levels)
    ]

    def soc_to_index(soc: float) -> int:
        """Find closest SoC index on grid."""
        best_idx = 0
        min_diff = abs(soc_grid[0] - soc)
        for idx, val in enumerate(soc_grid):
            diff = abs(val - soc)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        return best_idx

    # DP Table: dp[stage_idx][soc_idx] = (min_cost, prev_stage_idx, prev_soc_idx, charge_kwh)
    # Stage 0: Origin (Chennai) at km 0.0
    # Stage 1..N: Stations ST-01..ST-N
    # Stage N+1: Destination (Coimbatore) at km 552.0
    num_stages = len(sorted_stations) + 2
    INF = float("inf")

    dp: List[Dict[int, Tuple[float, int, int, float]]] = [{} for _ in range(num_stages)]

    # Initial state at Stage 0 (Origin)
    start_soc = max(BATTERY_BUFFER_PERCENT, min(100.0, ev.current_soc_percent))
    start_soc_idx = soc_to_index(start_soc)
    dp[0][start_soc_idx] = (0.0, -1, -1, 0.0)

    # All stage locations: [0.0, st1_km, st2_km, ..., 552.0]
    stage_kms = [0.0] + [s.distance_from_start_km for s in sorted_stations] + [route_dist]

    for stage in range(num_stages - 1):
        current_km = stage_kms[stage]
        current_station = sorted_stations[stage - 1] if stage > 0 else None

        for curr_soc_idx, (curr_cost, _, _, _) in list(dp[stage].items()):
            curr_soc = soc_grid[curr_soc_idx]

            # Forward transitions from current stage to next stage
            for next_stage in range(stage + 1, num_stages):
                next_km = stage_kms[next_stage]
                dist = next_km - current_km
                if dist < 0:
                    continue

                nodes_explored += 1

                # Check driving energy needed
                energy_needed = ev.energy_needed(dist)
                soc_drop = (energy_needed / ev.battery_capacity_kwh) * 100.0
                arrival_soc = curr_soc - soc_drop

                # Validate buffer requirement (>= 10%)
                if arrival_soc < BATTERY_BUFFER_PERCENT - 1e-3:
                    # Battery drops below buffer -> cannot reach next_stage without charging first
                    break  # Further stages will be even farther

                # If next_stage is Destination
                if next_stage == num_stages - 1:
                    if arrival_soc >= DESTINATION_RESERVE_PERCENT - 1e-3:
                        # Reached destination safely
                        travel_time_mins = (dist / max(10.0, average_speed_kmh)) * 60.0
                        norm_time = travel_time_mins / 450.0
                        leg_cost = alpha * norm_time

                        total_cost = curr_cost + leg_cost
                        dest_soc_idx = soc_to_index(arrival_soc)

                        if dest_soc_idx not in dp[next_stage] or total_cost < dp[next_stage][dest_soc_idx][0]:
                            dp[next_stage][dest_soc_idx] = (total_cost, stage, curr_soc_idx, 0.0)
                    continue

                # If next_stage is a charging station
                next_station = sorted_stations[next_stage - 1]

                # Check if station can serve vehicle
                if not next_station.can_serve(ev):
                    continue

                # Explore charging target SoC at next_station
                arr_soc_idx = soc_to_index(arrival_soc)

                for target_soc_idx in range(arr_soc_idx, soc_levels):
                    target_soc = soc_grid[target_soc_idx]
                    needed_soc = target_soc - arrival_soc
                    charge_kwh = (needed_soc / 100.0) * ev.battery_capacity_kwh

                    # Calculate charging cost & stay time
                    stay_mins = next_station.total_stay_time_minutes(charge_kwh, ev)
                    c_cost_inr = next_station.charging_cost(charge_kwh)
                    t_mins = (dist / max(10.0, average_speed_kmh)) * 60.0 + stay_mins

                    norm_t = t_mins / 450.0
                    norm_c = c_cost_inr / 800.0 if c_cost_inr > 0 else 0.0

                    stage_utility = (alpha * norm_t) + (beta * norm_c)
                    total_cost = curr_cost + stage_utility

                    if target_soc_idx not in dp[next_stage] or total_cost < dp[next_stage][target_soc_idx][0]:
                        dp[next_stage][target_soc_idx] = (total_cost, stage, curr_soc_idx, charge_kwh)

    # Backtrack optimal solution from destination stage
    dest_stage = num_stages - 1
    if not dp[dest_stage]:
        # Fallback: if no feasible path discovered, pick reachable stations iteratively
        exec_time = time.perf_counter() - start_time
        return [], {}, 999.0, exec_time, nodes_explored

    # Pick state with minimum cost at destination
    best_dest_soc_idx = min(dp[dest_stage].keys(), key=lambda k: dp[dest_stage][k][0])
    
    # Trace back
    curr_s = dest_stage
    curr_soc_i = best_dest_soc_idx
    best_stops_reversed: List[str] = []
    charge_amounts_map: Dict[str, float] = {}

    while curr_s > 0:
        val = dp[curr_s].get(curr_soc_i)
        if not val:
            break
        cost, prev_s, prev_soc_i, charge_kwh = val

        if curr_s > 0 and curr_s <= len(sorted_stations):
            st = sorted_stations[curr_s - 1]
            best_stops_reversed.append(st.id)
            if charge_kwh > 0:
                charge_amounts_map[st.id] = round(charge_kwh, 2)

        curr_s = prev_s
        curr_soc_i = prev_soc_i

    best_stops = list(reversed(best_stops_reversed))

    # Calculate final exact multi-objective score using core objective engine
    obj_res = calculate_objective(ev, best_stops, charge_amounts_map, sorted_stations, route_dist, alpha, beta, average_speed_kmh)
    obj_value = obj_res["combined_score"]

    exec_time = time.perf_counter() - start_time
    return best_stops, charge_amounts_map, obj_value, round(exec_time, 4), nodes_explored
