"""
EVolve: Intelligent EV Route Charging Optimizer
Quantum-Inspired QUBO Optimization Engine (D-Wave Simulated Annealing)
"""

import time
import math
import random
from typing import List, Dict, Any, Tuple, Optional
# pyrefly: ignore [missing-import]
import neal

from core.ev_model import EV
from core.station_model import Station
from core.validator import validate_solution
from core.objective import calculate_objective
from config import (
    BATTERY_BUFFER_PERCENT,
    DESTINATION_RESERVE_PERCENT,
    QUBO_P1,
    QUBO_P2,
    QUBO_P3,
    QUBO_NUM_READS,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)


def build_qubo(
    ev: EV,
    stations: List[Station],
    route: Optional[List[Tuple[float, float, float]]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    P1: float = QUBO_P1,
    P2: float = QUBO_P2,
    P3: float = QUBO_P3,
    min_battery_buffer_percent: float = BATTERY_BUFFER_PERCENT,
    min_destination_reserve_percent: float = DESTINATION_RESERVE_PERCENT,
) -> Dict[Tuple[int, int], float]:
    """
    Constructs the Quadratic Unconstrained Binary Optimization (QUBO) Q-matrix:
    x_i = 1 if stopping at station i, 0 otherwise.

    Objective Terms:
    - Minimize station wait time, charging duration, and dynamic price cost.

    Penalty Terms:
    - P1: Battery empty / buffer violation penalty (< min buffer SoC)
    - P2: Offline station / zero bay availability penalty
    - P3: Destination reserve non-attainment penalty (< min reserve SoC at destination)
    """
    Q: Dict[Tuple[int, int], float] = {}
    N = len(stations)

    if N == 0:
        return Q

    # EV Physical Range Parameters
    safe_start_range_km = ev.range_remaining_km(min_soc_percent=min_battery_buffer_percent)
    consumption_rate = max(0.01, ev.consumption_kwh_per_100km / 100.0)
    full_safe_range_km = ev.battery_capacity_kwh * (1.0 - min_battery_buffer_percent / 100.0) / consumption_rate

    route_dist = route[-1][2] if route else 552.0
    
    # Required range to reach destination with reserve
    energy_req_dest = ev.energy_needed(route_dist)
    soc_req_dest = (energy_req_dest / ev.battery_capacity_kwh) * 100.0
    soc_shortfall = (ev.current_soc_percent - soc_req_dest)

    # Estimate required total kWh top-up across journey
    total_kwh_needed = max(0.0, (min_destination_reserve_percent - (ev.current_soc_percent - soc_req_dest)) * ev.battery_capacity_kwh / 100.0)

    # --- 1. LINEAR DIAGONAL TERMS Q(i, i) ---
    for i in range(N):
        st = stations[i]
        
        # Estimated charge amount if stopping here
        est_charge_kwh = min(
            ev.battery_capacity_kwh * 0.5,
            max(10.0, total_kwh_needed)
        )

        wait_mins = st.wait_time_minutes
        charge_mins = st.charging_time_minutes(est_charge_kwh, ev)
        c_cost = st.charging_cost(est_charge_kwh)

        t_total = wait_mins + charge_mins
        norm_t = t_total / 450.0
        norm_c = c_cost / 800.0 if c_cost > 0 else 0.0

        # Linear objective utility bias
        linear_utility = (alpha * norm_t) + (beta * norm_c)

        # P2 Penalty: Offline station or unusable connector/slots
        p2_penalty = 0.0
        is_usable = st.is_operational and st.available_slots > 0 and st.supports_connector(ev.connector_type)
        if not is_usable:
            p2_penalty += P2 * 5.0

        Q[(i, i)] = linear_utility + p2_penalty

        # Stations within safe initial range get first-leg incentive
        if is_usable and st.distance_from_start_km <= safe_start_range_km:
            Q[(i, i)] -= 60.0

        # Stations within safe range of destination get final-leg arrival incentive
        if is_usable and (route_dist - st.distance_from_start_km) <= full_safe_range_km:
            Q[(i, i)] -= 60.0

    # --- 2. QUADRATIC PAIRWISE COUPLING TERMS Q(i, j) ---
    for i in range(N):
        st_i = stations[i]
        for j in range(i + 1, N):
            st_j = stations[j]
            gap_km = st_j.distance_from_start_km - st_i.distance_from_start_km

            # If gap between stations i and j is too large (> max single charge range), penalize
            if gap_km > full_safe_range_km:
                Q[(i, j)] = Q.get((i, j), 0.0) + P1 * (gap_km / full_safe_range_km)
            elif gap_km < 45.0:
                # Discourage redundant stops that are too close to each other
                Q[(i, j)] = Q.get((i, j), 0.0) + 35.0
            else:
                # Cooperative coupling for reasonable spacing
                Q[(i, j)] = Q.get((i, j), 0.0) - 30.0

    # --- 3. GLOBAL DESTINATION PENALTY P3 ---
    # Penalize configs where zero stops are selected if current range cannot reach destination
    if ev.range_remaining_km(min_soc_percent=min_destination_reserve_percent) < route_dist:
        for i in range(N):
            st = stations[i]
            if not (st.is_operational and st.available_slots > 0 and st.supports_connector(ev.connector_type)):
                continue
            dist_from_mid = abs(st.distance_from_start_km - (route_dist / 2.0))
            Q[(i, i)] -= 15.0 * (1.0 - dist_from_mid / route_dist)

    return Q


def solve(
    ev: EV,
    stations: List[Station],
    route: Optional[List[Tuple[float, float, float]]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    num_reads: int = QUBO_NUM_READS,
    beta_range: Optional[Tuple[float, float]] = (0.1, 10.0),
    average_speed_kmh: float = 75.0,
    min_battery_buffer_percent: float = BATTERY_BUFFER_PERCENT,
    min_destination_reserve_percent: float = DESTINATION_RESERVE_PERCENT,
) -> Tuple[List[str], Dict[str, float], float, float, int, float]:
    """
    Solves the EV route charging stop selection problem using Quantum-Inspired
    Simulated Annealing over the QUBO matrix.

    Returns:
    - best_stops (List[str]): Station IDs chosen by QUBO sampler
    - charge_amounts (Dict[str, float]): Computed energy additions at selected stops
    - objective_value (float): Combined multi-objective utility score
    - execution_time_seconds (float): Execution time in seconds
    - num_reads (int): Total quantum annealing reads sampled
    - feasibility_rate (float): Percentage of sampled reads that were strictly feasible
    """
    start_time = time.perf_counter()

    # Sort stations by distance
    sorted_stations = sorted(stations, key=lambda s: s.distance_from_start_km)
    N = len(sorted_stations)

    if N == 0:
        exec_t = time.perf_counter() - start_time
        return [], {}, 999.0, exec_t, num_reads, 0.0

    route_dist = route[-1][2] if route else 552.0

    # Build QUBO Q matrix
    Q = build_qubo(
        ev, sorted_stations, route, alpha, beta,
        min_battery_buffer_percent=min_battery_buffer_percent,
        min_destination_reserve_percent=min_destination_reserve_percent,
    )

    # Initialize D-Wave Neal Simulated Annealing Sampler
    sampler = neal.SimulatedAnnealingSampler()

    # Execute Quantum-Inspired Annealing Reads with configured beta schedule
    sample_kwargs: Dict[str, Any] = {"num_reads": num_reads}
    if beta_range is not None:
        sample_kwargs["beta_range"] = beta_range

    response = sampler.sample_qubo(Q, **sample_kwargs)

    feasible_count = 0
    found_feasible = False
    best_candidate_stops: List[str] = []
    best_candidate_charges: Dict[str, float] = {}
    best_candidate_score = float("inf")

    # Evaluate Sampled Bitstrings
    for datum in response.data(["sample", "energy"]):
        sample = datum.sample

        # Extract selected station indices
        selected_indices = [i for i in range(N) if sample.get(i, 0) == 1]
        selected_stops = [sorted_stations[i].id for i in selected_indices]

        # Calculate exact charge amounts required for this stop selection
        charge_map = _compute_required_charges(
            ev, sorted_stations, selected_stops, route,
            min_battery_buffer_percent=min_battery_buffer_percent,
            min_destination_reserve_percent=min_destination_reserve_percent,
        )

        # Validate feasibility against all 6 constraints
        is_feasible, violations, _ = validate_solution(
            ev, sorted_stations, selected_stops, charge_map, route,
            average_speed_kmh=average_speed_kmh,
            min_battery_buffer_percent=min_battery_buffer_percent,
            min_destination_reserve_percent=min_destination_reserve_percent,
        )

        # Calculate objective score
        obj_res = calculate_objective(ev, selected_stops, charge_map, sorted_stations, route_dist, alpha, beta, average_speed_kmh)
        score = obj_res["combined_score"]

        if is_feasible:
            feasible_count += 1
            if not found_feasible or score < best_candidate_score:
                found_feasible = True
                best_candidate_score = score
                best_candidate_stops = selected_stops
                best_candidate_charges = charge_map
        elif not found_feasible:
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate_stops = selected_stops
                best_candidate_charges = charge_map

    # Fallback if no feasible read found: compute greedy minimum charge path
    if not found_feasible:
        best_candidate_stops, best_candidate_charges = _fallback_greedy_stops(
            ev, sorted_stations, route,
            min_battery_buffer_percent=min_battery_buffer_percent,
            min_destination_reserve_percent=min_destination_reserve_percent,
        )
        obj_res = calculate_objective(ev, best_candidate_stops, best_candidate_charges, sorted_stations, route_dist, alpha, beta)
        best_candidate_score = obj_res["combined_score"]

    feasibility_rate = (feasible_count / float(num_reads)) * 100.0
    execution_time = time.perf_counter() - start_time

    return (
        best_candidate_stops,
        best_candidate_charges,
        round(best_candidate_score, 4),
        round(execution_time, 4),
        num_reads,
        round(feasibility_rate, 1),
    )


def _compute_required_charges(
    ev: EV,
    stations: List[Station],
    stop_sequence: List[str],
    route: Optional[List[Tuple[float, float, float]]] = None,
    min_battery_buffer_percent: float = BATTERY_BUFFER_PERCENT,
    min_destination_reserve_percent: float = DESTINATION_RESERVE_PERCENT,
) -> Dict[str, float]:
    """
    Computes exact optimal energy (kWh) needed at each selected station in stop_sequence
    to safely reach destination with required reserve and safe buffer above minimum.
    """
    station_map = {st.id: st for st in stations}
    stop_stations = [station_map[sid] for sid in stop_sequence if sid in station_map]
    stop_stations.sort(key=lambda s: s.distance_from_start_km)

    charge_map: Dict[str, float] = {}
    sim_ev = ev.clone()
    curr_km = 0.0

    for idx, st in enumerate(stop_stations):
        dist_leg = st.distance_from_start_km - curr_km
        sim_ev.consume_distance(dist_leg)

        route_dist = route[-1][2] if route else 552.0
        # Distance remaining from this stop to destination
        dist_remaining = route_dist - st.distance_from_start_km

        # Energy needed to reach destination (or next stop) preserving reserve with safety margin
        if idx < len(stop_stations) - 1:
            next_st = stop_stations[idx + 1]
            dist_to_next = next_st.distance_from_start_km - st.distance_from_start_km
            target_kwh_needed = sim_ev.energy_needed(dist_to_next) + ((min_battery_buffer_percent + 3.0) / 100.0) * sim_ev.battery_capacity_kwh
        else:
            target_kwh_needed = sim_ev.energy_needed(dist_remaining) + ((min_destination_reserve_percent + 2.0) / 100.0) * sim_ev.battery_capacity_kwh

        curr_kwh = sim_ev.get_current_kwh()
        shortfall_kwh = max(0.0, target_kwh_needed - curr_kwh)
        actual_charged = sim_ev.charge(shortfall_kwh)
        
        if actual_charged > 0:
            charge_map[st.id] = round(actual_charged, 2)

        curr_km = st.distance_from_start_km

    return charge_map


def _fallback_greedy_stops(
    ev: EV,
    stations: List[Station],
    route: Optional[List[Tuple[float, float, float]]] = None,
    min_battery_buffer_percent: float = BATTERY_BUFFER_PERCENT,
    min_destination_reserve_percent: float = DESTINATION_RESERVE_PERCENT,
) -> Tuple[List[str], Dict[str, float]]:
    """Greedy fallback stop selector."""
    route_dist = route[-1][2] if route else 552.0
    sorted_st = sorted(stations, key=lambda s: s.distance_from_start_km)
    stops = []
    charges = {}
    sim_ev = ev.clone()
    curr_km = 0.0

    for st in sorted_st:
        if not st.is_operational or st.available_slots == 0:
            continue
        if not st.supports_connector(ev.connector_type):
            continue
        dist = st.distance_from_start_km - curr_km
        if dist <= 0:
            continue
        if sim_ev.can_reach(dist, min_arrival_soc=min_battery_buffer_percent):
            rem_dist = route_dist - st.distance_from_start_km
            if not sim_ev.can_reach(rem_dist, min_arrival_soc=min_destination_reserve_percent) or sim_ev.current_soc_percent < 30.0:
                sim_ev.consume_distance(dist)
                curr_km = st.distance_from_start_km
                needed_kwh = min(sim_ev.battery_capacity_kwh * 0.85 - sim_ev.get_current_kwh(), sim_ev.battery_capacity_kwh * 0.6)
                needed_kwh = max(5.0, needed_kwh)
                added = sim_ev.charge(needed_kwh)
                stops.append(st.id)
                charges[st.id] = round(added, 2)
                if sim_ev.can_reach(route_dist - curr_km, min_arrival_soc=min_destination_reserve_percent):
                    break

    return stops, charges
