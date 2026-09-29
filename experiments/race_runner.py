"""
EVolve: Intelligent EV Route Charging Optimizer
Parallel Algorithm Race Runner & Scalability Experiment Suite
"""

import threading
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from core.ev_model import EV
from core.station_model import Station
from core.validator import validate_solution
from core.objective import calculate_objective
from data.station_finder import get_stations_along_route
from engines.classical_dp import solve as solve_classical_dp
from engines.quantum_qubo import solve as solve_quantum_qubo
from config import DEFAULT_ALPHA, DEFAULT_BETA


def run_race(
    ev: EV,
    stations: List[Station],
    route: Optional[List[Tuple[float, float, float]]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    average_speed_kmh: float = 75.0,
) -> Dict[str, Any]:
    """
    Races Classical Dynamic Programming against Quantum-Inspired QUBO in parallel threads.

    Returns combined results dictionary containing:
    - classical: detailed metric results dict for DP engine
    - quantum: detailed metric results dict for QUBO engine
    - summary: comparison winners on cost, time, runtime, and overall utility score
    """
    classical_result: Dict[str, Any] = {}
    quantum_result: Dict[str, Any] = {}

    def worker_dp():
        nonlocal classical_result
        stops, charges, obj, exec_t, nodes = solve_classical_dp(
            ev, stations, route, alpha, beta, average_speed_kmh=average_speed_kmh
        )
        is_feasible, violations, profile = validate_solution(
            ev, stations, stops, charges, route, average_speed_kmh=average_speed_kmh
        )
        obj_details = calculate_objective(
            ev, stops, charges, stations, alpha, beta, average_speed_kmh=average_speed_kmh
        )
        classical_result = {
            "engine": "Classical Dynamic Programming",
            "stops": stops,
            "charges": charges,
            "objective_score": obj,
            "execution_time_sec": exec_t,
            "nodes_explored": nodes,
            "is_feasible": is_feasible,
            "violations": violations,
            "battery_profile": profile,
            "obj_details": obj_details,
        }

    def worker_qubo():
        nonlocal quantum_result
        stops, charges, obj, exec_t, reads, rate = solve_quantum_qubo(
            ev, stations, route, alpha, beta, average_speed_kmh=average_speed_kmh
        )
        is_feasible, violations, profile = validate_solution(
            ev, stations, stops, charges, route, average_speed_kmh=average_speed_kmh
        )
        obj_details = calculate_objective(
            ev, stops, charges, stations, alpha, beta, average_speed_kmh=average_speed_kmh
        )
        quantum_result = {
            "engine": "Quantum-Inspired QUBO",
            "stops": stops,
            "charges": charges,
            "objective_score": obj,
            "execution_time_sec": exec_t,
            "num_reads": reads,
            "feasibility_rate_percent": rate,
            "is_feasible": is_feasible,
            "violations": violations,
            "battery_profile": profile,
            "obj_details": obj_details,
        }

    # Create parallel execution threads
    thread_dp = threading.Thread(target=worker_dp)
    thread_qubo = threading.Thread(target=worker_qubo)

    start_race = time.perf_counter()
    thread_dp.start()
    thread_qubo.start()

    thread_dp.join()
    thread_qubo.join()
    race_elapsed = round(time.perf_counter() - start_race, 4)

    # Determine Winners
    dp_cost = classical_result["obj_details"]["total_cost_inr"]
    qubo_cost = quantum_result["obj_details"]["total_cost_inr"]

    dp_time = classical_result["obj_details"]["total_time_minutes"]
    qubo_time = quantum_result["obj_details"]["total_time_minutes"]

    dp_score = classical_result["objective_score"]
    qubo_score = quantum_result["objective_score"]

    winner_cost = "Classical DP" if dp_cost <= qubo_cost else "Quantum-Inspired"
    winner_time = "Classical DP" if dp_time <= qubo_time else "Quantum-Inspired"
    winner_overall = "Classical DP" if dp_score <= qubo_score else "Quantum-Inspired"
    winner_speed = (
        "Classical DP"
        if classical_result["execution_time_sec"] <= quantum_result["execution_time_sec"]
        else "Quantum-Inspired"
    )

    return {
        "classical": classical_result,
        "quantum": quantum_result,
        "race_wall_time_sec": race_elapsed,
        "winners": {
            "cost": winner_cost,
            "time": winner_time,
            "overall": winner_overall,
            "runtime_speed": winner_speed,
        },
    }


def run_experiments(
    ev: Optional[EV] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    runs_per_size: int = 5,
) -> pd.DataFrame:
    """
    Executes scalability experiment benchmarks across 3 problem corridor sizes:
    - 5 stations
    - 10 stations
    - 14 stations

    Records mean/best/worst runtime, feasibility rates, and objective values.
    Returns a clean pandas DataFrame.
    """
    if ev is None:
        ev = EV()

    full_stations = get_stations_along_route()
    problem_sizes = [5, 10, 14]
    rows: List[Dict[str, Any]] = []

    for size in problem_sizes:
        # Sample subset evenly spaced along route
        if size == 14:
            sub_stations = full_stations
        else:
            indices = np.linspace(0, len(full_stations) - 1, size, dtype=int)
            sub_stations = [full_stations[i] for i in indices]

        dp_runtimes = []
        dp_objs = []
        dp_feas_count = 0

        qubo_runtimes = []
        qubo_objs = []
        qubo_feas_count = 0
        qubo_rates = []

        for _ in range(runs_per_size):
            # DP Run
            stops_dp, charges_dp, obj_dp, t_dp, _ = solve_classical_dp(ev, sub_stations, alpha=alpha, beta=beta)
            feas_dp, _, _ = validate_solution(ev, sub_stations, stops_dp, charges_dp)
            dp_runtimes.append(t_dp)
            dp_objs.append(obj_dp)
            if feas_dp:
                dp_feas_count += 1

            # QUBO Run
            stops_q, charges_q, obj_q, t_q, reads_q, rate_q = solve_quantum_qubo(ev, sub_stations, alpha=alpha, beta=beta)
            feas_q, _, _ = validate_solution(ev, sub_stations, stops_q, charges_q)
            qubo_runtimes.append(t_q)
            qubo_objs.append(obj_q)
            qubo_rates.append(rate_q)
            if feas_q:
                qubo_feas_count += 1

        rows.append({
            "Problem Size (Stations)": size,
            "Classical Mean Time (s)": round(float(np.mean(dp_runtimes)), 4),
            "Classical Best Time (s)": round(float(np.min(dp_runtimes)), 4),
            "Classical Worst Time (s)": round(float(np.max(dp_runtimes)), 4),
            "Classical Feasible %": round((dp_feas_count / runs_per_size) * 100.0, 1),
            "Classical Mean Obj": round(float(np.mean(dp_objs)), 4),
            "Quantum Mean Time (s)": round(float(np.mean(qubo_runtimes)), 4),
            "Quantum Best Time (s)": round(float(np.min(qubo_runtimes)), 4),
            "Quantum Worst Time (s)": round(float(np.max(qubo_runtimes)), 4),
            "Quantum Feasible %": round((qubo_feas_count / runs_per_size) * 100.0, 1),
            "Quantum Mean Obj": round(float(np.mean(qubo_objs)), 4),
            "Quantum Sampler Feasibility %": round(float(np.mean(qubo_rates)), 1),
        })

    return pd.DataFrame(rows)
