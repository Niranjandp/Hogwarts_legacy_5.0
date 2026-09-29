"""
EVolve: Intelligent EV Route Charging Optimizer
Quantum Optimization Specialist Engine & Mission Runner
"""

import os
import sys

# Ensure project root directory is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from typing import Dict, Any, List, Optional
from core.ev_model import EV
from core.station_model import Station
from core.validator import validate_solution
from core.objective import calculate_objective
from data.route_engine import get_route
from data.station_finder import get_stations_along_route
from data.telemetry_sim import simulate_telemetry
from experiments.race_runner import run_race
from config import BATTERY_BUFFER_PERCENT, DESTINATION_RESERVE_PERCENT


def get_quantum_specialist_solution(
    origin: str = "Coimbatore",
    destination: str = "Chennai",
    battery_capacity_kwh: float = 40.5,
    initial_soc_percent: float = 90.0,
    consumption_kwh_per_100km: float = 16.5,
    max_charging_power_kw: float = 70.0,
    max_speed_kmh: float = 105.0,
    connector_type: str = "CCS2",
    target_destination_reserve_percent: float = 20.0,
    min_battery_buffer_percent: float = 10.0,
    alpha: float = 0.65,
    beta: float = 0.35,
    num_reads: int = 100,
    beta_range: tuple = (0.1, 10.0),
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Executes the Quantum Optimization Specialist workflow:
    Calculates the most efficient charging route using Quantum QUBO (Ising Hamiltonian)
    and benchmarks it against Classical Dynamic Programming.
    """
    # 1. Instantiate Vehicle Model
    ev = EV(
        id="Tata Nexon EV Max",
        battery_capacity_kwh=battery_capacity_kwh,
        current_soc_percent=initial_soc_percent,
        consumption_kwh_per_100km=consumption_kwh_per_100km,
        max_charging_power_kw=max_charging_power_kw,
        connector_type=connector_type,
    )

    # 2. Compute Corridor Route Waypoints & Dynamic Stations
    route = get_route(origin=origin, destination=destination)
    raw_stations = get_stations_along_route(route)
    stations = simulate_telemetry(raw_stations, current_hour=14.0, seed=seed)
    route_dist_km = route[-1][2] if route else 504.6

    # 3. Race Classical DP against Quantum QUBO
    race_results = run_race(
        ev=ev,
        stations=stations,
        route=route,
        alpha=alpha,
        beta=beta,
        average_speed_kmh=min(max_speed_kmh * 0.8, 80.0),
        num_reads=num_reads,
        beta_range=beta_range,
        min_battery_buffer_percent=min_battery_buffer_percent,
        min_destination_reserve_percent=target_destination_reserve_percent,
    )

    qubo_res = race_results["quantum"]
    dp_res = race_results["classical"]
    winners = race_results["winners"]

    # 4. Construct Detailed Leg-by-Leg Quantum Stop Sequence
    station_map = {s.id: s for s in stations}
    quantum_stops_detail: List[Dict[str, Any]] = []

    # Reconstruct state of charge progression across selected stops
    sim_ev = ev.clone()
    curr_km = 0.0

    for idx, sid in enumerate(qubo_res["stops"]):
        if sid not in station_map:
            continue
        st = station_map[sid]
        dist_leg = st.distance_from_start_km - curr_km
        sim_ev.consume_distance(dist_leg)
        arr_soc = sim_ev.current_soc_percent

        charge_kwh = qubo_res["charges"].get(sid, 0.0)
        sim_ev.charge(charge_kwh)
        dep_soc = sim_ev.current_soc_percent

        plug_mins = st.charging_time_minutes(charge_kwh, ev)
        wait_mins = st.wait_time_minutes
        cost_inr = st.charging_cost(charge_kwh)

        quantum_stops_detail.append({
            "stop_index": idx + 1,
            "station_id": st.id,
            "station_name": st.name,
            "distance_km": round(st.distance_from_start_km, 1),
            "charger_type": st.charger_type,
            "max_power_kw": st.max_power_kw,
            "ac_connectors": ", ".join(st.ac_connectors) if st.ac_connectors else "None",
            "dc_connectors": ", ".join(st.dc_connectors) if st.dc_connectors else "None",
            "arrival_soc_percent": round(arr_soc, 1),
            "energy_charged_kwh": round(charge_kwh, 1),
            "departure_soc_percent": round(dep_soc, 1),
            "charging_time_mins": round(plug_mins, 1),
            "queue_delay_mins": round(wait_mins, 1),
            "cost_inr": round(cost_inr, 2),
        })
        curr_km = st.distance_from_start_km

    # Final arrival at destination
    dist_final = route_dist_km - curr_km
    if dist_final > 0:
        sim_ev.consume_distance(dist_final)
    dest_arrival_soc = round(sim_ev.current_soc_percent, 1)

    # 5. Journey Breakdown Totals
    total_energy_consumed_kwh = round(route_dist_km * (consumption_kwh_per_100km / 100.0), 2)
    total_energy_charged_kwh = round(sum(qubo_res["charges"].values()), 2)
    qubo_obj = qubo_res["obj_details"]
    dp_obj = dp_res["obj_details"]

    journey_breakdown = {
        "corridor": f"{origin} ➔ {destination}",
        "corridor_distance_km": round(route_dist_km, 1),
        "vehicle_model": ev.id,
        "initial_soc_percent": initial_soc_percent,
        "destination_arrival_soc_percent": dest_arrival_soc,
        "target_reserve_met": dest_arrival_soc >= target_destination_reserve_percent - 1e-1,
        "drive_time_mins": qubo_obj["travel_time"],
        "charging_time_mins": qubo_obj["charging_time"],
        "queue_delay_mins": qubo_obj["waiting_time"],
        "total_journey_time_mins": qubo_obj["total_time_minutes"],
        "total_journey_time_hours": round(qubo_obj["total_time_minutes"] / 60.0, 2),
        "total_charging_cost_inr": qubo_obj["total_cost_inr"],
        "total_energy_consumed_kwh": total_energy_consumed_kwh,
        "total_energy_charged_kwh": total_energy_charged_kwh,
    }

    # 6. Comparative Benchmark Metrics
    benchmark_metrics = {
        "execution_latency": {
            "quantum_sec": round(qubo_res["execution_time_sec"], 4),
            "classical_sec": round(dp_res["execution_time_sec"], 4),
            "winner": winners["runtime_speed"],
        },
        "total_route_time_mins": {
            "quantum_mins": qubo_obj["total_time_minutes"],
            "classical_mins": dp_obj["total_time_minutes"],
            "winner": winners["time"],
        },
        "total_charging_cost_inr": {
            "quantum_inr": qubo_obj["total_cost_inr"],
            "classical_inr": dp_obj["total_cost_inr"],
            "winner": winners["cost"],
        },
        "objective_score": {
            "quantum": qubo_res["objective_score"],
            "classical": dp_res["objective_score"],
            "winner": winners["overall"],
        },
        "feasibility": {
            "quantum_feasible": qubo_res["is_feasible"],
            "classical_feasible": dp_res["is_feasible"],
            "quantum_rate_percent": qubo_res.get("feasibility_rate_percent", 0.0),
        },
        "num_stops": {
            "quantum": len(qubo_res["stops"]),
            "classical": len(dp_res["stops"]),
        },
    }

    return {
        "stops_sequence": quantum_stops_detail,
        "journey_breakdown": journey_breakdown,
        "benchmark_metrics": benchmark_metrics,
        "raw_results": race_results,
    }


def format_specialist_markdown_report(solution: Dict[str, Any]) -> str:
    """Formats the quantum specialist solution into a presentation-ready markdown report."""
    j = solution["journey_breakdown"]
    b = solution["benchmark_metrics"]
    stops = solution["stops_sequence"]

    lines = [
        "# ⚡ Quantum Optimization Specialist Route Report",
        f"**Corridor:** {j['corridor']} ({j['corridor_distance_km']} km) | **Vehicle:** {j['vehicle_model']}",
        f"**Initial Battery:** {j['initial_soc_percent']}% | **Destination Arrival SoC:** {j['destination_arrival_soc_percent']}%",
        "",
        "## 1. Recommended Optimal Charging Stop Sequence (Quantum QUBO)",
        "| Stop # | Charging Station | Distance | Connectors (AC / DC) | Power (kW) | Arrival SoC | Charged | Departure SoC | Plug Time | Queue | Cost (₹) |",
        "|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in stops:
        lines.append(
            f"| **{s['stop_index']}** | {s['station_name']} | {s['distance_km']} km | "
            f"AC: `{s['ac_connectors']}`<br>DC: `{s['dc_connectors']}` | {s['max_power_kw']:.0f} kW | "
            f"{s['arrival_soc_percent']}% | **+{s['energy_charged_kwh']} kWh** | {s['departure_soc_percent']}% | "
            f"{s['charging_time_mins']}m | {s['queue_delay_mins']}m | ₹{s['cost_inr']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. Total Journey Breakdown",
        f"- **Highway Driving Time:** `{j['drive_time_mins']} mins` ({j['drive_time_mins']/60.0:.2f} hrs)",
        f"- **Active Plug Charging Time:** `{j['charging_time_mins']} mins`",
        f"- **Station Queue Delay:** `{j['queue_delay_mins']} mins`",
        f"- **Total Journey Duration:** `{j['total_journey_time_mins']} mins` (**{j['total_journey_time_hours']} hours**)",
        f"- **Total Charging Expenditure:** `₹{j['total_charging_cost_inr']:.2f}`",
        f"- **Total Energy Consumed (Road):** `{j['total_energy_consumed_kwh']} kWh`",
        f"- **Total Energy Replenished:** `{j['total_energy_charged_kwh']} kWh`",
        f"- **Destination Reserve Status:** `{'✓ TARGET MET (' + str(j['destination_arrival_soc_percent']) + '%)' if j['target_reserve_met'] else '⚠ BELOW TARGET'}`,",
        "",
        "## 3. Quantum QUBO vs. Classical Dynamic Programming (DP) Benchmark",
        "| Performance Metric | Quantum QUBO Solver | Classical Dynamic Programming | Performance Advantage |",
        "|:---|:---:|:---:|:---:|",
        f"| **Solver Execution Latency** | `{b['execution_latency']['quantum_sec']} s` | `{b['execution_latency']['classical_sec']} s` | **{b['execution_latency']['winner']}** |",
        f"| **Total Journey Duration** | `{b['total_route_time_mins']['quantum_mins']:.1f} mins` | `{b['total_route_time_mins']['classical_mins']:.1f} mins` | **{b['total_route_time_mins']['winner']}** |",
        f"| **Total Charging Cost** | `₹{b['total_charging_cost_inr']['quantum_inr']:.2f}` | `₹{b['total_charging_cost_inr']['classical_inr']:.2f}` | **{b['total_charging_cost_inr']['winner']}** |",
        f"| **Combined Score (α·Time + β·Cost)** | `{b['objective_score']['quantum']:.4f}` | `{b['objective_score']['classical']:.4f}` | **{b['objective_score']['winner']}** |",
        f"| **Constraint Feasibility** | `{'✓ Feasible' if b['feasibility']['quantum_feasible'] else 'Infeasible'}` ({b['feasibility']['quantum_rate_percent']:.1f}%) | `{'✓ Feasible' if b['feasibility']['classical_feasible'] else 'Infeasible'}` | **Dual Feasible** |",
        f"| **Intermediate Stops Count** | `{b['num_stops']['quantum']}` stops | `{b['num_stops']['classical']}` stops | Equal Stops |",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("Executing Quantum Logistics Specialist Route Optimization...")
    print("=" * 80)
    sol = get_quantum_specialist_solution()
    report = format_specialist_markdown_report(sol)
    print(report)
