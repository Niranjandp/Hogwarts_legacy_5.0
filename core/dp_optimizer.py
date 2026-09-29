"""Classical dynamic-programming optimizer for route charging plans."""

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Tuple

from config import (
    BATTERY_BUFFER_PERCENT,
    DEFAULT_ALPHA,
    DEFAULT_AVERAGE_SPEED_KMH,
    DEFAULT_BETA,
    DESTINATION_RESERVE_PERCENT,
    DP_SOC_DISCRETIZATION_LEVELS,
    TOTAL_ROUTE_DISTANCE_KM,
)
from core.ev_model import EV
from core.objective import calculate_objective
from core.station_model import Station
from core.validator import validate_solution


_TIME_BUCKET_MINUTES = 5


@dataclass(frozen=True)
class ChargingPlan:
    """A charging plan together with its feasibility report and objective metrics."""

    stop_sequence: List[str]
    charge_amounts: Dict[str, float]
    metrics: Dict[str, float]
    is_feasible: bool
    violations: List[str]
    battery_profile: List[Dict[str, Any]]


@dataclass(frozen=True)
class _Label:
    soc_percent: float
    time_hour: float
    score: float
    stop_sequence: Tuple[str, ...]
    charge_amounts: Tuple[Tuple[str, float], ...]


def optimize_charging_plan(
    ev: EV,
    stations: List[Station],
    start_time_hour: float = 8.0,
    average_speed_kmh: float = DEFAULT_AVERAGE_SPEED_KMH,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    soc_levels: int = DP_SOC_DISCRETIZATION_LEVELS,
) -> ChargingPlan:
    """Find a feasible, low-cost charging plan using a discretized SoC dynamic program.

    The search may skip any station and only charges at stations that can serve the
    EV at the projected arrival time. Time buckets preserve distinct scheduling
    states, while the final candidate is checked by the shared solution validator.
    """
    if average_speed_kmh <= 0:
        raise ValueError("average_speed_kmh must be positive")
    if soc_levels < 2:
        raise ValueError("soc_levels must be at least 2")
    if alpha < 0 or beta < 0 or alpha + beta == 0:
        raise ValueError("alpha and beta must be non-negative and not both zero")

    route_stations = sorted(
        (
            station
            for station in stations
            if 0.0 < station.distance_from_start_km < TOTAL_ROUTE_DISTANCE_KM
        ),
        key=lambda station: station.distance_from_start_km,
    )
    soc_grid = [
        BATTERY_BUFFER_PERCENT
        + index
        * (100.0 - BATTERY_BUFFER_PERCENT)
        / (soc_levels - 1)
        for index in range(soc_levels)
    ]
    speed = max(10.0, average_speed_kmh)

    origin = _Label(
        soc_percent=ev.current_soc_percent,
        time_hour=start_time_hour,
        score=0.0,
        stop_sequence=(),
        charge_amounts=(),
    )
    labels_by_location: List[Dict[Tuple[int, int], _Label]] = [{(-1, 0): origin}]

    for station_index, station in enumerate(route_stations):
        station_labels: Dict[Tuple[int, int], _Label] = {}
        for previous_index in range(station_index + 1):
            previous_km = (
                0.0
                if previous_index == 0
                else route_stations[previous_index - 1].distance_from_start_km
            )
            for label in labels_by_location[previous_index].values():
                distance_km = station.distance_from_start_km - previous_km
                arrival_soc = label.soc_percent - (
                    ev.energy_needed(distance_km) / ev.battery_capacity_kwh * 100.0
                )
                if arrival_soc < BATTERY_BUFFER_PERCENT - 1e-9:
                    continue

                arrival_time = label.time_hour + distance_km / speed
                if not station.can_serve(ev, arrival_time):
                    continue

                for target_index, target_soc in enumerate(soc_grid):
                    if target_soc <= arrival_soc + 1e-9:
                        continue

                    charge_kwh = (
                        (target_soc - arrival_soc)
                        / 100.0
                        * ev.battery_capacity_kwh
                    )
                    stay_minutes = station.total_stay_time_minutes(charge_kwh, ev)
                    incremental_score = (
                        alpha * stay_minutes / 450.0
                        + beta * station.charging_cost(charge_kwh) / 800.0
                    )
                    departure_time = arrival_time + stay_minutes / 60.0
                    elapsed_minutes = (departure_time - start_time_hour) * 60.0
                    time_bucket = ceil(elapsed_minutes / _TIME_BUCKET_MINUTES)
                    state_key = (target_index, time_bucket)
                    candidate = _Label(
                        soc_percent=target_soc,
                        time_hour=departure_time,
                        score=label.score + incremental_score,
                        stop_sequence=label.stop_sequence + (station.id,),
                        charge_amounts=label.charge_amounts + ((station.id, charge_kwh),),
                    )
                    current = station_labels.get(state_key)
                    if current is None or (candidate.score, candidate.time_hour) < (
                        current.score,
                        current.time_hour,
                    ):
                        station_labels[state_key] = candidate

        labels_by_location.append(station_labels)

    destination_candidates: List[_Label] = []
    for location_index, labels in enumerate(labels_by_location):
        location_km = (
            0.0
            if location_index == 0
            else route_stations[location_index - 1].distance_from_start_km
        )
        distance_km = TOTAL_ROUTE_DISTANCE_KM - location_km
        for label in labels.values():
            arrival_soc = label.soc_percent - (
                ev.energy_needed(distance_km) / ev.battery_capacity_kwh * 100.0
            )
            if arrival_soc >= DESTINATION_RESERVE_PERCENT - 1e-9:
                destination_candidates.append(label)

    destination_candidates.sort(key=lambda label: (label.score, label.time_hour))
    best_plan: ChargingPlan | None = None
    for label in destination_candidates:
        stops = list(label.stop_sequence)
        charges = dict(label.charge_amounts)
        is_feasible, violations, battery_profile = validate_solution(
            ev,
            stations,
            stops,
            charges,
            start_time_hour=start_time_hour,
            average_speed_kmh=average_speed_kmh,
        )
        metrics = calculate_objective(
            ev,
            stops,
            charges,
            stations,
            alpha=alpha,
            beta=beta,
            average_speed_kmh=average_speed_kmh,
        )
        plan = ChargingPlan(
            stop_sequence=stops,
            charge_amounts=charges,
            metrics=metrics,
            is_feasible=is_feasible,
            violations=violations,
            battery_profile=battery_profile,
        )
        if is_feasible:
            return plan
        if best_plan is None:
            best_plan = plan

    if best_plan is not None:
        return best_plan
    return ChargingPlan(
        stop_sequence=[],
        charge_amounts={},
        metrics={},
        is_feasible=False,
        violations=[
            "No feasible route plan found with the available stations and SoC discretization."
        ],
        battery_profile=[],
    )