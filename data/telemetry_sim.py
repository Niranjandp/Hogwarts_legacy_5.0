"""
EVolve: Intelligent EV Route Charging Optimizer
Live Telemetry & Stochastic Dynamic Station Simulator
"""

import random
import math
from typing import List, Optional
from core.station_model import Station
from config import MIN_PRICE_PER_KWH, MAX_PRICE_PER_KWH, PEAK_HOURS_SURCHARGE_FACTOR


def simulate_telemetry(
    stations: List[Station],
    current_hour: float = 14.0,
    seed: Optional[int] = None
) -> List[Station]:
    """
    Simulate real-time live telemetry for a collection of charging stations.
    Occupancy, wait times, pricing, and operational status dynamically shift based on:
    - Time of day (peak vs off-peak commuting hours)
    - Random fluctuations (queuing spikes, minor maintenance)
    """
    if seed is not None:
        random.seed(seed)

    updated_stations: List[Station] = []

    # Peak hours: 08:00 - 11:00 (morning rush) and 17:00 - 21:00 (evening rush)
    is_peak = (8.0 <= current_hour <= 11.0) or (17.0 <= current_hour <= 21.0)

    for st in stations:
        station_copy = st.clone()

        # 1. Operational status: ~7% probability of maintenance outage unless forced
        # Keep primary fast chargers reliable but simulate realistic uptime (93%)
        if random.random() < 0.07:
            station_copy.is_operational = False
            station_copy.available_slots = 0
            station_copy.wait_time_minutes = 999.0
            updated_stations.append(station_copy)
            continue
        else:
            station_copy.is_operational = True

        # 2. Occupancy & Available Slots
        # Peak hours see higher occupancy (50% - 100%), off-peak (10% - 70%)
        if is_peak:
            occupancy_factor = random.uniform(0.50, 0.95)
        else:
            occupancy_factor = random.uniform(0.10, 0.65)

        used_slots = int(math.floor(station_copy.num_slots * occupancy_factor))
        available = max(0, station_copy.num_slots - used_slots)
        station_copy.available_slots = available

        # 3. Queuing Wait Time (0 to 45 minutes based on occupancy)
        if available == 0:
            # All bays full -> queue wait time increases proportionally to charger power
            base_wait = random.uniform(15.0, 45.0)
            # Faster chargers clear queues faster
            wait_mult = 60.0 / max(30.0, station_copy.max_power_kw)
            station_copy.wait_time_minutes = round(base_wait * wait_mult, 1)
        elif available == 1 and is_peak:
            station_copy.wait_time_minutes = round(random.uniform(2.0, 10.0), 1)
        else:
            station_copy.wait_time_minutes = 0.0

        # 4. Time-of-Use Dynamic Tariff Pricing (₹8 - ₹18 / kWh)
        # Base price per station charger tier (DC fast chargers priced higher)
        if station_copy.max_power_kw >= 120.0:
            base_price = 16.0
        elif station_copy.max_power_kw >= 60.0:
            base_price = 14.0
        else:
            base_price = 9.5

        # Apply time-of-day peak multiplier
        if is_peak:
            dynamic_price = base_price * PEAK_HOURS_SURCHARGE_FACTOR
        else:
            dynamic_price = base_price * random.uniform(0.95, 1.05)

        # Clamp price strictly between MIN_PRICE_PER_KWH and MAX_PRICE_PER_KWH
        station_copy.price_per_kwh = round(
            max(MIN_PRICE_PER_KWH, min(MAX_PRICE_PER_KWH, dynamic_price)), 2
        )

        updated_stations.append(station_copy)

    return updated_stations


def update_single_station(
    station: Station,
    is_operational: Optional[bool] = None,
    available_slots: Optional[int] = None,
    wait_time_minutes: Optional[float] = None,
    price_per_kwh: Optional[float] = None
) -> Station:
    """
    Explicitly mutate a single station's state (used for dynamic event simulation,
    e.g., user clicking 'Station X Goes Offline' in the dashboard).
    """
    updated = station.clone()

    if is_operational is not None:
        updated.is_operational = is_operational
        if not is_operational:
            updated.available_slots = 0
            updated.wait_time_minutes = 999.0

    if available_slots is not None and updated.is_operational:
        updated.available_slots = max(0, min(updated.num_slots, available_slots))

    if wait_time_minutes is not None and updated.is_operational:
        updated.wait_time_minutes = max(0.0, wait_time_minutes)

    if price_per_kwh is not None:
        updated.price_per_kwh = max(0.0, price_per_kwh)

    return updated


def tick_live_telemetry(
    stations: List[Station],
    current_hour: float = 14.0,
    offline_id: Optional[str] = None
) -> List[Station]:
    """
    Simulate real-time micro-fluctuations over a 25-second interval:
    - Vehicles arrive or depart (available slots shift realistically by +/- 1)
    - Queue waiting times update accordingly
    - Dynamic energy tariffs fluctuate slightly based on real-time grid load (+/- 0.10 to 0.30 INR)
    - Preserves user-triggered offline states
    """
    is_peak = (8.0 <= current_hour <= 11.0) or (17.0 <= current_hour <= 21.0)
    updated: List[Station] = []

    for st in stations:
        copy_st = st.clone()

        # If manually marked offline, preserve offline status
        if offline_id and str(offline_id).strip() != "None" and copy_st.id == str(offline_id).strip():
            copy_st.is_operational = False
            copy_st.available_slots = 0
            copy_st.wait_time_minutes = 999.0
            updated.append(copy_st)
            continue

        if not copy_st.is_operational:
            # 15% chance of maintenance finishing on tick, otherwise stay offline
            if random.random() < 0.15:
                copy_st.is_operational = True
                copy_st.available_slots = random.randint(1, copy_st.num_slots)
                copy_st.wait_time_minutes = 0.0
            updated.append(copy_st)
            continue

        # Real-time event roll:
        # In a 25s window, ~40% chance of a bay transition
        event_roll = random.random()
        arrival_prob = 0.25 if is_peak else 0.15
        departure_prob = 0.15 if is_peak else 0.25

        if event_roll < arrival_prob and copy_st.available_slots > 0:
            # Car arrived & started charging
            copy_st.available_slots -= 1
        elif event_roll > (1.0 - departure_prob) and copy_st.available_slots < copy_st.num_slots:
            # Car finished charging & departed
            copy_st.available_slots += 1

        # Queue wait time updates
        if copy_st.available_slots == 0:
            # Bays full -> queue delay
            base_wait = copy_st.wait_time_minutes if copy_st.wait_time_minutes > 0 else random.uniform(10.0, 25.0)
            shift = random.choice([-2.0, -1.0, 0.0, 1.0, 3.0])
            copy_st.wait_time_minutes = round(max(3.0, min(50.0, base_wait + shift)), 0)
        elif copy_st.available_slots == 1 and is_peak:
            copy_st.wait_time_minutes = round(random.choice([0.0, 3.0, 5.0]), 0)
        else:
            copy_st.wait_time_minutes = 0.0

        # Dynamic tariff pricing micro-fluctuations (+/- 0.15 INR)
        price_delta = round(random.uniform(-0.20, 0.20), 2)
        new_price = round(copy_st.price_per_kwh + price_delta, 2)
        copy_st.price_per_kwh = max(MIN_PRICE_PER_KWH, min(MAX_PRICE_PER_KWH, new_price))

        updated.append(copy_st)

    return updated
