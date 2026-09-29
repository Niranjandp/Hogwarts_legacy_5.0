"""
EVolve: Intelligent EV Route Charging Optimizer
EV State and Dynamics Model
"""

from dataclasses import dataclass, field
import copy
from typing import Optional

from config import (
    BATTERY_BUFFER_PERCENT,
    DEFAULT_CONSUMPTION_KWH_PER_100KM,
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_CURRENT_SOC_PERCENT,
    DEFAULT_MAX_CHARGING_POWER_KW,
    DEFAULT_CONNECTOR_TYPE,
)


@dataclass
class EV:
    """
    Electric Vehicle data model capturing battery dynamics, consumption,
    connector compatibility, and range calculations.
    """
    id: str = "EV-01"
    battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH
    current_soc_percent: float = DEFAULT_CURRENT_SOC_PERCENT
    consumption_kwh_per_100km: float = DEFAULT_CONSUMPTION_KWH_PER_100KM
    max_charging_power_kw: float = DEFAULT_MAX_CHARGING_POWER_KW
    connector_type: str = DEFAULT_CONNECTOR_TYPE  # "CCS2", "CHAdeMO", "Type2", etc.
    priority: int = 1  # 1 (standard) to 3 (emergency/critical)

    def __post_init__(self) -> None:
        """Validate and clamp initial physical state."""
        self.current_soc_percent = max(0.0, min(100.0, float(self.current_soc_percent)))
        self.battery_capacity_kwh = max(1.0, float(self.battery_capacity_kwh))
        self.consumption_kwh_per_100km = max(0.1, float(self.consumption_kwh_per_100km))
        self.max_charging_power_kw = max(1.0, float(self.max_charging_power_kw))

    def energy_needed(self, distance_km: float) -> float:
        """
        Calculate total energy (kWh) consumed over a specified driving distance.
        """
        if distance_km <= 0:
            return 0.0
        return (distance_km * self.consumption_kwh_per_100km) / 100.0

    def range_remaining_km(self, min_soc_percent: float = 0.0) -> float:
        """
        Calculate remaining driving range (km) down to a specified target minimum SoC.
        Default evaluates theoretical maximum range down to 0% SoC.
        """
        usable_soc = max(0.0, self.current_soc_percent - min_soc_percent)
        usable_kwh = (usable_soc / 100.0) * self.battery_capacity_kwh
        if self.consumption_kwh_per_100km <= 0:
            return 0.0
        return (usable_kwh * 100.0) / self.consumption_kwh_per_100km

    def safe_range_km(self) -> float:
        """
        Calculate remaining safe driving range (km) maintaining the standard 10% buffer.
        """
        return self.range_remaining_km(min_soc_percent=BATTERY_BUFFER_PERCENT)

    def charge(self, kwh: float) -> float:
        """
        Charge battery by given energy amount in kWh.
        Clamps at 100% capacity and updates current_soc_percent.
        Returns the actual kWh added.
        """
        if kwh <= 0:
            return 0.0

        current_energy_kwh = (self.current_soc_percent / 100.0) * self.battery_capacity_kwh
        new_energy_kwh = min(self.battery_capacity_kwh, current_energy_kwh + kwh)
        actual_kwh_added = new_energy_kwh - current_energy_kwh
        
        self.current_soc_percent = (new_energy_kwh / self.battery_capacity_kwh) * 100.0
        return actual_kwh_added

    def charge_to_soc(self, target_soc_percent: float) -> float:
        """
        Charge battery up to a specific target SoC percentage (capped at 100%).
        Returns the actual kWh added.
        """
        target = min(100.0, max(self.current_soc_percent, target_soc_percent))
        needed_soc = target - self.current_soc_percent
        needed_kwh = (needed_soc / 100.0) * self.battery_capacity_kwh
        return self.charge(needed_kwh)

    def consume_distance(self, distance_km: float) -> float:
        """
        Simulate vehicle driving distance_km.
        Drains battery based on consumption rate.
        Returns the updated current_soc_percent.
        """
        energy_used_kwh = self.energy_needed(distance_km)
        soc_drop = (energy_used_kwh / self.battery_capacity_kwh) * 100.0
        self.current_soc_percent = max(0.0, self.current_soc_percent - soc_drop)
        return self.current_soc_percent

    def is_battery_safe(self, buffer_percent: float = BATTERY_BUFFER_PERCENT) -> bool:
        """
        Check if the current SoC satisfies the minimum safety buffer threshold.
        """
        return self.current_soc_percent >= buffer_percent

    def can_reach(self, distance_km: float, min_arrival_soc: float = BATTERY_BUFFER_PERCENT) -> bool:
        """
        Verify if the EV can reach a target distance while preserving min_arrival_soc.
        """
        energy_required = self.energy_needed(distance_km)
        soc_required = (energy_required / self.battery_capacity_kwh) * 100.0
        remaining_soc = self.current_soc_percent - soc_required
        return remaining_soc >= min_arrival_soc

    def get_current_kwh(self) -> float:
        """Return the current energy stored in the battery in kWh."""
        return (self.current_soc_percent / 100.0) * self.battery_capacity_kwh

    def clone(self) -> "EV":
        """Deep copy of the current vehicle state."""
        return copy.deepcopy(self)
