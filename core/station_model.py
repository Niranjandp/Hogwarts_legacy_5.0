"""
EVolve: Intelligent EV Route Charging Optimizer
Charging Station Data Model and Operations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import copy
from core.ev_model import EV


@dataclass
class Station:
    """
    Charging station model representing physical location, charger hardware capability,
    connector types, live occupancy telemetry, pricing, and operating constraints.
    """
    id: str
    name: str
    lat: float
    lon: float
    distance_from_start_km: float
    charger_type: str  # "AC" or "DC"
    max_power_kw: float  # e.g., 22.0, 60.0, 120.0, 150.0 kW
    num_slots: int
    available_slots: int
    price_per_kwh: float  # in INR (₹)
    wait_time_minutes: float  # estimated queuing delay
    is_operational: bool = True
    connector_types: List[str] = field(default_factory=lambda: ["CCS2", "Type2"])
    operating_hours: str = "24/7"  # "24/7" or "06:00-22:00"

    def __post_init__(self) -> None:
        """Sanitize operational bounds."""
        self.num_slots = max(1, int(self.num_slots))
        self.available_slots = max(0, min(self.num_slots, int(self.available_slots)))
        self.max_power_kw = max(1.0, float(self.max_power_kw))
        self.price_per_kwh = max(0.0, float(self.price_per_kwh))
        self.wait_time_minutes = max(0.0, float(self.wait_time_minutes))

    @property
    def occupancy_rate(self) -> float:
        """Percentage of total charging bays currently in use."""
        if self.num_slots <= 0:
            return 1.0
        return (self.num_slots - self.available_slots) / float(self.num_slots)

    def is_open_at_hour(self, hour: float = 12.0) -> bool:
        """
        Check if station is within its functional operating hours window.
        hour is expressed as decimal 0.0 to 24.0.
        """
        if self.operating_hours == "24/7" or not self.operating_hours:
            return True

        try:
            # Parse format like "06:00-22:00"
            start_str, end_str = self.operating_hours.split("-")
            start_h = float(start_str.split(":")[0]) + float(start_str.split(":")[1]) / 60.0
            end_h = float(end_str.split(":")[0]) + float(end_str.split(":")[1]) / 60.0

            current = hour % 24.0
            if start_h <= end_h:
                return start_h <= current <= end_h
            else:  # Overnight window e.g. 22:00-06:00
                return current >= start_h or current <= end_h
        except Exception:
            return True  # Fallback to open if parsing fails

    @property
    def ac_connectors(self) -> List[str]:
        """Returns list of AC connectors available at this station."""
        return [
            c for c in self.connector_types
            if any(k in c.upper() for k in ["TYPE2", "TYPE 2", "TYPE1", "UNIVERSAL", "15A", "AC", "MENNEKES"])
        ]

    @property
    def dc_connectors(self) -> List[str]:
        """Returns list of DC Fast connectors available at this station."""
        return [
            c for c in self.connector_types
            if any(k in c.upper() for k in ["CCS", "CCS2", "CHADEMO", "GB/T", "NACS", "DC"])
        ]

    def supports_connector(self, ev_connector: str) -> bool:
        """
        Check if station offers a matching connector for the EV.
        Handles labels like 'CCS2 (DC Fast)' matching 'CCS2'.
        """
        if not ev_connector:
            return True
        # Extract base connector keyword like 'CCS2' from 'CCS2 (DC Fast)'
        ev_clean = ev_connector.split("(")[0].strip().upper()
        return any(
            ev_clean in conn.upper() or conn.upper() in ev_clean or conn.upper() == "UNIVERSAL"
            for conn in self.connector_types
        )

    def can_serve(self, ev: EV, current_time_hour: float = 12.0) -> bool:
        """
        Evaluate full service feasibility for a given EV:
        1. Station must be operational
        2. Station must have at least 1 available bay (or manageable queue)
        3. Connector compatibility match
        4. Operating hours match
        """
        if not self.is_operational:
            return False

        if self.available_slots <= 0 and self.wait_time_minutes > 60:
            return False

        if not self.supports_connector(ev.connector_type):
            return False

        if not self.is_open_at_hour(current_time_hour):
            return False

        return True

    def effective_charging_power_kw(self, ev: Optional[EV] = None) -> float:
        """
        Determine actual charging throughput rate in kW:
        Limited by min(station max power, EV max charging power).
        """
        if ev is not None:
            return min(self.max_power_kw, ev.max_charging_power_kw)
        return self.max_power_kw

    def charging_time_minutes(self, kwh: float, ev: Optional[EV] = None) -> float:
        """
        Calculate total plug time (in minutes) needed to deliver requested energy in kWh.
        """
        if kwh <= 0:
            return 0.0

        effective_power_kw = self.effective_charging_power_kw(ev)
        if effective_power_kw <= 0:
            return 9999.0

        # Plug time in hours = energy (kWh) / power (kW)
        time_hours = kwh / effective_power_kw
        return time_hours * 60.0

    def total_stay_time_minutes(self, kwh: float, ev: Optional[EV] = None) -> float:
        """
        Calculate total elapsed station time: queue wait time + plug charging time.
        """
        return self.wait_time_minutes + self.charging_time_minutes(kwh, ev)

    def charging_cost(self, kwh: float) -> float:
        """
        Calculate monetary charging cost in INR (₹) for requested energy (kWh).
        """
        if kwh <= 0:
            return 0.0
        return kwh * self.price_per_kwh

    def clone(self) -> "Station":
        """Deep copy station state."""
        return copy.deepcopy(self)
