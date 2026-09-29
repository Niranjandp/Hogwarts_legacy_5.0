"""
EVolve: Intelligent EV Route Charging Optimizer
Synthetic Charging Station Generator and Spatial Finder
"""

from typing import List, Optional, Tuple, Dict, Any
from core.station_model import Station
from data.route_engine import get_route, get_waypoint_at_km, interpolate_route_point

# Master list of 14 realistic synthetic charging stations along the Chennai -> Coimbatore highway
STATION_DEFINITIONS = [
    {
        "id": "ST-01",
        "name": "TATA Power EZ Charge - Sriperumbudur",
        "km_mark": 35.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 3,
        "price_per_kwh": 14.5,
        "wait_time_minutes": 0.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-02",
        "name": "Relux Electric - Kanchipuram Hub",
        "km_mark": 70.0,
        "charger_type": "DC",
        "max_power_kw": 120.0,
        "num_slots": 6,
        "available_slots": 4,
        "price_per_kwh": 16.0,
        "wait_time_minutes": 5.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-03",
        "name": "Ather Grid - Walajapet Plaza",
        "km_mark": 105.0,
        "charger_type": "AC",
        "max_power_kw": 22.0,
        "num_slots": 4,
        "available_slots": 2,
        "price_per_kwh": 9.5,
        "wait_time_minutes": 10.0,
        "connector_types": ["Type2", "UNIVERSAL"],
        "operating_hours": "06:00-22:00",
    },
    {
        "id": "ST-04",
        "name": "TATA Power EZ Charge - Vellore Fort",
        "km_mark": 140.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 2,
        "price_per_kwh": 14.0,
        "wait_time_minutes": 8.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-05",
        "name": "Zeon Charging - Pallikonda Supercharger",
        "km_mark": 175.0,
        "charger_type": "DC",
        "max_power_kw": 150.0,
        "num_slots": 8,
        "available_slots": 5,
        "price_per_kwh": 17.5,
        "wait_time_minutes": 0.0,
        "connector_types": ["CCS2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-06",
        "name": "Jio-bp pulse - Ambur Highway Rest Stop",
        "km_mark": 210.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 1,
        "price_per_kwh": 15.0,
        "wait_time_minutes": 15.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-07",
        "name": "ChargeZONE - Vaniyambadi Expressway",
        "km_mark": 245.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 3,
        "price_per_kwh": 13.5,
        "wait_time_minutes": 0.0,
        "connector_types": ["CCS2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-08",
        "name": "TATA Power EZ Charge - Krishnagiri Junction",
        "km_mark": 270.0,
        "charger_type": "DC",
        "max_power_kw": 120.0,
        "num_slots": 6,
        "available_slots": 2,
        "price_per_kwh": 16.5,
        "wait_time_minutes": 12.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-09",
        "name": "Kevic EV Park - Dharmapuri Bypass",
        "km_mark": 310.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 2,
        "price_per_kwh": 13.8,
        "wait_time_minutes": 5.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-10",
        "name": "Shell Recharge - Thoppur Ghats",
        "km_mark": 350.0,
        "charger_type": "DC",
        "max_power_kw": 150.0,
        "num_slots": 6,
        "available_slots": 4,
        "price_per_kwh": 18.0,
        "wait_time_minutes": 0.0,
        "connector_types": ["CCS2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-11",
        "name": "Ather Grid - Salem Expressway Junction",
        "km_mark": 390.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 6,
        "available_slots": 3,
        "price_per_kwh": 14.2,
        "wait_time_minutes": 5.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-12",
        "name": "Zeon Charging - Sankari Highway Station",
        "km_mark": 435.0,
        "charger_type": "DC",
        "max_power_kw": 150.0,
        "num_slots": 8,
        "available_slots": 6,
        "price_per_kwh": 17.0,
        "wait_time_minutes": 0.0,
        "connector_types": ["CCS2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-13",
        "name": "TATA Power EZ Charge - Perundurai SIPCOT",
        "km_mark": 475.0,
        "charger_type": "DC",
        "max_power_kw": 60.0,
        "num_slots": 4,
        "available_slots": 2,
        "price_per_kwh": 14.5,
        "wait_time_minutes": 8.0,
        "connector_types": ["CCS2", "Type2"],
        "operating_hours": "24/7",
    },
    {
        "id": "ST-14",
        "name": "Jio-bp pulse - Avinashi Highway Node",
        "km_mark": 515.0,
        "charger_type": "DC",
        "max_power_kw": 120.0,
        "num_slots": 6,
        "available_slots": 4,
        "price_per_kwh": 15.5,
        "wait_time_minutes": 2.0,
        "connector_types": ["CCS2"],
        "operating_hours": "24/7",
    },
]


def get_stations_along_route(route: Optional[List[Tuple[float, float, float]]] = None) -> List[Station]:
    """
    Constructs and returns the list of 14 synthetic charging stations along the Chennai-Coimbatore route.
    If route is provided, station GPS coordinates are interpolated precisely along the route path.
    """
    stations: List[Station] = []

    for defn in STATION_DEFINITIONS:
        km = defn["km_mark"]
        point = interpolate_route_point(km)
        lat, lon = point[0], point[1]

        station = Station(
            id=defn["id"],
            name=defn["name"],
            lat=lat,
            lon=lon,
            distance_from_start_km=km,
            charger_type=defn["charger_type"],
            max_power_kw=defn["max_power_kw"],
            num_slots=defn["num_slots"],
            available_slots=defn["available_slots"],
            price_per_kwh=defn["price_per_kwh"],
            wait_time_minutes=defn["wait_time_minutes"],
            is_operational=True,
            connector_types=defn["connector_types"],
            operating_hours=defn["operating_hours"],
        )
        stations.append(station)

    return stations
