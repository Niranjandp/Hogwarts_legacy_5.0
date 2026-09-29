"""
EVolve: Intelligent EV Route Charging Optimizer
Synthetic Charging Station Generator and Spatial Finder
"""

from typing import List, Optional, Tuple, Dict, Any
import requests
from core.station_model import Station
from data.route_engine import get_route, get_waypoint_at_km, interpolate_route_point, haversine_distance
from config import USE_OCM_API, OCM_API_KEY

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


def interpolate_route_list(route: List[Tuple[float, float, float]], target_km: float) -> Tuple[float, float]:
    if not route:
        return 0.0, 0.0
    if target_km <= route[0][2]:
        return route[0][0], route[0][1]
    if target_km >= route[-1][2]:
        return route[-1][0], route[-1][1]
        
    for i in range(len(route) - 1):
        wp1 = route[i]
        wp2 = route[i + 1]
        if wp1[2] <= target_km <= wp2[2]:
            segment = wp2[2] - wp1[2]
            if segment <= 1e-6:
                return wp1[0], wp1[1]
            frac = (target_km - wp1[2]) / segment
            lat = wp1[0] + frac * (wp2[0] - wp1[0])
            lon = wp1[1] + frac * (wp2[1] - wp1[1])
            return lat, lon
    return route[-1][0], route[-1][1]


def fetch_ocm_stations(route: List[Tuple[float, float, float]]) -> List[Station]:
    """Fetch live EV charging stations from OpenChargeMap along the route."""
    stations_dict = {}
    total_km = route[-1][2] if route else 0.0
    
    # Sample points along the route every ~100 km to query OCM
    sample_points = []
    current_km = 0.0
    while current_km <= total_km:
        lat, lon = interpolate_route_list(route, current_km)
        sample_points.append((lat, lon))
        current_km += 100.0
    # ensure destination is included
    if route:
        sample_points.append((route[-1][0], route[-1][1]))
    
    for lat, lon in sample_points:
        url = f"https://api.openchargemap.io/v3/poi?key={OCM_API_KEY}&latitude={lat}&longitude={lon}&distance=50&distanceunit=KM&maxresults=50"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    st_id = str(item.get("ID"))
                    if st_id in stations_dict:
                        continue
                        
                    # Extract basics
                    title = item.get("AddressInfo", {}).get("Title", f"Station {st_id}")
                    s_lat = item.get("AddressInfo", {}).get("Latitude", lat)
                    s_lon = item.get("AddressInfo", {}).get("Longitude", lon)
                    
                    # Project onto route to find distance_from_start_km
                    min_d = float('inf')
                    best_km = 0.0
                    for r_lat, r_lon, r_km in route:
                        d = haversine_distance(s_lat, s_lon, r_lat, r_lon)
                        if d < min_d:
                            min_d = d
                            best_km = r_km
                            
                    # Connections
                    connections = item.get("Connections", [])
                    max_power = 22.0
                    num_slots = item.get("NumberOfPoints", 1) or 1
                    charger_type = "AC"
                    conn_types = []
                    
                    for c in connections:
                        power = c.get("PowerKW")
                        if power and power > max_power:
                            max_power = power
                        lvl = c.get("LevelID")
                        if lvl == 3:
                            charger_type = "DC"
                        type_id = c.get("ConnectionTypeID")
                        if type_id == 33:
                            conn_types.append("CCS2")
                        elif type_id == 25:
                            conn_types.append("Type2")
                        elif type_id == 2:
                            conn_types.append("CHAdeMO")
                    
                    if not conn_types:
                        conn_types = ["CCS2", "Type2"]
                        
                    price = 15.0  # fallback simulated price
                    
                    st = Station(
                        id=f"OCM-{st_id}",
                        name=title,
                        lat=s_lat,
                        lon=s_lon,
                        distance_from_start_km=best_km,
                        charger_type=charger_type,
                        max_power_kw=max_power,
                        num_slots=num_slots,
                        available_slots=num_slots,  # Telemetry_sim will overwrite this
                        price_per_kwh=price,
                        wait_time_minutes=0.0,
                        is_operational=True,
                        connector_types=conn_types,
                        operating_hours="24/7"
                    )
                    stations_dict[st_id] = st
        except Exception as e:
            print(f"OCM Fetch Error: {e}")
            
    # Sort by route distance
    sorted_stations = sorted(list(stations_dict.values()), key=lambda x: x.distance_from_start_km)
    return sorted_stations

def get_stations_along_route(route: Optional[List[Tuple[float, float, float]]] = None) -> List[Station]:
    """
    Constructs and returns the list of synthetic charging stations.
    If route is provided, station GPS coordinates are dynamically generated along the route path.
    """
    if USE_OCM_API and route:
        ocm_stations = fetch_ocm_stations(route)
        if ocm_stations:
            return ocm_stations

    stations: List[Station] = []
    
    if route and len(route) > 0:
        total_km = route[-1][2]
        import random
        # Seed for consistency across runs so stations don't move randomly on every render
        random.seed(42)
        
        current_km = 0.0
        station_id = 1
        
        while current_km < total_km:
            # Place a station every 40 to 60 km
            step = random.uniform(40.0, 60.0)
            current_km += step
            
            if current_km >= total_km:
                break
                
            lat, lon = interpolate_route_list(route, current_km)
            
            charger_type = random.choice(["AC", "DC", "DC", "DC"])
            max_power = random.choice([50.0, 60.0, 120.0, 150.0]) if charger_type == "DC" else 22.0
            num_slots = random.randint(2, 8)
            
            station = Station(
                id=f"ST-{station_id:03d}",
                name=f"Highway Route Station {station_id}",
                lat=lat,
                lon=lon,
                distance_from_start_km=current_km,
                charger_type=charger_type,
                max_power_kw=max_power,
                num_slots=num_slots,
                available_slots=random.randint(1, num_slots),
                price_per_kwh=round(random.uniform(12.0, 18.0), 2),
                wait_time_minutes=random.choice([0.0, 5.0, 10.0, 15.0]),
                is_operational=True,
                connector_types=["CCS2", "Type2"] if charger_type == "DC" else ["Type2"],
                operating_hours="24/7",
            )
            stations.append(station)
            station_id += 1
            
        return stations

    # Fallback to the original Chennai-Coimbatore definitions if no route is provided
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
