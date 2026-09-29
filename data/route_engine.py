"""
EVolve: Intelligent EV Route Charging Optimizer
Route Engine and Geographic Coordinate Services with optional Google Maps Platform integration
"""

import math
import requests
from typing import List, Tuple, Dict, Any, Optional
from config import ROUTE_WAYPOINTS, TOTAL_ROUTE_DISTANCE_KM, GOOGLE_MAPS_API_KEY


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface (in km).
    Uses the standard Haversine formula.
    """
    R = 6371.0  # Earth's mean radius in kilometers

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_route(api_key: str = "") -> List[Tuple[float, float, float]]:
    """
    Returns the Chennai to Coimbatore route as a list of 3-tuples:
    [(latitude, longitude, km_from_start), ...]

    If a valid Google Maps API Key is passed (or set in environment),
    it fetches live route polyline waypoints via Google Routes API.
    Otherwise, it returns the built-in 31-point GPS NH48/NH44/NH544 highway corridor route.
    """
    key = api_key or GOOGLE_MAPS_API_KEY
    if key:
        google_route = fetch_google_maps_route("Chennai", "Coimbatore", key)
        if google_route:
            return google_route

    return [(wp[0], wp[1], wp[2]) for wp in ROUTE_WAYPOINTS]


def fetch_google_maps_route(origin: str, destination: str, api_key: str) -> Optional[List[Tuple[float, float, float]]]:
    """
    Fetches live directions and route waypoints from Google Routes API v2.
    """
    try:
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline",
        }
        body = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
        }

        response = requests.post(url, json=body, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            routes = data.get("routes", [])
            if routes and "polyline" in routes[0]:
                encoded = routes[0]["polyline"]["encodedPolyline"]
                decoded_coords = _decode_polyline(encoded)
                
                # Convert to (lat, lon, cumulative_km)
                points: List[Tuple[float, float, float]] = []
                cum_km = 0.0
                if decoded_coords:
                    points.append((decoded_coords[0][0], decoded_coords[0][1], 0.0))
                    for i in range(1, len(decoded_coords)):
                        prev = decoded_coords[i - 1]
                        curr = decoded_coords[i]
                        d = haversine_distance(prev[0], prev[1], curr[0], curr[1])
                        cum_km += d
                        points.append((curr[0], curr[1], round(cum_km, 2)))
                return points
    except Exception:
        pass  # Graceful fallback to default route

    return None


def _decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """Decodes Google Maps Encoded Polyline algorithm into (lat, lon) list."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)

    while index < length:
        b, shift, result = 0, 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / 100000.0, lng / 100000.0))

    return coordinates


def get_route_waypoints() -> List[Tuple[float, float, float, str]]:
    """
    Returns the complete route waypoints metadata including names:
    [(latitude, longitude, km_from_start, location_name), ...]
    """
    return list(ROUTE_WAYPOINTS)


def get_route_distance() -> float:
    """
    Returns total highway route distance in kilometers (Chennai to Coimbatore).
    """
    return TOTAL_ROUTE_DISTANCE_KM


def get_waypoint_at_km(target_km: float) -> Tuple[float, float, float, str]:
    """
    Finds and returns the nearest route waypoint for a given kilometer marker:
    (latitude, longitude, km_from_start, location_name)
    """
    target_km = max(0.0, min(TOTAL_ROUTE_DISTANCE_KM, target_km))
    
    best_wp = ROUTE_WAYPOINTS[0]
    min_diff = abs(ROUTE_WAYPOINTS[0][2] - target_km)

    for wp in ROUTE_WAYPOINTS[1:]:
        diff = abs(wp[2] - target_km)
        if diff < min_diff:
            min_diff = diff
            best_wp = wp

    return best_wp


def interpolate_route_point(target_km: float) -> Tuple[float, float, float, str]:
    """
    Interpolates exact (lat, lon) coordinates at a specific kilometer marker
    between adjacent route waypoints.
    """
    target_km = max(0.0, min(TOTAL_ROUTE_DISTANCE_KM, target_km))

    if target_km <= ROUTE_WAYPOINTS[0][2]:
        return ROUTE_WAYPOINTS[0]
    if target_km >= ROUTE_WAYPOINTS[-1][2]:
        return ROUTE_WAYPOINTS[-1]

    for i in range(len(ROUTE_WAYPOINTS) - 1):
        wp1 = ROUTE_WAYPOINTS[i]
        wp2 = ROUTE_WAYPOINTS[i + 1]

        if wp1[2] <= target_km <= wp2[2]:
            segment_length = wp2[2] - wp1[2]
            if segment_length <= 1e-6:
                return wp1

            fraction = (target_km - wp1[2]) / segment_length
            lat = wp1[0] + fraction * (wp2[0] - wp1[0])
            lon = wp1[1] + fraction * (wp2[1] - wp1[1])
            name = f"Interpolated Corridor Point @ {target_km:.1f} km ({wp1[3]} -> {wp2[3]})"
            return (lat, lon, target_km, name)

    return ROUTE_WAYPOINTS[-1]
