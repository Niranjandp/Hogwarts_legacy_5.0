"""
EVolve: Intelligent EV Route Charging Optimizer
Route Engine and Geographic Coordinate Services
Fallback chain: Google Maps Routes API → OSRM (free, no key) → Static hardcoded waypoints
"""

import math
from typing import List, Tuple, Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None
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


def get_route(api_key: str = "", origin: str = "Chennai", destination: str = "Coimbatore") -> List[Tuple[float, float, float]]:
    """
    Returns the route as a list of 3-tuples:
    [(latitude, longitude, km_from_start), ...]

    Priority fallback chain:
    1. Google Maps Routes API (if api_key is valid and quota is available)
    2. OSRM public routing API + Nominatim geocoding (free, no key required)
    3. Built-in 31-point GPS static highway corridor as last resort
    """
    # 1. Try Google Maps first
    key = api_key or GOOGLE_MAPS_API_KEY
    if key:
        google_route = fetch_google_maps_route(origin, destination, key)
        if google_route:
            return google_route

    # 2. Fall back to OSRM (free, unlimited, uses OpenStreetMap data)
    osrm_route = fetch_osrm_route(origin, destination)
    if osrm_route:
        return osrm_route

    # 3. Last resort: hardcoded Chennai → Coimbatore static waypoints
    return [(wp[0], wp[1], wp[2]) for wp in ROUTE_WAYPOINTS]


def _geocode_city(city_name: str) -> Optional[Tuple[float, float]]:
    """
    Converts a city name to (lat, lon) using Nominatim (OpenStreetMap geocoder).
    Free to use, no API key required.
    """
    if requests is None:
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "EVolve-EV-Optimizer/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def fetch_osrm_route(origin: str, destination: str) -> Optional[List[Tuple[float, float, float]]]:
    """
    Fetches a real driving route between two city names using:
    - Nominatim (OSM) for free geocoding of city names
    - OSRM public demo server for free turn-by-turn routing
    Returns [(lat, lon, cumulative_km), ...] or None on failure.
    """
    if requests is None:
        return None
    try:
        # Geocode both cities
        origin_coords = _geocode_city(origin)
        dest_coords = _geocode_city(destination)
        if not origin_coords or not dest_coords:
            return None

        o_lat, o_lon = origin_coords
        d_lat, d_lon = dest_coords

        # Call OSRM public routing API
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{o_lon},{o_lat};{d_lon},{d_lat}"
            f"?overview=full&geometries=geojson&steps=false"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        coordinates = data["routes"][0]["geometry"]["coordinates"]
        # coordinates are [lon, lat] pairs in GeoJSON format
        points: List[Tuple[float, float, float]] = []
        cum_km = 0.0
        if coordinates:
            points.append((coordinates[0][1], coordinates[0][0], 0.0))
            for i in range(1, len(coordinates)):
                prev = coordinates[i - 1]
                curr = coordinates[i]
                d = haversine_distance(prev[1], prev[0], curr[1], curr[0])
                cum_km += d
                points.append((curr[1], curr[0], round(cum_km, 2)))
        return points if len(points) > 1 else None
    except Exception:
        return None


def _parse_waypoint(wp_str: str) -> dict:
    parts = wp_str.split(',')
    if len(parts) == 2:
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return {
                "location": {
                    "latLng": {
                        "latitude": lat,
                        "longitude": lon
                    }
                }
            }
        except ValueError:
            pass
    return {"address": wp_str}

def fetch_google_maps_route(origin: str, destination: str, api_key: str) -> Optional[List[Tuple[float, float, float]]]:
    """
    Fetches live directions and route waypoints from Google Routes API v2.
    """
    if requests is None:
        return None

    try:
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline",
        }
        body = {
            "origin": _parse_waypoint(origin),
            "destination": _parse_waypoint(destination),
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
