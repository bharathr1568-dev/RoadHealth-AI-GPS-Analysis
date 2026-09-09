"""
Location Service
Validates GPS coordinates from the browser Geolocation API and
generates map URLs for the citizen report.
"""


def validate_coordinates(latitude: float, longitude: float,
                         accuracy: float = None) -> dict:
    """
    Validate and sanitize GPS coordinates.
    Returns a dict with validation status and cleaned values.
    """
    errors = []

    if latitude is None or longitude is None:
        return {"valid": False, "errors": ["Latitude and longitude are required."]}

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (ValueError, TypeError):
        return {"valid": False, "errors": ["Invalid coordinate format."]}

    if not (-90.0 <= lat <= 90.0):
        errors.append(f"Latitude {lat} is out of range (-90 to 90).")
    if not (-180.0 <= lon <= 180.0):
        errors.append(f"Longitude {lon} is out of range (-180 to 180).")

    if lat == 0.0 and lon == 0.0:
        errors.append("Coordinates (0, 0) likely indicate a GPS error.")

    if errors:
        return {"valid": False, "errors": errors}

    result = {
        "valid": True,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
    }

    if accuracy is not None:
        try:
            acc = float(accuracy)
            result["accuracy"] = round(acc, 1)
        except (ValueError, TypeError):
            result["accuracy"] = None
    else:
        result["accuracy"] = None

    return result


def get_google_maps_url(latitude: float, longitude: float) -> str:
    """Generate a Google Maps URL for the given coordinates."""
    return f"https://www.google.com/maps?q={latitude},{longitude}"


def get_openstreetmap_url(latitude: float, longitude: float,
                          zoom: int = 17) -> str:
    """Generate an OpenStreetMap URL for the given coordinates."""
    return (
        f"https://www.openstreetmap.org/"
        f"?mlat={latitude}&mlon={longitude}#map={zoom}/{latitude}/{longitude}"
    )


def format_location_response(latitude: float, longitude: float,
                              accuracy: float = None,
                              timestamp: str = None) -> dict:
    """
    Build the location section of the API response.
    """
    return {
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "accuracy": round(float(accuracy), 1) if accuracy is not None else None,
        "timestamp": timestamp,
        "google_maps_url": get_google_maps_url(latitude, longitude),
        "openstreetmap_url": get_openstreetmap_url(latitude, longitude),
    }
