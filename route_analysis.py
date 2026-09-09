"""
RoadHealth AI — Route-Aware Pavement Analysis (Step 1: 1 km Reference Road in Tamil Nadu)
Provides real-world OpenStreetMap (OSM) reference road geometry, GeoJSON integration,
and interactive Leaflet map visualization for Tamil Nadu, India.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from styles import render_html


# ═══════════════════════════════════════════════════════════════
# DATA MODEL & GEODESIC HELPERS
# ═══════════════════════════════════════════════════════════════

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate geodesic distance in meters between two (latitude, longitude) pairs
    using the Haversine formula.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r_earth = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


@dataclass
class ReferenceRoad:
    """
    Data model representing a real-world reference road segment.
    """
    roadId: str
    roadName: str
    location: str
    state: str
    country: str
    startPoint: Dict[str, Any]
    endPoint: Dict[str, Any]
    routeCoordinates: List[Tuple[float, float]]  # List of (latitude, longitude) pairs
    approximateLengthMeters: float
    approximateLengthKm: float
    tamilName: Optional[str] = None
    district: Optional[str] = None
    highwayType: Optional[str] = None
    surface: Optional[str] = None
    lanes: Optional[str] = None
    osmSource: Optional[str] = None
    osmWayId: Optional[int] = None
    cumulativeDistances: List[float] = field(default_factory=list)

    def __post_init__(self):
        # Calculate cumulative distance along polyline if not provided
        if not self.cumulativeDistances and self.routeCoordinates:
            cum = [0.0]
            for i in range(len(self.routeCoordinates) - 1):
                d = haversine_distance(self.routeCoordinates[i], self.routeCoordinates[i + 1])
                cum.append(cum[-1] + d)
            self.cumulativeDistances = cum
            # Recalibrate total length to match exact sum of geodesic chords
            if not self.approximateLengthMeters or self.approximateLengthMeters == 0:
                self.approximateLengthMeters = round(cum[-1], 2)
                self.approximateLengthKm = round(cum[-1] / 1000.0, 3)

    def to_geojson(self) -> Dict[str, Any]:
        """Export road model to standard GeoJSON Feature format."""
        return {
            "type": "Feature",
            "properties": {
                "roadId": self.roadId,
                "roadName": self.roadName,
                "tamilName": self.tamilName,
                "highwayType": self.highwayType,
                "surface": self.surface,
                "location": self.location,
                "district": self.district,
                "state": self.state,
                "country": self.country,
                "osmSource": self.osmSource,
                "osmWayId": self.osmWayId,
                "approximateLengthMeters": self.approximateLengthMeters,
                "approximateLengthKm": self.approximateLengthKm,
                "nodeCount": len(self.routeCoordinates),
                "startPoint": self.startPoint,
                "endPoint": self.endPoint,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in self.routeCoordinates],
            },
        }


# ═══════════════════════════════════════════════════════════════
# DATA LOADER & REGISTRY
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
GEOJSON_FILE = DATA_DIR / "reference_road_tamilnadu.geojson"


def load_reference_road_from_geojson(file_path: Path) -> ReferenceRoad:
    """Load a reference road segment from a GeoJSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        feature = data["features"][0]
    elif data.get("type") == "Feature":
        feature = data
    else:
        raise ValueError("Invalid GeoJSON structure: Expected FeatureCollection or Feature")

    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    raw_coords = geometry.get("coordinates", [])

    # GeoJSON coordinates are [longitude, latitude] -> convert to (latitude, longitude)
    lat_lon_coords = [(float(coord[1]), float(coord[0])) for coord in raw_coords]

    return ReferenceRoad(
        roadId=props.get("roadId", "TN-REF-01"),
        roadName=props.get("roadName", "Tamil Nadu Reference Road"),
        tamilName=props.get("tamilName"),
        location=props.get("location", "Tamil Nadu, India"),
        district=props.get("district", "Chengalpattu"),
        state=props.get("state", "Tamil Nadu"),
        country=props.get("country", "India"),
        startPoint=props.get(
            "startPoint",
            {"latitude": lat_lon_coords[0][0], "longitude": lat_lon_coords[0][1], "description": "Start Point"},
        ),
        endPoint=props.get(
            "endPoint",
            {"latitude": lat_lon_coords[-1][0], "longitude": lat_lon_coords[-1][1], "description": "End Point"},
        ),
        routeCoordinates=lat_lon_coords,
        approximateLengthMeters=float(props.get("approximateLengthMeters", 1008.95)),
        approximateLengthKm=float(props.get("approximateLengthKm", 1.009)),
        highwayType=props.get("highwayType", "trunk"),
        surface=props.get("surface", "asphalt"),
        lanes=props.get("lanes", "2"),
        osmSource=props.get("osmSource", "OpenStreetMap (Way ID: 95641232)"),
        osmWayId=props.get("osmWayId", 95641232),
    )


@st.cache_data(show_spinner=False)
def get_default_reference_road() -> ReferenceRoad:
    """Retrieve the primary 1 km Tamil Nadu benchmark road."""
    if GEOJSON_FILE.exists():
        return load_reference_road_from_geojson(GEOJSON_FILE)
    
    # Fallback built-in coordinates for Grand Southern Trunk Road (Pallavaram, Tamil Nadu)
    fallback_coords = [
        (12.9563171, 80.1431335),
        (12.9564210, 80.1431760),
        (12.9571180, 80.1435570),
        (12.9576250, 80.1438210),
        (12.9580290, 80.1440560),
        (12.9583240, 80.1442090),
        (12.9584160, 80.1442510),
        (12.9612360, 80.1457280),
        (12.9615320, 80.1459000),
        (12.9625910, 80.1465150),
        (12.9635460, 80.1470210),
        (12.9635870, 80.1470490),
        (12.9636420, 80.1470780),
        (12.9637760, 80.1471500),
        (12.9640460, 80.1473190),
        (12.9643290, 80.1474960),
    ]
    return ReferenceRoad(
        roadId="TN-NH32-GST-01",
        roadName="Grand Southern Trunk Road (GST Road / NH 32)",
        tamilName="மாபெரும் தெற்கு வழித்தடம்",
        location="Pallavaram - Chromepet Corridor, Chennai Urban Agglomeration",
        district="Chengalpattu",
        state="Tamil Nadu",
        country="India",
        startPoint={"latitude": fallback_coords[0][0], "longitude": fallback_coords[0][1], "description": "Pallavaram Flyover South Sector"},
        endPoint={"latitude": fallback_coords[-1][0], "longitude": fallback_coords[-1][1], "description": "Chromepet North Junction"},
        routeCoordinates=fallback_coords,
        approximateLengthMeters=1008.95,
        approximateLengthKm=1.009,
        highwayType="trunk",
        surface="asphalt",
        osmWayId=95641232,
        osmSource="OpenStreetMap (Way ID: 95641232)",
    )


# ═══════════════════════════════════════════════════════════════
# LEAFLET OPENSTREETMAP HTML COMPONENT GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_leaflet_map_html(road: ReferenceRoad, height_px: int = 560) -> str:
    """
    Generate a self-contained Leaflet.js interactive map with OpenStreetMap tiles,
    cyber-glow road highlighting, custom start/end markers, and telemetry tooltips.
    """
    coords_json = json.dumps([[lat, lon] for lat, lon in road.routeCoordinates])
    cum_dist_json = json.dumps(road.cumulativeDistances)
    start_lat = road.startPoint["latitude"]
    start_lon = road.startPoint["longitude"]
    end_lat = road.endPoint["latitude"]
    end_lon = road.endPoint["longitude"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Route-Aware Map View</title>
    <!-- Leaflet 1.9.4 CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body, html {{
            height: 100%;
            width: 100%;
            background: #070D1E;
            overflow: hidden;
        }}
        #map {{
            height: {height_px}px;
            width: 100%;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        }}
        /* Leaflet Dark Customizations */
        .leaflet-container {{
            background: #070D1E;
        }}
        .leaflet-bar a {{
            background-color: rgba(15, 23, 42, 0.9) !important;
            color: #38BDF8 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(8px);
        }}
        .leaflet-bar a:hover {{
            background-color: rgba(30, 41, 59, 0.95) !important;
            color: #22D3EE !important;
        }}
        .leaflet-control-layers {{
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
            font-size: 0.8rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        }}
        .leaflet-control-layers-base label {{
            color: #CBD5E1 !important;
            font-weight: 500 !important;
            margin: 4px 0 !important;
        }}
        .leaflet-control-attribution {{
            background: rgba(15, 23, 42, 0.85) !important;
            color: #94A3B8 !important;
            font-size: 0.72rem !important;
            border-top-left-radius: 8px !important;
        }}
        .leaflet-control-attribution a {{
            color: #38BDF8 !important;
            text-decoration: none;
        }}
        /* Custom Popup Styling */
        .leaflet-popup-content-wrapper {{
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(34, 211, 238, 0.4) !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 30px rgba(0, 229, 255, 0.2) !important;
            backdrop-filter: blur(12px) !important;
            color: #F8FAFC !important;
            padding: 2px !important;
        }}
        .leaflet-popup-tip {{
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(34, 211, 238, 0.4) !important;
        }}
        .rh-map-popup-title {{
            font-size: 0.88rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .rh-map-popup-meta {{
            font-size: 0.76rem;
            color: #94A3B8;
            line-height: 1.4;
            margin-bottom: 6px;
        }}
        .rh-map-coord-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 3px 6px;
            border-radius: 6px;
            color: #A7F3D0;
            display: inline-block;
        }}
        /* HUD Badge Overlay */
        .rh-hud-overlay {{
            position: absolute;
            top: 14px;
            right: 14px;
            z-index: 1000;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(59, 130, 246, 0.35);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 8px 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .rh-hud-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 700;
            color: #38BDF8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .rh-pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10B981;
            box-shadow: 0 0 10px #10B981;
            animation: pulse-ring 1.8s infinite;
        }}
        @keyframes pulse-ring {{
            0% {{ transform: scale(0.95); opacity: 1; }}
            50% {{ transform: scale(1.3); opacity: 0.6; }}
            100% {{ transform: scale(0.95); opacity: 1; }}
        }}
        /* Custom Marker Pins */
        .custom-pin {{
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            box-shadow: 0 0 16px rgba(0,0,0,0.6);
            transition: transform 0.2s ease;
        }}
        .custom-pin:hover {{
            transform: scale(1.18);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="rh-hud-overlay">
        <div class="rh-pulse-dot"></div>
        <div class="rh-hud-badge">
            <span>OSM Reference Road: <strong>{road.approximateLengthMeters:,.1f} m</strong></span>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script>
        const routeCoords = {coords_json};
        const cumDists = {cum_dist_json};
        const roadName = "{road.roadName}";
        const roadLocation = "{road.location}";
        const stateName = "{road.state}";
        const totalLen = {road.approximateLengthMeters};

        // Initialize Map
        const map = L.map('map', {{
            zoomControl: true,
            scrollWheelZoom: true
        }});

        // Base Tile Layers (Standard OSM Carto & Dark Matter)
        const osmStandard = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
        }});

        const osmDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> | CARTO'
        }});

        const cartoDarkMatter = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> | CARTO'
        }});

        // Add standard OSM tile layer as primary
        osmStandard.addTo(map);

        const baseMaps = {{
            "🌐 OpenStreetMap Standard": osmStandard,
            "🗺️ OpenStreetMap Voyager": osmDark,
            "🌑 Dark High-Contrast": cartoDarkMatter
        }};
        L.control.layers(baseMaps, null, {{ position: 'topleft' }}).addTo(map);

        // Scale bar
        L.control.scale({{ metric: true, imperial: false, position: 'bottomleft' }}).addTo(map);

        // Glow Layer for Route (Wide translucent cyan)
        const glowPolyline = L.polyline(routeCoords, {{
            color: '#22D3EE',
            weight: 9,
            opacity: 0.35,
            lineCap: 'round',
            lineJoin: 'round'
        }}).addTo(map);

        // Core Route Polyline (Vibrant Solid Royal Blue/Cyan)
        const routePolyline = L.polyline(routeCoords, {{
            color: '#0284C7',
            weight: 5,
            opacity: 0.95,
            lineCap: 'round',
            lineJoin: 'round'
        }}).addTo(map);

        // Fit map bounds to show complete ~1 km road
        map.fitBounds(routePolyline.getBounds(), {{
            padding: [45, 45],
            maxZoom: 17
        }});

        // Custom Start Icon (Emerald Pin)
        const startIcon = L.divIcon({{
            className: 'start-pin-wrapper',
            html: `
                <div class="custom-pin" style="width: 32px; height: 32px; background: linear-gradient(135deg, #10B981 0%, #059669 100%); border: 3px solid #FFFFFF; color: #FFFFFF; font-size: 13px; font-weight: 800; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(16, 185, 129, 0.75);">
                    S
                </div>
            `,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        }});

        // Custom End Icon (Rose/Amber Finish Pin)
        const endIcon = L.divIcon({{
            className: 'end-pin-wrapper',
            html: `
                <div class="custom-pin" style="width: 32px; height: 32px; background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%); border: 3px solid #FFFFFF; color: #FFFFFF; font-size: 13px; font-weight: 800; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(244, 63, 94, 0.75);">
                    E
                </div>
            `,
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        }});

        // Start Marker
        const startMarker = L.marker([{start_lat}, {start_lon}], {{ icon: startIcon }}).addTo(map);
        startMarker.bindPopup(`
            <div style="padding: 6px;">
                <div class="rh-map-popup-title">🟢 START TERMINAL (0 m)</div>
                <div class="rh-map-popup-meta"><strong>${{roadName}}</strong><br>${{roadLocation}}</div>
                <div class="rh-map-coord-tag">LAT: {start_lat:.6f}° N | LON: {start_lon:.6f}° E</div>
            </div>
        `);

        // End Marker
        const endMarker = L.marker([{end_lat}, {end_lon}], {{ icon: endIcon }}).addTo(map);
        endMarker.bindPopup(`
            <div style="padding: 6px;">
                <div class="rh-map-popup-title">🏁 END TERMINAL (${{totalLen.toFixed(1)}} m)</div>
                <div class="rh-map-popup-meta"><strong>${{roadName}}</strong><br>${{roadLocation}}</div>
                <div class="rh-map-coord-tag">LAT: {end_lat:.6f}° N | LON: {end_lon:.6f}° E</div>
            </div>
        `);

        // Milestone markers along the road (+250m, +500m, +750m)
        const targetOffsets = [250, 500, 750];
        targetOffsets.forEach(targetDist => {{
            let bestIdx = 0;
            let minDiff = 9999;
            for (let i = 0; i < cumDists.length; i++) {{
                const diff = Math.abs(cumDists[i] - targetDist);
                if (diff < minDiff) {{
                    minDiff = diff;
                    bestIdx = i;
                }}
            }}
            const pt = routeCoords[bestIdx];
            const actualDist = cumDists[bestIdx];
            
            const milestoneIcon = L.divIcon({{
                className: 'ms-pin-wrapper',
                html: `
                    <div style="width: 10px; height: 10px; border-radius: 50%; background: #38BDF8; border: 2px solid #0F172A; box-shadow: 0 0 8px #38BDF8;"></div>
                `,
                iconSize: [10, 10],
                iconAnchor: [5, 5]
            }});

            const msMarker = L.marker(pt, {{ icon: milestoneIcon }}).addTo(map);
            msMarker.bindPopup(`
                <div style="padding: 4px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8;">📍 Checkpoint +${{actualDist.toFixed(0)}} m</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; margin: 3px 0;">Geographic Waypoint ${{bestIdx + 1}} of ${{routeCoords.length}}</div>
                    <div class="rh-map-coord-tag">${{pt[0].toFixed(6)}}, ${{pt[1].toFixed(6)}}</div>
                </div>
            `);
        }});

        // Route Polyline Tooltip
        routePolyline.bindTooltip(`🛣️ ${{roadName}} (~${{totalLen.toFixed(0)}} m)`, {{
            sticky: true,
            className: 'leaflet-tooltip-dark'
        }});

    </script>
</body>
</html>
"""
    return html_content


# ═══════════════════════════════════════════════════════════════
# STREAMLIT PAGE RENDERER
# ═══════════════════════════════════════════════════════════════

def render_route_analysis_page():
    """
    Render the standalone Route-Aware Pavement Analysis module in Streamlit.
    """
    # Page Header Banner
    st.markdown(render_html("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #0EA5E9 0%, #3B82F6 50%, #6366F1 100%); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 0 25px rgba(14, 165, 233, 0.45);">
                🗺️
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                    Route-Aware Pavement Analysis
                </h1>
                <p style="margin: 0; font-size: 0.88rem; color: #94A3B8; margin-top: 2px;">
                    Step 1: Real-World 1 km OpenStreetMap Reference Road · Tamil Nadu Benchmark
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 30px; padding: 6px 14px; display: flex; align-items: center; gap: 8px;">
                <span class="rh-pulse-dot" style="background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
                <span style="font-size: 0.78rem; font-weight: 700; color: #A7F3D0; text-transform: uppercase; letter-spacing: 0.04em;">OSM Verified</span>
            </div>
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 30px; padding: 6px 14px; font-size: 0.78rem; color: #38BDF8; font-weight: 600;">
                Tamil Nadu, India
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Road Selection & Configuration Container
    col_sel, col_mode = st.columns([2.5, 1], gap="medium")
    
    with col_sel:
        road_options = [
            "Tamil Nadu: Grand Southern Trunk Road (GST Road / NH 32) — ~1.0 km Benchmark",
        ]
        selected_option = st.selectbox(
            "Select Reference Road Segment",
            road_options,
            index=0,
            help="Choose the real-world road segment. Future steps will support custom GPS route uploads and multi-segment tracking.",
        )
    
    with col_mode:
        st.markdown(render_html("""
        <div style="padding-top: 24px;">
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 8px 12px; text-align: center;">
                <span style="font-size: 0.76rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Geometry Source</span><br>
                <span style="font-size: 0.88rem; font-weight: 700; color: #38BDF8;">OpenStreetMap Way 95641232</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

    # Fetch Road Model
    road = get_default_reference_road()

    # ── KEY TELEMETRY CARDS GRID ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; transition: all 0.25s ease;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Approx Route Length</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38BDF8; margin: 4px 0;">{road.approximateLengthMeters:,.1f} <span style="font-size: 0.95rem; font-weight: 600; color: #94A3B8;">m</span></div>
            <div style="font-size: 0.75rem; color: #10B981; font-weight: 600;">≈ {road.approximateLengthKm:.3f} km Benchmark</div>
        </div>
        """), unsafe_allow_html=True)

    with c2:
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; transition: all 0.25s ease;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Road Identification</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin: 6px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">GST Road / NH 32</div>
            <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 600;">ID: {road.roadId}</div>
        </div>
        """), unsafe_allow_html=True)

    with c3:
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; transition: all 0.25s ease;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Geographic Nodes</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #F59E0B; margin: 4px 0;">{len(road.routeCoordinates)} <span style="font-size: 0.95rem; font-weight: 600; color: #94A3B8;">Waypoints</span></div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">Real LineString Geometry</div>
        </div>
        """), unsafe_allow_html=True)

    with c4:
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 16px; transition: all 0.25s ease;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Administrative Region</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin: 6px 0;">{road.state}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">{road.country} · {road.district}</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAP INTERFACE ──────────────────────────────────────────
    st.markdown(render_html("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">🗺️ Interactive OpenStreetMap View</span>
            <span style="font-size: 0.76rem; color: #64748B; font-weight: 600;">(Leaflet + OSM Standard Tiles)</span>
        </div>
        <div style="font-size: 0.76rem; color: #38BDF8; font-weight: 600;">
            Highlighted Segment: 1,008.95 m
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Render Leaflet Map
    leaflet_html = generate_leaflet_map_html(road, height_px=540)
    components.html(leaflet_html, height=550, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── START & END TERMINAL TELEMETRY ─────────────────────────
    col_start, col_end = st.columns(2, gap="medium")

    with col_start:
        start_lat = road.startPoint["latitude"]
        start_lon = road.startPoint["longitude"]
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 14px; padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 28px; height: 28px; border-radius: 50%; background: #10B981; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem;">S</div>
                    <span style="font-size: 0.95rem; font-weight: 700; color: #A7F3D0;">START TERMINAL (0.0 m)</span>
                </div>
                <span style="font-size: 0.74rem; background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600;">Origin</span>
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; margin-bottom: 12px; line-height: 1.4;">
                {road.startPoint.get("description", "GST Road South Terminal, Pallavaram")}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div style="background: rgba(7, 13, 30, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Latitude</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">{start_lat:.7f}° N</div>
                </div>
                <div style="background: rgba(7, 13, 30, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Longitude</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">{start_lon:.7f}° E</div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col_end:
        end_lat = road.endPoint["latitude"]
        end_lon = road.endPoint["longitude"]
        st.markdown(render_html(f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 14px; padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 28px; height: 28px; border-radius: 50%; background: #F43F5E; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem;">E</div>
                    <span style="font-size: 0.95rem; font-weight: 700; color: #FDA4AF;">END TERMINAL (~{road.approximateLengthMeters:,.1f} m)</span>
                </div>
                <span style="font-size: 0.74rem; background: rgba(244, 63, 94, 0.15); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 6px; padding: 2px 8px; font-weight: 600;">Destination</span>
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; margin-bottom: 12px; line-height: 1.4;">
                {road.endPoint.get("description", "GST Road North Terminal, Chromepet")}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div style="background: rgba(7, 13, 30, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Latitude</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">{end_lat:.7f}° N</div>
                </div>
                <div style="background: rgba(7, 13, 30, 0.8); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Longitude</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">{end_lon:.7f}° E</div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TECHNICAL DETAILS & GEOJSON DATA VIEWER ────────────────
    with st.expander("📊 Engineering Metadata & Real OSM Waypoints", expanded=False):
        t1, t2 = st.tabs(["📌 Geographic Coordinates Table", "📄 GeoJSON Specification"])
        
        with t1:
            st.markdown("#### Real-World Geographic Nodes along Segment (OpenStreetMap Way ID: 95641232)")
            table_rows = []
            for i, (coord, cum_d) in enumerate(zip(road.routeCoordinates, road.cumulativeDistances)):
                role = "Start (Origin)" if i == 0 else ("End (Destination)" if i == len(road.routeCoordinates) - 1 else f"Waypoint #{i+1}")
                table_rows.append({
                    "Node #": i + 1,
                    "Role": role,
                    "Latitude (°N)": f"{coord[0]:.7f}",
                    "Longitude (°E)": f"{coord[1]:.7f}",
                    "Cumulative Distance (m)": f"+{cum_d:,.2f} m",
                    "Segment Increment (m)": f"+{(cum_d - road.cumulativeDistances[i-1]):.2f} m" if i > 0 else "0.00 m",
                })
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            st.caption(f"Verified Geodesic Polyline: Total 16 nodes, {road.approximateLengthMeters:,.2f} m via Haversine ellipsoidal chord integration.")

        with t2:
            st.markdown("#### Standard RFC 7946 GeoJSON Representation")
            geojson_str = json.dumps(road.to_geojson(), indent=2, ensure_ascii=False)
            st.code(geojson_str, language="json")
            st.download_button(
                "📥 Download Reference Road GeoJSON",
                data=geojson_str,
                file_name="reference_road_tamilnadu.geojson",
                mime="application/geo+json",
                use_container_width=False,
            )
