import json
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from shapely.geometry import LineString, MultiLineString

GEOJSON_PATH = os.path.join("data", "lk_roads.geojson")
OUTPUT_DIR = "images"
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_CACHE_PATH = os.path.join("data", "elevation_cache.json")
LOCATION_CACHE_PATH = os.path.join("data", "location_cache.json")
LOCATION_PROXIMITY_KM = 5.0
ELEVATION_WINDOW = 10


def _load_location_index():
    """Load curated location index: {name: [lat, lon]}"""
    if os.path.exists(LOCATION_CACHE_PATH):
        with open(LOCATION_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _get_annotation_places(coords, threshold_km=LOCATION_PROXIMITY_KM):
    """
    For each place in location_cache.json, find its closest point on the road.
    If within threshold_km, include it as (road_idx, place_lon, place_lat, name).
    Returns list sorted by road_idx.
    """
    index = _load_location_index()
    results = []
    for name, (place_lat, place_lon) in index.items():
        best_idx, best_dist = min(
            (
                (i, _haversine_km(place_lat, place_lon, lat, lon))
                for i, (lon, lat) in enumerate(coords)
            ),
            key=lambda x: x[1],
        )
        if best_dist <= threshold_km:
            results.append((best_idx, place_lon, place_lat, name))
    results.sort(key=lambda x: x[0])
    return results


def plot_road(road_id):
    gdf = gpd.read_file(GEOJSON_PATH)
    all_roads = gdf[gdf["CLASS"].notna()]
    road = all_roads[all_roads["CLASS"] == road_id]

    if road.empty:
        print(f"No road found with id '{road_id}'")
        return

    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_aspect("equal")
    ax.axis("off")

    road.plot(ax=ax, color="crimson", linewidth=1.5, zorder=2)

    dissolved = road.dissolve()
    centroid = dissolved.geometry.iloc[0].centroid
    ax.text(
        centroid.x,
        centroid.y,
        road_id,
        fontsize=10,
        ha="center",
        va="center",
        color="crimson",
        fontweight="bold",
        zorder=3,
    )

    # Annotate places from location index that are close to this road
    road_coords = _extract_coords(dissolved.geometry.iloc[0])
    annotation_places = _get_annotation_places(road_coords)
    for _, lon, lat, place in annotation_places:
        ax.plot(lon, lat, "o", color="darkred", markersize=4, zorder=4)
        ax.text(
            lon,
            lat,
            place,
            fontsize=6,
            ha="center",
            va="bottom",
            color="darkred",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75
            ),
            zorder=5,
        )

    ax.set_title(f"Sri Lanka Road: {road_id}", fontsize=16, pad=12)

    road_dir = os.path.join(OUTPUT_DIR, road_id.lower())
    os.makedirs(road_dir, exist_ok=True)
    output_path = os.path.join(road_dir, "road.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map to {output_path}")


def _extract_coords(geometry):
    """Return a flat list of (lon, lat) tuples from a LineString or MultiLineString."""
    if isinstance(geometry, LineString):
        return list(geometry.coords)
    if isinstance(geometry, MultiLineString):
        coords = []
        for part in geometry.geoms:
            coords.extend(part.coords)
        return coords
    return []


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def _elev_key(lat, lon):
    return f"{lat:.6f},{lon:.6f}"


def _load_elevation_cache():
    if os.path.exists(ELEVATION_CACHE_PATH):
        with open(ELEVATION_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_elevation_cache(cache):
    with open(ELEVATION_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _fetch_elevations(coords):
    """POST (lon, lat) coords to open-elevation and return elevations in metres."""
    locations = [{"latitude": lat, "longitude": lon} for lon, lat in coords]
    response = requests.post(
        ELEVATION_API,
        json={"locations": locations},
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=60,
    )
    response.raise_for_status()
    return [r["elevation"] for r in response.json()["results"]]


def plot_elevation(road_id):
    gdf = gpd.read_file(GEOJSON_PATH)
    road = gdf[gdf["CLASS"] == road_id]

    if road.empty:
        print(f"No road found with id '{road_id}'")
        return

    geometry = road.dissolve().geometry.iloc[0]
    coords = _extract_coords(geometry)

    distances = [0.0]
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        distances.append(distances[-1] + _haversine_km(lat1, lon1, lat2, lon2))

    cache = _load_elevation_cache()
    missing = [
        (i, lon, lat)
        for i, (lon, lat) in enumerate(coords)
        if _elev_key(lat, lon) not in cache
    ]
    if missing:
        print(
            f"Fetching elevation for {len(missing)} new points on road {road_id}..."
        )
        missing_coords = [(lon, lat) for _, lon, lat in missing]
        new_elevations = _fetch_elevations(missing_coords)
        for (_, lon, lat), elev in zip(missing, new_elevations):
            cache[_elev_key(lat, lon)] = float(elev)
        _save_elevation_cache(cache)
    else:
        print(f"Loading elevation for road {road_id} from cache...")
    raw_elevations = [cache[_elev_key(lat, lon)] for lon, lat in coords]
    n = len(raw_elevations)
    elevations = [
        sum(
            w := raw_elevations[
                max(0, i - ELEVATION_WINDOW) : min(n, i + ELEVATION_WINDOW + 1)
            ]
        )
        / len(w)
        for i in range(n)
    ]

    annotation_places = _get_annotation_places(coords)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(distances, elevations, color="steelblue", linewidth=1.5)
    ax.fill_between(distances, elevations, alpha=0.2, color="steelblue")
    ax.set_xlabel("Distance from origin (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Elevation Profile: Road {road_id}")
    ax.grid(True, alpha=0.3)

    ymax = max(elevations)
    for idx, _, _, place in annotation_places:
        d = distances[idx]
        ax.axvline(d, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
        ax.text(
            d,
            ymax,
            place,
            fontsize=7,
            ha="right",
            va="top",
            rotation=90,
            color="dimgray",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75
            ),
        )

    road_dir = os.path.join(OUTPUT_DIR, road_id.lower())
    os.makedirs(road_dir, exist_ok=True)
    output_path = os.path.join(road_dir, "elevation.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved elevation profile to {output_path}")


def analyze(road_id):
    plot_road(road_id)
    plot_elevation(road_id)


if __name__ == "__main__":
    road_id = sys.argv[1]
    analyze(road_id)
