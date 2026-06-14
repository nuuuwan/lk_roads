import json
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

GEOJSON_PATH = os.path.join("data", "lk_roads.geojson")
OUTPUT_DIR = "images"
ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_CACHE_PATH = os.path.join("data", "elevation_cache.json")
LOCATION_CACHE_PATH = os.path.join("data", "location_cache.json")
LOCATION_PROXIMITY_KM = 2
ELEVATION_WINDOW = 20
DISTRICTS_URL = (
    "https://raw.githubusercontent.com/nuuuwan/lk_admin_regions"
    "/refs/heads/main/data/geo/topojson/e4_medium/districts.topojson"
)
PROVINCES_URL = (
    "https://raw.githubusercontent.com/nuuuwan/lk_admin_regions"
    "/refs/heads/main/data/geo/topojson/e4_medium/provinces.topojson"
)


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


def _get_district_data(road_id):
    """Return (touched_gdf, district_color_map) for districts the road passes through."""
    gdf_roads = gpd.read_file(GEOJSON_PATH)
    road = gdf_roads[gdf_roads["CLASS"] == road_id]
    if road.empty:
        return None, {}
    districts_gdf = _load_admin_gdf(DISTRICTS_URL)
    provinces_gdf = _load_admin_gdf(PROVINCES_URL)
    road_geom = road.dissolve().to_crs("EPSG:4326").geometry.iloc[0]
    touched = districts_gdf[districts_gdf.intersects(road_geom)].copy()
    prov_name_map = dict(zip(provinces_gdf["id"], provinces_gdf["name"]))
    touched["province_name"] = touched["province_id"].map(prov_name_map)
    district_names = sorted(touched["name"].dropna().unique())
    cmap = plt.colormaps.get_cmap("Set2").resampled(
        max(len(district_names), 1)
    )
    district_color = {d: cmap(i) for i, d in enumerate(district_names)}
    return touched, district_color


def plot_road(road_id):
    gdf = gpd.read_file(GEOJSON_PATH)
    all_roads = gdf[gdf["CLASS"].notna()]
    road = all_roads[all_roads["CLASS"] == road_id]

    if road.empty:
        print(f"No road found with id '{road_id}'")
        return

    touched, district_color = _get_district_data(road_id)

    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_aspect("equal")
    ax.axis("off")

    # Shade districts individually in the background
    if touched is not None and not touched.empty:
        for _, row in touched.iterrows():
            color = district_color.get(row["name"], "#cccccc")
            gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").plot(
                ax=ax,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                row["name"],
                fontsize=6,
                ha="center",
                va="center",
                color="#333333",
                fontweight="bold",
                zorder=2,
            )

        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=district_color[d], label=d, alpha=0.7)
            for d in sorted(district_color)
        ]
        ax.legend(
            handles=handles, title="District", loc="lower left", fontsize=7
        )

    road.plot(ax=ax, color="crimson", linewidth=1.5, zorder=4)

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
        zorder=5,
    )

    road_coords = _extract_coords(dissolved.geometry.iloc[0])
    annotation_places = _get_annotation_places(road_coords)
    for _, lon, lat, place in annotation_places:
        ax.plot(lon, lat, "o", color="darkred", markersize=4, zorder=6)
        ax.text(
            lon,
            lat,
            place,
            fontsize=6,
            ha="center",
            va="bottom",
            color="darkred",
            zorder=7,
        )

    ax.set_title(f"Sri Lanka Road: {road_id}", fontsize=16, pad=12)

    road_dir = os.path.join(OUTPUT_DIR, road_id.lower())
    os.makedirs(road_dir, exist_ok=True)
    output_path = os.path.join(road_dir, "road.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map to {output_path}")
    os.system(f"open {output_path}")


def _ordered_coords_and_distances(geometry):
    """
    Return (coords, distances) with coords stitched into the best-ordered path
    by greedily connecting nearest endpoints between disconnected parts.
    """
    if isinstance(geometry, MultiLineString):
        geometry = linemerge(geometry)

    if isinstance(geometry, LineString):
        parts = [list(geometry.coords)]
    else:
        parts = [list(g.coords) for g in geometry.geoms]

    # Greedy stitch: repeatedly connect the nearest unvisited endpoint
    def _endpoint_dist(a, b):
        lon1, lat1 = a
        lon2, lat2 = b
        return _haversine_km(lat1, lon1, lat2, lon2)

    stitched = parts[0]
    remaining = [list(p) for p in parts[1:]]
    while remaining:
        end = stitched[-1]
        best_i, best_rev, best_d = 0, False, float("inf")
        for i, seg in enumerate(remaining):
            d_fwd = _endpoint_dist(end, seg[0])
            d_rev = _endpoint_dist(end, seg[-1])
            if d_fwd < best_d:
                best_i, best_rev, best_d = i, False, d_fwd
            if d_rev < best_d:
                best_i, best_rev, best_d = i, True, d_rev
        seg = remaining.pop(best_i)
        if best_rev:
            seg = seg[::-1]
        stitched.extend(seg)

    distances = [0.0]
    for i in range(1, len(stitched)):
        lon1, lat1 = stitched[i - 1]
        lon2, lat2 = stitched[i]
        distances.append(distances[-1] + _haversine_km(lat1, lon1, lat2, lon2))

    return stitched, distances


def _extract_coords(geometry):
    """Return a flat ordered list of (lon, lat) tuples."""
    coords, _ = _ordered_coords_and_distances(geometry)
    return coords


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
    coords, distances = _ordered_coords_and_distances(geometry)

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

    ax.plot(distances, elevations, color="steelblue", linewidth=1.5, zorder=2)
    ax.fill_between(
        distances, elevations, alpha=0.2, color="steelblue", zorder=1
    )
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
    os.system(f"open {output_path}")


def _load_admin_gdf(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    import tempfile

    tmp = tempfile.mktemp(suffix=".topojson")
    with open(tmp, "wb") as f:
        f.write(r.content)
    return gpd.read_file(tmp).set_crs("EPSG:4326", allow_override=True)


def get_districts(latlngs):
    """
    Given a list of (lat, lon) tuples, return a list of district names
    (or None) for each point.
    """
    gdf = _load_admin_gdf(DISTRICTS_URL)
    points = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in latlngs],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        gdf[["id", "name", "province_id", "geometry"]],
        how="left",
        predicate="within",
    )
    return joined["name"].tolist()


def analyze(road_id):
    plot_road(road_id)
    plot_elevation(road_id)


if __name__ == "__main__":
    road_id = sys.argv[1]
    analyze(road_id)
