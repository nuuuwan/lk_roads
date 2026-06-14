import os

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

SHP_PATH = os.path.join(
    "original_data", "lka_rdsl_250k_sdlka", "lka_rdsl_250k_sdlka.shp"
)
OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "lk_roads.geojson")


def _stitch_parts(parts):
    """Greedily stitch a list of coord-lists into one continuous LineString
    by connecting nearest endpoints (inserting bridge segments as needed)."""
    def _dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    stitched = list(parts[0])
    remaining = [list(p) for p in parts[1:]]
    while remaining:
        end = stitched[-1]
        best_i, best_rev, best_d = 0, False, float("inf")
        for i, seg in enumerate(remaining):
            d_fwd = _dist(end, seg[0])
            d_rev = _dist(end, seg[-1])
            if d_fwd < best_d:
                best_i, best_rev, best_d = i, False, d_fwd
            if d_rev < best_d:
                best_i, best_rev, best_d = i, True, d_rev
        seg = remaining.pop(best_i)
        if best_rev:
            seg = seg[::-1]
        stitched.extend(seg)
    return LineString(stitched)


def _merge_road(segments):
    """Merge all segments for a single road into one contiguous LineString."""
    # linemerge only works on collections; wrap single LineStrings
    collection = MultiLineString([s for s in segments if isinstance(s, LineString)] +
                                 [l for s in segments if isinstance(s, MultiLineString)
                                  for l in s.geoms])
    merged = linemerge(collection)
    if isinstance(merged, LineString):
        return merged
    parts = [list(g.coords) for g in merged.geoms]
    return _stitch_parts(parts)


def build_geojson():
    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf.to_crs(epsg=4326)

    # Build one contiguous LineString per named road class
    named = gdf[gdf["CLASS"].notna()].copy()
    rows = []
    for road_class, group in named.groupby("CLASS"):
        geom = _merge_road(list(group.geometry))
        rows.append({"CLASS": road_class, "geometry": geom})

    # Unnamed segments pass through as-is
    unnamed = gdf[gdf["CLASS"].isna()].copy()

    result = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    result = gpd.GeoDataFrame(
        gpd.pd.concat([result, unnamed[["CLASS", "geometry"]]], ignore_index=True),
        crs="EPSG:4326",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_file(OUTPUT_PATH, driver="GeoJSON")
    named_count = len(rows)
    unnamed_count = len(unnamed)
    print(f"Saved {named_count} named roads + {unnamed_count} unnamed segments to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_geojson()

