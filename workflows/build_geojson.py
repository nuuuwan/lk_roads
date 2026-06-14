import os

import geopandas as gpd

SHP_PATH = os.path.join(
    "original_data", "lka_rdsl_250k_sdlka", "lka_rdsl_250k_sdlka.shp"
)
OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "lk_roads.geojson")


def build_geojson():
    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf.to_crs(epsg=4326)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved {len(gdf)} road features to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_geojson()
