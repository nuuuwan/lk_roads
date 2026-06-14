import os

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

GEOJSON_PATH = os.path.join("data", "lk_roads.geojson")
OUTPUT_DIR = "images"


def plot_roads(output_path, title="Sri Lanka Roads", road_filter=None):
    """
    Plot Sri Lanka roads and save to output_path.

    Args:
        output_path: Path to save the PNG image.
        title: Title for the plot.
        road_filter: Optional callable that takes a road CLASS string and
                     returns True to include it. E.g. lambda id: id.startswith('A').
                     If None, all named roads are included.
    """
    gdf = gpd.read_file(GEOJSON_PATH)

    named = gdf[gdf["CLASS"].notna()].copy()
    unnamed = gdf[gdf["CLASS"].isna()].copy()

    if road_filter is not None:
        named = named[named["CLASS"].apply(road_filter)]

    classes = sorted(named["CLASS"].unique())
    n = max(len(classes), 1)
    cmap = cm.get_cmap("hsv", n)
    color_map = {cls: cmap(i) for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw all unnamed roads as grey background context
    if not unnamed.empty:
        unnamed.plot(ax=ax, color="#cccccc", linewidth=0.4, zorder=1)

    # Draw filtered named roads with unique colors and labels
    for cls in classes:
        subset = named[named["CLASS"] == cls]
        color = color_map[cls]
        subset.plot(ax=ax, color=color, linewidth=1.2, zorder=2)

        dissolved = subset.dissolve()
        centroid = dissolved.geometry.iloc[0].centroid
        ax.text(
            centroid.x,
            centroid.y,
            cls,
            fontsize=5.5,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            zorder=3,
        )

    ax.set_title(title, fontsize=16, pad=12)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map to {output_path}")


if __name__ == "__main__":
    plot_roads(
        output_path=os.path.join(OUTPUT_DIR, "lk_roads.png"),
        title="Sri Lanka Roads by Road ID",
    )
    plot_roads(
        output_path=os.path.join(OUTPUT_DIR, "roads_a.png"),
        title="Sri Lanka A-Grade Roads",
        road_filter=lambda road_id: road_id.startswith("A"),
    )
