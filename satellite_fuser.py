from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


RNG_SEED: int = 42


def _load_base_data(
    data_dir: Path,
    location_prefix: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load base node and edge tables for a given location and convert them
    into GeoPandas GeoDataFrames.

    This function is robust to slight schema differences between
    locations (e.g., latitude/longitude vs. y/x column naming).
    """
    nodes_path: Path = data_dir / f"{location_prefix}_nodes_base.csv"
    edges_path: Path = data_dir / f"{location_prefix}_edges_base.csv"

    nodes_df: pd.DataFrame = pd.read_csv(nodes_path)
    edges_df: pd.DataFrame = pd.read_csv(edges_path)

    # Nodes: support either (latitude, longitude) or (y, x) columns.
    if {"latitude", "longitude"}.issubset(nodes_df.columns):
        xs = nodes_df["longitude"]
        ys = nodes_df["latitude"]
    else:
        xs = nodes_df["x"]
        ys = nodes_df["y"]

    nodes_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:4326",
    )

    # Edges: geometry stored as WKT strings; convert back to shapes.
    edges_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        edges_df,
        geometry=gpd.GeoSeries.from_wkt(edges_df["geometry"]),
        crs="EPSG:4326",
    )

    return nodes_gdf, edges_gdf


def _compute_bounds(nodes_gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """
    Compute the geographic bounding box of all nodes.
    """
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _create_synthetic_flood_mask(
    bounds: Tuple[float, float, float, float],
    resolution_deg: float,
    output_path: Path,
) -> None:
    """
    Create a synthetic flood mask as a GeoTIFF aligned with the given
    bounding box.

    The mask is a simple "river" feature represented as a band of ones
    (flooded pixels) running roughly through the center of the region.
    """
    minx, miny, maxx, maxy = bounds

    width: int = max(1, int(np.ceil((maxx - minx) / resolution_deg)))
    height: int = max(1, int(np.ceil((maxy - miny) / resolution_deg)))

    # Initialize with zeros (dry land).
    mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)

    # Create a synthetic "river" as a vertical band in the middle of the
    # raster footprint.
    center_col: int = width // 2
    band_half_width: int = max(1, width // 20)
    left_col: int = max(0, center_col - band_half_width)
    right_col: int = min(width, center_col + band_half_width)
    mask[:, left_col:right_col] = 1

    transform = from_origin(
        west=minx,
        north=maxy,
        xsize=resolution_deg,
        ysize=resolution_deg,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mask, 1)


def _sample_raster_at_points(
    dataset: rasterio.io.DatasetReader,
    xs: List[float],
    ys: List[float],
) -> np.ndarray:
    """
    Sample the raster at a list of point coordinates and return the
    resulting values as a NumPy array.
    """
    coords = list(zip(xs, ys))
    samples = list(dataset.sample(coords))
    # Each sample is an array with shape (bands,); we use the first band.
    return np.array([float(s[0]) for s in samples], dtype=float)


def _update_edges_with_flood_mask(
    edges_gdf: gpd.GeoDataFrame, flood_path: Path
) -> gpd.GeoDataFrame:
    """
    For each edge geometry, mark an edge as blocked if any sampled point
    along the edge lies on a flooded pixel in the flood mask raster.
    """
    with rasterio.open(flood_path) as src:
        blocked_flags: List[bool] = []
        for geom in edges_gdf.geometry:
            if geom is None or geom.is_empty:
                blocked_flags.append(False)
                continue

            # Sample a fixed number of points along the edge geometry.
            num_samples: int = 20
            distances = np.linspace(0.0, geom.length, num_samples)
            xs: List[float] = []
            ys: List[float] = []
            for d in distances:
                point = geom.interpolate(d)
                xs.append(float(point.x))
                ys.append(float(point.y))

            values = _sample_raster_at_points(src, xs, ys)
            is_blocked: bool = bool(np.any(values >= 0.5))
            blocked_flags.append(is_blocked)

    edges_gdf = edges_gdf.copy()
    edges_gdf["is_blocked"] = blocked_flags
    return edges_gdf


def _update_nodes_with_flood_mask(
    nodes_gdf: gpd.GeoDataFrame, flood_path: Path
) -> gpd.GeoDataFrame:
    """
    For each node point, if it falls on a flooded pixel, increase its
    damage_severity to a value in [0.7, 1.0] and reduce its capacity by
    90%.
    """
    rng = np.random.default_rng(RNG_SEED)

    with rasterio.open(flood_path) as src:
        xs = nodes_gdf.geometry.x.to_numpy()
        ys = nodes_gdf.geometry.y.to_numpy()
        values = _sample_raster_at_points(
            src,
            xs.tolist(),
            ys.tolist(),
        )

    flooded_mask: np.ndarray = values >= 0.5

    nodes_gdf = nodes_gdf.copy()

    # Initialize damage_severity column if absent.
    if "damage_severity" not in nodes_gdf.columns:
        nodes_gdf["damage_severity"] = 0.0

    # Sample new damage severities for flooded nodes.
    num_flooded: int = int(flooded_mask.sum())
    if num_flooded > 0:
        new_damage = rng.uniform(low=0.7, high=1.0, size=num_flooded)
        nodes_gdf.loc[flooded_mask, "damage_severity"] = new_damage

        # Reduce capacity by 90% for flooded nodes.
        if "current_capacity" in nodes_gdf.columns:
            nodes_gdf.loc[flooded_mask, "current_capacity"] = (
                nodes_gdf.loc[flooded_mask, "current_capacity"] * 0.1
            ).astype(int)

    return nodes_gdf


def main() -> None:
    """
    Simulate ingestion of a satellite flood mask and fuse it with
    OpenStreetMap infrastructure data for multiple locations (Wayanad
    and Puri).

    For each location, the output consists of:
    - A synthetic flood mask GeoTIFF.
    - Fused node and edge CSVs with updated damage and blockage
      attributes.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for location_prefix in ["wayanad", "puri"]:
        nodes_gdf, edges_gdf = _load_base_data(data_dir, location_prefix)
        bounds = _compute_bounds(nodes_gdf)

        flood_path: Path = data_dir / f"{location_prefix}_flood_mask.tif"

        # Use a modest spatial resolution to keep raster size manageable.
        _create_synthetic_flood_mask(
            bounds=bounds,
            resolution_deg=0.001,
            output_path=flood_path,
        )

        # Update edges and nodes according to flood extents.
        fused_edges_gdf = _update_edges_with_flood_mask(edges_gdf, flood_path)
        fused_nodes_gdf = _update_nodes_with_flood_mask(nodes_gdf, flood_path)

        # Persist fused data as plain CSV for downstream pipeline steps.
        fused_nodes_path: Path = data_dir / f"{location_prefix}_fused_nodes.csv"
        fused_edges_path: Path = data_dir / f"{location_prefix}_fused_edges.csv"

        fused_nodes_gdf.drop(columns=["geometry"]).to_csv(
            fused_nodes_path,
            index=False,
        )
        fused_edges_gdf.drop(columns=["geometry"]).to_csv(
            fused_edges_path,
            index=False,
        )

        print(f"[{location_prefix}] Saved synthetic flood mask to: {flood_path}")
        print(f"[{location_prefix}] Saved fused node data to:      {fused_nodes_path}")
        print(f"[{location_prefix}] Saved fused edge data to:      {fused_edges_path}")


if __name__ == "__main__":
    main()

