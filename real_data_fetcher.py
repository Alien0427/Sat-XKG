from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


RNG_SEED: int = 42
PLACE_NAME: str = "Wayanad, Kerala, India"
PURI_PLACE_NAME: str = "Puri, Odisha, India"


@dataclass(frozen=True)
class NodeRecord:
    node_id: int
    latitude: float
    longitude: float
    node_type: str
    current_capacity: int
    damage_severity: float


def _ensure_data_dir() -> Path:
    """
    Ensure that the `data/` directory exists and return its path.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_graph() -> nx.MultiDiGraph:
    """
    Download the drivable road network for the specified region using
    OSMnx.
    """
    graph: nx.MultiDiGraph = ox.graph_from_place(
        PLACE_NAME,
        network_type="drive",
    )
    return graph


def _download_health_facilities() -> pd.DataFrame:
    """
    Download point features corresponding to hospitals and clinics for
    the specified region.

    Returns a GeoDataFrame-like DataFrame with at least:
    - osmid (or index) as node identifier
    - amenity field indicating 'hospital' or 'clinic'
    """
    tags = {"amenity": ["hospital", "clinic"]}
    gdf = ox.features_from_place(PLACE_NAME, tags=tags)
    # Restrict to points only; polygons are not used as distinct nodes here.
    gdf_points = gdf[gdf.geometry.geom_type == "Point"].copy()

    # Reset the index to avoid MultiIndex casting issues and standardize
    # the node identifier column to 'node_id'.
    gdf_points = gdf_points.reset_index()

    if "osmid" in gdf_points.columns:
        gdf_points["node_id"] = gdf_points["osmid"].astype(int)
    elif "element_id" in gdf_points.columns:
        gdf_points["node_id"] = gdf_points["element_id"].astype(int)
    else:
        gdf_points["node_id"] = gdf_points.index.astype(int)

    return gdf_points


def _build_node_dataframe(
    nodes_gdf: pd.DataFrame,
    health_gdf: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Construct the base node table for Wayanad with infrastructure types
    derived from both the road network and health facility layers.
    """
    # Map from node_id to health amenity type, if available.
    amenity_by_node: Dict[int, str] = {}
    for row in health_gdf.itertuples(index=False):
        node_id: int = int(getattr(row, "node_id"))
        amenity: str = str(getattr(row, "amenity"))
        if amenity in {"hospital", "clinic"}:
            amenity_by_node[node_id] = amenity

    records: List[NodeRecord] = []

    for idx, row in enumerate(nodes_gdf.itertuples(index=True)):
        node_id: int = int(row.Index)
        latitude: float = float(getattr(row, "y"))
        longitude: float = float(getattr(row, "x"))

        if node_id in amenity_by_node:
            node_type: str = amenity_by_node[node_id]
        else:
            node_type = "road_intersection"

        current_capacity: int = int(rng.integers(low=50, high=501))
        damage_severity: float = 0.0

        records.append(
            NodeRecord(
                node_id=node_id,
                latitude=latitude,
                longitude=longitude,
                node_type=node_type,
                current_capacity=current_capacity,
                damage_severity=damage_severity,
            )
        )

    nodes_df = pd.DataFrame(
        {
            "node_id": [rec.node_id for rec in records],
            "latitude": [rec.latitude for rec in records],
            "longitude": [rec.longitude for rec in records],
            "type": [rec.node_type for rec in records],
            "current_capacity": [rec.current_capacity for rec in records],
            "damage_severity": [rec.damage_severity for rec in records],
        }
    )

    # Add 5 random relief camps by re-typing existing nodes.
    if len(nodes_df) >= 5:
        relief_indices = rng.choice(
            nodes_df.index.to_numpy(),
            size=5,
            replace=False,
        )
        nodes_df.loc[relief_indices, "type"] = "relief_camp"

    return nodes_df


def _build_edge_dataframe(edges_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the base edge table with geometric and distance
    information.
    """
    # Reset the index so that 'u' and 'v' become ordinary columns even
    # when the GeoDataFrame uses a MultiIndex (u, v, key).
    edges_gdf = edges_gdf.reset_index()

    source_ids: List[int] = []
    target_ids: List[int] = []
    distances_km: List[float] = []
    geometries: List[object] = []
    is_blocked_flags: List[bool] = []

    for row in edges_gdf.itertuples(index=False):
        source_id: int = int(getattr(row, "u"))
        target_id: int = int(getattr(row, "v"))

        length_m: float = float(getattr(row, "length", 0.0))
        distance_km: float = length_m / 1000.0

        geometry = getattr(row, "geometry", None)

        source_ids.append(source_id)
        target_ids.append(target_id)
        distances_km.append(distance_km)
        geometries.append(geometry)
        is_blocked_flags.append(False)

    edges_df = pd.DataFrame(
        {
            "source_id": source_ids,
            "target_id": target_ids,
            "distance_km": distances_km,
            "geometry": geometries,
            "is_blocked": is_blocked_flags,
        }
    )
    return edges_df


def main() -> None:
    """
    Fetch real road network and health facility data for Wayanad,
    construct node and edge tables compatible with the Sat-XKG
    pipeline, and persist them as base CSV files for downstream
    disaster simulations.
    """
    rng: np.random.Generator = np.random.default_rng(RNG_SEED)
    data_dir: Path = _ensure_data_dir()

    # ------------------------------------------------------------------
    # 1) Wayanad, Kerala, India (already implemented earlier)
    # ------------------------------------------------------------------
    graph: nx.MultiDiGraph = _download_graph()
    health_gdf: pd.DataFrame = _download_health_facilities()

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

    nodes_df: pd.DataFrame = _build_node_dataframe(nodes_gdf, health_gdf, rng)
    edges_df: pd.DataFrame = _build_edge_dataframe(edges_gdf)

    wayanad_nodes_path: Path = data_dir / "wayanad_nodes_base.csv"
    wayanad_edges_path: Path = data_dir / "wayanad_edges_base.csv"

    nodes_df.to_csv(wayanad_nodes_path, index=False)
    edges_df.to_csv(wayanad_edges_path, index=False)

    print(f"Saved base node data to: {wayanad_nodes_path}")
    print(f"Saved base edge data to: {wayanad_edges_path}")

    # ------------------------------------------------------------------
    # 2) Puri, Odisha, India (new region as requested)
    # ------------------------------------------------------------------
    puri_graph: nx.MultiDiGraph = ox.graph_from_place(
        PURI_PLACE_NAME,
        network_type="drive",
    )
    puri_nodes_gdf, puri_edges_gdf = ox.graph_to_gdfs(puri_graph)

    # Nodes: keep y (latitude) and x (longitude), rename identifier column to node_id.
    puri_nodes_reset = puri_nodes_gdf.reset_index()
    if "osmid" in puri_nodes_reset.columns:
        puri_nodes_reset = puri_nodes_reset.rename(columns={"osmid": "node_id"})
    elif "index" in puri_nodes_reset.columns:
        puri_nodes_reset = puri_nodes_reset.rename(columns={"index": "node_id"})

    puri_nodes = puri_nodes_reset[["node_id", "y", "x"]].copy()

    # Randomly assign type; ensure relief_camp share is < 5%.
    node_count: int = len(puri_nodes)
    num_camps: int = max(1, int(0.03 * node_count))

    all_indices = np.arange(node_count)
    camp_indices = rng.choice(all_indices, size=num_camps, replace=False)
    non_camp_mask = np.ones(node_count, dtype=bool)
    non_camp_mask[camp_indices] = False

    types = np.empty(node_count, dtype=object)
    types[camp_indices] = "relief_camp"

    # For remaining nodes, randomly assign hospital vs road_intersection.
    remaining_indices = all_indices[non_camp_mask]
    remaining_types = rng.choice(
        ["hospital", "road_intersection"],
        size=remaining_indices.size,
        p=[0.1, 0.9],
    )
    types[remaining_indices] = remaining_types

    puri_nodes["type"] = types
    puri_nodes["current_capacity"] = rng.integers(
        low=50,
        high=501,
        size=node_count,
    )
    puri_nodes["damage_severity"] = 0.0

    # Edges: keep u, v, length, geometry, and initialize is_blocked = False.
    puri_edges_reset = puri_edges_gdf.reset_index()
    puri_edges = puri_edges_reset[["u", "v", "length", "geometry"]].copy()
    puri_edges["is_blocked"] = False

    puri_nodes_path: Path = data_dir / "puri_nodes_base.csv"
    puri_edges_path: Path = data_dir / "puri_edges_base.csv"

    puri_nodes.to_csv(puri_nodes_path, index=False)
    puri_edges.to_csv(puri_edges_path, index=False)

    print(f"Saved base node data to: {puri_nodes_path}")
    print(f"Saved base edge data to: {puri_edges_path}")


if __name__ == "__main__":
    main()

