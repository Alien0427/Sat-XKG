from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Column mapping dictionaries to support heterogeneous schemas across
# different data sources (synthetic vs. real-world OSM data).
NODE_ID_CANDIDATES: List[str] = ["node_id", "id"]
NODE_LAT_CANDIDATES: List[str] = ["latitude", "lat", "y"]
NODE_LON_CANDIDATES: List[str] = ["longitude", "lon", "x"]
EDGE_SRC_CANDIDATES: List[str] = ["source_id", "u", "source"]
EDGE_TGT_CANDIDATES: List[str] = ["target_id", "v", "target"]
EDGE_DIST_CANDIDATES: List[str] = ["distance_km", "length", "weight"]
EDGE_BLOCKED_CANDIDATES: List[str] = ["is_blocked", "blocked"]


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Select the first column name from `candidates` that exists in
    `df.columns`. Raises a KeyError if none are found.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns {candidates} found in DataFrame.")


def _load_dataframes(
    nodes_path: Path, edges_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load node and edge tables from CSV files.
    """
    nodes_df: pd.DataFrame = pd.read_csv(nodes_path)
    edges_df: pd.DataFrame = pd.read_csv(edges_path)
    return nodes_df, edges_df


def build_graph(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame
) -> nx.Graph:
    """
    Construct a Spatial Knowledge Graph from tabular node and edge data.

    Node attributes:
    - damage_severity
    - current_capacity
    - type
    - latitude
    - longitude

    Edge attributes:
    - distance_km
    - is_blocked
    - effective_distance
    """
    graph: nx.Graph = nx.Graph()

    # Resolve column names for node attributes in a schema-agnostic way.
    node_id_col: str = _pick_column(nodes_df, NODE_ID_CANDIDATES)
    lat_col: str = _pick_column(nodes_df, NODE_LAT_CANDIDATES)
    lon_col: str = _pick_column(nodes_df, NODE_LON_CANDIDATES)

    for row in nodes_df.itertuples(index=False):
        node_id: int = int(getattr(row, node_id_col))
        node_type: str = str(getattr(row, "type"))

        latitude: float = float(getattr(row, lat_col))
        longitude: float = float(getattr(row, lon_col))

        damage_severity: float = float(getattr(row, "damage_severity"))
        current_capacity: int = int(getattr(row, "current_capacity"))

        graph.add_node(
            node_id,
            type=node_type,
            latitude=latitude,
            longitude=longitude,
            damage_severity=damage_severity,
            current_capacity=current_capacity,
        )

    # Resolve column names for edge attributes.
    src_col: str = _pick_column(edges_df, EDGE_SRC_CANDIDATES)
    tgt_col: str = _pick_column(edges_df, EDGE_TGT_CANDIDATES)
    dist_col: str = _pick_column(edges_df, EDGE_DIST_CANDIDATES)
    blocked_col: str | None = None
    for candidate in EDGE_BLOCKED_CANDIDATES:
        if candidate in edges_df.columns:
            blocked_col = candidate
            break

    for row in edges_df.itertuples(index=False):
        source_id: int = int(getattr(row, src_col))
        target_id: int = int(getattr(row, tgt_col))
        distance_km: float = float(getattr(row, dist_col))

        if blocked_col is not None:
            raw_is_blocked = getattr(row, blocked_col)
            if isinstance(raw_is_blocked, str):
                is_blocked: bool = raw_is_blocked.strip().lower() == "true"
            else:
                is_blocked = bool(raw_is_blocked)
        else:
            is_blocked = False

        effective_distance: float = (
            distance_km * 100.0 if is_blocked else distance_km
        )

        graph.add_edge(
            source_id,
            target_id,
            distance_km=distance_km,
            is_blocked=is_blocked,
            effective_distance=effective_distance,
        )

    return graph


def visualize_graph(graph: nx.Graph) -> plt.Figure:
    """
    Visualize the graph using matplotlib.

    - Nodes are placed according to geographic coordinates
      (longitude on the x-axis, latitude on the y-axis).
    - Node color encodes damage_severity from green (0) to red (1).
    - Node shape encodes infrastructure type where possible.
    """
    # Create an explicit Figure/Axes so NetworkX draws into `ax`
    # (avoid pyplot global state, which Streamlit may cache).
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Positions in geographic space.
    positions: Dict[int, Tuple[float, float]] = {
        node_id: (
            float(data.get("longitude", 0.0)),
            float(data.get("latitude", 0.0)),
        )
        for node_id, data in graph.nodes(data=True)
    }

    # Use the modern colormap API to avoid deprecation warnings.
    cmap = plt.colormaps.get("RdYlGn_r")  # low damage -> green, high -> red
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Define shapes per node type; default to circle for unknown types.
    type_to_shape: Dict[str, str] = {
        "hospital": "s",          # square
        "relief_camp": "^",       # triangle
        "road_intersection": "o", # circle
    }

    # Draw edges first so nodes appear on top.
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        alpha=0.3,
        width=1.0,
        ax=ax,
    )

    # Draw each node type with its own marker shape.
    unique_types = sorted(
        {str(data.get("type", "")) for _, data in graph.nodes(data=True)}
    )
    for node_type in unique_types:
        nodelist: List[int] = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if str(data.get("type", "")) == node_type
        ]
        if not nodelist:
            continue

        node_damage: List[float] = [
            float(graph.nodes[node_id].get("damage_severity", 0.0))
            for node_id in nodelist
        ]
        node_colors = [cmap(norm(value)) for value in node_damage]

        shape: str = type_to_shape.get(node_type, "o")

        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=nodelist,
            node_color=node_colors,
            node_shape=shape,
            node_size=80,
            linewidths=0.5,
            edgecolors="black",
            ax=ax,
        )

    colorbar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=ax,
        label="Damage severity",
        shrink=0.8,
    )
    cbar.ax.set_ylabel("Damage severity")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Knowledge Graph ({len(graph.nodes)} nodes)")
    ax.axis('equal')
    fig.tight_layout()
    return fig


def to_pyg_data(graph: nx.Graph) -> Data:
    """
    Convert a NetworkX graph into a PyTorch Geometric Data object.

    Node features (x):
    - damage_severity (float)
    - current_capacity (float, unnormalized)
    - one-hot encoding of node type

    Edge attributes (edge_attr):
    - distance_km
    - effective_distance
    - is_blocked (0.0 or 1.0)
    """
    # Fix an ordering over nodes for deterministic indexing.
    node_ids: List[int] = sorted(graph.nodes())
    node_id_to_index: Dict[int, int] = {
        node_id: idx for idx, node_id in enumerate(node_ids)
    }

    # Determine the type vocabulary across all nodes.
    unique_types = sorted(
        {str(data.get("type", "")) for _, data in graph.nodes(data=True)}
    )
    type_to_index: Dict[str, int] = {
        node_type: idx for idx, node_type in enumerate(unique_types)
    }

    num_nodes: int = len(node_ids)
    num_type_features: int = len(unique_types)
    feature_dimension: int = 2 + num_type_features

    x = torch.zeros(
        (num_nodes, feature_dimension),
        dtype=torch.float32,
    )

    for node_id, data in graph.nodes(data=True):
        node_index: int = node_id_to_index[node_id]

        damage_severity: float = float(data.get("damage_severity", 0.0))
        current_capacity: float = float(data.get("current_capacity", 0))

        # Continuous features.
        x[node_index, 0] = damage_severity
        x[node_index, 1] = current_capacity

        # One-hot type encoding.
        node_type: str = str(data.get("type", ""))
        type_index: int = type_to_index.get(node_type, -1)
        if type_index >= 0:
            x[node_index, 2 + type_index] = 1.0

    # Build undirected edges as bidirectional pairs.
    edge_indices: List[List[int]] = [[], []]
    edge_attributes: List[List[float]] = []

    for source_id, target_id, edge_data in graph.edges(data=True):
        for u, v in ((source_id, target_id), (target_id, source_id)):
            u_index: int = node_id_to_index[u]
            v_index: int = node_id_to_index[v]

            distance_km: float = float(edge_data.get("distance_km", 0.0))
            effective_distance: float = float(
                edge_data.get("effective_distance", distance_km)
            )
            is_blocked_bool: bool = bool(edge_data.get("is_blocked", False))
            is_blocked_float: float = 1.0 if is_blocked_bool else 0.0

            edge_indices[0].append(u_index)
            edge_indices[1].append(v_index)
            edge_attributes.append(
                [distance_km, effective_distance, is_blocked_float]
            )

    edge_index = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )
    return data


def main() -> None:
    """
    Entry point used for quick manual inspection.

    The function:
    1. Loads node and edge CSV files.
    2. Builds the NetworkX spatial knowledge graph.
    3. Visualizes the graph.
    4. Converts it into a PyTorch Geometric Data object and prints a
       summary.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    # Prefer the requested data/ directory, but fall back to the project root
    # if the files were generated there.
    nodes_path: Path = data_dir / "nodes.csv"
    edges_path: Path = data_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        nodes_path = base_dir / "nodes.csv"
        edges_path = base_dir / "edges.csv"

    nodes_df, edges_df = _load_dataframes(nodes_path, edges_path)
    graph = build_graph(nodes_df, edges_df)

    fig = visualize_graph(graph)
    plt.show()
    pyg_data: Data = to_pyg_data(graph)
    print(pyg_data)


if __name__ == "__main__":
    main()

