from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


RNG_SEED: int = 42
NUM_NODES: int = 50


@dataclass(frozen=True)
class NodeAttributes:
    node_id: int
    node_type: str
    latitude: float
    longitude: float
    damage_severity: float
    current_capacity: int


def _haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Compute the great-circle distance between two points on Earth using
    the haversine formula.

    All angles are in degrees; the result is returned in kilometers.
    """
    radius_earth_km: float = 6371.0

    phi1: float = math.radians(lat1)
    phi2: float = math.radians(lat2)
    d_phi: float = math.radians(lat2 - lat1)
    d_lambda: float = math.radians(lon2 - lon1)

    a: float = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c: float = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return radius_earth_km * c


def _generate_nodes(rng: np.random.Generator) -> List[NodeAttributes]:
    """
    Generate a list of nodes representing post-disaster infrastructure.

    Node types:
    - hospital
    - road_intersection
    - relief_camp

    Geographic coordinates are sampled around a notional disaster region.
    """
    # Center of the fictional disaster area (latitude, longitude).
    center_lat: float = 27.70
    center_lon: float = 85.30

    # Random offsets in degrees (~10–20 km scale region).
    lat_offsets: np.ndarray = rng.uniform(low=-0.1, high=0.1, size=NUM_NODES)
    lon_offsets: np.ndarray = rng.uniform(low=-0.1, high=0.1, size=NUM_NODES)

    node_types: List[str] = ["hospital", "road_intersection", "relief_camp"]
    # Slightly favor road intersections, as they are more numerous in road networks.
    type_probabilities: List[float] = [0.15, 0.55, 0.30]

    sampled_types: np.ndarray = rng.choice(
        node_types, size=NUM_NODES, p=type_probabilities
    )
    damage_severities: np.ndarray = rng.uniform(low=0.0, high=1.0, size=NUM_NODES)

    nodes: List[NodeAttributes] = []
    for node_idx in range(NUM_NODES):
        node_type: str = str(sampled_types[node_idx])
        lat: float = center_lat + float(lat_offsets[node_idx])
        lon: float = center_lon + float(lon_offsets[node_idx])
        damage: float = float(damage_severities[node_idx])

        if node_type == "hospital":
            # Hospitals have finite patient capacity.
            capacity: int = int(rng.integers(low=50, high=201))
        elif node_type == "relief_camp":
            # Relief camps can host displaced civilians.
            capacity = int(rng.integers(low=30, high=301))
        else:
            # Road intersections represent routing nodes; capacity can
            # represent the number of vehicles that can queue or pass
            # through concurrently.
            capacity = int(rng.integers(low=0, high=51))

        nodes.append(
            NodeAttributes(
                node_id=node_idx,
                node_type=node_type,
                latitude=lat,
                longitude=lon,
                damage_severity=damage,
                current_capacity=capacity,
            )
        )

    return nodes


def _build_connected_edges(
    rng: np.random.Generator, nodes: List[NodeAttributes]
) -> List[Tuple[int, int, float, bool]]:
    """
    Construct a connected undirected graph over the nodes and derive edge
    attributes (distance and blockage status).

    Connectivity is guaranteed by first linking nodes in a simple chain
    and then adding extra random edges to introduce redundancy.
    """
    graph: nx.Graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.node_id)

    # Step 1: ensure connectivity via a simple chain.
    for node_id in range(1, NUM_NODES):
        graph.add_edge(node_id - 1, node_id)

    # Step 2: add additional random edges to enrich connectivity.
    num_extra_edges: int = NUM_NODES  # one extra edge per node on average
    for _ in range(num_extra_edges):
        source: int = int(rng.integers(low=0, high=NUM_NODES))
        target: int = int(rng.integers(low=0, high=NUM_NODES))
        if source == target:
            continue
        graph.add_edge(source, target)

    # Map node_id to coordinates and damage severity for attribute computation.
    node_lookup: dict[int, NodeAttributes] = {
        node.node_id: node for node in nodes
    }

    edges: List[Tuple[int, int, float, bool]] = []
    for source_id, target_id in graph.edges():
        source_node: NodeAttributes = node_lookup[source_id]
        target_node: NodeAttributes = node_lookup[target_id]

        distance_km: float = _haversine_distance_km(
            source_node.latitude,
            source_node.longitude,
            target_node.latitude,
            target_node.longitude,
        )

        # Blockage probability reflects the average damage of the two
        # incident nodes. This encodes the idea that edges passing through
        # highly damaged regions are more likely to be unusable.
        mean_damage: float = (
            source_node.damage_severity + target_node.damage_severity
        ) / 2.0
        base_block_prob: float = 0.1
        variable_block_prob: float = 0.4 * mean_damage
        block_probability: float = min(
            0.9, base_block_prob + variable_block_prob
        )
        is_blocked: bool = bool(rng.random() < block_probability)

        edges.append((source_id, target_id, distance_km, is_blocked))

    return edges


def _nodes_to_dataframe(nodes: List[NodeAttributes]) -> pd.DataFrame:
    """
    Convert the list of node attributes into a pandas DataFrame with the
    required schema.
    """
    data = {
        "node_id": [node.node_id for node in nodes],
        "type": [node.node_type for node in nodes],
        "latitude": [node.latitude for node in nodes],
        "longitude": [node.longitude for node in nodes],
        "damage_severity": [node.damage_severity for node in nodes],
        "current_capacity": [node.current_capacity for node in nodes],
    }
    return pd.DataFrame(data)


def _edges_to_dataframe(
    edges: List[Tuple[int, int, float, bool]]
) -> pd.DataFrame:
    """
    Convert the list of edges into a pandas DataFrame with the required
    schema.
    """
    data = {
        "source_id": [edge[0] for edge in edges],
        "target_id": [edge[1] for edge in edges],
        "distance_km": [edge[2] for edge in edges],
        "is_blocked": [edge[3] for edge in edges],
    }
    return pd.DataFrame(data)


def main() -> None:
    """
    Generate mock post-disaster infrastructure data and save it as
    `nodes.csv` and `edges.csv` in the same directory as this script.
    """
    rng: np.random.Generator = np.random.default_rng(RNG_SEED)

    nodes: List[NodeAttributes] = _generate_nodes(rng)
    edges = _build_connected_edges(rng, nodes)

    nodes_df: pd.DataFrame = _nodes_to_dataframe(nodes)
    edges_df: pd.DataFrame = _edges_to_dataframe(edges)

    base_dir: Path = Path(__file__).resolve().parent
    nodes_path: Path = base_dir / "nodes.csv"
    edges_path: Path = base_dir / "edges.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)


if __name__ == "__main__":
    main()

