from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from graph_builder import _load_dataframes, build_graph


ClusterId = int


def _load_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load node and edge tables from CSV files.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    nodes_path: Path = data_dir / "nodes.csv"
    edges_path: Path = data_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        nodes_path = base_dir / "nodes.csv"
        edges_path = base_dir / "edges.csv"

    return _load_dataframes(nodes_path, edges_path)


def _cluster_nodes(
    nodes_df: pd.DataFrame, num_clusters: int = 3
) -> Dict[int, ClusterId]:
    """
    Cluster nodes geographically using latitude and longitude.

    Returns a mapping from node_id to cluster identifier in
    {0, 1, ..., num_clusters-1}.
    """
    coordinates = nodes_df[["latitude", "longitude"]].to_numpy(dtype=float)

    model = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels: np.ndarray = model.fit_predict(coordinates)

    node_ids: List[int] = nodes_df["node_id"].astype(int).tolist()
    node_to_cluster: Dict[int, ClusterId] = {
        node_id: int(label) for node_id, label in zip(node_ids, labels, strict=True)
    }
    return node_to_cluster


def _build_subgraphs(
    graph: nx.Graph, node_to_cluster: Dict[int, ClusterId], num_clusters: int
) -> List[nx.Graph]:
    """
    Build per-cluster subgraphs from the global graph.

    For this prototype, an edge is included in a subgraph only if both
    endpoints are assigned to the same cluster. Cross-cluster edges are
    dropped, which simulates communication boundaries between edge
    domains.
    """
    subgraphs: List[nx.Graph] = [nx.Graph() for _ in range(num_clusters)]

    # Add nodes with attributes to their respective cluster graph.
    for node_id, data in graph.nodes(data=True):
        cluster_id: ClusterId = node_to_cluster[int(node_id)]
        subgraphs[cluster_id].add_node(node_id, **data)

    # Add edges where both endpoints lie in the same cluster.
    for u, v, data in graph.edges(data=True):
        cluster_u: ClusterId = node_to_cluster[int(u)]
        cluster_v: ClusterId = node_to_cluster[int(v)]
        if cluster_u == cluster_v:
            subgraphs[cluster_u].add_edge(u, v, **data)

    return subgraphs


def _save_subgraphs(
    subgraphs: List[nx.Graph],
    data_dir: Path,
) -> None:
    """
    Export each subgraph's nodes and edges to CSV files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for idx, subgraph in enumerate(subgraphs, start=1):
        nodes_records = []
        for node_id, data in subgraph.nodes(data=True):
            nodes_records.append(
                {
                    "node_id": int(node_id),
                    "type": data.get("type"),
                    "latitude": float(data.get("latitude", 0.0)),
                    "longitude": float(data.get("longitude", 0.0)),
                    "damage_severity": float(data.get("damage_severity", 0.0)),
                    "current_capacity": int(data.get("current_capacity", 0)),
                }
            )

        edges_records = []
        for u, v, data in subgraph.edges(data=True):
            edges_records.append(
                {
                    "source_id": int(u),
                    "target_id": int(v),
                    "distance_km": float(data.get("distance_km", 0.0)),
                    "is_blocked": bool(data.get("is_blocked", False)),
                    "effective_distance": float(
                        data.get("effective_distance", data.get("distance_km", 0.0))
                    ),
                }
            )

        nodes_df = pd.DataFrame(nodes_records)
        edges_df = pd.DataFrame(edges_records)

        nodes_path: Path = data_dir / f"edge_{idx}_nodes.csv"
        edges_path: Path = data_dir / f"edge_{idx}_edges.csv"

        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)


def _visualize_clusters(
    graph: nx.Graph, node_to_cluster: Dict[int, ClusterId], output_path: Path
) -> None:
    """
    Plot the global graph with nodes colored by their assigned edge
    cluster and save the result to disk.
    """
    positions: Dict[int, Tuple[float, float]] = {
        node_id: (
            float(data.get("longitude", 0.0)),
            float(data.get("latitude", 0.0)),
        )
        for node_id, data in graph.nodes(data=True)
    }

    unique_clusters = sorted(set(node_to_cluster.values()))
    color_palette = plt.colormaps.get("tab10")

    plt.figure(figsize=(8, 6))

    # Draw edges in light gray for context.
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        alpha=0.2,
        width=0.8,
        edge_color="lightgray",
    )

    # Draw nodes colored by cluster assignment.
    for cluster_id in unique_clusters:
        nodelist: List[int] = [
            node_id
            for node_id, cid in node_to_cluster.items()
            if cid == cluster_id
        ]
        if not nodelist:
            continue

        color = color_palette(cluster_id % 10)
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=nodelist,
            node_color=[color],
            node_size=60,
            linewidths=0.3,
            edgecolors="black",
            label=f"Edge Node {cluster_id + 1}",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Edge AI Partitions of Disaster Zone")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """
    Simulate three Edge AI nodes (e.g., UAVs or satellites) by
    partitioning the disaster region into spatial clusters, exporting
    per-partition subgraphs, and visualizing the resulting allocation.
    """
    nodes_df, edges_df = _load_tables()

    # Cluster nodes geographically into three edge domains.
    node_to_cluster: Dict[int, ClusterId] = _cluster_nodes(nodes_df, num_clusters=3)

    # Build the full spatial knowledge graph and the cluster-specific
    # subgraphs.
    graph: nx.Graph = build_graph(nodes_df, edges_df)
    subgraphs: List[nx.Graph] = _build_subgraphs(
        graph,
        node_to_cluster,
        num_clusters=3,
    )

    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    _save_subgraphs(subgraphs, data_dir=data_dir)

    # Visualize the global graph with per-edge-node color coding.
    output_path: Path = base_dir / "edge_clusters.png"
    _visualize_clusters(graph, node_to_cluster, output_path)

    print("Edge partitions created and saved:")
    for idx in range(1, 4):
        print(f"  - data/edge_{idx}_nodes.csv")
        print(f"  - data/edge_{idx}_edges.csv")
    print(f"Cluster visualization saved to: {output_path.name}")


if __name__ == "__main__":
    main()

