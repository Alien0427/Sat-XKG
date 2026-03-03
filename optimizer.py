from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from graph_builder import _load_dataframes, build_graph


def _load_graph() -> nx.Graph:
    """
    Load node and edge tables from CSV files and construct the spatial
    knowledge graph.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    nodes_path: Path = data_dir / "nodes.csv"
    edges_path: Path = data_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        nodes_path = base_dir / "nodes.csv"
        edges_path = base_dir / "edges.csv"

    nodes_df, edges_df = _load_dataframes(nodes_path, edges_path)
    graph: nx.Graph = build_graph(nodes_df, edges_df)
    return graph


def _identify_relief_camps(graph: nx.Graph) -> List[int]:
    """
    Return the list of node IDs corresponding to relief camps.
    """
    return [
        node_id
        for node_id, data in graph.nodes(data=True)
        if str(data.get("type", "")) == "relief_camp"
    ]


def _compute_cost_matrix(
    graph: nx.Graph,
    camp_nodes: List[int],
    critical_nodes: List[int],
) -> np.ndarray:
    """
    Compute the shortest-path distance from each relief camp to each
    critical node using `effective_distance` as the edge weight.
    """
    num_camps: int = len(camp_nodes)
    num_critical: int = len(critical_nodes)
    cost_matrix: np.ndarray = np.zeros((num_camps, num_critical), dtype=float)

    for i, camp_id in enumerate(camp_nodes):
        for j, critical_id in enumerate(critical_nodes):
            distance: float = nx.shortest_path_length(
                graph,
                source=camp_id,
                target=critical_id,
                weight="effective_distance",
            )
            cost_matrix[i, j] = distance

    return cost_matrix


def optimize_deployment(
    graph: nx.Graph,
    critical_nodes: List[int],
) -> List[Tuple[int, int, float]]:
    """
    Solve an assignment problem that deploys teams from relief camps to
    critical nodes while minimizing total travel distance.

    Returns a list of (camp_node_id, critical_node_id, distance).
    """
    camp_nodes: List[int] = _identify_relief_camps(graph)
    if len(camp_nodes) < len(critical_nodes):
        raise ValueError(
            "Not enough relief camps to cover all critical nodes. "
            f"Found {len(camp_nodes)} camps for {len(critical_nodes)} critical nodes."
        )

    cost_matrix: np.ndarray = _compute_cost_matrix(
        graph,
        camp_nodes,
        critical_nodes,
    )

    # Hungarian algorithm: rows = camps, columns = critical nodes.
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    assignments: List[Tuple[int, int, float]] = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        camp_id: int = camp_nodes[row_idx]
        critical_id: int = critical_nodes[col_idx]
        distance: float = float(cost_matrix[row_idx, col_idx])
        assignments.append((camp_id, critical_id, distance))

    return assignments


def main() -> None:
    """
    Entry point for computing and displaying an optimal deployment plan
    for rescue teams.
    """
    graph: nx.Graph = _load_graph()

    # Critical node IDs can be provided dynamically from a GNN prediction
    # pipeline. For this prototype, we rely on fixed IDs derived from a
    # prior run.
    critical_nodes: List[int] = [28, 27, 29, 8, 31]

    assignments: List[Tuple[int, int, float]] = optimize_deployment(
        graph,
        critical_nodes,
    )

    print("Rescue Team Deployment Manifest")
    print("-" * 40)
    for camp_id, critical_id, distance in assignments:
        print(
            f"Team from Camp [{camp_id}] deployed to Critical Node "
            f"[{critical_id}]. Route distance: {distance:.2f}"
        )


if __name__ == "__main__":
    main()

