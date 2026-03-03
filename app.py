from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import torch
from torch import Tensor

from graph_builder import build_graph, to_pyg_data, visualize_graph
from gnn_model import GCN, _build_mock_targets, _normalize_features
from optimizer import optimize_deployment
from telecom_simulator import (
    _compute_bandwidth_savings,
    _compute_cloud_payload,
    _compute_edge_payload,
    _compute_latency,
)
from energy_simulator import (
    _compute_cloud_energy,
    _compute_edge_energy,
    _compute_energy_savings,
)


def _load_fused_data(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load fused node and edge tables for a given disaster zone prefix.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    nodes_path: Path = data_dir / f"{prefix}_fused_nodes.csv"
    edges_path: Path = data_dir / f"{prefix}_fused_edges.csv"

    nodes_df: pd.DataFrame = pd.read_csv(nodes_path)
    edges_df: pd.DataFrame = pd.read_csv(edges_path)
    return nodes_df, edges_df


def _train_gnn_on_graph(graph: nx.Graph) -> Tuple[GCN, List[int], Tensor]:
    """
    Train a GCN on the provided graph using the same synthetic target
    definition as in the standalone training script.

    Returns:
    - model: trained GCN instance
    - node_ids: list mapping tensor index -> original node_id
    - predictions: 1D tensor of predicted criticality scores
    """
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    node_ids: List[int] = sorted(graph.nodes())
    node_types: List[str] = [
        str(graph.nodes[node_id].get("type", "")) for node_id in node_ids
    ]

    data = to_pyg_data(graph)

    # Normalize node features.
    x_norm, _, _ = _normalize_features(data.x)
    data.x = x_norm
    data = data.to(device)

    # Synthetic supervision.
    targets: Tensor = _build_mock_targets(data, node_types)

    model = GCN(in_channels=data.num_node_features, hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    num_epochs: int = 50
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out: Tensor = model(data.x, data.edge_index)
        loss: Tensor = criterion(out, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions: Tensor = model(data.x, data.edge_index).view(-1).cpu()

    return model, node_ids, predictions


def _compute_top_critical_nodes(
    node_ids: List[int],
    predictions: Tensor,
    k: int = 5,
) -> pd.DataFrame:
    """
    Build a small DataFrame summarizing the top-k most critical nodes.
    """
    top_k: int = min(k, predictions.numel())
    top_values, top_indices = torch.topk(predictions, k=top_k)

    records: List[dict] = []
    for rank in range(top_k):
        node_index = int(top_indices[rank].item())
        node_id = node_ids[node_index]
        score = float(top_values[rank].item())
        records.append(
            {
                "Rank": rank + 1,
                "Node ID": node_id,
                "Predicted Criticality": round(score, 4),
            }
        )

    df = pd.DataFrame(records)
    # Ensure node identifiers are represented using an integer dtype
    # that preserves precision for large OSM-style IDs.
    df["Node ID"] = df["Node ID"].astype("Int64")
    return df


def _build_deployment_manifest(
    graph: nx.Graph,
    critical_node_ids: List[int],
) -> str:
    """
    Use the optimizer to derive a deployment manifest and format it for
    display.
    """
    assignments = optimize_deployment(graph, critical_node_ids)

    lines = ["Rescue Team Deployment Manifest", "-" * 40]
    for camp_id, critical_id, distance in assignments:
        lines.append(
            f"Team from Camp [{camp_id}] deployed to Critical Node "
            f"[{critical_id}]. Route distance: {distance:.2f}"
        )
    return "\n".join(lines)


def _build_explanation_sentence(
    graph: nx.Graph,
    critical_node_id: int,
) -> str:
    """
    Construct a lightweight, human-readable explanation for why a given
    node is considered critical, based on its attributes and local
    connectivity in the fused graph.
    """
    data = graph.nodes[critical_node_id]
    node_type = str(data.get("type", "unknown"))
    damage = float(data.get("damage_severity", 0.0))

    # Count incident blocked edges as a proxy for disrupted access.
    blocked_neighbors = [
        neighbor
        for neighbor in graph.neighbors(critical_node_id)
        if bool(graph.edges[critical_node_id, neighbor].get("is_blocked", False))
    ]

    if blocked_neighbors:
        neighbor_summary = (
            f"with blocked access to {len(blocked_neighbors)} neighboring nodes"
        )
    else:
        neighbor_summary = "with intact but vulnerable connectivity"

    return (
        f"Node {critical_node_id} is prioritized because it is a "
        f"{node_type.replace('_', ' ')} with high inferred damage "
        f"(damage_severity ≈ {damage:.2f}) and {neighbor_summary}, "
        "indicating that reaching this asset is both critical and "
        "operationally constrained."
    )


def _compute_telemetry_metrics(num_nodes: int) -> Tuple[float, float, float]:
    """
    Compute latency and bandwidth savings metrics for display.

    Returns:
    - cloud_latency_hours
    - edge_latency_ms
    - bandwidth_saved_percent
    """
    cloud_payload = _compute_cloud_payload(num_nodes=num_nodes)
    edge_payload = _compute_edge_payload()
    latency = _compute_latency(cloud_payload, edge_payload)
    bandwidth_saved = _compute_bandwidth_savings(cloud_payload, edge_payload)
    return (
        latency.cloud_latency_hours,
        latency.edge_latency_milliseconds,
        bandwidth_saved,
    )


def _compute_energy_metrics(num_nodes: int) -> Tuple[float, float, float]:
    """
    Compute energy consumption metrics for display.

    Returns:
    - cloud_energy_total
    - edge_energy_total
    - energy_saved_percent
    """
    cloud_profile = _compute_cloud_energy(num_nodes=num_nodes)
    edge_profile = _compute_edge_energy()
    energy_saved = _compute_energy_savings(cloud_profile, edge_profile)
    return (
        cloud_profile.total_energy_joules,
        edge_profile.total_energy_joules,
        energy_saved,
    )


def main() -> None:
    """
    Streamlit application entry point.
    """
    st.set_page_config(
        page_title="Sat-XKG: 6G Edge AI Disaster Response",
        layout="wide",
    )

    st.title("Sat-XKG: 6G Edge AI Disaster Response")

    # Sidebar controls.
    st.sidebar.header("Simulation Controls")
    zone_label = st.sidebar.selectbox(
        "Disaster Zone",
        ["Wayanad, Kerala", "Puri, Odisha"],
    )
    zone_prefix = "wayanad" if zone_label.startswith("Wayanad") else "puri"

    execute = st.sidebar.button("Execute Simulation")

    if not execute:
        st.info("Select a disaster zone and click **Execute Simulation** to run Sat-XKG.")
        return

    # ------------------------------------------------------------------
    # 1. Load fused data and construct the spatial knowledge graph.
    # ------------------------------------------------------------------
    nodes_df, edges_df = _load_fused_data(zone_prefix)
    graph = build_graph(nodes_df, edges_df)

    st.subheader("Spatial Knowledge Graph")
    # Use the visualization helper and render the returned figure.
    fig = visualize_graph(graph)
    st.pyplot(fig)
    # Explicitly close this figure to prevent Streamlit from reusing the
    # previous Matplotlib state when switching locations.
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. GNN inference to estimate node criticality.
    # ------------------------------------------------------------------
    st.subheader("GNN Inference: Critical Node Ranking")
    model, node_ids, predictions = _train_gnn_on_graph(graph)
    top_nodes_df = _compute_top_critical_nodes(node_ids, predictions, k=5)
    st.dataframe(top_nodes_df, use_container_width=True)

    critical_node_ids: List[int] = top_nodes_df["Node ID"].tolist()

    # ------------------------------------------------------------------
    # 3. Operations optimizer: resource deployment manifest.
    # ------------------------------------------------------------------
    st.subheader("Operations Optimizer: Resource Deployment")
    manifest_text = _build_deployment_manifest(graph, critical_node_ids)
    st.text(manifest_text)

    # ------------------------------------------------------------------
    # 4. XAI explainer: narrative for the top critical node.
    # ------------------------------------------------------------------
    st.subheader("Explainable AI (XAI)")
    if critical_node_ids:
        explanation_sentence = _build_explanation_sentence(
            graph,
            critical_node_ids[0],
        )
        st.success(explanation_sentence)
    else:
        st.warning("No critical nodes were identified for explanation.")

    # ------------------------------------------------------------------
    # 5. Telemetry and hardware efficiency metrics.
    # ------------------------------------------------------------------
    st.subheader("6G Telemetry & Hardware Efficiency")
    col1, col2 = st.columns(2)

    num_nodes: int = int(len(nodes_df))
    cloud_latency_h, edge_latency_ms, bandwidth_saved = _compute_telemetry_metrics(
        num_nodes=num_nodes
    )
    cloud_energy_j, edge_energy_j, energy_saved = _compute_energy_metrics(
        num_nodes=num_nodes
    )

    with col1:
        st.metric(
            label="Latency: Edge vs Cloud",
            value=f"{edge_latency_ms:,.1f} ms",
            delta=f"-{cloud_latency_h:,.2f} hours vs cloud",
        )
        st.metric(
            label="Bandwidth Saved",
            value=f"{bandwidth_saved:,.2f} %",
        )

    with col2:
        st.metric(
            label="Energy per Tile (Edge)",
            value=f"{edge_energy_j:,.3f} J",
            delta=f"-{(cloud_energy_j - edge_energy_j):,.3f} J vs cloud",
        )
        st.metric(
            label="Battery Life Saved",
            value=f"{energy_saved:,.2f} %",
        )


if __name__ == "__main__":
    main()

