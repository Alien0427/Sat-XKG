from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.explain import Explainer, GNNExplainer

from gnn_model import (
    GCN,
    _build_mock_targets,
    _normalize_features,
    _prepare_data,
)


def _train_model(num_epochs: int = 50) -> Tuple[GCN, Tensor, List[int], List[str]]:
    """
    Re-instantiate and quickly train the GCN model so that we have
    meaningful weights to explain.

    Returns:
    - model: trained GCN instance
    - x: normalized node feature matrix (Tensor)
    - node_ids: mapping from PyG indices to original node IDs
    - node_types: list of node type strings aligned with PyG indices
    """
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    data, node_ids, node_types = _prepare_data()

    # Build synthetic training targets from raw (unnormalized) features.
    y: Tensor = _build_mock_targets(data, node_types)

    # Normalize node features.
    x_norm, _, _ = _normalize_features(data.x)
    data.x = x_norm

    data = data.to(device)
    y = y.to(device)

    model = GCN(in_channels=data.num_node_features, hidden_channels=16).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out: Tensor = model(data.x, data.edge_index)
        loss: Tensor = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"[Training] Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    return model, data.x, node_ids, node_types


def _build_feature_names(node_types: List[str]) -> List[str]:
    """
    Reconstruct human-readable feature names aligned with the encoding
    used in `to_pyg_data`.

    - Index 0: damage_severity
    - Index 1: current_capacity
    - Remaining indices: one-hot type indicators ordered by sorted
      unique node type strings.
    """
    unique_types = sorted(set(node_types))
    feature_names: List[str] = ["damage_severity", "current_capacity"]
    feature_names.extend([f"is_type_{node_type}" for node_type in unique_types])
    return feature_names


def explain_node(target_node_id: int = 28) -> None:
    """
    Generate a human-readable explanation for the model's prediction on
    a specific node.

    The explanation is based on:
    - The most influential input features.
    - The most important incident edges (connections) according to the
      GNNExplainer mask.
    """
    model, x, node_ids, node_types = _train_model(num_epochs=50)

    # Map the external node ID to the internal PyG node index.
    if target_node_id not in node_ids:
        raise ValueError(
            f"Target node ID {target_node_id} does not exist in the graph."
        )
    node_index: int = node_ids.index(target_node_id)

    device = next(model.parameters()).device
    x = x.to(device)

    # Recreate the edge_index by calling _prepare_data again, ensuring
    # consistency with the features and node ordering.
    data, _, _ = _prepare_data()
    edge_index = data.edge_index.to(device)

    # Configure the PyG Explainer with GNNExplainer as the underlying algorithm.
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(x=x, edge_index=edge_index, index=node_index)

    # Feature importance for the target node.
    # node_mask has shape [num_nodes, num_features]; we select the row
    # corresponding to the node of interest.
    node_mask: Tensor = explanation.node_mask[node_index]
    feature_importances = node_mask.detach().cpu().numpy()

    feature_names: List[str] = _build_feature_names(node_types)
    top_feature_indices = feature_importances.argsort()[::-1][:2]
    top_features: List[Tuple[str, float]] = [
        (feature_names[idx], float(feature_importances[idx]))
        for idx in top_feature_indices
    ]

    # Edge importance, restricted to edges incident to the target node.
    edge_index_expl = explanation.edge_index.cpu()
    edge_mask: Tensor = explanation.edge_mask.detach().cpu()

    incident_edges: List[Tuple[int, int, float]] = []
    for edge_pos in range(edge_index_expl.size(1)):
        src = int(edge_index_expl[0, edge_pos].item())
        dst = int(edge_index_expl[1, edge_pos].item())
        if src == node_index or dst == node_index:
            importance: float = float(edge_mask[edge_pos].item())
            incident_edges.append((src, dst, importance))

    incident_edges.sort(key=lambda e: e[2], reverse=True)
    top_incident_edges = incident_edges[:2]

    # Map edge endpoints back to original node IDs for readability.
    top_edge_descriptions: List[str] = []
    for src_idx, dst_idx, importance in top_incident_edges:
        other_idx: int = dst_idx if src_idx == node_index else src_idx
        other_node_id: int = node_ids[other_idx]
        top_edge_descriptions.append(
            f"connection to Node {other_node_id} (importance={importance:.3f})"
        )

    # Compose a human-readable explanation.
    print(f"\nEXPLANATION FOR NODE {target_node_id}")
    print("-" * 60)

    if top_features:
        feature_text = ", ".join(
            f"{name} (importance={importance:.3f})"
            for name, importance in top_features
        )
        print(f"Most influential features: {feature_text}.")

    if top_edge_descriptions:
        edge_text = "; ".join(top_edge_descriptions)
        print(
            f"The model also relied on this node's {edge_text}, "
            "which shapes how information flows from neighboring regions."
        )

    if top_features:
        primary_feature_name: str = top_features[0][0]
        print(
            f"\nSummary: The model prioritized Node {target_node_id} primarily "
            f"due to its '{primary_feature_name}' and its structural "
            f"relationships captured by the highlighted connections."
        )


def main() -> None:
    """
    Script entry point: train the GNN briefly and generate an
    explanation for the critical node with ID 28.
    """
    explain_node(target_node_id=28)


if __name__ == "__main__":
    main()

