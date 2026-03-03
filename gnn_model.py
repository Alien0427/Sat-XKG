from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import MSELoss, Module
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from graph_builder import _load_dataframes, build_graph, to_pyg_data


class GCN(Model := Module):  # type: ignore[misc]
    """
    Two-layer Graph Convolutional Network for node-level regression.

    The network maps node features to a single scalar criticality score
    in the range [0, 1] via a final Sigmoid activation.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.conv1: GCNConv = GCNConv(in_channels, hidden_channels)
        self.conv2: GCNConv = GCNConv(hidden_channels, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        return x


def _normalize_features(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Standardize node features to zero mean and unit variance per feature.

    Returns the normalized features together with the mean and standard
    deviation, which can be useful for future inverse transforms.
    """
    mean: Tensor = x.mean(dim=0, keepdim=True)
    std: Tensor = x.std(dim=0, keepdim=True)

    # Avoid division by zero in case a feature is constant.
    std = torch.where(std == 0, torch.ones_like(std), std)

    x_norm: Tensor = (x - mean) / std
    return x_norm, mean, std


def _build_mock_targets(
    data: Data, graph_node_types: List[str]
) -> Tensor:
    """
    Construct synthetic criticality scores for each node.

    The target is defined as:

        criticality = 0.7 * damage_severity + 0.3 * is_hospital,

    where `is_hospital` is 1 for hospital nodes and 0 otherwise.

    The function assumes that:
    - data.x[:, 0] encodes damage_severity
    - graph_node_types[i] contains the node type string for node index i
    """
    num_nodes: int = data.num_nodes
    device: torch.device = data.x.device

    damage_severity: Tensor = data.x[:, 0]

    is_hospital_flags: List[float] = [
        1.0 if node_type == "hospital" else 0.0
        for node_type in graph_node_types
    ]
    is_hospital: Tensor = torch.tensor(
        is_hospital_flags,
        dtype=torch.float32,
        device=device,
    )

    targets: Tensor = 0.7 * damage_severity + 0.3 * is_hospital
    return targets.view(num_nodes, 1)


def _prepare_data() -> Tuple[Data, List[int], List[str]]:
    """
    Load tabular data, build the NetworkX graph, and convert it to a
    PyTorch Geometric Data object.

    Returns:
    - data: PyG Data object with node features and edge structure.
    - node_ids: List mapping PyG node indices back to original node IDs.
    - node_types: List of node type strings aligned with PyG indices.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    nodes_path: Path = data_dir / "nodes.csv"
    edges_path: Path = data_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        nodes_path = base_dir / "nodes.csv"
        edges_path = base_dir / "edges.csv"

    nodes_df, edges_df = _load_dataframes(nodes_path, edges_path)
    graph = build_graph(nodes_df, edges_df)

    # Maintain the same deterministic ordering as in to_pyg_data.
    node_ids: List[int] = sorted(graph.nodes())
    node_types: List[str] = [
        str(graph.nodes[node_id].get("type", "")) for node_id in node_ids
    ]

    data: Data = to_pyg_data(graph)
    return data, node_ids, node_types


def train_gnn() -> None:
    """
    Build and train a GCN to regress node-level criticality scores.

    The model is trained on synthetic labels derived from node damage
    severity and hospital status. This setup mimics a supervised
    learning scenario where criticality is a continuous risk score.
    """
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    data, node_ids, node_types = _prepare_data()

    # Normalize input features.
    x_norm, _, _ = _normalize_features(data.x)
    data.x = x_norm

    data = data.to(device)

    # Prepare synthetic targets.
    y: Tensor = _build_mock_targets(data, node_types)

    model = GCN(in_channels=data.num_node_features, hidden_channels=16).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = MSELoss()

    num_epochs: int = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out: Tensor = model(data.x, data.edge_index)  # shape: [num_nodes, 1]
        loss: Tensor = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    # Evaluate trained model and report top-5 most critical nodes.
    model.eval()
    with torch.no_grad():
        predictions: Tensor = model(data.x, data.edge_index).view(-1)

    # Identify indices of the five largest predicted criticality scores.
    top_k: int = min(5, predictions.numel())
    top_values, top_indices = torch.topk(predictions, k=top_k)

    print("\nTop predicted critical nodes:")
    for rank in range(top_k):
        node_index: int = int(top_indices[rank].item())
        node_id: int = node_ids[node_index]
        score: float = float(top_values[rank].item())
        print(f"Rank {rank + 1}: Node ID {node_id} | Predicted criticality = {score:.4f}")


def main() -> None:
    """
    Script entry point for training and inspecting the GNN model.
    """
    train_gnn()


if __name__ == "__main__":
    main()

