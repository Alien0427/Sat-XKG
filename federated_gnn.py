from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import MSELoss, Module
from torch.optim import Adam
from torch_geometric.data import Data

from graph_builder import _load_dataframes, build_graph, to_pyg_data
from gnn_model import GCN, _build_mock_targets, _normalize_features


def _prepare_partition(
    partition_index: int,
    device: torch.device,
) -> Tuple[Data, Tensor]:
    """
    Load a single edge partition (sub-graph) and convert it into a PyG
    Data object together with synthetic target scores.

    The synthetic targets are computed from the raw (unnormalized)
    features, and then the features are standardized for training.
    """
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"

    nodes_path: Path = data_dir / f"edge_{partition_index}_nodes.csv"
    edges_path: Path = data_dir / f"edge_{partition_index}_edges.csv"

    nodes_df, edges_df = _load_dataframes(nodes_path, edges_path)
    graph = build_graph(nodes_df, edges_df)

    # Maintain deterministic node ordering.
    node_ids: List[int] = sorted(graph.nodes())
    node_types: List[str] = [
        str(graph.nodes[node_id].get("type", "")) for node_id in node_ids
    ]

    data: Data = to_pyg_data(graph)

    # Build synthetic targets using the original features.
    y: Tensor = _build_mock_targets(data, node_types).to(device)

    # Normalize node features for training.
    x_norm, _, _ = _normalize_features(data.x)
    data.x = x_norm
    data = data.to(device)

    return data, y


def _federated_averaging(
    local_models: List[Module],
    global_model: Module,
) -> None:
    """
    Aggregate local model parameters into the global model using the
    standard Federated Averaging (FedAvg) rule, i.e., arithmetic mean
    over corresponding parameters.
    """
    if not local_models:
        return

    global_state = global_model.state_dict()
    local_states = [model.state_dict() for model in local_models]
    num_models: int = len(local_states)

    # Initialize the aggregated state with zeros of the same shape.
    aggregated_state = {key: torch.zeros_like(value) for key, value in global_state.items()}

    for key in aggregated_state.keys():
        for state in local_states:
            aggregated_state[key] += state[key]
        aggregated_state[key] /= float(num_models)

    global_model.load_state_dict(aggregated_state)


def run_federated_training() -> None:
    """
    Simulate Federated Graph Learning over three edge partitions using a
    simple FedAvg protocol.

    Each communication round consists of:
    A. Broadcasting global parameters to all local models.
    B. Training each local model on its own sub-graph.
    C. Averaging local parameters back into the global model.

    After every aggregation step, the global model is evaluated on the
    union of all sub-graphs (treated independently but combined in the
    loss metric).
    """
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Load all edge partitions.
    partition_indices: List[int] = [1, 2, 3]
    partition_data: List[Tuple[Data, Tensor]] = [
        _prepare_partition(idx, device) for idx in partition_indices
    ]

    # Initialize global and local models.
    example_data: Data = partition_data[0][0]
    in_channels: int = example_data.num_node_features

    global_model: GCN = GCN(in_channels=in_channels, hidden_channels=16).to(device)
    local_models: List[GCN] = [
        GCN(in_channels=in_channels, hidden_channels=16).to(device)
        for _ in partition_indices
    ]

    local_optimizers: List[Adam] = [
        Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for model in local_models
    ]
    criterion = MSELoss()

    num_rounds: int = 10
    local_epochs: int = 5

    for round_idx in range(1, num_rounds + 1):
        # Step A: broadcast global weights to local models.
        global_state = global_model.state_dict()
        for local_model in local_models:
            local_model.load_state_dict(global_state)

        # Step B: local training on each partition.
        for (local_model, optimizer, (data, targets)) in zip(
            local_models, local_optimizers, partition_data
        ):
            local_model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                out: Tensor = local_model(data.x, data.edge_index)
                loss: Tensor = criterion(out, targets)
                loss.backward()
                optimizer.step()

        # Step C: Federated Averaging to update the global model.
        _federated_averaging(local_models, global_model)

        # Evaluate global model across all partitions.
        global_model.eval()
        total_loss: float = 0.0
        total_nodes: int = 0
        with torch.no_grad():
            for data, targets in partition_data:
                out_global: Tensor = global_model(data.x, data.edge_index)
                loss_global: Tensor = criterion(out_global, targets)
                num_nodes: int = data.num_nodes
                total_loss += float(loss_global.item()) * float(num_nodes)
                total_nodes += num_nodes

        average_loss: float = total_loss / float(total_nodes)
        print(f"Round {round_idx:02d} | Global model average loss: {average_loss:.6f}")


def main() -> None:
    """
    Script entry point to launch the federated GNN training simulation.
    """
    run_federated_training()


if __name__ == "__main__":
    main()

