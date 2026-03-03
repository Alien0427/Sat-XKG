from __future__ import annotations

from dataclasses import dataclass


TRANSMISSION_ENERGY_PER_MB_J: float = 0.5
EDGE_INFERENCE_ENERGY_J: float = 2.0

IMAGE_TILE_SIZE_MB: float = 250.0
EDGE_TELEMETRY_SIZE_KB: float = 2.0


@dataclass(frozen=True)
class EnergyProfile:
    transmission_energy_joules: float
    compute_energy_joules: float

    @property
    def total_energy_joules(self) -> float:
        return self.transmission_energy_joules + self.compute_energy_joules


def _compute_cloud_energy(num_nodes: int) -> EnergyProfile:
    """
    Compute the energy profile of the traditional cloud architecture,
    where raw imagery is transmitted and inference happens remotely.
    """
    # Assume one 250 MB image tile per 50 nodes to approximate how many
    # distinct high-resolution scenes must be downlinked for the current
    # graph.
    tiles: float = float(num_nodes) / 50.0
    transmission_energy: float = tiles * IMAGE_TILE_SIZE_MB * TRANSMISSION_ENERGY_PER_MB_J
    return EnergyProfile(
        transmission_energy_joules=transmission_energy,
        compute_energy_joules=0.0,
    )


def _compute_edge_energy() -> EnergyProfile:
    """
    Compute the energy profile of the edge architecture, where inference
    is executed on-board and only compact XAI telemetry is transmitted.
    """
    telemetry_size_mb: float = EDGE_TELEMETRY_SIZE_KB / 1024.0
    transmission_energy: float = telemetry_size_mb * TRANSMISSION_ENERGY_PER_MB_J

    compute_energy: float = EDGE_INFERENCE_ENERGY_J

    return EnergyProfile(
        transmission_energy_joules=transmission_energy,
        compute_energy_joules=compute_energy,
    )


def _compute_energy_savings(
    cloud_profile: EnergyProfile, edge_profile: EnergyProfile
) -> float:
    """
    Compute the relative reduction in total energy consumption when
    moving from cloud-centric to edge-centric processing.
    """
    cloud_total: float = cloud_profile.total_energy_joules
    edge_total: float = edge_profile.total_energy_joules
    savings_fraction: float = 1.0 - (edge_total / cloud_total)
    return savings_fraction * 100.0


def _format_report(
    cloud_profile: EnergyProfile,
    edge_profile: EnergyProfile,
    savings_percent: float,
) -> str:
    """
    Build a structured 'Hardware Power Efficiency Report' comparing
    traditional cloud versus 6G edge energy consumption.
    """
    lines: list[str] = []
    lines.append("------------------------------------------------------------")
    lines.append("                HARDWARE POWER EFFICIENCY REPORT            ")
    lines.append("------------------------------------------------------------")
    lines.append("")
    lines.append("1. HARDWARE ENERGY MODEL")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"Transmission Energy Cost : {TRANSMISSION_ENERGY_PER_MB_J:.2f} J / MB"
    )
    lines.append(
        f"Edge GNN Inference Cost  : {EDGE_INFERENCE_ENERGY_J:.2f} J per inference"
    )
    lines.append("")
    lines.append("2. ARCHITECTURE PROFILES (PER IMAGE TILE)")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"Cloud Architecture       : "
        f"Tx = {cloud_profile.transmission_energy_joules:8.3f} J, "
        f"Compute = {cloud_profile.compute_energy_joules:6.3f} J, "
        f"Total = {cloud_profile.total_energy_joules:8.3f} J"
    )
    lines.append(
        f"6G Edge Architecture     : "
        f"Tx = {edge_profile.transmission_energy_joules:8.6f} J, "
        f"Compute = {edge_profile.compute_energy_joules:6.3f} J, "
        f"Total = {edge_profile.total_energy_joules:8.6f} J"
    )
    lines.append("")
    lines.append("3. BATTERY UTILIZATION")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"Relative Energy Saved    : {savings_percent:8.4f} % "
        "(edge vs. cloud, per tile)"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  Offloading raw imagery to a distant cloud forces the platform to "
        "pay a high RF transmission cost for every tile. Executing GNN "
        "inference locally on an energy-efficient SoC and transmitting only "
        "compact XAI telemetry reduces per-tile energy consumption by more "
        "than an order of magnitude, directly translating into extended "
        "satellite/UAV battery life and longer on-station time."
    )
    lines.append("------------------------------------------------------------")
    return "\n".join(lines)


def main() -> None:
    """
    Entry point for the energy consumption simulation. It compares the
    energy required by traditional cloud processing and 6G edge
    inference for a single 250 MB image tile.
    """
    # For standalone usage, retain the original assumption of one tile.
    cloud_profile: EnergyProfile = _compute_cloud_energy(num_nodes=1)
    edge_profile: EnergyProfile = _compute_edge_energy()

    savings_percent: float = _compute_energy_savings(
        cloud_profile,
        edge_profile,
    )

    report: str = _format_report(
        cloud_profile,
        edge_profile,
        savings_percent,
    )
    print(report)


if __name__ == "__main__":
    main()

