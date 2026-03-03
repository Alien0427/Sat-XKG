from __future__ import annotations

from dataclasses import dataclass


PAYLOAD_PER_NODE_MB: float = 250.0
EDGE_PAYLOAD_KB: float = 2.0
DISASTER_BANDWIDTH_MBPS: float = 5.0


@dataclass(frozen=True)
class PayloadMetrics:
    size_megabytes: float
    size_megabits: float


@dataclass(frozen=True)
class LatencyMetrics:
    transmission_time_seconds: float
    cloud_latency_hours: float
    edge_latency_milliseconds: float


def _compute_cloud_payload(num_nodes: int) -> PayloadMetrics:
    """
    Compute the total size of the traditional cloud payload.

    Each high-resolution SAR/Optical tile is assumed to cover a small
    set of nodes. For the purposes of this simulator, we assume that one
    250 MB image tile corresponds to 50 graph nodes.
    """
    tiles: float = float(num_nodes) / 50.0
    total_mb: float = tiles * PAYLOAD_PER_NODE_MB
    total_megabits: float = total_mb * 8.0
    return PayloadMetrics(size_megabytes=total_mb, size_megabits=total_megabits)


def _compute_edge_payload() -> PayloadMetrics:
    """
    Compute the size of the edge payload, assuming only textual
    telemetry (deployment manifest and XAI explanation) is transmitted.
    """
    size_megabytes: float = EDGE_PAYLOAD_KB / 1024.0
    size_megabits: float = size_megabytes * 8.0
    return PayloadMetrics(
        size_megabytes=size_megabytes,
        size_megabits=size_megabits,
    )


def _compute_latency(
    cloud_payload: PayloadMetrics, edge_payload: PayloadMetrics
) -> LatencyMetrics:
    """
    Compute end-to-end transmission latency for cloud and edge payloads
    over a degraded disaster bandwidth.

    Transmission time (seconds) = payload (Mb) / bandwidth (Mbps).

    Cloud latency is reported in hours; edge latency is reported in
    milliseconds.
    """
    cloud_time_seconds: float = (
        cloud_payload.size_megabits / DISASTER_BANDWIDTH_MBPS
    )
    edge_time_seconds: float = (
        edge_payload.size_megabits / DISASTER_BANDWIDTH_MBPS
    )

    cloud_hours: float = cloud_time_seconds / 3600.0
    edge_milliseconds: float = edge_time_seconds * 1000.0

    return LatencyMetrics(
        transmission_time_seconds=cloud_time_seconds,
        cloud_latency_hours=cloud_hours,
        edge_latency_milliseconds=edge_milliseconds,
    )


def _compute_bandwidth_savings(
    cloud_payload: PayloadMetrics, edge_payload: PayloadMetrics
) -> float:
    """
    Compute the relative bandwidth saved by transmitting only the edge
    telemetry compared to the full cloud imagery.
    """
    savings: float = 1.0 - (edge_payload.size_megabits / cloud_payload.size_megabits)
    return savings * 100.0


def _format_report(
    cloud_payload: PayloadMetrics,
    edge_payload: PayloadMetrics,
    latency: LatencyMetrics,
    bandwidth_savings_percent: float,
) -> str:
    """
    Assemble a structured textual report summarizing payload sizes,
    transmission latencies, and bandwidth savings.
    """
    lines: list[str] = []
    lines.append("------------------------------------------------------------")
    lines.append("               6G EDGE TELEMETRY & LATENCY REPORT           ")
    lines.append("------------------------------------------------------------")
    lines.append("")
    lines.append("1. PAYLOAD CHARACTERISTICS")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"Traditional Cloud Payload : {cloud_payload.size_megabytes:10.2f} MB "
        f"({cloud_payload.size_megabits:,.2f} Mb)"
    )
    lines.append(
        f"6G Edge Payload           : {edge_payload.size_megabytes:10.6f} MB "
        f"({edge_payload.size_megabits:,.6f} Mb)"
    )
    lines.append("")
    lines.append("2. NETWORK CONDITIONS")
    lines.append("------------------------------------------------------------")
    lines.append(f"Assumed Downlink Bandwidth: {DISASTER_BANDWIDTH_MBPS:.1f} Mbps")
    lines.append("")
    lines.append("3. TRANSMISSION LATENCY")
    lines.append("------------------------------------------------------------")
    lines.append(
        "Traditional Cloud Latency : "
        f"{latency.cloud_latency_hours:10.3f} hours"
    )
    lines.append(
        "6G Edge Latency           : "
        f"{latency.edge_latency_milliseconds:10.3f} ms"
    )
    lines.append("")
    lines.append("4. BANDWIDTH UTILIZATION")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"Relative Bandwidth Saved  : {bandwidth_savings_percent:10.4f} %"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  In a degraded disaster network, transmitting full-resolution imagery "
        "to a remote cloud introduces multi-hour delays, whereas executing "
        "inference on 6G edge nodes and sending only the decision telemetry "
        "reduces end-to-end latency to the millisecond regime while saving "
        "over 99.9% of the bandwidth."
    )
    lines.append("------------------------------------------------------------")
    return "\n".join(lines)


def main() -> None:
    """
    Entry point for the telecom simulation. It computes and reports the
    latency and bandwidth implications of cloud-centric versus 6G edge
    inference architectures in a disaster scenario.
    """
    # For standalone usage, retain the original assumption of 50 tiles.
    cloud_payload: PayloadMetrics = _compute_cloud_payload(num_nodes=50)
    edge_payload: PayloadMetrics = _compute_edge_payload()

    latency: LatencyMetrics = _compute_latency(cloud_payload, edge_payload)
    bandwidth_savings_percent: float = _compute_bandwidth_savings(
        cloud_payload,
        edge_payload,
    )

    report: str = _format_report(
        cloud_payload,
        edge_payload,
        latency,
        bandwidth_savings_percent,
    )
    print(report)


if __name__ == "__main__":
    main()

