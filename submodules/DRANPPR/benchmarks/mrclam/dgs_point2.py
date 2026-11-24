"""Protocol helpers for the MR.CLAM DGS-Point2 extension."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmarks.slideslam.run import CommandSpec


@dataclass(frozen=True)
class IterationCommunication:
    point2_separator_blocks: int
    point2_separator_bytes: int
    linear_system_packets: int
    linear_system_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class CommunicationAccounting:
    """Logical payload implied by the documented owner/observer protocol."""

    remote_observer_object_pairs: int

    def __post_init__(self) -> None:
        if self.remote_observer_object_pairs < 0:
            raise ValueError("remote observer/object pair count must be nonnegative")

    def for_outer_iteration(self, inner_sweeps: int) -> IterationCommunication:
        if inner_sweeps < 1:
            raise ValueError("inner sweeps must be positive")
        pairs = self.remote_observer_object_pairs
        separator_blocks = pairs * inner_sweeps
        separator_bytes = separator_blocks * 2 * 8
        # One five-double (symmetric H_ll and g_l) packet at relinearization,
        # followed by a two-double pose-dependent RHS packet per inner sweep.
        linear_packets = pairs * (1 + inner_sweeps)
        linear_bytes = pairs * (5 + 2 * inner_sweeps) * 8
        return IterationCommunication(
            separator_blocks,
            separator_bytes,
            linear_packets,
            linear_bytes,
            separator_bytes + linear_bytes,
        )


@dataclass(frozen=True)
class DGSPoint2TraceRow:
    outer_iteration: int
    compute_seconds: float
    native_cost: float
    point2_separator_bytes: int
    linear_system_bytes: int
    total_comm_bytes: int
    cumulative_total_comm_bytes: int


def build_dgs_point2_command(
    *,
    binary: Path,
    canonical_dir: Path,
    output_dir: Path,
    seed: int,
    outer_iterations: int,
    inner_sweeps: int,
    checkpoints: Sequence[int],
) -> CommandSpec:
    if seed < 0 or outer_iterations < 1 or inner_sweeps < 1:
        raise ValueError("invalid DGS-Point2 iteration configuration")
    checkpoints = tuple(int(value) for value in checkpoints)
    if (
        not checkpoints
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[0] < 1
        or checkpoints[-1] > outer_iterations
    ):
        raise ValueError("invalid DGS-Point2 checkpoints")
    return CommandSpec(
        name=f"dgs_point2_extension_seed_{seed}",
        method="dgs_point2_extension",
        seed=seed,
        iteration=outer_iterations,
        argv=(
            str(Path(binary)),
            "--canonical-dir",
            str(Path(canonical_dir)),
            "--output-dir",
            str(Path(output_dir)),
            "--seed",
            str(seed),
            "--outer-iterations",
            str(outer_iterations),
            "--inner-sweeps",
            str(inner_sweeps),
            "--checkpoints",
            ",".join(str(value) for value in checkpoints),
        ),
        environment={},
    )


def read_dgs_point2_trace(path: Path) -> tuple[DGSPoint2TraceRow, ...]:
    rows = []
    previous_time = -math.inf
    previous_cumulative = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = DGSPoint2TraceRow(
                outer_iteration=int(raw["outer_iteration"]),
                compute_seconds=float(raw["compute_seconds"]),
                native_cost=float(raw["native_cost"]),
                point2_separator_bytes=int(raw["point2_separator_bytes"]),
                linear_system_bytes=int(raw["linear_system_bytes"]),
                total_comm_bytes=int(raw["total_comm_bytes"]),
                cumulative_total_comm_bytes=int(raw["cumulative_total_comm_bytes"]),
            )
            if not math.isfinite(row.compute_seconds) or row.compute_seconds < previous_time:
                raise ValueError("DGS-Point2 compute time must be finite and monotonic")
            if row.point2_separator_bytes + row.linear_system_bytes != row.total_comm_bytes:
                raise ValueError("DGS-Point2 communication total is inconsistent")
            if (
                row.cumulative_total_comm_bytes
                != previous_cumulative + row.total_comm_bytes
            ):
                raise ValueError("DGS-Point2 cumulative communication is inconsistent")
            rows.append(row)
            previous_time = row.compute_seconds
            previous_cumulative = row.cumulative_total_comm_bytes
    if not rows:
        raise ValueError("DGS-Point2 trace is empty")
    return tuple(rows)


__all__ = [
    "CommunicationAccounting",
    "DGSPoint2TraceRow",
    "IterationCommunication",
    "build_dgs_point2_command",
    "read_dgs_point2_trace",
]
