"""Protocol helpers for Edge-wise DRAN++-Point2 Schur on MR.CLAM."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmarks.slideslam.run import CommandSpec


@dataclass(frozen=True)
class DranPoint2SchurTraceRow:
    round: int
    compute_seconds: float
    native_cost: float
    directed_point2_blocks: int
    communication_bytes: int
    cumulative_communication_bytes: int


def build_dran_point2_schur_command(
    *,
    binary: Path,
    canonical_dir: Path,
    output_dir: Path,
    seed: int,
    rounds: int,
    beta: float,
    checkpoints: Sequence[int],
    consensus_projection: bool = True,
    penalty_information: str = "observation",
    warm_start: Path | None = None,
) -> CommandSpec:
    checkpoints = tuple(int(value) for value in checkpoints)
    if seed < 0 or rounds < 1 or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("invalid Point2-Schur DRAN configuration")
    if (
        not checkpoints
        or checkpoints != tuple(sorted(set(checkpoints)))
        or checkpoints[0] < 1
        or checkpoints[-1] > rounds
    ):
        raise ValueError("invalid Point2-Schur DRAN checkpoints")
    if penalty_information not in {"observation", "marginal"}:
        raise ValueError("invalid Point2-Schur DRAN penalty information")
    arguments = [
        str(Path(binary)),
        "--canonical-dir",
        str(Path(canonical_dir)),
        "--output-dir",
        str(Path(output_dir)),
        "--seed",
        str(seed),
        "--rounds",
        str(rounds),
        "--beta",
        f"{beta:g}",
        "--consensus-projection",
        "true" if consensus_projection else "false",
        "--penalty-information",
        penalty_information,
        "--checkpoints",
        ",".join(str(value) for value in checkpoints),
    ]
    if warm_start is not None:
        arguments.extend(("--warm-start", str(Path(warm_start))))
    return CommandSpec(
        name=f"edgewise_dranpp_point2_schur_seed_{seed}",
        method="edgewise_dranpp_point2_schur",
        seed=seed,
        iteration=rounds,
        argv=tuple(arguments),
        environment={},
    )


def read_dran_point2_schur_trace(
    path: Path,
) -> tuple[DranPoint2SchurTraceRow, ...]:
    rows = []
    previous_time = -math.inf
    previous_cumulative = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = DranPoint2SchurTraceRow(
                round=int(raw["round"]),
                compute_seconds=float(raw["compute_seconds"]),
                native_cost=float(raw["native_cost"]),
                directed_point2_blocks=int(raw["directed_point2_blocks"]),
                communication_bytes=int(raw["communication_bytes"]),
                cumulative_communication_bytes=int(
                    raw["cumulative_communication_bytes"]
                ),
            )
            if row.round != len(rows):
                raise ValueError("Point2-Schur DRAN trace has a missing round")
            if (
                not math.isfinite(row.compute_seconds)
                or row.compute_seconds < previous_time
            ):
                raise ValueError(
                    "Point2-Schur DRAN compute time must be finite and monotonic"
                )
            if not math.isfinite(row.native_cost) or row.native_cost < 0.0:
                raise ValueError("Point2-Schur DRAN cost must be finite")
            if row.communication_bytes != row.directed_point2_blocks * 2 * 8:
                raise ValueError("Point2-Schur DRAN Point2 payload is inconsistent")
            if (
                row.cumulative_communication_bytes
                != previous_cumulative + row.communication_bytes
            ):
                raise ValueError(
                    "Point2-Schur DRAN cumulative communication is inconsistent"
                )
            rows.append(row)
            previous_time = row.compute_seconds
            previous_cumulative = row.cumulative_communication_bytes
    if not rows:
        raise ValueError("Point2-Schur DRAN trace is empty")
    return tuple(rows)


__all__ = [
    "DranPoint2SchurTraceRow",
    "build_dran_point2_schur_command",
    "read_dran_point2_schur_trace",
]
