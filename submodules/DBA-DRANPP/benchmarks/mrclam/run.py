"""Checkpoint-only full-communication runner for the native MR.CLAM benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.slideslam.evaluate import (
    read_dran_state,
    read_distributed_mapper_state,
    read_larpg_state,
    read_sesync_state,
    relative_cost_gap,
)
from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.run import (
    BenchmarkRunConfig,
    CommandSpec,
    ExecutableSet,
    build_command_specs,
    default_executables,
    execute_command,
    load_canonical_package,
)
from benchmarks.slideslam.schema import ObjectVertex, PoseObjectFactor, PoseVertex
from benchmarks.slideslam.se3 import SE3

from .evaluate import (
    GroundTruthData,
    evaluate_point2_state,
    lifted_objective_breakdown,
    load_ground_truth,
    physical_metrics,
    read_imesa_point_state,
    read_dgs_point2_state,
)
from .dgs_point2 import (
    CommunicationAccounting,
    build_dgs_point2_command,
    read_dgs_point2_trace,
)
from .source import audit_dataset, load_manifest


class MRCLAMRunnerError(RuntimeError):
    """Raised when a solver violates the MR.CLAM benchmark protocol."""


TopologyRow = tuple[int, int, int, float]


def read_dran_topology_schedule(
    path: Path, num_robots: int
) -> tuple[TopologyRow, ...]:
    """Read and strictly validate the undirected topology schedule CSV."""
    path = Path(path)
    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["round", "src", "dst", "weight"]:
            raise MRCLAMRunnerError(
                "topology header must be round,src,dst,weight"
            )
        for line_number, record in enumerate(reader, start=2):
            try:
                round_index = int(record["round"])
                src = int(record["src"])
                dst = int(record["dst"])
                weight = float(record["weight"])
            except (TypeError, ValueError) as error:
                raise MRCLAMRunnerError(
                    f"invalid topology value at line {line_number}"
                ) from error
            if round_index < 0:
                raise MRCLAMRunnerError("topology round must be nonnegative")
            if src < 0 or dst < 0 or src >= num_robots or dst >= num_robots:
                raise MRCLAMRunnerError("topology robot id is out of range")
            if src == dst:
                raise MRCLAMRunnerError("topology self edges are not allowed")
            if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0:
                raise MRCLAMRunnerError(
                    "topology weight must be finite and in (0, 1]"
                )
            edge = (round_index, min(src, dst), max(src, dst))
            if edge in seen:
                raise MRCLAMRunnerError("duplicate undirected topology edge")
            seen.add(edge)
            rows.append((edge[0], edge[1], edge[2], weight))
    if not rows:
        raise MRCLAMRunnerError("topology schedule has no edges")
    rounds = sorted({row[0] for row in rows})
    if rounds != list(range(rounds[-1] + 1)):
        raise MRCLAMRunnerError(
            "topology rounds must be zero-based and contiguous"
        )
    return tuple(rows)


def dran_topology_summary_pair_counts(
    rows: Sequence[TopologyRow], checkpoint: int
) -> tuple[int, ...]:
    """Map native summary rows 0..checkpoint to active undirected pairs."""
    if checkpoint < 0:
        raise MRCLAMRunnerError("DRAN checkpoint must be nonnegative")
    if not rows:
        raise MRCLAMRunnerError("DRAN topology schedule has no rows")
    round_count = max(row[0] for row in rows) + 1
    pairs_by_round = tuple(
        sum(row[0] == round_index for row in rows)
        for round_index in range(round_count)
    )
    if any(pair_count == 0 for pair_count in pairs_by_round):
        raise MRCLAMRunnerError("DRAN topology schedule contains an empty round")
    return (pairs_by_round[0],) + tuple(
        pairs_by_round[(iteration - 1) % round_count]
        for iteration in range(1, checkpoint + 1)
    )


@dataclass(frozen=True)
class CommunicationTotals:
    logical_bytes: int
    payload_blocks: int
    messages: int
    solver_wall_time_s: float


@dataclass(frozen=True)
class MRCLAMRunConfig:
    repository_root: Path
    manifest_path: Path
    data_root: Path
    canonical_dir: Path
    output_dir: Path
    seeds: tuple[int, ...] = (0, 1, 2)
    checkpoints: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
    timeout_seconds: float = 1800.0
    dry_run: bool = False
    dran_local_max_iterations: int = 3
    dran_beta: float = 1.0
    dran_reduced_rotation_preconditioner: str = "cholesky"
    dran_initial_local_max_iterations: int = 0
    dran_initial_local_rounds: int = 0
    require_certified_reference: bool = True
    dgs_point2_inner_sweeps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "manifest_path",
            "data_root",
            "canonical_dir",
            "output_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if not self.seeds or tuple(sorted(set(self.seeds))) != self.seeds:
            raise MRCLAMRunnerError("seeds must be sorted and unique")
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise MRCLAMRunnerError("checkpoints must be sorted and unique")
        if self.checkpoints[0] <= 0:
            raise MRCLAMRunnerError("checkpoints must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise MRCLAMRunnerError("timeout must be finite and positive")
        if self.dran_local_max_iterations < 1:
            raise MRCLAMRunnerError("DRAN local iterations must be positive")
        if self.dgs_point2_inner_sweeps < 1:
            raise MRCLAMRunnerError("DGS-Point2 inner sweeps must be positive")
        if not math.isfinite(self.dran_beta) or self.dran_beta <= 0.0:
            raise MRCLAMRunnerError("DRAN beta must be finite and positive")
        if (
            self.dran_initial_local_max_iterations < 0
            or self.dran_initial_local_rounds < 0
        ):
            raise MRCLAMRunnerError(
                "DRAN initial local iterations and rounds must be nonnegative"
            )
        if (self.dran_initial_local_max_iterations > 0) != (
            self.dran_initial_local_rounds > 0
        ):
            raise MRCLAMRunnerError(
                "DRAN initial local iterations and rounds must both be zero or positive"
            )
        if self.dran_reduced_rotation_preconditioner not in {
            "none",
            "jacobi",
            "schur_jacobi",
            "cholesky",
            "portfolio",
        }:
            raise MRCLAMRunnerError("invalid DRAN reduced-rotation preconditioner")


def initial_state_audit(
    graph: CanonicalGraph, seeds: Sequence[int]
) -> tuple[dict[str, float | int], ...]:
    """Report one common iteration-zero objective with native conventions."""

    rows = []
    for seed in seeds:
        state = {
            record.variable_id: record.pose
            for record in graph.initial_states
            if record.seed == seed
        }
        if len(state) != len(graph.vertices):
            raise MRCLAMRunnerError(
                f"canonical initial state is incomplete for seed {seed}"
            )
        evaluation = evaluate_point2_state(graph, state)
        rows.append(
            {
                "seed": int(seed),
                "external_cost": evaluation.cost,
                "external_gradient_norm": evaluation.gradient_norm,
                "odometry_cost": evaluation.odometry_cost,
                "observation_cost": evaluation.observation_cost,
                "larpg_native_objective": 0.5 * evaluation.cost,
                "dran_native_objective": evaluation.cost,
                "distributed_mapper_augmented_objective": evaluation.cost,
            }
        )
    return tuple(rows)


def _rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            result = list(csv.DictReader(stream, delimiter=delimiter))
    except OSError as error:
        raise MRCLAMRunnerError(f"cannot read solver log {path}: {error}") from error
    if not result:
        raise MRCLAMRunnerError(f"solver log is empty: {path}")
    return result


def _integer(row: Mapping[str, str], field: str, context: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise MRCLAMRunnerError(f"{context} has invalid integer field {field}") from error
    if value < 0:
        raise MRCLAMRunnerError(f"{context} has negative field {field}")
    return value


def _number(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise MRCLAMRunnerError(f"{context} has invalid numeric field {field}") from error
    if not math.isfinite(value):
        raise MRCLAMRunnerError(f"{context} has non-finite field {field}")
    return value


def _indexed_prefix(
    rows: Sequence[Mapping[str, str]],
    *,
    field: str,
    checkpoint: int,
    first_index: int,
    context: str,
) -> list[Mapping[str, str]]:
    indexed = {_integer(row, field, context): row for row in rows}
    if len(indexed) != len(rows):
        raise MRCLAMRunnerError(f"{context} contains a duplicate iteration")
    observed = sorted(indexed)
    expected_all = list(range(first_index, first_index + len(rows)))
    if observed != expected_all or len(rows) < checkpoint:
        raise MRCLAMRunnerError(
            f"{context} iterations must be a continuous prefix from {first_index} "
            f"through at least {first_index + checkpoint - 1}"
        )
    return [indexed[index] for index in observed[:checkpoint]]


def accumulate_larpg_communication(
    path: Path, checkpoint: int, *, expected_active_blocks: int
) -> CommunicationTotals:
    prefix = _indexed_prefix(
        _rows(path),
        field="iteration",
        checkpoint=checkpoint,
        first_index=1,
        context="LARPG",
    )
    logical_bytes = 0
    payload_blocks = 0
    messages = 0
    initialization_bytes = 0
    wall_time = 0.0
    for row in prefix:
        schur = _integer(row, "uploaded_schur_blocks", "LARPG")
        gradients = _integer(row, "uploaded_gradient_blocks", "LARPG")
        broadcasts = _integer(row, "broadcast_object_blocks", "LARPG")
        if (schur, gradients, broadcasts) != (
            expected_active_blocks,
            expected_active_blocks,
            expected_active_blocks,
        ):
            raise MRCLAMRunnerError(
                "LARPG full communication omitted an active agent-object block"
            )
        upload = _integer(row, "logical_upload_bytes", "LARPG")
        broadcast = _integer(row, "logical_broadcast_bytes", "LARPG")
        if upload != schur * 42 * 8 or broadcast != broadcasts * 6 * 8:
            raise MRCLAMRunnerError("LARPG logical payload accounting mismatch")
        logical_bytes += upload + broadcast
        payload_blocks += schur + gradients + broadcasts
        messages += schur + broadcasts
        wall_time += _number(row, "wall_seconds", "LARPG")
    return CommunicationTotals(logical_bytes, payload_blocks, messages, wall_time)


def accumulate_dran_communication(
    path: Path,
    checkpoint: int,
    *,
    expected_blocks_per_round: int | Sequence[int],
    expected_edge_hypotheses: int | Sequence[int],
    object_block_bytes: int = 3 * 4 * 8,
    require_shared_only: bool = False,
) -> CommunicationTotals:
    prefix = _indexed_prefix(
        _rows(path),
        field="iter",
        checkpoint=checkpoint + 1,
        first_index=0,
        context="Edge-wise DRAN++(Object)",
    )
    logical_bytes = 0
    payload_blocks = 0
    messages = 0
    if object_block_bytes <= 0:
        raise MRCLAMRunnerError("DRAN object block bytes must be positive")
    pose_block_bytes = object_block_bytes
    def per_row_expectations(
        value: int | Sequence[int], name: str
    ) -> tuple[int, ...]:
        if isinstance(value, int):
            expectations = (value,) * len(prefix)
        else:
            expectations = tuple(value)
            if len(expectations) != len(prefix):
                raise MRCLAMRunnerError(
                    f"DRAN {name} expectation length does not match summary rows"
                )
        if any(item < 0 for item in expectations):
            raise MRCLAMRunnerError(f"DRAN {name} expectations must be nonnegative")
        return expectations

    block_expectations = per_row_expectations(
        expected_blocks_per_round, "block count"
    )
    hypothesis_expectations = per_row_expectations(
        expected_edge_hypotheses, "edge hypothesis"
    )
    for row_index, (row, expected_blocks, expected_hypotheses) in enumerate(
        zip(prefix, block_expectations, hypothesis_expectations)
    ):
        # Shared-state-only DRAN keeps one locally reproducible directed dual
        # incidence at each endpoint of every physical association edge.  The
        # caller supplies the physical (undirected) edge-hypothesis count.
        expected_active_hypotheses = (
            2 * expected_hypotheses
            if require_shared_only
            else expected_hypotheses
        )
        sent = _integer(row, "comm_pose_count", "Edge-wise DRAN++(Object)")
        if sent != expected_blocks:
            raise MRCLAMRunnerError(
                "DRAN full communication count differs from the topology expectation"
            )
        expected_pose_bytes = sent * pose_block_bytes
        logged_pose_bytes = _number(
            row, "iter_comm_mb", "Edge-wise DRAN++(Object)"
        ) * 1024 * 1024
        if abs(logged_pose_bytes - expected_pose_bytes) > 1e-6:
            raise MRCLAMRunnerError("DRAN pose payload accounting mismatch")
        active = _integer(
            row,
            "lower_shared_object_consensus_active_hypotheses",
            "Edge-wise DRAN++(Object)",
        )
        targets = _integer(
            row,
            "lower_shared_object_consensus_target_messages",
            "Edge-wise DRAN++(Object)",
        )
        delivered = _integer(
            row,
            "lower_shared_object_consensus_delivered_messages",
            "Edge-wise DRAN++(Object)",
        )
        certificates = _integer(
            row,
            "lower_shared_object_consensus_certificate_messages",
            "Edge-wise DRAN++(Object)",
        )
        stale_certificates = _integer(
            row,
            "lower_shared_object_consensus_stale_certificates",
            "Edge-wise DRAN++(Object)",
        )
        current_certificates = _integer(
            row,
            "lower_shared_object_consensus_current_incident_certificates",
            "Edge-wise DRAN++(Object)",
        )
        expected_certificates = _integer(
            row,
            "lower_shared_object_consensus_expected_incident_certificates",
            "Edge-wise DRAN++(Object)",
        )
        outstanding = _integer(
            row,
            "lower_shared_object_consensus_outstanding_messages",
            "Edge-wise DRAN++(Object)",
        )
        dropped = _integer(
            row,
            "lower_shared_object_consensus_dropped_attempts",
            "Edge-wise DRAN++(Object)",
        )
        retries = _integer(
            row,
            "lower_shared_object_consensus_retry_attempts",
            "Edge-wise DRAN++(Object)",
        )
        lower_payload = _integer(
            row,
            "lower_shared_object_consensus_payload_bytes",
            "Edge-wise DRAN++(Object)",
        )
        shared_edge_payload = _integer(
            row,
            "association_shared_edge_payload_bytes",
            "Edge-wise DRAN++(Object)",
        )
        if require_shared_only:
            if (
                active != expected_active_hypotheses
                or targets != 0
                or delivered != 0
                or certificates != 0
                or stale_certificates != 0
                or current_certificates != 0
                or expected_certificates != 0
                or outstanding != 0
                or dropped != 0
                or retries != 0
                or lower_payload != 0
                or shared_edge_payload != 0
            ):
                raise MRCLAMRunnerError(
                    "DRAN shared-only refresh contains target, certificate, "
                    "retry, or auxiliary payload"
                )
            row_bytes = expected_pose_bytes
            row_payload_blocks = sent
            row_messages = sent
        else:
            if (
                active != expected_active_hypotheses
                or targets < expected_hypotheses
                or targets % expected_hypotheses != 0
                or delivered != targets
                or certificates < expected_hypotheses
                or stale_certificates != 0
                or current_certificates != expected_certificates
                or expected_certificates != 2 * expected_hypotheses
                or outstanding != 0
                or dropped != 0
            ):
                raise MRCLAMRunnerError(
                    "DRAN edge-wise refresh was incomplete, stale, or dropped"
                )
            minimum_lower_payload = (
                targets * pose_block_bytes + certificates * 6 * 8
            )
            if lower_payload < minimum_lower_payload:
                raise MRCLAMRunnerError(
                    "DRAN edge-wise payload accounting mismatch"
                )
            row_bytes = expected_pose_bytes + lower_payload + shared_edge_payload
            row_payload_blocks = sent + targets + certificates + retries
            row_messages = sent + targets + certificates + retries
        if row_index == 0:
            initialization_bytes = row_bytes
        else:
            logical_bytes += row_bytes
            payload_blocks += row_payload_blocks
            messages += row_messages
        logged_cumulative_bytes = _number(
            row, "cumulative_total_comm_mb", "Edge-wise DRAN++(Object)"
        ) * 1024 * 1024
        if abs(logged_cumulative_bytes - initialization_bytes - logical_bytes) > 1e-5:
            raise MRCLAMRunnerError("DRAN cumulative payload accounting mismatch")
    wall_time = _number(
        prefix[-1], "time", "Edge-wise DRAN++(Object)"
    )
    return CommunicationTotals(logical_bytes, payload_blocks, messages, wall_time)


def accumulate_imesa_communication(
    path: Path,
    checkpoint: int,
    *,
    expected_pairs: int,
    expected_shared_objects: int,
) -> CommunicationTotals:
    prefix = _indexed_prefix(
        _rows(path),
        field="round",
        checkpoint=checkpoint,
        first_index=1,
        context="iMESA(Point2)",
    )
    expected_blocks = 2 * expected_pairs * expected_shared_objects
    expected_messages = 4 * expected_pairs
    logical_bytes = 0
    payload_blocks = 0
    messages = 0
    wall_time = 0.0
    for row in prefix:
        declared = _integer(row, "declared_key_blocks", "iMESA(Point2)")
        states = _integer(row, "state_point2_blocks", "iMESA(Point2)")
        row_messages = _integer(row, "directional_messages", "iMESA(Point2)")
        row_bytes = _integer(row, "logical_bytes", "iMESA(Point2)")
        if (
            declared != expected_blocks
            or states != expected_blocks
            or row_messages != expected_messages
            or row_bytes != declared * 8 + states * 2 * 8
        ):
            raise MRCLAMRunnerError(
                "iMESA full communication omitted a pair, key, or Point2 state"
            )
        logical_bytes += row_bytes
        payload_blocks += declared + states
        messages += row_messages
        wall_time += _number(row, "wall_seconds", "iMESA(Point2)")
    return CommunicationTotals(logical_bytes, payload_blocks, messages, wall_time)


def accumulate_dgs_point2_communication(
    path: Path,
    checkpoint: int,
    *,
    remote_observer_object_pairs: int,
    inner_sweeps: int,
) -> CommunicationTotals:
    rows = read_dgs_point2_trace(path)
    if tuple(row.outer_iteration for row in rows[: checkpoint + 1]) != tuple(
        range(checkpoint + 1)
    ):
        raise MRCLAMRunnerError("DGS-Point2 trace has a missing outer iteration")
    if len(rows) <= checkpoint:
        raise MRCLAMRunnerError("DGS-Point2 trace is shorter than the checkpoint")
    expected = CommunicationAccounting(
        remote_observer_object_pairs
    ).for_outer_iteration(inner_sweeps)
    for row in rows[1 : checkpoint + 1]:
        if (
            row.point2_separator_bytes != expected.point2_separator_bytes
            or row.linear_system_bytes != expected.linear_system_bytes
            or row.total_comm_bytes != expected.total_bytes
        ):
            raise MRCLAMRunnerError(
                "DGS-Point2 omitted a Point2 separator or linear-system packet"
            )
    return CommunicationTotals(
        logical_bytes=checkpoint * expected.total_bytes,
        payload_blocks=checkpoint
        * (expected.point2_separator_blocks + expected.linear_system_packets),
        messages=checkpoint
        * (expected.point2_separator_blocks + expected.linear_system_packets),
        solver_wall_time_s=rows[checkpoint].compute_seconds,
    )


def build_edgewise_dran_command(
    *,
    binary: Path,
    data_dir: Path,
    association_file: Path,
    output_dir: Path,
    num_robots: int,
    num_objects: int,
    seed: int,
    iterations: int,
    local_max_iterations: int = 3,
    reduced_rotation_preconditioner: str = "cholesky",
    initial_local_max_iterations: int = 0,
    initial_local_rounds: int = 0,
    topology: str = "complete",
    topology_file: Path | None = None,
    topology_weight_mode: str = "unit",
    beta: float = 1.0,
    adaptive_beta: bool = False,
    adaptive_beta_min: float = 0.05,
    adaptive_beta_max: float = 0.5,
    adaptive_beta_imbalance_ratio: float = 5.0,
    adaptive_beta_update_factor: float = 1.25,
    adaptive_beta_ema_decay: float = 0.8,
    adaptive_beta_update_period: int = 5,
    exact_proximal_curvature: bool = False,
    proximal_curvature_scale: float = 1.0,
    dual_step_size: float = 1.0,
    dual_relaxation_residual_threshold: float = 0.0,
    spectral_dual_step: bool = False,
    global_spectral_dual_step: bool = False,
    spectral_dual_step_min: float = 0.75,
    spectral_dual_step_max: float = 1.5,
    spectral_dual_step_ema_decay: float = 0.5,
    point2_consensus_projection: bool = False,
    point2_consensus_projection_relaxation: float = 1.0,
    point2_consensus_projection_rounds: int = 0,
    point2_consensus_reset_dual_after_warmup: bool = False,
    exclude_summary_diagnostics_from_solver_time: bool = False,
    association_aware_ted_cci_initialization: bool = False,
    exact_point2_owner: bool = False,
    exact_point2_owner_optimize_trajectories: bool = True,
    exact_point2_schur_oracle: bool = False,
    exact_point2_owner_pcg: bool = False,
    exact_point2_owner_pcg_iterations: int = 1,
    exact_point2_owner_pcg_warm_start: bool = True,
    exact_point2_owner_pcg_warm_start_scale: float = 1.0,
    exact_point2_owner_pcg_late_warm_start_round: int = 0,
    exact_point2_owner_pcg_late_warm_start_scale: float = 1.0,
    exact_point2_owner_pcg_matrix_free: bool = False,
    exact_point2_owner_pcg_quadratic_warm_start: bool = False,
    exact_point2_owner_pcg_preconditioner_refresh_residual: float = 0.0,
    exact_point2_owner_pcg_preconditioner_refresh_min_rounds: int = 5,
    exact_point2_gauge_sync_once: bool = False,
    exact_point2_gauge_sync_rounds: int = 0,
    save_iteration_estimate_snapshots: bool = False,
) -> CommandSpec:
    if sum(
        (exact_point2_owner, exact_point2_schur_oracle, exact_point2_owner_pcg)
    ) > 1:
        raise MRCLAMRunnerError(
            "DRAN exact Point2 modes are mutually exclusive"
        )
    exact_point2_mode = (
        exact_point2_owner
        or exact_point2_schur_oracle
        or exact_point2_owner_pcg
    )
    if exact_point2_gauge_sync_once and not exact_point2_mode:
        raise MRCLAMRunnerError(
            "one-shot gauge synchronization requires an exact Point2 mode"
        )
    if exact_point2_gauge_sync_rounds < 0 or (
        exact_point2_gauge_sync_rounds > 0 and not exact_point2_mode
    ):
        raise MRCLAMRunnerError(
            "finite gauge synchronization rounds require an exact Point2 mode"
        )
    if exact_point2_gauge_sync_once and exact_point2_gauge_sync_rounds > 0:
        raise MRCLAMRunnerError(
            "one-shot and finite-round gauge synchronization are mutually exclusive"
        )
    if exact_point2_owner_pcg_matrix_free and not exact_point2_owner_pcg:
        raise MRCLAMRunnerError(
            "DRAN matrix-free Point2 products require exact owner-PCG mode"
        )
    if exact_point2_owner_pcg_quadratic_warm_start and not exact_point2_owner_pcg:
        raise MRCLAMRunnerError(
            "DRAN quadratic warm start requires exact owner-PCG mode"
        )
    if (
        not math.isfinite(
            exact_point2_owner_pcg_preconditioner_refresh_residual
        )
        or exact_point2_owner_pcg_preconditioner_refresh_residual < 0.0
        or exact_point2_owner_pcg_preconditioner_refresh_min_rounds < 1
    ):
        raise MRCLAMRunnerError(
            "DRAN owner-PCG preconditioner refresh parameters are invalid"
        )
    if (
        exact_point2_owner_pcg_preconditioner_refresh_residual > 0.0
        and not exact_point2_owner_pcg_matrix_free
    ):
        raise MRCLAMRunnerError(
            "DRAN owner-PCG preconditioner refresh requires matrix-free mode"
        )
    if (
        num_robots < 2
        or num_objects < 1
        or iterations < 1
        or local_max_iterations < 1
    ):
        raise MRCLAMRunnerError("invalid Edge-wise DRAN++(Object) command size")
    if not math.isfinite(beta) or beta <= 0.0:
        raise MRCLAMRunnerError("DRAN beta must be finite and positive")
    if (
        not math.isfinite(proximal_curvature_scale)
        or proximal_curvature_scale < 0.5
        or proximal_curvature_scale > 1.0
    ):
        raise MRCLAMRunnerError(
            "DRAN proximal curvature scale must be in [0.5, 1]"
        )
    if not math.isfinite(dual_step_size) or not 0.0 < dual_step_size <= 1.6:
        raise MRCLAMRunnerError("DRAN dual step size must be in (0, 1.6]")
    if (
        not math.isfinite(dual_relaxation_residual_threshold)
        or dual_relaxation_residual_threshold < 0.0
    ):
        raise MRCLAMRunnerError(
            "DRAN dual relaxation residual threshold must be nonnegative"
        )
    if (
        not math.isfinite(spectral_dual_step_min)
        or spectral_dual_step_min <= 0.0
        or not math.isfinite(spectral_dual_step_max)
        or spectral_dual_step_max < spectral_dual_step_min
        or not math.isfinite(spectral_dual_step_ema_decay)
        or spectral_dual_step_ema_decay < 0.0
        or spectral_dual_step_ema_decay >= 1.0
    ):
        raise MRCLAMRunnerError("DRAN spectral dual-step parameters are invalid")
    if (spectral_dual_step or global_spectral_dual_step) and adaptive_beta:
        raise MRCLAMRunnerError("DRAN spectral dual steps require fixed beta")
    if spectral_dual_step and global_spectral_dual_step:
        raise MRCLAMRunnerError(
            "DRAN edge-wise and global spectral dual steps are mutually exclusive"
        )
    if global_spectral_dual_step and topology != "complete":
        raise MRCLAMRunnerError(
            "DRAN global spectral dual steps require complete topology"
        )
    if (
        not math.isfinite(point2_consensus_projection_relaxation)
        or point2_consensus_projection_relaxation <= 0.0
        or point2_consensus_projection_relaxation > 1.0
    ):
        raise MRCLAMRunnerError(
            "DRAN Point2 consensus projection relaxation must be in (0, 1]"
        )
    if point2_consensus_projection_rounds < 0:
        raise MRCLAMRunnerError(
            "DRAN Point2 consensus projection rounds must be nonnegative"
        )
    if point2_consensus_reset_dual_after_warmup and (
        not point2_consensus_projection or point2_consensus_projection_rounds == 0
    ):
        raise MRCLAMRunnerError(
            "DRAN Point2 dual reset requires a finite projection warmup"
        )
    if (
        not math.isfinite(adaptive_beta_min)
        or adaptive_beta_min <= 0.0
        or not math.isfinite(adaptive_beta_max)
        or adaptive_beta_max < adaptive_beta_min
        or not math.isfinite(adaptive_beta_imbalance_ratio)
        or adaptive_beta_imbalance_ratio <= 1.0
        or not math.isfinite(adaptive_beta_update_factor)
        or adaptive_beta_update_factor <= 1.0
        or not math.isfinite(adaptive_beta_ema_decay)
        or adaptive_beta_ema_decay < 0.0
        or adaptive_beta_ema_decay >= 1.0
        or adaptive_beta_update_period < 1
    ):
        raise MRCLAMRunnerError("DRAN adaptive beta parameters are invalid")
    if initial_local_max_iterations < 0 or initial_local_rounds < 0:
        raise MRCLAMRunnerError(
            "initial local iterations and rounds must be nonnegative"
        )
    if (initial_local_max_iterations > 0) != (initial_local_rounds > 0):
        raise MRCLAMRunnerError(
            "initial local iterations and rounds must both be zero or positive"
        )
    if exact_point2_owner_pcg_iterations < 1:
        raise MRCLAMRunnerError("DRAN exact Point2 owner-PCG iterations must be positive")
    if (
        not math.isfinite(exact_point2_owner_pcg_warm_start_scale)
        or exact_point2_owner_pcg_warm_start_scale < 0.0
        or exact_point2_owner_pcg_late_warm_start_round < 0
        or not math.isfinite(exact_point2_owner_pcg_late_warm_start_scale)
        or exact_point2_owner_pcg_late_warm_start_scale < 0.0
    ):
        raise MRCLAMRunnerError(
            "DRAN exact Point2 owner-PCG warm-start schedule is invalid"
        )
    if topology not in {"complete", "ring"}:
        raise MRCLAMRunnerError("DRAN topology must be complete or ring")
    if topology_weight_mode not in {"unit", "matrix"}:
        raise MRCLAMRunnerError("DRAN topology weight mode must be unit or matrix")
    if topology_file is None and topology_weight_mode != "unit":
        raise MRCLAMRunnerError(
            "DRAN topology weight mode requires a topology schedule file"
        )
    if exact_point2_mode and (topology != "complete" or topology_file is not None):
        raise MRCLAMRunnerError(
            "DRAN exact Point2 modes require complete topology"
        )
    if exact_point2_mode and (
        adaptive_beta
        or point2_consensus_projection
        or point2_consensus_reset_dual_after_warmup
    ):
        raise MRCLAMRunnerError(
            "DRAN exact Point2 modes do not accept direct-edge options"
        )
    topology_arguments = ()
    if topology_file is not None:
        topology_arguments = (
            "--topology_file",
            str(Path(topology_file)),
            "--topology_weight_mode",
            topology_weight_mode,
        )
    direct_edge_arguments = ()
    if not exact_point2_mode:
        direct_edge_arguments = (
            "--beta",
            f"{beta:g}",
            "--association_lower_direct_edge_synchronous_shared_only",
            "true",
            "--association_lower_direct_edge_exact_proximal_curvature",
            "true" if exact_proximal_curvature else "false",
            "--association_lower_direct_edge_proximal_curvature_scale",
            f"{proximal_curvature_scale:g}",
            "--association_lower_direct_edge_dual_step_size",
            f"{dual_step_size:g}",
            "--association_lower_direct_edge_dual_relaxation_residual_threshold",
            f"{dual_relaxation_residual_threshold:g}",
            "--association_lower_direct_edge_spectral_dual_step",
            "true" if spectral_dual_step else "false",
            "--association_lower_direct_edge_global_spectral_dual_step",
            "true" if global_spectral_dual_step else "false",
            "--association_lower_direct_edge_spectral_dual_step_min",
            f"{spectral_dual_step_min:g}",
            "--association_lower_direct_edge_spectral_dual_step_max",
            f"{spectral_dual_step_max:g}",
            "--association_lower_direct_edge_spectral_dual_step_ema_decay",
            f"{spectral_dual_step_ema_decay:g}",
            "--association_lower_direct_edge_point2_consensus_projection",
            "true" if point2_consensus_projection else "false",
            "--association_lower_direct_edge_point2_consensus_projection_relaxation",
            f"{point2_consensus_projection_relaxation:g}",
            "--association_lower_direct_edge_point2_consensus_projection_rounds",
            str(point2_consensus_projection_rounds),
            "--association_lower_direct_edge_point2_consensus_reset_dual_after_warmup",
            "true" if point2_consensus_reset_dual_after_warmup else "false",
            "--association_lower_direct_edge_adaptive_beta",
            "true" if adaptive_beta else "false",
            "--association_lower_direct_edge_beta_min",
            f"{adaptive_beta_min:g}",
            "--association_lower_direct_edge_beta_max",
            f"{adaptive_beta_max:g}",
            "--association_lower_direct_edge_beta_imbalance_ratio",
            f"{adaptive_beta_imbalance_ratio:g}",
            "--association_lower_direct_edge_beta_update_factor",
            f"{adaptive_beta_update_factor:g}",
            "--association_lower_direct_edge_beta_ema_decay",
            f"{adaptive_beta_ema_decay:g}",
            "--association_lower_direct_edge_beta_update_period",
            str(adaptive_beta_update_period),
        )
    owner_pcg_arguments = ()
    if exact_point2_owner_pcg:
        owner_pcg_arguments = (
            "--exact_point2_owner_pcg_iterations",
            str(exact_point2_owner_pcg_iterations),
            "--exact_point2_owner_pcg_warm_start",
            "true" if exact_point2_owner_pcg_warm_start else "false",
            "--exact_point2_owner_pcg_warm_start_scale",
            f"{exact_point2_owner_pcg_warm_start_scale:g}",
            "--exact_point2_owner_pcg_late_warm_start_round",
            str(exact_point2_owner_pcg_late_warm_start_round),
            "--exact_point2_owner_pcg_late_warm_start_scale",
            f"{exact_point2_owner_pcg_late_warm_start_scale:g}",
            "--exact_point2_owner_pcg_matrix_free",
            "true" if exact_point2_owner_pcg_matrix_free else "false",
            "--exact_point2_owner_pcg_quadratic_warm_start",
            "true" if exact_point2_owner_pcg_quadratic_warm_start else "false",
            "--exact_point2_owner_pcg_preconditioner_refresh_residual",
            f"{exact_point2_owner_pcg_preconditioner_refresh_residual:g}",
            "--exact_point2_owner_pcg_preconditioner_refresh_min_rounds",
            str(exact_point2_owner_pcg_preconditioner_refresh_min_rounds),
        )
    exact_owner_arguments = ()
    if exact_point2_owner:
        exact_owner_arguments = (
            "--exact_point2_owner_optimize_trajectories",
            "true" if exact_point2_owner_optimize_trajectories else "false",
        )
    exact_point2_arguments = ()
    if exact_point2_mode:
        exact_point2_arguments = (
            "--exact_point2_gauge_sync_rounds",
            str(exact_point2_gauge_sync_rounds),
        )
    if exact_point2_owner_pcg:
        method = "dran_exact_point2_owner_pcg"
    elif exact_point2_schur_oracle:
        method = "dran_exact_point2_schur_oracle"
    elif exact_point2_owner:
        method = "dran_exact_point2"
    else:
        method = "edgewise_dranpp_object"
    if exact_point2_owner_pcg:
        shared_latent_mode = "exact_point2_owner_pcg"
    elif exact_point2_schur_oracle:
        shared_latent_mode = "exact_point2_schur_oracle"
    elif exact_point2_owner:
        shared_latent_mode = "exact_point2_owner"
    else:
        shared_latent_mode = "direct_edge_dual"
    return CommandSpec(
        name=f"{method}_seed_{seed}_round_{iterations}",
        method=method,
        seed=seed,
        iteration=iterations,
        argv=(
            str(Path(binary)),
            "--data_dir",
            str(Path(data_dir)),
            "--num_robots",
            str(num_robots),
            "--num_objects",
            str(num_objects),
            "--iters",
            str(iterations),
            "--quiet",
            "true",
            "--topology",
            topology,
            *topology_arguments,
            "--communication_policy",
            "all",
            "--object_initialization",
            "input_state",
            "--association_aware_ted_cci_initialization",
            "true" if association_aware_ted_cci_initialization else "false",
            "--object_initialization_anchor_mode",
            "constant",
            "--object_orientation_observed",
            "false",
            "--object_communication_representation",
            "point2",
            "--object_interface_mode",
            "copy_baseline",
            "--local_solver",
            "reduced_rotation",
            "--local_max_iterations",
            str(local_max_iterations),
            "--local_initial_max_iterations",
            str(initial_local_max_iterations),
            "--local_initial_rounds",
            str(initial_local_rounds),
            "--reduced_rotation_preconditioner",
            reduced_rotation_preconditioner,
            "--association_policy",
            "manual",
            "--association_mode",
            "fixed_file_soft",
            "--association_file",
            str(Path(association_file)),
            "--association_freeze_upper_variables",
            "true",
            "--association_lower_shared_latent_mode",
            shared_latent_mode,
            *exact_point2_arguments,
            *exact_owner_arguments,
            *owner_pcg_arguments,
            *direct_edge_arguments,
            "--exclude_summary_diagnostics_from_solver_time",
            "true" if exclude_summary_diagnostics_from_solver_time else "false",
            "--association_require_observed_objects",
            "true",
            "--association_residual_use_transform_alignment",
            "true",
            "--association_transform_dimension",
            "3",
            "--association_gauge_synchronization_enabled",
            "true",
            "--association_gauge_synchronization_apply_to_states",
            "false" if exact_point2_gauge_sync_once else "true",
            "--association_gauge_synchronization_apply_once_to_initial_states",
            "true" if exact_point2_gauge_sync_once else "false",
            "--object_pose_tol",
            "0",
            "--object_max_age",
            "0",
            "--diagnostic_profile",
            "lightweight",
            "--association_save_iteration_estimate_snapshots",
            "true" if save_iteration_estimate_snapshots else "false",
            "--output_dir",
            str(Path(output_dir)),
            "--save",
            "true",
        ),
        environment={},
    )


def build_imesa_command(
    *,
    binary: Path,
    canonical_dir: Path,
    output_dir: Path,
    seed: int,
    rounds: int,
    checkpoints: Sequence[int],
    topology: str = "complete",
) -> CommandSpec:
    if topology not in {"complete", "ring"}:
        raise MRCLAMRunnerError("iMESA topology must be complete or ring")
    return CommandSpec(
        name=f"imesa_point2_seed_{seed}_round_{rounds}",
        method="imesa_point2",
        seed=seed,
        iteration=rounds,
        argv=(
            str(binary),
            "--canonical-dir",
            str(canonical_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--rounds",
            str(rounds),
            "--checkpoints",
            ",".join(str(item) for item in checkpoints),
            "--topology",
            topology,
            "--initial-penalty",
            "0.001",
            "--penalty",
            "1",
        ),
        environment={},
    )


def accumulate_distributed_mapper_communication(
    communication_path: Path, loss_path: Path, *, checkpoint: int
) -> CommunicationTotals:
    loss_prefix = _indexed_prefix(
        _rows(loss_path, "\t"),
        field="iter",
        checkpoint=checkpoint,
        first_index=0,
        context="Distributed Mapper loss",
    )
    grouped: dict[int, list[int]] = {}
    for row in _rows(communication_path, "\t"):
        iteration = _integer(row, "iter", "Distributed Mapper communication")
        totals = grouped.setdefault(iteration, [0, 0])
        totals[0] += _integer(row, "messages", "Distributed Mapper communication")
        totals[1] += _integer(row, "bytes", "Distributed Mapper communication")
    logical_bytes = 0
    messages = 0
    previous_time = -math.inf
    for iteration, row in enumerate(loss_prefix):
        round_messages, round_bytes = grouped.get(iteration, (0, 0))
        logged = _integer(row, "iter_comm_poses", "Distributed Mapper loss")
        if round_messages <= 0 or logged != round_messages:
            raise MRCLAMRunnerError(
                "Distributed Mapper did not perform a full non-stopped exchange round"
            )
        messages += round_messages
        logical_bytes += round_bytes
        elapsed = _number(
            row, "cumulative_solver_time_s", "Distributed Mapper loss"
        )
        if elapsed < previous_time:
            raise MRCLAMRunnerError(
                "Distributed Mapper cumulative solver time is not monotonic"
            )
        previous_time = elapsed
    return CommunicationTotals(logical_bytes, messages, messages, previous_time)


def _legacy_config(config: MRCLAMRunConfig) -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        repository_root=config.repository_root,
        manifest_path=config.manifest_path,
        data_root=config.data_root,
        canonical_dir=config.canonical_dir,
        output_dir=config.output_dir,
        seeds=config.seeds,
        checkpoints=config.checkpoints,
        max_iterations=config.checkpoints[-1],
        timeout_seconds=config.timeout_seconds,
        dry_run=config.dry_run,
    )


def default_mrclam_executables(repository_root: Path) -> ExecutableSet:
    root = Path(repository_root).resolve()
    return replace(
        default_executables(root),
        dran=root / "build/bin/manual-object-dran-example",
    )


def _validate_fixed_associations(path: Path, graph: Any) -> tuple[int, int]:
    objects = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    expected = {
        (receiver, sender, object_index, object_index)
        for object_index, vertex in enumerate(objects)
        for receiver in vertex.observing_robots
        for sender in vertex.observing_robots
        if receiver != sender
    }
    observed = set()
    for row in _rows(path):
        try:
            entry = (
                int(row["receiver_robot"]),
                int(row["sender_robot"]),
                int(row["receiver_object"]),
                int(row["sender_object"]),
            )
            weight = float(row["weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise MRCLAMRunnerError("fixed association file is malformed") from error
        if entry in observed or weight != 1.0:
            raise MRCLAMRunnerError(
                "fixed association file must contain unique unit barcode matches"
            )
        observed.add(entry)
    if observed != expected:
        raise MRCLAMRunnerError(
            "fixed association file does not match the canonical observer incidence"
        )
    if len(observed) % 2 != 0:
        raise MRCLAMRunnerError("fixed association file is not reciprocal")
    return len(observed), len(observed) // 2


def _prepare_sesync_input(source: Path, target: Path) -> None:
    records = [
        line
        for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(records) + "\n", encoding="utf-8")


def _write_state(path: Path, checkpoint: int, state: Mapping[int, SE3]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for variable_id, pose in sorted(state.items()):
            stream.write(
                json.dumps(
                    {
                        "iteration": checkpoint,
                        "id": variable_id,
                        "pose": pose.to_dict(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _checkpoint_result(
    *,
    method: str,
    seed: int,
    checkpoint: int,
    graph: Any,
    state: Mapping[int, SE3],
    ground_truth: GroundTruthData,
    reference_cost: float,
    native_objective: float,
    communication: CommunicationTotals,
) -> dict[str, Any]:
    common = evaluate_point2_state(graph, state)
    breakdown = lifted_objective_breakdown(graph, state)
    physical = physical_metrics(graph, state, ground_truth)
    if abs(common.cost - breakdown.total_cost) > 1e-7 * max(1.0, common.cost):
        raise MRCLAMRunnerError("lifted objective breakdown does not match common cost")
    return {
        "method": method,
        "seed": seed,
        "iteration": checkpoint,
        "external_cost": common.cost,
        "factor_count": len(graph.factors),
        "mean_factor_cost": common.cost / len(graph.factors),
        "external_gradient_norm": common.gradient_norm,
        "relative_cost_gap_to_certified_reference": relative_cost_gap(
            common.cost, reference_cost
        ),
        "native_objective": native_objective,
        "odometry_cost": breakdown.odometry_cost,
        "observation_translation_cost": breakdown.observation_translation_cost,
        "observation_rotation_regularizer_cost": (
            breakdown.observation_rotation_regularizer_cost
        ),
        "robot_translation_rmse_m": physical.robot_translation_rmse_m,
        "robot_yaw_rmse_rad": physical.robot_yaw_rmse_rad,
        "object_position_rmse_m": physical.object_position_rmse_m,
        "object_position_permutation_invariant_rmse_m": (
            physical.object_position_permutation_invariant_rmse_m
        ),
        "cumulative_logical_bytes": communication.logical_bytes,
        "cumulative_payload_blocks": communication.payload_blocks,
        "cumulative_messages": communication.messages,
        "cumulative_solver_wall_time_s": communication.solver_wall_time_s,
    }


def _final_row(path: Path, field: str, value: int, delimiter: str = ",") -> Mapping[str, str]:
    matches = [row for row in _rows(path, delimiter) if _integer(row, field, str(path)) == value]
    if len(matches) != 1:
        raise MRCLAMRunnerError(f"{path} does not contain exactly one requested final row")
    return matches[0]


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in records:
        method = row.get("method")
        iteration = row.get("iteration")
        seed = row.get("seed")
        if (
            not isinstance(method, str)
            or isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise MRCLAMRunnerError("aggregate record has invalid identity fields")
        grouped.setdefault((method, iteration), []).append(row)

    result = []
    identity_fields = {"method", "seed", "iteration"}
    for (method, iteration), group in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(set(seeds)) != len(seeds):
            raise MRCLAMRunnerError("aggregate group contains a duplicate seed")
        metric_fields = sorted(set(group[0]) - identity_fields)
        if any(set(row) - identity_fields != set(metric_fields) for row in group):
            raise MRCLAMRunnerError("aggregate group has inconsistent metric fields")
        aggregate: dict[str, Any] = {
            "method": method,
            "iteration": iteration,
            "seed_count": len(group),
        }
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            if any(not math.isfinite(value) for value in values):
                raise MRCLAMRunnerError(f"aggregate metric {field} is non-finite")
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            aggregate[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(aggregate)
    return tuple(result)


def _write_summaries(output: Path, records: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(records, key=lambda row: (row["method"], row["seed"], row["iteration"]))
    aggregates = aggregate_records(ordered)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    (output / "summary.json").write_text(
        json.dumps(ordered, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (output / "aggregate.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MR.CLAM Native Five-Robot Object Backend Benchmark",
        "",
        "| Method | Repeat | Round | Mean factor cost | Robot RMSE m | Object RMSE m | PI Object RMSE m | Bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {method} | {seed} | {iteration} | {mean_factor_cost:.6g} | "
            "{robot_translation_rmse_m:.6g} | {object_position_rmse_m:.6g} | "
            "{object_position_permutation_invariant_rmse_m:.6g} | "
            "{cumulative_logical_bytes} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Across-Repeat Aggregate",
            "",
            "All repeats use the same unperturbed initial state. Sample standard deviation is reported; a single-repeat smoke run has zero standard deviation.",
            "",
            "| Method | Round | Repeats | Mean factor cost | Cost std | Robot RMSE mean m | PI Object RMSE mean m | Bytes mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            "| {method} | {iteration} | {seed_count} | {mean_factor_cost_mean:.6g} | "
            "{mean_factor_cost_std:.6g} | "
            "{robot_translation_rmse_m_mean:.6g} | "
            "{object_position_permutation_invariant_rmse_m_mean:.6g} | "
            "{cumulative_logical_bytes_mean:.6g} |".format(**row)
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    config: MRCLAMRunConfig,
    *,
    executables: ExecutableSet | None = None,
    imesa_executable: Path | None = None,
    dgs_point2_executable: Path | None = None,
) -> Mapping[str, Any]:
    if config.output_dir.exists():
        raise MRCLAMRunnerError(f"output already exists: {config.output_dir}")
    manifest = load_manifest(config.manifest_path)
    if any(seed not in manifest.evaluation.seeds for seed in config.seeds):
        raise MRCLAMRunnerError("requested seed is not pinned by the manifest")
    if any(item not in manifest.evaluation.fixed_checkpoints for item in config.checkpoints):
        raise MRCLAMRunnerError("requested checkpoint is not pinned by the manifest")
    audit = audit_dataset(manifest, config.data_root)
    package = load_canonical_package(config.canonical_dir)
    if package.manifest.get("benchmark_id") != manifest.benchmark_id:
        raise MRCLAMRunnerError("canonical package and source manifest differ")
    if package.graph.metadata.get("ground_truth_used_for_initialization") is not False:
        raise MRCLAMRunnerError("canonical graph does not prove ground-truth-free initialization")
    if executables is None:
        executables = default_mrclam_executables(config.repository_root)
    if imesa_executable is None:
        imesa_executable = (
            config.repository_root
            / "baselines/iMESA_mrclam/build-local/imesa-mrclam"
        )
    if dgs_point2_executable is None:
        dgs_point2_executable = (
            config.repository_root
            / "baselines/DGS_Point2/build-local/dgs-point2-mrclam"
        )
    for name in ("larpg", "dran", "distributed_mapper", "sesync"):
        path = Path(getattr(executables, name))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise MRCLAMRunnerError(f"required executable is unavailable: {path}")
    if not imesa_executable.is_file() or not os.access(imesa_executable, os.X_OK):
        raise MRCLAMRunnerError(
            f"required iMESA executable is unavailable: {imesa_executable}"
        )
    if not dgs_point2_executable.is_file() or not os.access(
        dgs_point2_executable, os.X_OK
    ):
        raise MRCLAMRunnerError(
            "required DGS-Point2 extension executable is unavailable: "
            f"{dgs_point2_executable}"
        )

    robot_by_vertex_id = {
        vertex.global_id: vertex.robot_id
        for vertex in package.graph.vertices
        if isinstance(vertex, PoseVertex)
    }
    active_agent_object_blocks = len(
        {
            (robot_by_vertex_id[factor.from_id], factor.to_id)
            for factor in package.graph.factors
            if isinstance(factor, PoseObjectFactor)
        }
    )
    robot_ids = sorted(set(robot_by_vertex_id.values()))
    num_robots = len(robot_ids)
    num_objects = sum(
        isinstance(vertex, ObjectVertex) for vertex in package.graph.vertices
    )
    remote_observer_object_pairs = sum(
        len(vertex.observing_robots) - 1
        for vertex in package.graph.vertices
        if isinstance(vertex, ObjectVertex)
    )
    if robot_ids != list(range(num_robots)) or num_objects < 1:
        raise MRCLAMRunnerError("canonical robot/object IDs are not adapter-compatible")
    fixed_association_file = (
        config.canonical_dir / "adapters/dran/fixed_associations.csv"
    )
    expected_full_dran_blocks, expected_edge_hypotheses = (
        _validate_fixed_associations(fixed_association_file, package.graph)
    )
    legacy = _legacy_config(config)
    command_preview = []
    for seed in config.seeds:
        for checkpoint in config.checkpoints:
            command_set = build_command_specs(
                legacy,
                executables,
                seed=seed,
                iterations=checkpoint,
                work_root=config.output_dir,
                package=package,
            )
            dran_spec = build_edgewise_dran_command(
                binary=executables.dran,
                data_dir=(
                    config.canonical_dir / f"adapters/dran/seed_{seed}"
                ),
                association_file=fixed_association_file,
                output_dir=(
                    config.output_dir
                    / f"native/edgewise_dranpp_object_seed_{seed}_round_{checkpoint}"
                ),
                num_robots=num_robots,
                num_objects=num_objects,
                seed=seed,
                iterations=checkpoint,
                local_max_iterations=config.dran_local_max_iterations,
                reduced_rotation_preconditioner=(
                    config.dran_reduced_rotation_preconditioner
                ),
                initial_local_max_iterations=(
                    config.dran_initial_local_max_iterations
                ),
                initial_local_rounds=config.dran_initial_local_rounds,
                beta=config.dran_beta,
            )
            command_preview.extend(
                [list(dran_spec.argv), list(command_set["distributed_mapper"].argv)]
            )
        command_set = build_command_specs(
            legacy,
            executables,
            seed=seed,
            iterations=config.checkpoints[-1],
            work_root=config.output_dir,
            package=package,
        )
        command_preview.append(list(command_set["larpg"].argv))
        command_preview.append(
            list(
                build_imesa_command(
                    binary=imesa_executable,
                    canonical_dir=config.canonical_dir,
                    output_dir=config.output_dir / f"native/imesa_point2_seed_{seed}",
                    seed=seed,
                    rounds=config.checkpoints[-1],
                    checkpoints=config.checkpoints,
                ).argv
            )
        )
        command_preview.append(
            list(
                build_dgs_point2_command(
                    binary=dgs_point2_executable,
                    canonical_dir=config.canonical_dir,
                    output_dir=(
                        config.output_dir
                        / f"native/dgs_point2_extension_seed_{seed}"
                    ),
                    seed=seed,
                    outer_iterations=config.checkpoints[-1],
                    inner_sweeps=config.dgs_point2_inner_sweeps,
                    checkpoints=config.checkpoints,
                ).argv
            )
        )
    run_manifest: dict[str, Any] = {
        "schema_version": "mrclam_native_backend_run_v2",
        "benchmark_id": manifest.benchmark_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": config.dry_run,
        "native_multi_robot_eligible": audit.native_multi_robot_eligible,
        "physical_robot_count": audit.physical_robot_count,
        "common_sensor_duration_s": audit.common_sensor_duration_s,
        "archive_sha256": audit.archive_sha256,
        "canonical_factor_hash": package.manifest["canonical_factor_hash"],
        "canonical_graph_sha256": package.manifest["canonical_graph_sha256"],
        "ground_truth_used_for_initialization": False,
        "seeds": list(config.seeds),
        "checkpoints": list(config.checkpoints),
        "communication_policy": "full",
        "dgs_point2_extension": {
            "name": "DGS-Point2 extension",
            "outer_relinearization": True,
            "inner_solver": "distributed_block_gauss_seidel",
            "inner_sweeps": config.dgs_point2_inner_sweeps,
            "remote_observer_object_pairs": remote_observer_object_pairs,
            "counts_point2_separator_and_linear_system_communication": True,
        },
        "larpg_active_agent_object_blocks_per_round": active_agent_object_blocks,
        "dran_backend": "edgewise_dranpp_object_direct_edge_dual_point2_shared_only",
        "dran_association_policy": "manual_fixed_file_soft",
        "dran_public_object_state": (
            "arithmetic mean of all observer copies after root-frame transport"
        ),
        "dran_fixed_association_file": str(fixed_association_file),
        "dran_fixed_association_sha256": hashlib.sha256(
            fixed_association_file.read_bytes()
        ).hexdigest(),
        "dran_full_object_blocks_per_round": expected_full_dran_blocks,
        "dran_object_block_bytes": 16,
        "dran_protocol_payload_per_round": 0,
        "dran_edge_hypotheses_per_refresh": expected_edge_hypotheses,
        "dran_local_max_iterations": config.dran_local_max_iterations,
        "dran_reduced_rotation_preconditioner": (
            config.dran_reduced_rotation_preconditioner
        ),
        "dran_initial_local_max_iterations": (
            config.dran_initial_local_max_iterations
        ),
        "dran_initial_local_rounds": config.dran_initial_local_rounds,
        "object_state_semantics": package.manifest.get(
            "object_state_semantics", "position_only"
        ),
        "imesa_backend": "official_imesa_heterogeneous_pose2_point2",
        "imesa_commit": "693c1ad2e20d88f97b480060c4830f0545599f7b",
        "imesa_communication": "synchronous_complete_pair_sweep_no_filter",
        "imesa_penalty": 1.0,
        "distributed_mapper_object_model_compatibility": package.manifest.get(
            "distributed_mapper_object_model_compatibility",
            "diagnostic_only_requires_full_se3_object_observations",
        ),
        "distributed_mapper_result_status": "diagnostic_only",
        "commands": command_preview,
    }
    common_initial_audit = initial_state_audit(package.graph, config.seeds)
    run_manifest["common_initial_state_audit"] = list(common_initial_audit)
    config.output_dir.mkdir(parents=True)
    for directory in ("commands", "logs", "native", "reference", "states", "inputs"):
        (config.output_dir / directory).mkdir()
    with (config.output_dir / "initial_state_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(common_initial_audit[0]))
        writer.writeheader()
        writer.writerows(common_initial_audit)
    if config.dry_run:
        (config.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_manifest

    ground_truth = load_ground_truth(manifest, audit.dataset_directory)
    first_commands = build_command_specs(
        legacy,
        executables,
        seed=config.seeds[0],
        iterations=config.checkpoints[-1],
        work_root=config.output_dir,
        package=package,
    )
    reference_spec = first_commands["sesync"]
    reference_input = Path(reference_spec.argv[reference_spec.argv.index("--input") + 1])
    _prepare_sesync_input(config.canonical_dir / "global_graph.g2o", reference_input)
    reference_result = execute_command(
        reference_spec,
        command_dir=config.output_dir / "commands",
        log_dir=config.output_dir / "logs",
        timeout_seconds=config.timeout_seconds,
    )
    reference_path = Path(reference_spec.argv[reference_spec.argv.index("--output") + 1])
    reference_state = read_sesync_state(reference_path, package.graph)
    reference_common = evaluate_point2_state(package.graph, reference_state)
    reference_physical = physical_metrics(package.graph, reference_state, ground_truth)
    reference_summary = next(
        (
            line
            for line in reference_result.stdout_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("SUMMARY")
        ),
        "",
    )
    reference_certified = "status=GlobalOpt" in reference_summary
    if config.require_certified_reference and not reference_certified:
        raise MRCLAMRunnerError("SE-Sync did not certify the centralized reference")
    run_manifest["certified_reference"] = {
        "certified": reference_certified,
        "summary": reference_summary,
        "external_cost": reference_common.cost,
        "external_gradient_norm": reference_common.gradient_norm,
        **{
            key: value
            for key, value in asdict(reference_physical).items()
            if key != "ground_truth_from_estimate"
        },
    }

    records: list[dict[str, Any]] = []
    for seed in config.seeds:
        mapper_max_trace_paths: tuple[Path, Path] | None = None
        max_checkpoint = config.checkpoints[-1]
        max_commands = build_command_specs(
            legacy,
            executables,
            seed=seed,
            iterations=max_checkpoint,
            work_root=config.output_dir,
            package=package,
        )
        larpg_spec = max_commands["larpg"]
        execute_command(
            larpg_spec,
            command_dir=config.output_dir / "commands",
            log_dir=config.output_dir / "logs",
            timeout_seconds=config.timeout_seconds,
        )
        larpg_output = Path(
            larpg_spec.argv[larpg_spec.argv.index("--output-dir") + 1]
        )
        larpg_csv = larpg_output / "iterations.csv"
        for checkpoint in config.checkpoints:
            state = read_larpg_state(
                larpg_output / f"states/iteration_{checkpoint:03d}.jsonl",
                package.graph,
            )
            communication = accumulate_larpg_communication(
                larpg_csv,
                checkpoint,
                expected_active_blocks=active_agent_object_blocks,
            )
            native_row = _final_row(larpg_csv, "iteration", checkpoint)
            records.append(
                _checkpoint_result(
                    method="larpg",
                    seed=seed,
                    checkpoint=checkpoint,
                    graph=package.graph,
                    state=state,
                    ground_truth=ground_truth,
                    reference_cost=reference_common.cost,
                    native_objective=_number(native_row, "objective_after", "LARPG"),
                    communication=communication,
                )
            )
            _write_state(config.output_dir / f"states/larpg_{seed}.jsonl", checkpoint, state)

        imesa_output = config.output_dir / f"native/imesa_point2_seed_{seed}"
        imesa_spec = build_imesa_command(
            binary=imesa_executable,
            canonical_dir=config.canonical_dir,
            output_dir=imesa_output,
            seed=seed,
            rounds=max_checkpoint,
            checkpoints=config.checkpoints,
        )
        execute_command(
            imesa_spec,
            command_dir=config.output_dir / "commands",
            log_dir=config.output_dir / "logs",
            timeout_seconds=config.timeout_seconds,
        )
        imesa_csv = imesa_output / "iteration_summary.csv"
        for checkpoint in config.checkpoints:
            imesa_state = read_imesa_point_state(
                imesa_output / f"states/round_{checkpoint:03d}.txt",
                package.graph,
            )
            communication = accumulate_imesa_communication(
                imesa_csv,
                checkpoint,
                expected_pairs=num_robots * (num_robots - 1) // 2,
                expected_shared_objects=num_objects,
            )
            native_row = _final_row(imesa_csv, "round", checkpoint)
            records.append(
                _checkpoint_result(
                    method="imesa_point2",
                    seed=seed,
                    checkpoint=checkpoint,
                    graph=package.graph,
                    state=imesa_state,
                    ground_truth=ground_truth,
                    reference_cost=reference_common.cost,
                    native_objective=_number(
                        native_row, "native_position_cost", "iMESA(Point2)"
                    ),
                    communication=communication,
                )
            )
            _write_state(
                config.output_dir / f"states/imesa_point2_{seed}.jsonl",
                checkpoint,
                imesa_state,
            )

        dgs_output = (
            config.output_dir / f"native/dgs_point2_extension_seed_{seed}"
        )
        dgs_spec = build_dgs_point2_command(
            binary=dgs_point2_executable,
            canonical_dir=config.canonical_dir,
            output_dir=dgs_output,
            seed=seed,
            outer_iterations=max_checkpoint,
            inner_sweeps=config.dgs_point2_inner_sweeps,
            checkpoints=config.checkpoints,
        )
        execute_command(
            dgs_spec,
            command_dir=config.output_dir / "commands",
            log_dir=config.output_dir / "logs",
            timeout_seconds=config.timeout_seconds,
        )
        dgs_csv = dgs_output / "iteration_summary.csv"
        dgs_rows = read_dgs_point2_trace(dgs_csv)
        for checkpoint in config.checkpoints:
            dgs_state = read_dgs_point2_state(
                dgs_output / f"states/outer_{checkpoint:03d}.txt",
                package.graph,
            )
            communication = accumulate_dgs_point2_communication(
                dgs_csv,
                checkpoint,
                remote_observer_object_pairs=remote_observer_object_pairs,
                inner_sweeps=config.dgs_point2_inner_sweeps,
            )
            records.append(
                _checkpoint_result(
                    method="dgs_point2_extension",
                    seed=seed,
                    checkpoint=checkpoint,
                    graph=package.graph,
                    state=dgs_state,
                    ground_truth=ground_truth,
                    reference_cost=reference_common.cost,
                    native_objective=dgs_rows[checkpoint].native_cost,
                    communication=communication,
                )
            )
            _write_state(
                config.output_dir / f"states/dgs_point2_extension_{seed}.jsonl",
                checkpoint,
                dgs_state,
            )

        for checkpoint in config.checkpoints:
            command_set = build_command_specs(
                legacy,
                executables,
                seed=seed,
                iterations=checkpoint,
                work_root=config.output_dir,
                package=package,
            )
            dran_output = (
                config.output_dir
                / f"native/edgewise_dranpp_object_seed_{seed}_round_{checkpoint}"
            )
            dran_spec = build_edgewise_dran_command(
                binary=executables.dran,
                data_dir=(
                    config.canonical_dir / f"adapters/dran/seed_{seed}"
                ),
                association_file=fixed_association_file,
                output_dir=dran_output,
                num_robots=num_robots,
                num_objects=num_objects,
                seed=seed,
                iterations=checkpoint,
                local_max_iterations=config.dran_local_max_iterations,
                reduced_rotation_preconditioner=(
                    config.dran_reduced_rotation_preconditioner
                ),
                initial_local_max_iterations=(
                    config.dran_initial_local_max_iterations
                ),
                initial_local_rounds=config.dran_initial_local_rounds,
                beta=config.dran_beta,
            )
            execute_command(
                dran_spec,
                command_dir=config.output_dir / "commands",
                log_dir=config.output_dir / "logs",
                timeout_seconds=config.timeout_seconds,
            )
            dran_state = read_dran_state(
                dran_output / "object_poses.txt",
                config.canonical_dir
                / f"adapters/dran/seed_{seed}/adapter_manifest.json",
                package.graph,
                transform_path=(
                    dran_output / "final_association_transforms.csv"
                ),
                common_frame_robot=0,
                point_objects=True,
            )
            dran_csv = dran_output / "iteration_summary.csv"
            communication = accumulate_dran_communication(
                dran_csv,
                checkpoint,
                expected_blocks_per_round=expected_full_dran_blocks,
                expected_edge_hypotheses=expected_edge_hypotheses,
                object_block_bytes=2 * 8,
                require_shared_only=True,
            )
            native_row = _final_row(dran_csv, "iter", checkpoint)
            records.append(
                _checkpoint_result(
                    method="edgewise_dranpp_object",
                    seed=seed,
                    checkpoint=checkpoint,
                    graph=package.graph,
                    state=dran_state,
                    ground_truth=ground_truth,
                    reference_cost=reference_common.cost,
                    native_objective=_number(
                        native_row,
                        "measurement_cost",
                        "Edge-wise DRAN++(Object)",
                    ),
                    communication=communication,
                )
            )
            _write_state(
                config.output_dir
                / f"states/edgewise_dranpp_object_{seed}.jsonl",
                checkpoint,
                dran_state,
            )

            mapper_input = config.output_dir / f"inputs/dm_seed_{seed}_round_{checkpoint}"
            shutil.copytree(
                config.canonical_dir / f"adapters/distributed_mapper/seed_{seed}",
                mapper_input,
            )
            mapper_spec = command_set["distributed_mapper"]
            mapper_argv = list(mapper_spec.argv)
            data_index = mapper_argv.index("--dataDir") + 1
            mapper_argv[data_index] = f"{mapper_input}{os.sep}"
            mapper_spec = replace(mapper_spec, argv=tuple(mapper_argv))
            execute_command(
                mapper_spec,
                command_dir=config.output_dir / "commands",
                log_dir=config.output_dir / "logs",
                timeout_seconds=config.timeout_seconds,
            )
            mapper_state = read_distributed_mapper_state(
                mapper_input,
                mapper_input / "adapter_manifest.json",
                package.graph,
            )
            comm_path = Path(
                mapper_spec.argv[mapper_spec.argv.index("--commCsv") + 1]
            )
            loss_path = Path(
                mapper_spec.argv[mapper_spec.argv.index("--lossCsv") + 1]
            )
            communication = accumulate_distributed_mapper_communication(
                comm_path, loss_path, checkpoint=checkpoint
            )
            if checkpoint == max_checkpoint:
                mapper_max_trace_paths = (comm_path, loss_path)
            native_row = _final_row(loss_path, "iter", checkpoint - 1, "\t")
            records.append(
                _checkpoint_result(
                    method="distributed_mapper",
                    seed=seed,
                    checkpoint=checkpoint,
                    graph=package.graph,
                    state=mapper_state,
                    ground_truth=ground_truth,
                    reference_cost=reference_common.cost,
                    native_objective=_number(
                        native_row, "cost", "Distributed Mapper"
                    ),
                    communication=communication,
                )
            )
            _write_state(
                config.output_dir / f"states/distributed_mapper_{seed}.jsonl",
                checkpoint,
                mapper_state,
            )
            shutil.rmtree(mapper_input)

        if mapper_max_trace_paths is None:
            raise MRCLAMRunnerError("Distributed Mapper maximum trace is missing")
        max_comm_path, max_loss_path = mapper_max_trace_paths
        for row in records:
            if row["method"] != "distributed_mapper" or row["seed"] != seed:
                continue
            trace_totals = accumulate_distributed_mapper_communication(
                max_comm_path, max_loss_path, checkpoint=int(row["iteration"])
            )
            row["cumulative_solver_wall_time_s"] = trace_totals.solver_wall_time_s

    _write_summaries(config.output_dir, records)
    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["record_count"] = len(records)
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_manifest


__all__ = [
    "CommunicationTotals",
    "MRCLAMRunConfig",
    "MRCLAMRunnerError",
    "accumulate_distributed_mapper_communication",
    "accumulate_dran_communication",
    "accumulate_imesa_communication",
    "accumulate_dgs_point2_communication",
    "accumulate_larpg_communication",
    "aggregate_records",
    "build_edgewise_dran_command",
    "build_imesa_command",
    "build_dgs_point2_command",
    "default_mrclam_executables",
    "initial_state_audit",
    "run_benchmark",
]
