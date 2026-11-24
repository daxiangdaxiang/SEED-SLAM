"""Reproducible orchestration for the SlideSLAM backend benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .acquire import acquire_sources, load_manifest
from .associate import (
    build_global_objects,
    compose_robot_alignments,
    maximum_support_tree,
    run_pair_registration,
)
from .evaluate import (
    EvaluationError,
    IterationRecord,
    detect_convergence,
    evaluate_state,
    parse_distributed_mapper_iterations,
    parse_dran_iterations,
    parse_larpg_iterations,
    read_dran_state,
    read_distributed_mapper_state,
    read_larpg_state,
    read_sesync_state,
    reference_metrics,
    relative_cost_gap,
    require_checkpoints,
)
from .materialize import (
    CanonicalGraph,
    MaterializedPackage,
    build_canonical_graph,
    canonical_factor_hash,
    legacy_canonical_factor_hash,
    distributed_mapper_preflight,
    inspect_distributed_mapper_adapter,
    inspect_dran_adapter,
    inspect_larpg_adapter,
    materialize_graph,
)
from .extract import extract_robot_bag
from .schema import (
    AssociationRecord,
    InitialStateRecord,
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    PoseVertex,
    canonical_fingerprint,
    read_jsonl,
)
from .se3 import SE3


class RunnerError(RuntimeError):
    """Raised when a benchmark run is not execution-valid."""


@dataclass(frozen=True)
class ExecutableSet:
    larpg: Path
    dran: Path
    distributed_mapper: Path
    sesync: Path
    official_adapter: Path


@dataclass(frozen=True)
class BenchmarkRunConfig:
    repository_root: Path
    manifest_path: Path
    data_root: Path
    canonical_dir: Path
    output_dir: Path
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    max_iterations: int
    timeout_seconds: float
    dry_run: bool = False

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
            raise RunnerError("seeds must be sorted and unique")
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or self.checkpoints[0] < 1
        ):
            raise RunnerError("checkpoints must be sorted, unique, and positive")
        if self.max_iterations < self.checkpoints[-1]:
            raise RunnerError("max_iterations is smaller than the final checkpoint")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise RunnerError("timeout_seconds must be finite and positive")


@dataclass(frozen=True)
class LoadedCanonicalPackage:
    root: Path
    graph: CanonicalGraph
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    method: str
    seed: Optional[int]
    iteration: Optional[int]
    argv: tuple[str, ...]
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.name or not self.method or not self.argv:
            raise RunnerError("command name, method, and argv are required")
        if any(not isinstance(value, str) or not value for value in self.argv):
            raise RunnerError("command argv must contain non-empty strings")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    wall_seconds: float
    stdout_path: Path
    stderr_path: Path
    command_path: Path


@dataclass(frozen=True)
class CommandSet(MappingABC[str, CommandSpec]):
    commands: Mapping[str, CommandSpec]
    expected_full_dran_blocks_per_round: int
    canonical_factor_count: int

    def __getitem__(self, key: str) -> CommandSpec:
        return self.commands[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.commands)

    def __len__(self) -> int:
        return len(self.commands)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RunnerError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{description} must be a JSON object: {path}")
    return value


def load_canonical_package(root: Path) -> LoadedCanonicalPackage:
    root = Path(root).resolve()
    manifest = _read_json(root / "manifest.json", "canonical manifest")
    audit = _read_json(root / "audit.json", "canonical audit")
    vertices = tuple(
        sorted(
            [
                *read_jsonl(root / "vertices.jsonl", PoseVertex),
                *read_jsonl(root / "vertices.jsonl", ObjectVertex),
            ],
            key=lambda item: item.global_id,
        )
    )
    factors = tuple(
        sorted(
            [
                *read_jsonl(root / "factors.jsonl", OdometryFactor),
                *read_jsonl(root / "factors.jsonl", PoseObjectFactor),
            ],
            key=lambda item: item.factor_id,
        )
    )
    graph = CanonicalGraph(
        vertices=vertices,
        factors=factors,
        associations=tuple(
            read_jsonl(root / "associations.jsonl", AssociationRecord)
        ),
        initial_states=tuple(
            read_jsonl(root / "initial_state.jsonl", InitialStateRecord)
        ),
        audit=audit,
        metadata=manifest,
    )
    graph.validate()
    if manifest.get("vertex_count") != len(vertices):
        raise RunnerError("canonical vertex count does not match manifest")
    if manifest.get("factor_count") != len(factors):
        raise RunnerError("canonical factor count does not match manifest")
    factor_hash = canonical_factor_hash(factors)
    if manifest.get("canonical_factor_hash") not in {
        factor_hash,
        legacy_canonical_factor_hash(factors),
    }:
        raise RunnerError("canonical factor hash does not match manifest")
    graph_hash = canonical_fingerprint(
        [
            root / "vertices.jsonl",
            root / "factors.jsonl",
            root / "associations.jsonl",
            root / "initial_state.jsonl",
            root / "global_graph.g2o",
        ]
    )
    if manifest.get("canonical_graph_sha256") != graph_hash:
        raise RunnerError("canonical graph hash does not match manifest")
    return LoadedCanonicalPackage(root, graph, manifest)


def default_executables(repository_root: Path) -> ExecutableSet:
    root = Path(repository_root).resolve()
    return ExecutableSet(
        larpg=root / "baselines/LARPG/build-local/larpg_object_graph",
        dran=root / "build/bin/object-based-multi-robot-example",
        distributed_mapper=(
            root
            / "baselines/distributed-mapper/distributed_mapper_core/cpp/build-local/runDistributedMapper"
        ),
        sesync=root / "baselines/SESync/C++/build-local/bin/SE-Sync",
        official_adapter=(
            root
            / "baselines/object_association/adapters/slideslam_official/build-local/slideslam_semantic_clipper"
        ),
    )


def _git_state(checkout: Path, paths: Sequence[str] = ()) -> dict[str, Any]:
    checkout = Path(checkout).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout.decode("ascii").strip()
        status_command = [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ]
        if paths:
            status_command.extend(["--", *paths])
        status = subprocess.run(
            status_command,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerError(f"cannot inspect git checkout: {checkout}") from error
    return {
        "checkout": str(checkout),
        "commit": commit,
        "dirty": bool(status),
        "dirty_fingerprint": hashlib.sha256(status).hexdigest(),
    }


def _version_line(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RunnerError(f"cannot query tool version: {' '.join(command)}") from error
    lines = completed.stdout.splitlines()
    return lines[0] if lines else ""


def benchmark_preflight(
    config: BenchmarkRunConfig, executables: ExecutableSet
) -> dict[str, Any]:
    package = load_canonical_package(config.canonical_dir)
    manifest = load_manifest(config.manifest_path)
    executable_report = {}
    for name, path in asdict(executables).items():
        resolved = Path(path).resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise RunnerError(
                f"{name} executable is unavailable or not executable: {resolved}"
            )
        executable_report[name] = {
            "path": str(resolved),
            "sha256": _sha256(resolved),
        }
    official_checkout_value = manifest.source.get("official_checkout")
    if not isinstance(official_checkout_value, str):
        raise RunnerError("manifest official SlideSLAM checkout is missing")
    official = _git_state(config.repository_root / official_checkout_value)
    expected_official_commit = manifest.source.get("official_commit")
    if official["commit"] != expected_official_commit:
        raise RunnerError(
            "official SlideSLAM checkout commit does not match the manifest pin"
        )
    dm_checkout = config.repository_root / "baselines/distributed-mapper"
    dm = distributed_mapper_preflight(
        dm_checkout, binary=executables.distributed_mapper
    )
    adapter_report = {}
    for seed in config.seeds:
        inspections = {
            "larpg": inspect_larpg_adapter(
                package.root / "adapters/larpg", seed
            ),
            "dran": inspect_dran_adapter(
                package.root / "adapters/dran", seed
            ),
            "distributed_mapper": inspect_distributed_mapper_adapter(
                package.root / "adapters/distributed_mapper", seed
            ),
        }
        adapter_report[str(seed)] = {
            name: {
                "vertex_count": inspection.vertex_count,
                "factor_count": inspection.factor_count,
                "factor_hash": inspection.factor_hash,
            }
            for name, inspection in inspections.items()
        }
    root_paths = (
        "benchmarks/slideslam",
        "scripts/run_slideslam_object_benchmark.sh",
        "baselines/LARPG",
        "include/DPGO/ObjectCommunicationPolicy.h",
        "src/ObjectCommunicationPolicy.cpp",
        "examples/ObjectBasedMultiRobotExample.cpp",
        "tests/testObjectFullCommunication.cpp",
    )
    return {
        "root_git": _git_state(config.repository_root, root_paths),
        "official_slideslam": official,
        "distributed_mapper": {
            "checkout": str(dm.checkout),
            "binary": str(dm.binary),
            "commit": dm.commit,
            "dirty": dm.dirty,
            "dirty_fingerprint": dm.dirty_fingerprint,
        },
        "executables": executable_report,
        "adapters": adapter_report,
        "canonical": {
            "root": str(package.root),
            "graph_sha256": package.manifest["canonical_graph_sha256"],
            "factor_hash": package.manifest["canonical_factor_hash"],
            "association_sha256": package.manifest["association_sha256"],
            "vertex_count": len(package.graph.vertices),
            "factor_count": len(package.graph.factors),
            "seeds": list(package.manifest["seeds"]),
        },
        "source_audit": dict(package.graph.audit),
        "tools": {
            "cmake": _version_line(("cmake", "--version")),
            "compiler": _version_line(("c++", "--version")),
            "python": platform.python_version(),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }


def materialize_native_sources(
    manifest_path: Path,
    data_root: Path,
    *,
    output_dir: Optional[Path] = None,
    download: bool = False,
    association_timeout_seconds: float = 120.0,
) -> MaterializedPackage:
    """Audit, extract, associate, and freeze the official native team data."""

    manifest = load_manifest(manifest_path)
    data_root = Path(data_root).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    report = acquire_sources(manifest_path, data_root, download=download)
    inventory_by_robot = {
        inventory.robot_id: inventory for inventory in report.inventories
    }
    source_by_robot = {source.robot_id: source for source in manifest.sources}
    extractions = tuple(
        extract_robot_bag(
            inventory_by_robot[robot_id].path,
            manifest,
            source_by_robot[robot_id],
            repository_root,
        )
        for robot_id in sorted(source_by_robot)
    )
    tracks_by_robot = {
        extraction.robot_id: extraction.tracks.accepted
        for extraction in extractions
    }
    adapter_value = manifest.source.get("adapter_binary")
    official_commit = manifest.source.get("official_commit")
    if not isinstance(adapter_value, str) or not isinstance(official_commit, str):
        raise RunnerError("manifest official adapter provenance is incomplete")
    adapter_binary = repository_root / adapter_value
    pair_root = data_root / "preprocessing" / manifest.benchmark_id / "pairs"
    registrations = []
    for reference_robot, query_robot in combinations(sorted(tracks_by_robot), 2):
        registrations.append(
            run_pair_registration(
                reference_tracks=tracks_by_robot[reference_robot],
                query_tracks=tracks_by_robot[query_robot],
                pair_dir=pair_root / f"robot_{reference_robot}_{query_robot}",
                binary=adapter_binary,
                sigma=manifest.association["sigma"],
                epsilon=manifest.association["epsilon"],
                min_pairs=manifest.association["min_pairs"],
                min_map_objects=manifest.association["min_map_objects"],
                matching_threshold=manifest.association["matching_threshold"],
                projection_distance=manifest.association[
                    "hard_projection_distance_m"
                ],
                timeout_seconds=association_timeout_seconds,
                official_commit=official_commit,
            )
        )
    tree = maximum_support_tree(
        registrations, root_robot=int(manifest.association["root_robot"])
    )
    alignments = compose_robot_alignments(tree)
    tracks = tuple(
        track
        for robot_id in sorted(tracks_by_robot)
        for track in tracks_by_robot[robot_id]
    )
    matches = tuple(
        match
        for registration in registrations
        if registration.accepted
        for match in registration.projected_matches
    )
    association = build_global_objects(
        matches,
        tracks,
        minimum_observing_robots=int(
            manifest.association["minimum_observing_robots"]
        ),
    )
    graph = build_canonical_graph(
        extractions,
        association,
        alignments,
        manifest,
        audit=report.to_dict(),
    )
    if output_dir is None:
        output_dir = data_root / "canonical" / manifest.benchmark_id
    return materialize_graph(graph, output_dir)


def build_command_specs(
    config: BenchmarkRunConfig,
    executables: ExecutableSet,
    *,
    seed: int,
    iterations: int,
    work_root: Path,
    package: LoadedCanonicalPackage | None = None,
) -> CommandSet:
    if seed not in config.seeds:
        raise RunnerError(f"seed {seed} is not enabled")
    if iterations < 1 or iterations > config.max_iterations:
        raise RunnerError("command iteration count is outside configured bounds")
    if package is None:
        package = load_canonical_package(config.canonical_dir)
    elif package.root != config.canonical_dir:
        raise RunnerError("preloaded canonical package does not match the run config")
    work_root = Path(work_root).resolve()
    larpg_output = work_root / f"native/larpg_seed_{seed}"
    dran_csv = work_root / f"native/dran_seed_{seed}_round_{iterations}.csv"
    dran_trajectory = (
        work_root / f"native/dran_seed_{seed}_round_{iterations}_trajectory.txt"
    )
    dran_objects = (
        work_root / f"native/dran_seed_{seed}_round_{iterations}_objects.txt"
    )
    dm_seed_dir = (
        config.canonical_dir / f"adapters/distributed_mapper/seed_{seed}"
    )
    dm_comm = work_root / f"native/dm_seed_{seed}_round_{iterations}_comm.tsv"
    dm_loss = work_root / f"native/dm_seed_{seed}_round_{iterations}_loss.tsv"
    sesync_output = work_root / "reference/sesync.txt"
    sesync_input = work_root / "reference/global_graph.g2o"
    dran_manifest = _read_json(
        config.canonical_dir
        / f"adapters/dran/seed_{seed}/adapter_manifest.json",
        "DRAN adapter manifest",
    )
    num_robots = int(dran_manifest["num_robots"])
    num_objects = int(dran_manifest["num_objects"])
    commands = {
        "larpg": CommandSpec(
            name=f"larpg_seed_{seed}",
            method="larpg",
            seed=seed,
            iteration=iterations,
            argv=(
                str(executables.larpg),
                "--input",
                str(
                    config.canonical_dir
                    / f"adapters/larpg/problem_seed_{seed}.json"
                ),
                "--output-dir",
                str(larpg_output),
                "--iterations",
                str(iterations),
                "--communication",
                "full",
            ),
            environment={},
        ),
        "dran": CommandSpec(
            name=f"dran_seed_{seed}_round_{iterations}",
            method="dran",
            seed=seed,
            iteration=iterations,
            argv=(
                str(executables.dran),
                str(config.canonical_dir / f"adapters/dran/seed_{seed}"),
                str(num_robots),
                str(num_objects),
                str(iterations),
                "1",
                "1",
                str(dran_csv),
                "0",
                "0",
                str(dran_trajectory),
                "complete",
                "1",
                "5",
                "0.01",
                "raw",
            ),
            environment={
                "OBJECT_COMMUNICATION_POLICY": "full",
                "OBJECT_TOPOLOGY": "complete",
                "OBJECT_POSE_TOL": "0",
                "OBJECT_MAX_AGE": "0",
                "OBJECT_INNOVATION_THRESHOLD": "0",
                "OBJECT_BOUNDARY_BUDGET_MAX_POSES": "0",
                "OBJECT_OUTER_STOP_CONSENSUS_COST": "-1",
                "DPGO_OBJECT_POSE_OUTPUT": str(dran_objects),
            },
        ),
        "distributed_mapper": CommandSpec(
            name=f"distributed_mapper_seed_{seed}_round_{iterations}",
            method="distributed_mapper",
            seed=seed,
            iteration=iterations,
            argv=(
                str(executables.distributed_mapper),
                "--nrRobots",
                str(num_robots),
                "--dataDir",
                f"{dm_seed_dir}{os.sep}",
                "--useBetweenNoise",
                "true",
                "--useLandmarks",
                "true",
                "--maxIter",
                str(iterations),
                "--rthresh",
                "0",
                "--pthresh",
                "0",
                "--commCsv",
                str(dm_comm),
                "--lossCsv",
                str(dm_loss),
            ),
            environment={"LD_LIBRARY_PATH": str(executables.distributed_mapper.parent)},
        ),
        "sesync": CommandSpec(
            name="sesync_reference",
            method="sesync",
            seed=None,
            iteration=None,
            argv=(
                str(executables.sesync),
                "--input",
                str(sesync_input),
                "--output",
                str(sesync_output),
                "--threads",
                "1",
                "--quiet",
            ),
            environment={
                "LD_LIBRARY_PATH": os.pathsep.join(
                    [
                        str(
                            config.repository_root
                            / "baselines/SESync/C++/build-local/lib"
                        ),
                        str(
                            config.repository_root
                            / "baselines/SESync/C++/build/lib"
                        ),
                    ]
                )
            },
        ),
    }
    return CommandSet(
        commands=commands,
        expected_full_dran_blocks_per_round=(
            num_objects * num_robots * (num_robots - 1)
        ),
        canonical_factor_count=len(package.graph.factors),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _prepare_sesync_input(source: Path, target: Path) -> None:
    try:
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RunnerError(f"cannot read canonical SE-Sync input: {source}") from error
    records = [
        line
        for line in lines
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    ]
    _write_text_atomic(Path(target), "\n".join(records) + "\n")


def execute_command(
    spec: CommandSpec,
    *,
    command_dir: Path,
    log_dir: Path,
    timeout_seconds: float,
) -> CommandResult:
    executable = Path(spec.argv[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RunnerError(
            f"{spec.method} executable is unavailable or not executable: {executable}"
        )
    command_dir = Path(command_dir)
    log_dir = Path(log_dir)
    command_path = command_dir / f"{spec.name}.json"
    stdout_path = log_dir / f"{spec.name}.stdout.log"
    stderr_path = log_dir / f"{spec.name}.stderr.log"
    environment = os.environ.copy()
    for key, value in spec.environment.items():
        if key == "LD_LIBRARY_PATH" and environment.get(key):
            environment[key] = value + os.pathsep + environment[key]
        else:
            environment[key] = value
    started_wall = time.time()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(spec.argv),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as error:
        wall_seconds = time.monotonic() - started
        _write_text_atomic(stdout_path, error.stdout or "")
        _write_text_atomic(stderr_path, error.stderr or "")
        raise RunnerError(
            f"{spec.method} seed {spec.seed} iteration {spec.iteration} "
            f"timed out after {wall_seconds:.6f} seconds"
        ) from error
    except OSError as error:
        raise RunnerError(f"cannot execute {spec.method}: {error}") from error
    wall_seconds = time.monotonic() - started
    _write_text_atomic(stdout_path, completed.stdout)
    _write_text_atomic(stderr_path, completed.stderr)
    command_record = {
        "name": spec.name,
        "method": spec.method,
        "seed": spec.seed,
        "iteration": spec.iteration,
        "argv": list(spec.argv),
        "environment": dict(sorted(spec.environment.items())),
        "executable_sha256": _sha256(executable),
        "started_unix_seconds": started_wall,
        "wall_seconds": wall_seconds,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    _write_text_atomic(
        command_path,
        json.dumps(command_record, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    if completed.returncode != 0:
        raise RunnerError(
            f"{spec.method} seed {spec.seed} iteration {spec.iteration} "
            f"exited with status {completed.returncode}"
        )
    return CommandResult(
        completed.returncode,
        wall_seconds,
        stdout_path,
        stderr_path,
        command_path,
    )


def _json_text(value: Any, *, diagnostic: str) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{diagnostic} contains a non-finite or invalid value") from error


def _state_jsonl(states: Mapping[int, Mapping[int, SE3]]) -> str:
    lines = []
    for iteration, state in sorted(states.items()):
        for variable_id, pose in sorted(state.items()):
            if not isinstance(pose, SE3):
                raise RunnerError(
                    f"iteration {iteration} state ID {variable_id} is not SE3"
                )
            lines.append(
                json.dumps(
                    {
                        "iteration": iteration,
                        "id": variable_id,
                        "pose": pose.to_dict(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
    return "\n".join(lines) + ("\n" if lines else "")


def write_benchmark_artifacts(
    target: Path,
    *,
    run_manifest: Mapping[str, Any],
    records_by_run: Mapping[tuple[str, int], Sequence[IterationRecord]],
    states_by_run: Mapping[tuple[str, int], Mapping[int, Mapping[int, SE3]]],
    evaluations_by_run: Mapping[tuple[str, int], Mapping[str, Any]],
    checkpoints: Sequence[int],
    support_root: Optional[Path] = None,
) -> None:
    target = Path(target)
    if target.exists():
        raise RunnerError(f"run output already exists: {target}")
    for (method, seed), records in records_by_run.items():
        try:
            require_checkpoints(records, checkpoints)
        except EvaluationError as error:
            raise RunnerError(f"{method} seed {seed}: {error}") from error
        states = states_by_run.get((method, seed))
        if states is None:
            raise RunnerError(f"{method} seed {seed}: missing states")
        missing_states = [item for item in checkpoints if item not in states]
        if missing_states:
            raise RunnerError(
                f"{method} seed {seed}: missing checkpoint {missing_states[0]} state"
            )
        evaluation = evaluations_by_run.get((method, seed))
        if evaluation is None:
            raise RunnerError(f"{method} seed {seed}: missing evaluation")
        _json_text(evaluation, diagnostic=f"{method} seed {seed} evaluation")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        for directory in ("commands", "logs", "iterations", "states", "evaluations"):
            (temporary / directory).mkdir()
        if support_root is not None:
            support_root = Path(support_root)
            for source in support_root.iterdir():
                if source.name in {
                    "iterations",
                    "states",
                    "evaluations",
                    "inputs",
                }:
                    continue
                destination = temporary / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
        (temporary / "run_manifest.json").write_text(
            _json_text(run_manifest, diagnostic="run manifest"), encoding="utf-8"
        )
        summary_rows = []
        for method, seed in sorted(records_by_run):
            require_checkpoints(records_by_run[(method, seed)], checkpoints)
            records = tuple(
                sorted(records_by_run[(method, seed)], key=lambda item: item.iteration)
            )
            iteration_path = temporary / f"iterations/{method}_{seed}.csv"
            with iteration_path.open("w", newline="", encoding="utf-8") as stream:
                fieldnames = list(asdict(records[0]))
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    row = asdict(record)
                    writer.writerow(row)
                    summary_rows.append({"seed": seed, **row})
            (temporary / f"states/{method}_{seed}.jsonl").write_text(
                _state_jsonl(states_by_run[(method, seed)]), encoding="utf-8"
            )
            (temporary / f"evaluations/{method}_{seed}.json").write_text(
                _json_text(
                    evaluations_by_run[(method, seed)],
                    diagnostic=f"{method} seed {seed} evaluation",
                ),
                encoding="utf-8",
            )
        with (temporary / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fieldnames = list(summary_rows[0]) if summary_rows else []
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(summary_rows)
        markdown = [
            "# SlideSLAM Object Backend Benchmark",
            "",
            "| Method | Seed | Round | External cost | External gradient | Logical bytes |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            markdown.append(
                "| {method} | {seed} | {iteration} | {external_cost:.12g} | "
                "{external_gradient_norm:.12g} | {logical_bytes} |".format(**row)
            )
        (temporary / "summary.md").write_text(
            "\n".join(markdown) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _command_manifest(spec: CommandSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "method": spec.method,
        "seed": spec.seed,
        "iteration": spec.iteration,
        "argv": list(spec.argv),
        "environment": dict(sorted(spec.environment.items())),
    }


def _replace_option(spec: CommandSpec, option: str, value: str) -> CommandSpec:
    argv = list(spec.argv)
    try:
        index = argv.index(option)
    except ValueError as error:
        raise RunnerError(f"{spec.method} command has no {option} option") from error
    if index + 1 >= len(argv):
        raise RunnerError(f"{spec.method} command has no value for {option}")
    argv[index + 1] = value
    return replace(spec, argv=tuple(argv))


def _evaluation_report(
    records: Sequence[IterationRecord],
    states: Mapping[int, Mapping[int, SE3]],
    graph: CanonicalGraph,
    reference: Mapping[int, SE3],
    checkpoints: Sequence[int],
    *,
    relative_cost_tolerance: float,
    gradient_tolerance: float,
    consecutive: int,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda item: item.iteration))
    require_checkpoints(ordered, checkpoints)
    reference_evaluation = evaluate_state(graph, reference)
    cumulative_bytes = 0
    cumulative_blocks = 0
    cumulative_messages = 0
    rows = []
    for record in ordered:
        cumulative_bytes += record.logical_bytes
        cumulative_blocks += record.payload_blocks
        cumulative_messages += record.messages
        metrics = reference_metrics(graph, states[record.iteration], reference)
        rows.append(
            {
                **asdict(record),
                "relative_cost_gap_to_centralized_reference": relative_cost_gap(
                    record.external_cost, reference_evaluation.cost
                ),
                **asdict(metrics),
                "cumulative_logical_bytes": cumulative_bytes,
                "cumulative_payload_blocks": cumulative_blocks,
                "cumulative_messages": cumulative_messages,
            }
        )
    convergence = detect_convergence(
        ordered,
        relative_cost_tolerance=relative_cost_tolerance,
        gradient_tolerance=gradient_tolerance,
        consecutive=consecutive,
    )
    return {
        "finite": all(math.isfinite(row["external_cost"]) for row in rows),
        "checkpoints": list(checkpoints),
        "reference_label": "centralized_reference",
        "centralized_reference_cost": reference_evaluation.cost,
        "convergence": asdict(convergence),
        "iterations": rows,
    }


def run_benchmark(
    config: BenchmarkRunConfig,
    *,
    executables: Optional[ExecutableSet] = None,
) -> Mapping[str, Any]:
    """Preflight and execute the benchmark, or emit an exact dry-run manifest."""

    if config.output_dir.exists():
        raise RunnerError(f"run output already exists: {config.output_dir}")
    if executables is None:
        executables = default_executables(config.repository_root)
    preparation_started = time.monotonic()
    preflight = benchmark_preflight(config, executables)
    commands = []
    expected_full_blocks = {}
    sesync_added = False
    for seed in config.seeds:
        command_set = build_command_specs(
            config,
            executables,
            seed=seed,
            iterations=config.max_iterations,
            work_root=config.output_dir,
        )
        if not sesync_added:
            commands.append(_command_manifest(command_set["sesync"]))
            sesync_added = True
        for method in ("larpg", "dran", "distributed_mapper"):
            commands.append(_command_manifest(command_set[method]))
        expected_full_blocks[str(seed)] = (
            command_set.expected_full_dran_blocks_per_round
        )
    run_manifest = {
        "schema_version": "slideslam_backend_run_v1",
        "benchmark_id": preflight["canonical"].get(
            "benchmark_id", load_manifest(config.manifest_path).benchmark_id
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": config.dry_run,
        "seeds": list(config.seeds),
        "checkpoints": list(config.checkpoints),
        "max_iterations": config.max_iterations,
        "timeout_seconds": config.timeout_seconds,
        "data_root": str(config.data_root),
        "canonical_dir": str(config.canonical_dir),
        "output_dir": str(config.output_dir),
        "expected_full_dran_blocks_per_round": expected_full_blocks,
        "commands": commands,
        "preflight": preflight,
        "timing_seconds": {
            "preparation": time.monotonic() - preparation_started,
            "solver": 0.0,
        },
    }
    if config.dry_run:
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{config.output_dir.name}.tmp-",
                dir=config.output_dir.parent,
            )
        )
        try:
            (temporary / "run_manifest.json").write_text(
                _json_text(run_manifest, diagnostic="dry-run manifest"),
                encoding="utf-8",
            )
            os.replace(temporary, config.output_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return run_manifest

    package = load_canonical_package(config.canonical_dir)
    benchmark_manifest = load_manifest(config.manifest_path)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    execution_root = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_dir.name}.execution-",
            dir=config.output_dir.parent,
        )
    )
    solver_wall_seconds = 0.0
    records_by_run: dict[tuple[str, int], list[IterationRecord]] = {}
    states_by_run: dict[tuple[str, int], dict[int, Mapping[int, SE3]]] = {}
    evaluations_by_run: dict[tuple[str, int], Mapping[str, Any]] = {}
    try:
        for directory in ("commands", "logs", "native", "reference", "inputs"):
            (execution_root / directory).mkdir()
        reference_commands = build_command_specs(
            config,
            executables,
            seed=config.seeds[0],
            iterations=config.max_iterations,
            work_root=execution_root,
        )
        reference_input = Path(
            reference_commands["sesync"].argv[
                reference_commands["sesync"].argv.index("--input") + 1
            ]
        )
        _prepare_sesync_input(
            config.canonical_dir / "global_graph.g2o", reference_input
        )
        reference_result = execute_command(
            reference_commands["sesync"],
            command_dir=execution_root / "commands",
            log_dir=execution_root / "logs",
            timeout_seconds=config.timeout_seconds,
        )
        solver_wall_seconds += reference_result.wall_seconds
        reference_path = Path(
            reference_commands["sesync"].argv[
                reference_commands["sesync"].argv.index("--output") + 1
            ]
        )
        reference = read_sesync_state(reference_path, package.graph)
        reference_log = reference_result.stdout_path.read_text(encoding="utf-8")
        reference_summary = next(
            (line for line in reversed(reference_log.splitlines()) if line.startswith("SUMMARY")),
            "",
        )
        if not reference_summary:
            raise RunnerError("SE-Sync reference produced no SUMMARY line")

        evaluation_rounds = tuple(range(1, config.max_iterations + 1))
        for seed in config.seeds:
            larpg_commands = build_command_specs(
                config,
                executables,
                seed=seed,
                iterations=config.max_iterations,
                work_root=execution_root,
            )
            larpg_result = execute_command(
                larpg_commands["larpg"],
                command_dir=execution_root / "commands",
                log_dir=execution_root / "logs",
                timeout_seconds=config.timeout_seconds,
            )
            solver_wall_seconds += larpg_result.wall_seconds
            larpg_output = Path(
                larpg_commands["larpg"].argv[
                    larpg_commands["larpg"].argv.index("--output-dir") + 1
                ]
            )
            larpg_states = {
                iteration: read_larpg_state(
                    larpg_output / f"states/iteration_{iteration:03d}.jsonl",
                    package.graph,
                )
                for iteration in evaluation_rounds
            }
            try:
                larpg_records = list(
                    parse_larpg_iterations(
                        larpg_output / "iterations.csv",
                        package.graph,
                        larpg_states,
                    )
                )
            except EvaluationError as error:
                raise RunnerError(f"larpg seed {seed}: {error}") from error
            records_by_run[("larpg", seed)] = larpg_records
            states_by_run[("larpg", seed)] = larpg_states

            dran_records = []
            dran_states = {}
            mapper_records = []
            mapper_states = {}
            for iteration in evaluation_rounds:
                command_set = build_command_specs(
                    config,
                    executables,
                    seed=seed,
                    iterations=iteration,
                    work_root=execution_root,
                )
                dran_spec = command_set["dran"]
                dran_result = execute_command(
                    dran_spec,
                    command_dir=execution_root / "commands",
                    log_dir=execution_root / "logs",
                    timeout_seconds=config.timeout_seconds,
                )
                solver_wall_seconds += dran_result.wall_seconds
                object_state_path = Path(dran_spec.environment["DPGO_OBJECT_POSE_OUTPUT"])
                dran_state = read_dran_state(
                    object_state_path,
                    config.canonical_dir
                    / f"adapters/dran/seed_{seed}/adapter_manifest.json",
                    package.graph,
                )
                dran_states[iteration] = dran_state
                dran_csv = Path(dran_spec.argv[7])
                try:
                    dran_records.extend(
                        parse_dran_iterations(
                            dran_csv,
                            package.graph,
                            {iteration: dran_state},
                        )
                    )
                except EvaluationError as error:
                    raise RunnerError(
                        f"dran seed {seed} iteration {iteration}: {error}"
                    ) from error

                mapper_input = (
                    execution_root
                    / f"inputs/distributed_mapper_seed_{seed}_round_{iteration}"
                )
                shutil.copytree(
                    config.canonical_dir
                    / f"adapters/distributed_mapper/seed_{seed}",
                    mapper_input,
                )
                mapper_spec = _replace_option(
                    command_set["distributed_mapper"],
                    "--dataDir",
                    f"{mapper_input}{os.sep}",
                )
                mapper_result = execute_command(
                    mapper_spec,
                    command_dir=execution_root / "commands",
                    log_dir=execution_root / "logs",
                    timeout_seconds=config.timeout_seconds,
                )
                solver_wall_seconds += mapper_result.wall_seconds
                mapper_state = read_distributed_mapper_state(
                    mapper_input,
                    mapper_input / "adapter_manifest.json",
                    package.graph,
                )
                mapper_states[iteration] = mapper_state
                loss_path = Path(
                    mapper_spec.argv[mapper_spec.argv.index("--lossCsv") + 1]
                )
                comm_path = Path(
                    mapper_spec.argv[mapper_spec.argv.index("--commCsv") + 1]
                )
                try:
                    mapper_records.extend(
                        parse_distributed_mapper_iterations(
                            loss_path,
                            comm_path,
                            package.graph,
                            {iteration: mapper_state},
                            solver_wall_times={
                                iteration: mapper_result.wall_seconds
                            },
                        )
                    )
                except EvaluationError as error:
                    raise RunnerError(
                        "distributed_mapper seed "
                        f"{seed} iteration {iteration}: {error}"
                    ) from error
            records_by_run[("dran", seed)] = dran_records
            states_by_run[("dran", seed)] = dran_states
            records_by_run[("distributed_mapper", seed)] = mapper_records
            states_by_run[("distributed_mapper", seed)] = mapper_states

            for method in ("larpg", "dran", "distributed_mapper"):
                try:
                    evaluations_by_run[(method, seed)] = _evaluation_report(
                        records_by_run[(method, seed)],
                        states_by_run[(method, seed)],
                        package.graph,
                        reference,
                        config.checkpoints,
                        relative_cost_tolerance=(
                            benchmark_manifest.evaluation.relative_cost_tolerance
                        ),
                        gradient_tolerance=(
                            benchmark_manifest.evaluation.gradient_tolerance
                        ),
                        consecutive=(
                            benchmark_manifest.evaluation.consecutive_rounds
                        ),
                    )
                except EvaluationError as error:
                    raise RunnerError(f"{method} seed {seed}: {error}") from error

        run_manifest["dry_run"] = False
        run_manifest["centralized_reference"] = {
            "path": "reference/sesync.txt",
            "summary": reference_summary,
            "external": asdict(evaluate_state(package.graph, reference)),
        }
        run_manifest["centralized_reference"]["external"].pop("gradient")
        run_manifest["centralized_reference"]["external"]["variable_ids"] = list(
            run_manifest["centralized_reference"]["external"]["variable_ids"]
        )
        run_manifest["executed_command_count"] = 1 + len(config.seeds) * (
            1 + 2 * config.max_iterations
        )
        run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        run_manifest["timing_seconds"] = {
            "preparation": run_manifest["timing_seconds"]["preparation"],
            "solver": solver_wall_seconds,
        }
        write_benchmark_artifacts(
            config.output_dir,
            run_manifest=run_manifest,
            records_by_run=records_by_run,
            states_by_run=states_by_run,
            evaluations_by_run=evaluations_by_run,
            checkpoints=config.checkpoints,
            support_root=execution_root,
        )
    except Exception:
        shutil.rmtree(execution_root, ignore_errors=True)
        raise
    shutil.rmtree(execution_root, ignore_errors=True)
    return run_manifest


__all__ = [
    "BenchmarkRunConfig",
    "CommandResult",
    "CommandSet",
    "CommandSpec",
    "ExecutableSet",
    "LoadedCanonicalPackage",
    "RunnerError",
    "benchmark_preflight",
    "build_command_specs",
    "default_executables",
    "execute_command",
    "load_canonical_package",
    "materialize_native_sources",
    "run_benchmark",
    "write_benchmark_artifacts",
]
