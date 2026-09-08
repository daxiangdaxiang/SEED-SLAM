#!/usr/bin/env python3
"""Run the preregistered feedback-safe DBA ablation matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dba_fixed_point import file_sha256


VARIANTS = (
    "frozen",
    "joint_cavity",
    "joint_cavity_leave_pair_pairwise",
    "joint_cavity_leave_pair",
    "joint_cavity_leave_pair_no_intensity",
    "joint_cavity_leave_pair_diagonal_covariance",
)


@dataclass(frozen=True, order=True)
class FeedbackCase:
    dataset: str
    initializer: str
    variant: str

    def slug(self) -> str:
        return f"{self.dataset}/{self.initializer}/{self.variant}"


def plan_cases(
    starts: Mapping[str, Iterable[str]],
) -> tuple[FeedbackCase, ...]:
    return tuple(
        FeedbackCase(str(dataset), str(initializer), variant)
        for dataset, initializers in sorted(starts.items())
        for initializer in initializers
        for variant in VARIANTS
    )


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _write_associations(
    path: Path, rows: Iterable[Mapping[str, object]]
) -> None:
    fields = (
        "receiver_robot",
        "sender_robot",
        "receiver_object",
        "sender_object",
        "weight",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_support(
    initializations: Iterable[Path],
    output_path: Path,
) -> Path:
    """Freeze the label-free union supplied by the external front ends."""
    relations = set()
    for path in initializations:
        for row in _read_csv(path):
            if float(row.get("weight", 1.0)) < 0.5:
                continue
            first = (
                int(row["receiver_robot"]),
                int(row["receiver_object"]),
            )
            second = (
                int(row["sender_robot"]),
                int(row["sender_object"]),
            )
            if first != second:
                relations.add(tuple(sorted((first, second))))
    rows = tuple(
        {
            "receiver_robot": first[0],
            "sender_robot": second[0],
            "receiver_object": first[1],
            "sender_object": second[1],
            "weight": 1,
        }
        for relation in sorted(relations)
        for first, second in (relation, tuple(reversed(relation)))
    )
    _write_associations(output_path, rows)
    return output_path


def _metadata_support(path: Path) -> list[list[int]]:
    return [
        [robot, object_id]
        for robot, object_id in sorted(
        {
            (int(row["robot_id"]), int(row["local_slot_id"]))
            for row in _read_csv(path)
        }
        )
    ]


def _git_hash(repo_root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _case_hash(record: Mapping[str, object]) -> str:
    payload = json.dumps(
        {
            key: value
            for key, value in record.items()
            if key not in ("case_hash", "status")
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _terminal_summary(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return value.get("status") == "partition_fixed_point"


def execute_case_records(
    records: Iterable[Mapping[str, object]],
    *,
    dry_run: bool,
    resume: bool,
) -> list[dict[str, object]]:
    outcomes = []
    for supplied in records:
        record = dict(supplied)
        case_dir = Path(str(record["output_dir"]))
        control_dir = Path(str(record.get("control_dir", case_dir)))
        status_path = control_dir / "case_status.json"
        previous = {}
        if status_path.is_file():
            try:
                previous = json.loads(
                    status_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                previous = {}
        if (
            resume
            and previous.get("case_hash") == record["case_hash"]
            and previous.get("status") == "complete"
            and _terminal_summary(case_dir / "summary.json")
        ):
            outcomes.append(
                {**record, "status": "skipped_complete", "returncode": 0}
            )
            continue
        if dry_run:
            outcomes.append(
                {**record, "status": "dry_run", "returncode": None}
            )
            continue
        if case_dir.exists() and any(case_dir.iterdir()):
            if resume and status_path.is_file():
                shutil.rmtree(case_dir)
            else:
                raise ValueError(
                    f"case output directory is nonempty: {case_dir}"
                )
        case_dir.mkdir(parents=True, exist_ok=True)
        control_dir.mkdir(parents=True, exist_ok=True)
        (control_dir / "command.json").write_text(
            json.dumps(record["command"], indent=2) + "\n",
            encoding="utf-8",
        )
        start = time.perf_counter()
        with (control_dir / "stdout.log").open(
            "w", encoding="utf-8"
        ) as stdout, (control_dir / "stderr.log").open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                tuple(map(str, record["command"])),
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        status = (
            "complete"
            if completed.returncode == 0
            and _terminal_summary(case_dir / "summary.json")
            else "failed"
        )
        outcome = {
            **record,
            "status": status,
            "returncode": completed.returncode,
            "elapsed_seconds": time.perf_counter() - start,
        }
        status_path.write_text(
            json.dumps(
                {
                    "case_hash": record["case_hash"],
                    "status": status,
                    "returncode": completed.returncode,
                    "elapsed_seconds": outcome["elapsed_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outcomes.append(outcome)
    return outcomes


def _load_experiment_config(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(
        value.get("datasets"), list
    ):
        raise ValueError("experiment config must contain a dataset list")
    return value


def _variant_arguments(variant: str) -> tuple[str, ...]:
    if variant == "joint_cavity_leave_pair_no_intensity":
        return (
            "--upper-objective",
            "joint_cavity_leave_pair",
            "--disable-component-intensity",
        )
    if variant == "joint_cavity_leave_pair_diagonal_covariance":
        return (
            "--upper-objective",
            "joint_cavity_leave_pair",
            "--robot-covariance-mode",
            "diagonal",
        )
    return ("--upper-objective", variant)


def build_case_records(
    config: Mapping[str, object],
    *,
    output_dir: Path,
    repo_root: Path,
) -> list[dict[str, object]]:
    git_hash = _git_hash(repo_root)
    records = []
    for dataset in config["datasets"]:
        name = str(dataset["name"])
        initializers = {
            str(key): Path(str(value)).resolve()
            for key, value in dataset["initializers"].items()
        }
        candidate_support = _candidate_support(
            initializers.values(),
            output_dir / "inputs" / name / "candidate_support.csv",
        )
        common = {
            key: dataset[key]
            for key in (
                "data_dir",
                "metadata",
                "lower_topology",
                "upper_topology",
                "ground_truth",
                "num_robots",
                "num_objects",
                "plane_axes",
            )
        }
        for case in plan_cases({name: tuple(initializers)}):
            case_dir = output_dir / case.slug()
            manifest_path = (
                output_dir
                / "inputs"
                / name
                / case.initializer
                / "dataset_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": 1,
                "name": name,
                **common,
                "initial_association": str(
                    initializers[case.initializer]
                ),
                "candidate_support": str(candidate_support.resolve()),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            input_paths = {
                key: str(Path(str(value)).resolve())
                for key, value in manifest.items()
                if key
                in (
                    "metadata",
                    "lower_topology",
                    "upper_topology",
                    "ground_truth",
                    "initial_association",
                    "candidate_support",
                )
                and value is not None
            }
            input_hashes = {
                key: file_sha256(Path(value))
                for key, value in input_paths.items()
            }
            command = [
                sys.executable,
                str(repo_root / "scripts/run_dba_fixed_point.py"),
                "--dataset-manifest",
                str(manifest_path),
                "--lower-iterations",
                str(int(config.get("lower_iterations", 10))),
                "--safety-max-outer-iterations",
                str(int(config.get("safety_max_outer_iterations", 20))),
                "--binary",
                str(
                    Path(
                        str(
                            config.get(
                                "binary",
                                repo_root
                                / "build/bin/manual-object-dran-example",
                            )
                        )
                    ).resolve()
                ),
                "--output-dir",
                str(case_dir),
                *_variant_arguments(case.variant),
            ]
            record = {
                "case_id": case.slug(),
                "dataset": name,
                "initializer": case.initializer,
                "variant": case.variant,
                "output_dir": str(case_dir),
                "control_dir": str(output_dir / "_runner" / case.slug()),
                "command": command,
                "git_hash": git_hash,
                "input_hashes": input_hashes,
                "topology": str(Path(str(dataset["upper_topology"])).resolve()),
                "candidate_support": str(candidate_support.resolve()),
                "candidate_support_sha256": file_sha256(candidate_support),
                "ground_truth": str(
                    Path(str(dataset["ground_truth"])).resolve()
                ),
                "support": _metadata_support(
                    Path(str(dataset["metadata"]))
                ),
                "lower_iterations": int(
                    config.get("lower_iterations", 10)
                ),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                },
            }
            record["case_hash"] = _case_hash(record)
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = _load_experiment_config(args.experiment_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = build_case_records(
        config, output_dir=args.output_dir, repo_root=repo_root
    )
    manifest = {
        "schema_version": 1,
        "precommitted_variants": list(VARIANTS),
        "case_count": len(records),
        "cases": records,
        "algorithm_parameter_overrides": {},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outcomes = execute_case_records(
        records, dry_run=args.dry_run, resume=args.resume
    )
    (args.output_dir / "outcomes.json").write_text(
        json.dumps(outcomes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [value for value in outcomes if value["status"] == "failed"]
    print(
        json.dumps(
            {
                "case_count": len(outcomes),
                "failed": len(failed),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
