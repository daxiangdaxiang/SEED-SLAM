#!/usr/bin/env python3

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.dba_collapsed_profile_partition import ObjectCopy
from scripts.run_dba_fixed_point import (
    build_lower_command,
    canonical_partition,
    dataset_config,
    parse_args,
    parse_response_pattern,
    partition_sha256,
    read_candidate_support,
    run_fixed_point_protocol,
)


def _row(
    receiver_robot: int,
    receiver_object: int,
    sender_robot: int,
    sender_object: int,
    weight: float = 1.0,
) -> dict[str, object]:
    return {
        "receiver_robot": receiver_robot,
        "receiver_object": receiver_object,
        "sender_robot": sender_robot,
        "sender_object": sender_object,
        "weight": weight,
    }


def test_canonical_partition_hash_ignores_row_direction_and_order() -> None:
    support = ((0, 10), (1, 20), (2, 30), (2, 31))
    first = (
        _row(0, 10, 1, 20),
        _row(2, 30, 1, 20),
        _row(1, 20, 0, 10),
        _row(1, 20, 2, 30),
    )
    reordered = tuple(reversed(first))

    first_partition = canonical_partition(first, support=support)
    reordered_partition = canonical_partition(reordered, support=support)

    assert first_partition == (
        ((0, 10), (1, 20), (2, 30)),
        ((2, 31),),
    )
    assert reordered_partition == first_partition
    assert partition_sha256(reordered_partition) == partition_sha256(
        first_partition
    )


def test_canonical_partition_hash_changes_when_component_changes() -> None:
    support = ((0, 10), (1, 20), (2, 30))
    merged = canonical_partition(
        (_row(0, 10, 1, 20), _row(1, 20, 2, 30)), support=support
    )
    split = canonical_partition((_row(0, 10, 1, 20),), support=support)

    assert partition_sha256(merged) != partition_sha256(split)


def test_canonical_partition_ignores_subthreshold_soft_edges() -> None:
    support = ((0, 10), (1, 20))
    partition = canonical_partition(
        (_row(0, 10, 1, 20, weight=0.49),), support=support
    )

    assert partition == (((0, 10),), ((1, 20),))


def test_fixed_candidate_support_is_canonical_and_topology_checked(
    tmp_path: Path,
) -> None:
    first = ObjectCopy(0, 10, 4)
    second = ObjectCopy(1, 20, 4)
    path = tmp_path / "candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(_row(0, 0, 0, 0)))
        writer.writeheader()
        writer.writerows((_row(0, 10, 1, 20), _row(1, 20, 0, 10)))

    result = read_candidate_support(
        path, {(0, 10): first, (1, 20): second}, ((0, 1),)
    )

    assert result == ((first, second),)

    with pytest.raises(ValueError, match="outside upper topology"):
        read_candidate_support(
            path, {(0, 10): first, (1, 20): second}, ((0, 2),)
        )


def _write_associations(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "receiver_robot",
                "sender_robot",
                "receiver_object",
                "sender_object",
                "weight",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _protocol_callbacks(
    tmp_path: Path,
    outputs: tuple[tuple[dict[str, object], ...], ...],
):
    lower_inputs: list[Path] = []

    def run_lower(epoch: int, association: Path, lower_dir: Path) -> None:
        lower_dir.mkdir(parents=True)
        lower_inputs.append(association)
        shutil.copyfile(
            association, lower_dir / "final_association_weights.csv"
        )
        (lower_dir / "iteration_summary.csv").write_text(
            "iter,cumulative_total_comm_mb\n1,1.0\n", encoding="utf-8"
        )

    def run_upper(
        epoch: int,
        association: Path,
        lower_dir: Path,
        upper_dir: Path,
    ) -> Path:
        output = upper_dir / "associations.csv"
        _write_associations(output, outputs[epoch])
        (upper_dir / "posthoc_metrics.json").write_text(
            json.dumps({"association_f1": 0.75}) + "\n",
            encoding="utf-8",
        )
        (upper_dir / "summary.json").write_text(
            json.dumps(
                {
                    "distributed_selection_payload_bytes": 1024 * 1024,
                    "distributed_canonical_state_sync_payload_bytes": 768 * 1024,
                    "upper_objective": "joint_cavity_leave_pair",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    return lower_inputs, run_lower, run_upper


def test_protocol_stops_when_full_lower_response_partition_is_fixed(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    merged = (
        _row(0, 10, 1, 20),
        _row(1, 20, 0, 10),
    )
    lower_inputs, run_lower, run_upper = _protocol_callbacks(
        tmp_path, (merged, merged)
    )

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (1, 20)),
        output_dir=tmp_path / "run",
        safety_max_outer_iterations=5,
        run_lower=run_lower,
        run_upper=run_upper,
    )

    assert result.status == "partition_fixed_point"
    assert len(result.epochs) == 2
    assert result.final_association == result.epochs[-1].output_association
    assert result.final_lower_dir == result.epochs[-1].lower_dir
    assert lower_inputs == [
        result.epochs[0].input_association,
        result.epochs[0].output_association,
    ]
    assert result.attempted_outer_rounds == 2
    assert result.delivered_response_count == 2
    assert result.scheduled_missing_response_count == 0


def test_protocol_records_scheduled_missing_lower_responses(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    merged = (
        _row(0, 10, 1, 20),
        _row(1, 20, 0, 10),
    )
    lower_calls: list[int] = []
    upper_calls: list[int] = []

    def run_lower(epoch: int, association: Path, lower_dir: Path) -> None:
        lower_calls.append(epoch)
        lower_dir.mkdir(parents=True)
        shutil.copyfile(
            association, lower_dir / "final_association_weights.csv"
        )
        (lower_dir / "iteration_summary.csv").write_text(
            "iter,cumulative_total_comm_mb\n1,1.0\n", encoding="utf-8"
        )

    def run_upper(
        epoch: int,
        association: Path,
        lower_dir: Path,
        upper_dir: Path,
    ) -> Path:
        del association, lower_dir
        upper_calls.append(epoch)
        output = upper_dir / "associations.csv"
        _write_associations(output, merged)
        (upper_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        return output

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (1, 20)),
        output_dir=tmp_path / "run",
        safety_max_outer_iterations=8,
        run_lower=run_lower,
        run_upper=run_upper,
        delivered_response_pattern=(True, False, False, False),
    )

    assert lower_calls == [0, 1]
    assert upper_calls == [0, 1]
    assert result.attempted_outer_rounds == 5
    assert result.delivered_response_count == 2
    assert result.scheduled_missing_response_count == 3
    with (tmp_path / "run/availability_trace.csv").open(newline="") as handle:
        availability_rows = list(csv.DictReader(handle))
    assert [row["delivered"] for row in availability_rows] == [
        "1",
        "0",
        "0",
        "0",
        "1",
    ]
    assert not (tmp_path / "run/epoch_02").exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    (("1", (True,)), ("1110", (True, True, True, False)), ("10", (True, False))),
)
def test_parse_response_pattern(value: str, expected: tuple[bool, ...]) -> None:
    assert parse_response_pattern(value) == expected


@pytest.mark.parametrize("value", ("", "0", "000", "10x"))
def test_parse_response_pattern_rejects_invalid_schedule(value: str) -> None:
    with pytest.raises(ValueError, match="response pattern"):
        parse_response_pattern(value)


def test_protocol_binarizes_the_partition_before_lower_feedback(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(
        initial,
        (
            _row(0, 10, 1, 20, weight=0.75),
            _row(0, 11, 1, 21, weight=0.25),
        ),
    )
    selected = (_row(0, 10, 1, 20),)
    lower_inputs, run_lower, run_upper = _protocol_callbacks(
        tmp_path, (selected,)
    )

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (0, 11), (1, 20), (1, 21)),
        output_dir=tmp_path / "run",
        safety_max_outer_iterations=2,
        run_lower=run_lower,
        run_upper=run_upper,
    )

    with lower_inputs[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert result.status == "partition_fixed_point"
    assert lower_inputs[0].name == "initial_partition_associations.csv"
    assert len(rows) == 1
    assert rows[0]["weight"] == "1"


def test_protocol_reports_partition_cycle_without_claiming_convergence(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    merged = (
        _row(0, 10, 1, 20),
        _row(1, 20, 0, 10),
    )
    lower_inputs, run_lower, run_upper = _protocol_callbacks(
        tmp_path, (merged, ())
    )

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (1, 20)),
        output_dir=tmp_path / "run",
        safety_max_outer_iterations=5,
        run_lower=run_lower,
        run_upper=run_upper,
    )

    assert result.status == "partition_cycle"
    assert len(result.epochs) == 2
    assert result.final_lower_dir is None
    assert result.epochs[-1].status == "partition_cycle"
    assert len(lower_inputs) == 2


def test_protocol_reports_safety_limit_as_nonconvergence(tmp_path: Path) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    first = (_row(0, 10, 1, 20), _row(1, 20, 0, 10))
    second = (_row(1, 20, 2, 30), _row(2, 30, 1, 20))
    lower_inputs, run_lower, run_upper = _protocol_callbacks(
        tmp_path, (first, second)
    )

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (1, 20), (2, 30)),
        output_dir=tmp_path / "run",
        safety_max_outer_iterations=2,
        run_lower=run_lower,
        run_upper=run_upper,
    )

    assert result.status == "safety_limit"
    assert result.final_lower_dir is None
    assert result.epochs[-1].status == "safety_limit"


def test_protocol_rejects_lower_response_from_a_different_partition(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())

    def run_lower(epoch: int, association: Path, lower_dir: Path) -> None:
        del epoch, association
        _write_associations(
            lower_dir / "final_association_weights.csv",
            (_row(0, 10, 1, 20),),
        )

    def run_upper(
        epoch: int,
        association: Path,
        lower_dir: Path,
        upper_dir: Path,
    ) -> Path:
        del epoch, association, lower_dir
        output = upper_dir / "associations.csv"
        _write_associations(output, ())
        return output

    with pytest.raises(ValueError, match="lower response partition mismatch"):
        run_fixed_point_protocol(
            initial_association=initial,
            support=((0, 10), (1, 20)),
            output_dir=tmp_path / "run",
            safety_max_outer_iterations=2,
            run_lower=run_lower,
            run_upper=run_upper,
        )


def _option(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def test_lower_command_preserves_precommitted_fair_response_settings(
    tmp_path: Path,
) -> None:
    config = dataset_config("05", tmp_path / "repo")
    association = tmp_path / "association.csv"
    output_dir = tmp_path / "lower"
    command = build_lower_command(
        config,
        binary=tmp_path / "manual-object-dran-example",
        association=association,
        output_dir=output_dir,
    )

    assert _option(command, "--iters") == "10"
    assert _option(command, "--association_lower_shared_latent_mode") == (
        "exact_weighted"
    )
    assert _option(
        command,
        "--association_object_observation_track_empirical_information",
    ) == "true"
    assert _option(command, "--local_solver") == "reduced_rotation"
    assert _option(command, "--local_max_iterations") == "1"
    assert _option(command, "--association_freeze_upper_variables") == "true"
    assert _option(command, "--association_pair_cavity_diagnostic") == "true"
    assert _option(command, "--save") == "true"
    assert _option(command, "--association_file") == str(association.resolve())
    assert _option(command, "--output_dir") == str(output_dir.resolve())
    assert not any("ground_truth" in value for value in command)


def test_joint_cavity_lower_exports_leave_pair_out_states(
    tmp_path: Path,
) -> None:
    config = dataset_config("07", tmp_path / "repo")
    command = build_lower_command(
        config,
        binary=tmp_path / "manual-object-dran-example",
        association=tmp_path / "association.csv",
        output_dir=tmp_path / "lower",
        cavity_export_leave_pair_states=True,
    )

    assert _option(
        command, "--association_cavity_export_leave_pair_states"
    ) == "true"


def test_dataset_configs_keep_05_and_07_on_the_same_protocol(
    tmp_path: Path,
) -> None:
    first = dataset_config("05", tmp_path)
    second = dataset_config("07", tmp_path)

    assert first.lower_iterations == second.lower_iterations == 10
    assert first.num_robots == second.num_robots == 3
    assert first.plane_axes == second.plane_axes == "xy"
    assert first.num_objects == 259
    assert second.num_objects == 168
    assert first.initial_association.parts[-3:] == (
        "dba",
        "05",
        "initial_associations.csv",
    )
    assert second.initial_association.parts[-3:] == (
        "dba",
        "07",
        "initial_associations.csv",
    )
    assert first.candidate_support.name == "candidate_support.csv"
    assert second.candidate_support.name == "candidate_support.csv"


def test_cli_accepts_exactly_one_dataset_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    manifest = tmp_path / "dataset.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dba_fixed_point.py",
            "--dataset-manifest",
            str(manifest),
            "--lower-iterations",
            "12",
            "--output-dir",
            str(output),
        ],
    )

    args = parse_args()

    assert args.dataset is None
    assert args.dataset_manifest == manifest
    assert args.lower_iterations == 12

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dba_fixed_point.py",
            "--dataset",
            "05",
            "--dataset-manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_cli_exposes_feedback_and_calibration_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dba_fixed_point.py",
            "--dataset",
            "05",
            "--output-dir",
            str(tmp_path / "output"),
            "--upper-objective",
            "joint_cavity_leave_pair_pairwise",
            "--disable_component_intensity",
            "--robot_covariance_mode",
            "diagonal",
        ],
    )

    args = parse_args()

    assert args.upper_objective == "joint_cavity_leave_pair_pairwise"
    assert args.disable_component_intensity is True
    assert args.robot_covariance_mode == "diagonal"


def test_protocol_writes_incremental_audit_bundle(tmp_path: Path) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    merged = (_row(0, 10, 1, 20), _row(1, 20, 0, 10))
    _, run_lower, run_upper = _protocol_callbacks(
        tmp_path, (merged, merged)
    )
    output_dir = tmp_path / "run"

    result = run_fixed_point_protocol(
        initial_association=initial,
        support=((0, 10), (1, 20)),
        output_dir=output_dir,
        safety_max_outer_iterations=5,
        run_lower=run_lower,
        run_upper=run_upper,
    )

    assert result.status == "partition_fixed_point"
    assert (output_dir / "outer_trace.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "epoch_00/partition.sha256").is_file()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "partition_fixed_point"
    assert summary["full_lower_response_count"] == 2
    assert summary["upper_solve_count"] == 2
    assert summary["attempted_outer_rounds"] == 2
    assert summary["delivered_response_count"] == 2
    assert summary["scheduled_missing_response_count"] == 0
    assert summary["delivered_response_pattern"] == "1"
    assert summary["ground_truth_used_by_optimizer"] is False
    assert summary["method"] == "dba_joint_cavity_crossfit_fixed_point"
    assert summary["crossfit_block_surrogate"] is True
    assert summary["final_lower_dir"] == str(result.final_lower_dir)
    assert len(summary["partition_hashes"]) == 3
    assert summary["total_communication_mb"] == 5.5
    with (output_dir / "outer_trace.csv").open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    assert trace[-1]["association_f1"] == "0.75"


def test_protocol_rejects_existing_nonempty_output_directory(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.csv"
    _write_associations(initial, ())
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data\n")
    _, run_lower, run_upper = _protocol_callbacks(tmp_path, ((),))

    with pytest.raises(ValueError, match="nonempty"):
        run_fixed_point_protocol(
            initial_association=initial,
            support=((0, 10),),
            output_dir=output_dir,
            safety_max_outer_iterations=5,
            run_lower=run_lower,
            run_upper=run_upper,
        )

    assert (output_dir / "keep.txt").read_text() == "user data\n"


def test_script_help_exposes_fixed_point_protocol_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_dba_fixed_point.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
    assert "--safety-max-outer-iterations" in completed.stdout
