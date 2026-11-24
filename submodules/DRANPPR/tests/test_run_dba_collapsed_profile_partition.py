#!/usr/bin/env python3

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.dba_collapsed_profile_partition import (
    GaussianObjectObservation,
    NetworkPartition,
    ObjectCopy,
)
from scripts.dba_network_quotient_reference import (
    NetworkQuotientProblem,
    RobotCavityState,
    ScorerInputs,
)
import scripts.run_dba_collapsed_profile_partition as runner


def test_script_entrypoint_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_dba_collapsed_profile_partition.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _fixture() -> ScorerInputs:
    first = ObjectCopy(0, 10, 4)
    anchor_first = ObjectCopy(0, 11, 9)
    second = ObjectCopy(1, 20, 4)
    lonely = ObjectCopy(1, 21, 7)
    anchor_second = ObjectCopy(1, 22, 9)
    problem = NetworkQuotientProblem(
        robots=(
            RobotCavityState(
                0,
                (first, anchor_first),
                np.array([[1.0, 2.0], [0.0, 0.0]]),
                np.array([[0.04, 0.0], [0.0, 0.04]]),
            ),
            RobotCavityState(
                1,
                (second, lonely, anchor_second),
                np.array([[1.05, 1.98], [8.0, -3.0], [0.0, 0.0]]),
                np.array(
                    [[0.05, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.04]]
                ),
            ),
        ),
        partition=NetworkPartition.build(
            ((first,), (second,), (lonely,), (anchor_first, anchor_second))
        ),
    )
    return ScorerInputs(
        problem=problem,
        incumbent_edges=((anchor_first, anchor_second),),
        topology=((0, 1),),
        metadata_by_key={
            (0, 10): 4,
            (0, 11): 9,
            (1, 20): 4,
            (1, 21): 7,
            (1, 22): 9,
        },
        current_transforms={
            0: (0.0, np.zeros(2)),
            1: (0.0, np.zeros(2)),
        },
    )


def test_observations_use_local_marginal_uncertainty() -> None:
    observations = runner.observations_from_problem(_fixture().problem)
    assert np.allclose(observations[ObjectCopy(0, 10, 4)].covariance, 0.04 * np.eye(2))
    assert np.allclose(observations[ObjectCopy(1, 21, 7)].covariance, 0.09 * np.eye(2))


def test_metadata_attributes_exclude_identity_labels_and_calibrate_without_knobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "robot_id",
                "local_slot_id",
                "true_object_id",
                "category",
                "shape_x",
                "shape_y",
                "descriptor_0",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"robot_id": 0, "local_slot_id": 10, "true_object_id": 99, "category": 4, "shape_x": 1.0, "shape_y": 2.0, "descriptor_0": 0.1},
                {"robot_id": 0, "local_slot_id": 11, "true_object_id": 98, "category": 9, "shape_x": 3.0, "shape_y": 2.0, "descriptor_0": 0.3},
                {"robot_id": 1, "local_slot_id": 20, "true_object_id": 99, "category": 4, "shape_x": 1.1, "shape_y": 2.0, "descriptor_0": 0.1},
                {"robot_id": 1, "local_slot_id": 21, "true_object_id": 97, "category": 7, "shape_x": 8.0, "shape_y": 2.0, "descriptor_0": 0.8},
                {"robot_id": 1, "local_slot_id": 22, "true_object_id": 98, "category": 9, "shape_x": 3.1, "shape_y": 2.0, "descriptor_0": 0.3},
            )
        )

    names, keyed = runner.read_object_attributes(
        path,
        {(copy.robot, copy.object_id) for copy in _fixture().problem.partition.copies},
    )
    attributes = {
        copy: keyed[(copy.robot, copy.object_id)]
        for copy in _fixture().problem.partition.copies
    }
    noise = runner.calibrate_attribute_noise(
        _fixture().problem.partition, attributes
    )

    assert names == ("descriptor_0",)
    assert noise.shape == (1,)
    assert np.all(noise > 0.0)
    assert "true_object_id" not in names


def test_metadata_without_descriptors_disables_attribute_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "robot_id",
                "local_slot_id",
                "true_object_id",
                "category",
                "shape_x",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "robot_id": copy.robot,
                "local_slot_id": copy.object_id,
                "true_object_id": 99,
                "category": copy.semantic_class,
                "shape_x": 1.0,
            }
            for copy in _fixture().problem.partition.copies
        )

    names, keyed = runner.read_object_attributes(
        path,
        {(copy.robot, copy.object_id) for copy in _fixture().problem.partition.copies},
    )

    assert names == ()
    assert keyed == {}


def test_run_inputs_is_variable_cardinality_and_gt_free() -> None:
    result = runner.run_inputs(_fixture(), profile_transforms=True)
    assert result.initial_component_count == 4
    assert result.partition.component_count == 3
    selected = {
        (
            row["receiver_robot"],
            row["sender_robot"],
            row["receiver_object"],
            row["sender_object"],
        )
        for row in result.association_rows
    }
    assert (0, 1, 10, 20) in selected
    assert (1, 0, 20, 10) in selected
    assert result.objective_trace[-1] < result.objective_trace[0]


def test_explicit_empty_candidate_support_is_not_regenerated_from_state() -> None:
    result = runner.run_inputs(
        _fixture(), profile_transforms=True, candidate_edges=()
    )

    assert result.candidate_edge_count == 0
    assert result.partition.component_count == 4


def test_fixed_transform_ablation_uses_the_same_partition_objective() -> None:
    result = runner.run_inputs(_fixture(), profile_transforms=False)
    assert result.partition.component_count == 3
    assert result.mode == "fixed_transforms"
    assert result.objective_trace[-1] < result.objective_trace[0]


def test_joint_cavity_mode_uses_one_monotone_objective() -> None:
    result = runner.run_inputs(
        _fixture(),
        profile_transforms=True,
        upper_objective="joint_cavity",
    )

    assert result.mode == "joint_cavity_profile"
    assert result.objective_name == "joint_cavity"
    assert result.objective_evaluations > 0
    assert result.canonical_state_sync_payload_bytes > 0
    assert result.objective_terms["total"] == result.objective_trace[-1]
    assert all(
        after < before
        for before, after in zip(
            result.objective_trace, result.objective_trace[1:]
        )
    )


def test_joint_cavity_mode_includes_nonconstant_object_attributes() -> None:
    inputs = _fixture()
    attributes = {
        copy: np.array([0.1 * copy.object_id, float(copy.semantic_class)])
        for copy in inputs.problem.partition.copies
    }

    result = runner.run_inputs(
        inputs,
        profile_transforms=True,
        attributes=attributes,
        attribute_names=("descriptor_0", "descriptor_1"),
        upper_objective="joint_cavity",
    )

    assert result.attribute_names == ("descriptor_0", "descriptor_1")
    assert np.isfinite(result.objective_terms["attribute_nll"])
    assert result.objective_terms["attribute_nll"] != 0.0
    assert result.objective_terms["total"] == result.objective_trace[-1]


def test_leave_pair_joint_mode_requires_and_uses_topology_cavity(
    tmp_path: Path,
) -> None:
    inputs = _fixture()
    result = runner.run_inputs(
        inputs,
        profile_transforms=True,
        upper_objective="joint_cavity_leave_pair",
        leave_pair_problems={(0, 1): inputs.problem},
    )

    assert result.mode == "joint_cavity_leave_pair_crossfit_profile"
    assert result.objective_name == "joint_cavity_leave_pair"
    assert result.objective_evaluations > 0

    runner.write_result_bundle(result, tmp_path, input_manifest={})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["method"] == "distributed_joint_cavity_crossfit_partition"
    assert summary["crossfit_block_surrogate"] is True


def test_heldout_pairwise_control_uses_same_fold_without_network_consensus(
    tmp_path: Path,
) -> None:
    inputs = _fixture()
    result = runner.run_inputs(
        inputs,
        profile_transforms=True,
        upper_objective="joint_cavity_leave_pair_pairwise",
        leave_pair_problems={(0, 1): inputs.problem},
    )

    assert result.mode == "joint_cavity_leave_pair_pairwise_profile"
    assert result.feedback_variant == "heldout_pairwise"
    assert result.network_feasible is True
    assert result.duplicate_robot_component_count == 0
    assert result.consensus_rounds == 0
    assert result.objective_trace[-1] <= result.objective_trace[0]

    runner.write_result_bundle(result, tmp_path, input_manifest={})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["method"] == (
        "distributed_joint_cavity_crossfit_pairwise"
    )
    assert summary["feedback_variant"] == "heldout_pairwise"


def test_joint_result_bundle_records_probabilistic_calibration(
    tmp_path: Path,
) -> None:
    result = runner.run_inputs(
        _fixture(),
        profile_transforms=True,
        upper_objective="joint_cavity",
        include_component_intensity=False,
        robot_covariance_mode="diagonal",
    )
    runner.write_result_bundle(
        result,
        tmp_path,
        input_manifest={
            "candidate_support": "support.csv",
            "candidate_support_sha256": "abc",
            "initial_partition_sha256": "def",
        },
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["component_intensity"] == "disabled"
    assert summary["robot_covariance_mode"] == "diagonal"
    assert summary["candidate_support_path"] == "support.csv"
    assert summary["candidate_support_sha256"] == "abc"
    assert summary["initial_partition_sha256"] == "def"


def test_write_result_bundle_has_replay_and_audit_files(tmp_path: Path) -> None:
    result = runner.run_inputs(_fixture(), profile_transforms=True)
    runner.write_result_bundle(
        result,
        tmp_path,
        input_manifest={"cavity_dir": "state", "association": "initial.csv"},
    )

    required = {
        "associations.csv",
        "partition.csv",
        "transforms.csv",
        "objective_trace.csv",
        "accepted_edits.csv",
        "summary.json",
        "input_manifest.json",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    with (tmp_path / "associations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["hyperparameters"] == {}
    assert summary["final_component_count"] == 3
    assert summary["accepted_edit_count"] == 1
    assert summary["ground_truth_used_by_optimizer"] is False
    assert summary["component_intensity_model"] == "empirical_semantic_ppp"
    assert summary["component_intensities"] == {"4": 2.0, "7": 1.0, "9": 1.0}
    assert summary["distributed_selection_consensus_rounds"] > 0
    assert summary["distributed_selection_payload_bytes"] > 0
    assert summary["upper_objective"] == "collapsed_independent"
    assert summary["final_objective_terms"]["total"] == summary["final_objective"]


def test_joint_cavity_result_bundle_reports_objective_terms(tmp_path: Path) -> None:
    result = runner.run_inputs(
        _fixture(),
        profile_transforms=True,
        upper_objective="joint_cavity",
    )
    runner.write_result_bundle(result, tmp_path, input_manifest={})

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["upper_objective"] == "joint_cavity"
    assert summary["objective_evaluations"] > 0
    assert summary["distributed_canonical_state_sync_messages"] > 0
    assert summary["distributed_canonical_state_sync_payload_bytes"] > 0
    assert set(summary["final_objective_terms"]) == {
        "cardinality",
        "attribute_nll",
        "log_determinant",
        "mahalanobis",
        "normalization",
        "state_nll",
        "total",
    }


def test_posthoc_metrics_use_declared_topology_even_if_a_pair_emits_no_match() -> None:
    result = runner.run_inputs(_fixture(), profile_transforms=True)
    truth = [
        {
            "receiver_robot": "0",
            "sender_robot": "1",
            "receiver_object": "10",
            "sender_object": "20",
        },
        {
            "receiver_robot": "1",
            "sender_robot": "0",
            "receiver_object": "20",
            "sender_object": "10",
        },
        {
            "receiver_robot": "0",
            "sender_robot": "1",
            "receiver_object": "11",
            "sender_object": "22",
        },
        {
            "receiver_robot": "1",
            "sender_robot": "0",
            "receiver_object": "22",
            "sender_object": "11",
        },
        {
            "receiver_robot": "0",
            "sender_robot": "2",
            "receiver_object": "10",
            "sender_object": "30",
        },
    ]
    metrics = runner.posthoc_association_metrics(
        result.association_rows,
        truth,
        evaluated_pairs={(0, 1), (1, 0)},
    )
    assert metrics["association_f1"] == 1.0
    assert metrics["association_false_negative"] == 0
    assert metrics["network_partition_precision"] == 1.0
    assert metrics["network_partition_recall"] == 0.5
    assert metrics["network_partition_f1"] == 2.0 / 3.0
    assert metrics["network_partition_false_negative"] == 2
    empty = runner.posthoc_association_metrics(
        (), truth, evaluated_pairs={(0, 1), (1, 0)}
    )
    assert empty["association_false_negative"] == 4
    assert empty["network_partition_false_negative"] == 4
