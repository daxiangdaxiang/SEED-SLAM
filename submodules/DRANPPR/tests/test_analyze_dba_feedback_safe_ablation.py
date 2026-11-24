import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_dba_feedback_safe_ablation import (
    all_pairs_precision_recall_f1,
    bcubed_precision_recall_f1,
    directed_precision_recall_f1,
    duplicate_robot_component_count,
    main,
    shared_transform_error,
)


def _rows(edges):
    return tuple(
        {
            "receiver_robot": first[0],
            "receiver_object": first[1],
            "sender_robot": second[0],
            "sender_object": second[1],
            "weight": 1.0,
        }
        for first, second in edges
    )


def _write_csv(path: Path, rows) -> None:
    rows = tuple(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_partition_metrics_distinguish_edges_transitivity_and_conflicts():
    truth = _rows(
        (
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 1), (1, 1)),
        )
    )
    estimate = _rows(
        (
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 0), (1, 2)),
        )
    )
    support = tuple((robot, object_id) for robot in range(3) for object_id in range(3))

    directed = directed_precision_recall_f1(estimate, truth)
    all_pairs = all_pairs_precision_recall_f1(estimate, truth, support)
    bcubed = bcubed_precision_recall_f1(estimate, truth, support)

    assert directed == {
        "precision": pytest.approx(2.0 / 3.0),
        "recall": pytest.approx(2.0 / 3.0),
        "f1": pytest.approx(2.0 / 3.0),
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert all_pairs["true_positive"] == 3
    assert all_pairs["false_positive"] == 2
    assert all_pairs["false_negative"] == 1
    assert bcubed["precision"] < 1.0
    assert bcubed["recall"] < 1.0
    assert duplicate_robot_component_count(estimate, support) == 1


def test_shared_transform_error_removes_one_common_frame_offset():
    truth = {
        0: (0.0, (0.0, 0.0)),
        1: (0.2, (2.0, 1.0)),
        2: (-0.1, (-1.0, 3.0)),
    }
    estimate = {
        robot: (yaw + 0.3, (translation[0] + 4.0, translation[1] - 2.0))
        for robot, (yaw, translation) in truth.items()
    }

    error = shared_transform_error(estimate, truth, root_robot=0)

    assert error["translation_rmse"] > 0.0
    assert error["rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-12)

    # A true left-composed SE(2) gauge transform should align exactly.
    import numpy as np

    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    gauge = np.array([4.0, -2.0])
    gauged = {
        robot: (
            yaw + angle,
            tuple(rotation @ np.asarray(translation) + gauge),
        )
        for robot, (yaw, translation) in truth.items()
    }
    exact = shared_transform_error(gauged, truth, root_robot=0)
    assert exact["translation_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert exact["rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-12)


def test_analyzer_emits_case_aggregate_and_claim_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_root = tmp_path / "results"
    case_dir = result_root / "05/teaserpp/heldout_network"
    truth_path = tmp_path / "truth.csv"
    topology_path = tmp_path / "topology.csv"
    estimate_path = case_dir / "final.csv"
    rows = _rows((((0, 0), (1, 0)),))
    _write_csv(truth_path, rows)
    _write_csv(estimate_path, rows)
    _write_csv(
        topology_path,
        ({"first_robot": 0, "second_robot": 1},),
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "partition_fixed_point",
                "final_association": str(estimate_path.relative_to(tmp_path)),
                "total_communication_mb": 1.5,
                "feedback_variants": ["heldout_network"],
            }
        ),
        encoding="utf-8",
    )
    (result_root / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "05/teaserpp/heldout_network",
                        "dataset": "05",
                        "initializer": "teaserpp",
                        "variant": "heldout_network",
                        "output_dir": str(case_dir),
                        "ground_truth": str(truth_path),
                        "topology": str(topology_path),
                        "support": [[0, 0], [1, 0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_dba_feedback_safe_ablation.py",
            "--result-root",
            str(result_root),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.chdir(tmp_path)

    main()

    assert (output_dir / "case_results.csv").is_file()
    assert (output_dir / "aggregate_results.csv").is_file()
    assert (output_dir / "claim_evidence.csv").is_file()
    assert "heldout_network" in (output_dir / "summary.md").read_text()
