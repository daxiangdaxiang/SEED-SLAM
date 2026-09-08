#!/usr/bin/env python3

import json
from pathlib import Path

import pytest

from scripts.dba_protocol_manifest import load_dataset_manifest


def _manifest_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    data_dir = tmp_path / "object_pgo"
    data_dir.mkdir()
    for relative in (
        "object_pgo/local_object_metadata.csv",
        "complete_topology.csv",
        "chain_topology.csv",
        "candidate_support.csv",
        "object_pgo/gt_associations.csv",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n", encoding="utf-8")
    value: dict[str, object] = {
        "schema_version": 1,
        "name": "synthetic_seed_0",
        "data_dir": "object_pgo",
        "metadata": "object_pgo/local_object_metadata.csv",
        "lower_topology": "complete_topology.csv",
        "upper_topology": "chain_topology.csv",
        "ground_truth": "object_pgo/gt_associations.csv",
        "initial_association": "object_pgo/gt_associations.csv",
        "candidate_support": "candidate_support.csv",
        "num_robots": 6,
        "num_objects": 80,
        "plane_axes": "xy",
    }
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, value


def test_manifest_resolves_dataset_paths_without_algorithm_parameters(
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest_fixture(tmp_path)

    config = load_dataset_manifest(manifest, lower_iterations=12)

    assert config.name == "synthetic_seed_0"
    assert config.data_dir == (tmp_path / "object_pgo").resolve()
    assert config.metadata == (
        tmp_path / "object_pgo/local_object_metadata.csv"
    ).resolve()
    assert config.lower_topology == (
        tmp_path / "complete_topology.csv"
    ).resolve()
    assert config.upper_topology == (
        tmp_path / "chain_topology.csv"
    ).resolve()
    assert config.ground_truth == (
        tmp_path / "object_pgo/gt_associations.csv"
    ).resolve()
    assert config.initial_association == config.ground_truth
    assert config.candidate_support == (
        tmp_path / "candidate_support.csv"
    ).resolve()
    assert config.num_robots == 6
    assert config.num_objects == 80
    assert config.lower_iterations == 12
    assert config.plane_axes == "xy"


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"schema_version": 2}, "unsupported"),
        ({"num_robots": 0}, "positive"),
        ({"num_objects": -1}, "positive"),
        ({"plane_axes": "zx"}, "plane axes"),
        ({"lower_iterations": 10}, "algorithm parameters"),
        ({"association_temperature": 0.5}, "algorithm parameters"),
        ({"unexpected": True}, "unknown"),
    ),
)
def test_manifest_rejects_invalid_or_algorithmic_fields(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    manifest, value = _manifest_fixture(tmp_path)
    value.update(mutation)
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_dataset_manifest(manifest, lower_iterations=10)


def test_manifest_rejects_missing_required_key(tmp_path: Path) -> None:
    manifest, value = _manifest_fixture(tmp_path)
    del value["ground_truth"]
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        load_dataset_manifest(manifest, lower_iterations=10)


def test_manifest_rejects_missing_input_path(tmp_path: Path) -> None:
    manifest, value = _manifest_fixture(tmp_path)
    value["upper_topology"] = "absent.csv"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="absent.csv"):
        load_dataset_manifest(manifest, lower_iterations=10)


def test_manifest_allows_no_lower_topology(tmp_path: Path) -> None:
    manifest, value = _manifest_fixture(tmp_path)
    value["lower_topology"] = None
    manifest.write_text(json.dumps(value), encoding="utf-8")

    config = load_dataset_manifest(manifest, lower_iterations=10)

    assert config.lower_topology is None


def test_manifest_allows_no_explicit_candidate_support(tmp_path: Path) -> None:
    manifest, value = _manifest_fixture(tmp_path)
    del value["candidate_support"]
    manifest.write_text(json.dumps(value), encoding="utf-8")

    config = load_dataset_manifest(manifest, lower_iterations=10)

    assert config.candidate_support is None


def test_manifest_rejects_nonpositive_runtime_lower_iterations(
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest_fixture(tmp_path)

    with pytest.raises(ValueError, match="lower iterations"):
        load_dataset_manifest(manifest, lower_iterations=0)
