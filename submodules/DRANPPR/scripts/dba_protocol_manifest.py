#!/usr/bin/env python3
"""Dataset-only manifests for reproducible DBA protocols."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    metadata: Path
    lower_topology: Path | None
    upper_topology: Path
    ground_truth: Path
    initial_association: Path
    candidate_support: Path | None
    num_robots: int
    num_objects: int
    lower_iterations: int
    plane_axes: str


def load_dataset_manifest(
    path: Path, *, lower_iterations: int
) -> DatasetConfig:
    """Load dataset facts while keeping algorithm settings out of manifests."""

    manifest = Path(path).resolve()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dataset manifest must be a JSON object")

    forbidden = {
        key
        for key in value
        if key == "lower_iterations" or key.startswith("association_")
    }
    if forbidden:
        raise ValueError(
            "dataset manifests cannot contain algorithm parameters: "
            + ", ".join(sorted(forbidden))
        )

    allowed = {
        "schema_version",
        "name",
        "data_dir",
        "metadata",
        "lower_topology",
        "upper_topology",
        "ground_truth",
        "initial_association",
        "candidate_support",
        "num_robots",
        "num_objects",
        "plane_axes",
    }
    required = allowed - {"lower_topology", "candidate_support"}
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(
            "invalid dataset manifest keys: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if value["schema_version"] != 1:
        raise ValueError("unsupported dataset manifest schema")
    if lower_iterations <= 0:
        raise ValueError("lower iterations must be positive")

    num_robots = int(value["num_robots"])
    num_objects = int(value["num_objects"])
    if num_robots <= 0 or num_objects <= 0:
        raise ValueError("robot and object counts must be positive")
    if value["plane_axes"] not in ("xy", "xz", "yz"):
        raise ValueError("plane axes must be xy, xz, or yz")

    def resolve(raw: object) -> Path:
        candidate = Path(str(raw))
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (manifest.parent / candidate).resolve()
        )

    lower_topology = (
        None
        if value.get("lower_topology") is None
        else resolve(value["lower_topology"])
    )
    candidate_support = (
        None
        if value.get("candidate_support") is None
        else resolve(value["candidate_support"])
    )
    config = DatasetConfig(
        name=str(value["name"]),
        data_dir=resolve(value["data_dir"]),
        metadata=resolve(value["metadata"]),
        lower_topology=lower_topology,
        upper_topology=resolve(value["upper_topology"]),
        ground_truth=resolve(value["ground_truth"]),
        initial_association=resolve(value["initial_association"]),
        candidate_support=candidate_support,
        num_robots=num_robots,
        num_objects=num_objects,
        lower_iterations=int(lower_iterations),
        plane_axes=str(value["plane_axes"]),
    )
    paths = (
        config.data_dir,
        config.metadata,
        config.upper_topology,
        config.ground_truth,
        config.initial_association,
    ) + (() if config.lower_topology is None else (config.lower_topology,))
    if config.candidate_support is not None:
        paths += (config.candidate_support,)
    missing_paths = [str(candidate) for candidate in paths if not candidate.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "missing dataset manifest inputs: " + ", ".join(missing_paths)
        )
    return config
