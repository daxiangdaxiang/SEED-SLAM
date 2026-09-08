"""Pinned generation parameters for the synthetic Pose3 extension."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class MRCLAMSE3ConfigError(ValueError):
    """Raised when a derived benchmark manifest is incomplete or unsafe."""


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MRCLAMSE3ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MRCLAMSE3ConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MRCLAMSE3ConfigError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class GenerationConfig:
    dataset_number: int
    generator_seed: int
    z_amplitude_m: float
    roll_amplitude_deg: float
    pitch_amplitude_deg: float
    profile_cycles: float
    object_height_min_m: float
    object_height_max_m: float
    object_rotation_sigma_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_number", _integer(self.dataset_number, "dataset number", minimum=1)
        )
        object.__setattr__(
            self, "generator_seed", _integer(self.generator_seed, "generator seed", minimum=0)
        )
        for field, label in (
            ("z_amplitude_m", "z amplitude"),
            ("roll_amplitude_deg", "roll amplitude"),
            ("pitch_amplitude_deg", "pitch amplitude"),
            ("profile_cycles", "profile cycles"),
            ("object_height_min_m", "minimum object height"),
            ("object_height_max_m", "maximum object height"),
            ("object_rotation_sigma_rad", "object rotation sigma"),
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), label))
        if self.object_height_max_m <= self.object_height_min_m:
            raise MRCLAMSE3ConfigError(
                "maximum object height must exceed minimum object height"
            )

    @staticmethod
    def moderate_v1(dataset_number: int) -> "GenerationConfig":
        return GenerationConfig(
            dataset_number=dataset_number,
            generator_seed=731021,
            z_amplitude_m=0.6,
            roll_amplitude_deg=6.0,
            pitch_amplitude_deg=7.0,
            profile_cycles=2.0,
            object_height_min_m=0.4,
            object_height_max_m=1.6,
            object_rotation_sigma_rad=0.05,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_number": self.dataset_number,
            "generator_seed": self.generator_seed,
            "z_amplitude_m": self.z_amplitude_m,
            "roll_amplitude_deg": self.roll_amplitude_deg,
            "pitch_amplitude_deg": self.pitch_amplitude_deg,
            "profile_cycles": self.profile_cycles,
            "object_height_min_m": self.object_height_min_m,
            "object_height_max_m": self.object_height_max_m,
            "object_rotation_sigma_rad": self.object_rotation_sigma_rad,
        }

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> "GenerationConfig":
        if not isinstance(value, Mapping):
            raise MRCLAMSE3ConfigError("generation config must be an object")
        expected = {
            "dataset_number",
            "generator_seed",
            "z_amplitude_m",
            "roll_amplitude_deg",
            "pitch_amplitude_deg",
            "profile_cycles",
            "object_height_min_m",
            "object_height_max_m",
            "object_rotation_sigma_rad",
        }
        if set(value) != expected:
            raise MRCLAMSE3ConfigError(
                f"generation config fields must equal {sorted(expected)}"
            )
        return GenerationConfig(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class NonplanarPose:
    z_m: float
    roll_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class GenerationManifest:
    path: Path
    benchmark_id: str
    source_manifest: Path
    config: GenerationConfig
    object_identity_corrections: Mapping[int, int]
    object_identity_correction_basis: str
    repeats: tuple[int, ...]
    checkpoints: tuple[int, ...]


def nonplanar_profile(
    config: GenerationConfig, robot_subject: int, u: float
) -> NonplanarPose:
    robot_subject = _integer(robot_subject, "robot subject", minimum=1)
    u = float(u)
    if not math.isfinite(u) or u < 0.0 or u > 1.0:
        raise MRCLAMSE3ConfigError("normalized time must be finite and in [0, 1]")
    phase = 0.37 * config.dataset_number + 0.73 * robot_subject

    def zero_started(amplitude: float, cycles: float, offset: float) -> float:
        angle = phase + offset
        return 0.5 * amplitude * (
            math.sin(2.0 * math.pi * cycles * u + angle) - math.sin(angle)
        )

    return NonplanarPose(
        z_m=zero_started(config.z_amplitude_m, config.profile_cycles, 0.0),
        roll_rad=zero_started(
            math.radians(config.roll_amplitude_deg),
            config.profile_cycles + 0.5,
            1.1,
        ),
        pitch_rad=zero_started(
            math.radians(config.pitch_amplitude_deg),
            config.profile_cycles - 0.5,
            2.2,
        ),
    )


def _integer_tuple(value: Any, name: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MRCLAMSE3ConfigError(f"{name} must be a non-empty list")
    result = tuple(_integer(item, name, minimum=minimum) for item in value)
    if tuple(sorted(set(result))) != result:
        raise MRCLAMSE3ConfigError(f"{name} must be sorted and unique")
    return result


def normalize_object_identity_corrections(value: Any) -> dict[int, int]:
    """Validate an explicit, collision-free reported-to-physical permutation."""

    if not isinstance(value, Mapping):
        raise MRCLAMSE3ConfigError("object identity corrections must be an object")
    result: dict[int, int] = {}
    for reported_value, physical_value in value.items():
        try:
            reported = int(reported_value)
        except (TypeError, ValueError) as error:
            raise MRCLAMSE3ConfigError(
                "reported object identities must be positive integers"
            ) from error
        if isinstance(reported_value, bool) or reported < 1:
            raise MRCLAMSE3ConfigError(
                "reported object identities must be positive integers"
            )
        if isinstance(reported_value, str) and str(reported) != reported_value:
            raise MRCLAMSE3ConfigError(
                "reported object identities must use canonical integer strings"
            )
        physical = _integer(
            physical_value, "physical object identity", minimum=1
        )
        if reported == physical:
            raise MRCLAMSE3ConfigError(
                "object identity corrections must not contain no-op entries"
            )
        result[reported] = physical
    if set(result) != set(result.values()):
        raise MRCLAMSE3ConfigError(
            "object identity corrections must form a permutation"
        )
    return dict(sorted(result.items()))


def load_generation_manifest(path: str | Path) -> GenerationManifest:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MRCLAMSE3ConfigError(f"cannot read generation manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise MRCLAMSE3ConfigError("generation manifest must be an object")
    if value.get("schema_version") != "mrclam_synthetic_se3_manifest_v1":
        raise MRCLAMSE3ConfigError("unsupported generation manifest schema")
    benchmark_id = value.get("benchmark_id")
    source_value = value.get("source_manifest")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise MRCLAMSE3ConfigError("benchmark id must be a non-empty string")
    if not isinstance(source_value, str) or not source_value:
        raise MRCLAMSE3ConfigError("source manifest must be a non-empty path")
    source_manifest = (path.parent / source_value).resolve()
    if not source_manifest.is_file():
        raise MRCLAMSE3ConfigError(f"source manifest does not exist: {source_manifest}")
    config = GenerationConfig.from_dict(value.get("generation"))
    object_identity_corrections = normalize_object_identity_corrections(
        value.get("object_identity_corrections")
    )
    correction_basis = value.get("object_identity_correction_basis")
    if not isinstance(correction_basis, str) or not correction_basis.strip():
        raise MRCLAMSE3ConfigError(
            "object identity correction basis must be a non-empty string"
        )
    if bool(object_identity_corrections) == (correction_basis == "none"):
        raise MRCLAMSE3ConfigError(
            "object identity correction basis must agree with the correction map"
        )
    expected_id = (
        f"mrclam_dataset{config.dataset_number}_synthetic_se3_objectpose3_v2"
    )
    if benchmark_id != expected_id:
        raise MRCLAMSE3ConfigError(f"benchmark id must equal {expected_id}")
    return GenerationManifest(
        path=path,
        benchmark_id=benchmark_id,
        source_manifest=source_manifest,
        config=config,
        object_identity_corrections=object_identity_corrections,
        object_identity_correction_basis=correction_basis,
        repeats=_integer_tuple(value.get("repeats"), "repeats", minimum=0),
        checkpoints=_integer_tuple(value.get("checkpoints"), "checkpoints", minimum=1),
    )


__all__ = [
    "GenerationConfig",
    "GenerationManifest",
    "MRCLAMSE3ConfigError",
    "NonplanarPose",
    "load_generation_manifest",
    "normalize_object_identity_corrections",
    "nonplanar_profile",
]
