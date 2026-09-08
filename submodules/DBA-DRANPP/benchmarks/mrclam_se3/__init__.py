"""MR.CLAM-derived synthetic SE(3)-ObjectPose3 benchmark."""

from .config import (
    GenerationConfig,
    GenerationManifest,
    MRCLAMSE3ConfigError,
    NonplanarPose,
    load_generation_manifest,
    nonplanar_profile,
)

__all__ = [
    "GenerationConfig",
    "GenerationManifest",
    "MRCLAMSE3ConfigError",
    "NonplanarPose",
    "load_generation_manifest",
    "nonplanar_profile",
]
