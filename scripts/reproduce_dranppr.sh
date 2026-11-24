#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/submodules/DRANPPR"
out="${1:-results/release}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m experiments.audit_model_response --root fixtures/interface_models --output "$out/stress" --stress
python3 -m experiments.review_model_response --root "$out/stress" --output "$out/stress_review"
python3 -m experiments.audit_model_response --root fixtures/interface_models --output "$out/controls" \
  --scenarios ideal delay heterogeneous loss outage --seeds 35
python3 -m experiments.review_model_response --root "$out/controls" --output "$out/controls_review"
