#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/submodules/DRANPPR"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m pytest -q \
  tests/test_quadratic_response.py tests/test_prediction_error_budget.py \
  tests/test_review_error_budget.py tests/test_response_prediction.py \
  tests/test_neighbor_prediction.py tests/test_history_forecast_controls.py \
  tests/test_async_interface.py tests/test_async_interface_audit.py \
  tests/test_review_async_interface.py
