# Error-Calibrated Response Prediction Implementation Plan

> **For agentic workers:** Use executing-plans inline. The user approved implementation and validation on 2026-11-24. No delegation, commit, push, or production-default change.

**Goal:** Test whether packet-driven first-order prediction of neighbor gradient contributions is cheaper and more effective than the existing history-polynomial predictor.

**Architecture:** In the fixed chart, retain actual neighbor state separately from its gradient contribution `z=B_ij*x_j`. Fit one unanchored linear slope to the last eight actual timestamped contributions, only when a message arrives; anchor forecasts at the latest actual contribution. Before ingesting each new observation, compare its increment with the old slope prediction at the SAME source timestamp. Maintain two discounted scalar sufficient statistics for a clipped least-squares forecast gain in the receiver's existing inverse-majorizer metric. Keep the original delay-controlled optimizer unchanged.

**Tech Stack:** Existing NumPy/SciPy actor simulator and pytest. No dependency or native C++ changes.

## Frozen Design And Gates

- `response_linear`: packet-cached slope, unit forecast gain.
- `response_calibrated`: same slope; forecast gain `clip(S_pd/S_pp,0,1)`; hold until the first nonzero validated forecast. Before assimilation, p is old velocity and d is observed increment divided by source-time interval. Update S with discount `(size-1)/(size+1)`, using the same size=8 as the slope window. This avoids a separately tuned forgetting constant.
- Calibration is a retrospective regression on realized errors, NOT calibrated covariance, a future error bound, or a new convergence theorem. Irregular horizons are normalized under a local constant-velocity assumption. Long-horizon model bias remains an explicit stress-test target.
- Accepted observations only; no predicted sample enters history. `reset` invalidates all learned statistics. Actual cache and zero-delay behavior must remain unchanged.
- Cache B*x for hold too, avoiding an unfair implementation-speed comparison. Old predictor paths must reproduce their existing traces.
- Budget: one planned candidate plus the uncalibrated ablation. No dataset-specific tuning. A failed result is reportable; do not search indefinitely for a win.

## Tasks

- [x] Add failing tests in `tests/test_response_prediction.py`: irregular linear reproduction; causal pre-assimilation error; clipped gain formula; no fitting on predict; translation/scale invariance; reset and stale packets; zero-delay and prefix-budget parity.
- [x] Implement `ResponsePrediction` beside the existing predictor in `experiments/neighbor_prediction.py`; integrate optional response predictors and shared contribution cache in `experiments/async_interface.py`. Expose both names in `audit_neighbor_prediction.py`; snapshot new tests and plan in the audit driver.
- [x] Run focused tests, then D1/D4/SE3_31, three nonideal scenarios, seed31, 600 ticks. Record all outcomes without tuning to endpoints.
- [x] Run nine models, five networks, three seeds, 2000 ticks with hold/linear/calibrated. Require exact archived-hold parity and every-prefix byte/update equality. Compare against sealed prior secant/poly2/risk3 results without rerunning or altering them.
- [x] Add replay/synthetic tests that distinguish noise suppression from abrupt unobservable changes, and stress long delay/outage plus strong coupling. Include both failure and recovery intervals.
- [x] Benchmark all methods on the same machine with shuffled-order repetitions, recording service time and identical-target cost. Do not call synthetic ticks wall-clock time or claim native robot runtime.
- [x] Independently recompute objectives and inexact-gradient identities; verify source/input SHA ledgers and focused tests. Document scope, regressions, causal limitations, and whether the actual adaptive component improves on its ablation.

Commands: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_response_prediction.py tests/test_neighbor_prediction.py tests/test_async_interface.py tests/test_async_interface_audit.py tests/test_review_async_interface.py`.

Primary command: `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_neighbor_prediction --output results/dranppr_response_final_20261124 --predictions hold response_linear response_calibrated --ticks 2000 --parity-root results/dranppr_async_directional_20261123`.

Success requires reduced measured overhead plus useful matched-budget objective progress, not merely better prediction MSE. Universal superiority, native nonlinear SE3 performance, and riMESA superiority are out of scope for this fixed-model experiment.

Outcome: cheaper than risk3 but not an overall optimizer improvement. See
`docs/dranppr_response_prediction_report_20261124.md`. Added nondeployable
exact-neighbor/fresh-metric diagnostics to isolate the D1 bottleneck, without
changing the candidate. Primary: 405 runs; controls: 40 stress runs, 18 oracle
runs, 90 fixed-horizon and 45 same-accuracy timed runs; 86 focused tests.
