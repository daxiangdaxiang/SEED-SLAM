# Range Prediction Attribution Implementation Plan

> Execute inline with executing-plans in the existing isolated research
> worktree. The user approved continuation of the prior report's range
> mean/covariance attribution question. No agents, commits, pushes, production
> changes, new estimator parameters, or GT input. The later user-requested
> blind-spot extension adds only the explicitly declared 48 LM / 48 GN solves.

**Goal:** Explain which parts of the existing adaptive prediction account for
range-score regressions and whether harm is concentrated by object/frame.

**Architecture:** Reuse checksummed nominal and joint-QR forecasts from D1/D4.
Evaluate matched Gaussian distributions with archived means/covariances swapped,
then aggregate every robot, checkpoint, declared object, and frame. Separate
measurement covariance R from U = total covariance - R, where U includes BOTH
state/map and propagated process uncertainty. No corrective model is fitted.

**Tech Stack:** Existing `split_score`, `verify_archive`, JSON, NumPy, pytest.

## Frozen Design And Interpretation

- Source: `results/dranppr_network_noise_transfer_qr_20261003` (360 hashes).
  Verification source: prior motion audit (29 hashes). Do not change either.
- D1/D4, robots 1--5, original 30/60/90 s checkpoints and five-second forecast
  windows, all 533 observations. No subgroup selected to improve a score.
- Require exact keys `(robot, source_row, subject, timestamp)` and matching
  reconstructed observations. Compare bearing modulo 2*pi, not raw addition.
  Refuse duplicate/missing support or inputs labelled `truth_used=true`.
- Confirm 2D original scores and range-marginal group scores against archives.
- Evaluate the frozen 2x2x2 grid: mean m from nominal/adapted, U from
  nominal/adapted, R from nominal/adapted. Each covariance is `U + R`.
  Check symmetric positive definite totals/R and positive semidefinite U;
  no damping, flooring or repair. Store eight scores per observation, 4,264
  hypothetical scores on 533 measurements, not independent experiments.
- Primary 2x2 comparison: original, new mean/old covariance, old mean/new
  covariance, fully adapted. Report BOTH conditional paths and the interaction:
  `L11-L10-L01+L00`. Do not identify these as causal effects or deployable arms.
- Preserve full covariance coupling and both Gaussian conditioning orders.
  Range marginal is the primary attribution target, with joint and
  range-given-bearing scores retained for checks.
- Report mean predicted range variance components U and R separately. Shrinkage
  of U is not evidence that Q alone was estimated too small: it also changes
  with checkpoint covariance, observation weights, geometry and linearization.
- Aggregate dataset, robot, robot/checkpoint, robot/subject,
  robot/checkpoint/subject. Store every frame. Report observation-weighted and
  frame-equal means, counts, and signed innovation mean/RMSE/centred SD.
- Positive harm mass is `sum(max(adapted_range_NLL - nominal_range_NLL,0))`;
  negative mass is reported separately. No division by a near-zero net change,
  data deletion, independence assumption, p-value, or fitted bias correction.
- Available alternatives: rerun new noise models (premature) or repeat broad
  sweeps (confounded). Choose this no-fit attribution because it directly
  addresses the newly isolated range failure at negligible compute cost.

## Expanded Blind-Spot Audit Requested By The User

Cover all identified categories, but distinguish verified invariants from
unresolved scientific assumptions. Add range standardized energy, nominal 95%
marginal coverage (chi-square, 1 DOF; diagnostic only), covariance correlation,
and the log-volume/normalized-error terms of range NLL. Check every recorded
checkpoint's solver termination, gradient and covariance linear solve residual;
calibration rank/bounds/score/Fisher spectrum. These checks cannot certify the
nonlinear posterior or global optimum. Report time-window and frame-equal
counterexamples, not just all-observation averages.

The report must retain open risks: finite estimated noise-parameter uncertainty,
model misspecification/nonlinearity, unseen subjects and unsupported forecasts,
short held-out coverage, D2 high-noise negative result, lack of fresh matched
riMESA comparison, and batch diagnostic versus genuine distributed online
execution. Review prior archived tests for evidence, but never label these gaps
closed merely because another unit test passes. Do not add arbitrary inflation,
object biases, gates, or robust kernels to hide an unresolved issue.

## Controlled Re-Solve Extension

The covariance swaps exposed an attribution ambiguity: U contraction is not
equivalent to changing Q or R_range. Resolve that with actual fixed-weight
factorials, not just synthetic distributions. Freeze the original learned
five-vector and toggle three groups: Q_xyz, R_bearing, R_range. Each is either
the original nominal value or its existing learned value. Eight arms, D1/D4,
30/60/90 s: 48 LM checkpoint/forecast runs, then 48 same-initialization GN
controls. No refit, no new values, same complete graph, prefix nominal weights,
same initial states and cap 100. This is a mechanism experiment, not optimized
arm selection or an equal-compute distributed baseline comparison.

Create `experiments/audit_frozen_weight_factorial.py` and tests for the exact
weight grid and endpoint parity. Reuse the existing block-sensitivity
`predict_worker`; store scratch linearizations in `/tmp`. Verify binary and all
11 source library hashes. Record every command, exit status, runtime, gradients,
LM/GN state differences and covariance solve residual. Stop on a failed worker;
180-second worker timeout and 900-second producer budget prevent unbounded work.
Both all-nominal and all-adapted arms must reproduce old predictions within
1e-6, with exact differences reported rather than rounded away.

- [x] Test the frozen grid and parity rejection, then run all 48 predictions and
  same-start GN controls. Keep every arm; do not tune against held-out scores.
- [x] Archive compact inputs/states/forecasts, hashes of scratch Jacobians and
  code. Explain which ambiguity this closes and which statistical/model or
  deployment assumptions still require new evidence.

## Tasks

### 1. Tests First

Create `tests/test_range_prediction_attribution.py` for:

- `paired_prediction`: same observed measurement with different means; all
  eight covariance combinations preserve chain-rule score identities; original
  endpoints match stored scores; bearing wrapping is respected.
- Refuse duplicate/missing observations, changed values, invalid covariance
  decomposition, and truth-assisted forecasts. Inputs remain unchanged.
- `summarize_rows`: unequal object counts per frame must distinguish row and
  frame weighting; positive/negative masses reconcile the total score change.
- Mean/covariance interaction identity, including an independent scalar
  closed-form check of range scores.
- Full archived-fixture audit: 533 matched observations, 10 robot groups,
  386 frames, 8 hypothetical distributions per observation; exact endpoint
  reproduction and all decomposition identities.

- [x] Write tests and observe red failures for missing implementation.

### 2. Minimal Auditor

Create `experiments/audit_range_prediction_attribution.py`, exposing
`matched_predictions`, `paired_prediction`, `summarize_rows`, and `run`.
CLI flags: `--source`, `--previous`, `--output`. Refuse existing output.
Write manifest first, rows/groups/frame diagnostics second, completion last.
Reuse `split_score` instead of implementing another Gaussian likelihood.

- [x] Implement the tested audit and run focused tests to green.
- [x] Execute audit into `results/dranppr_range_prediction_attribution_20261005`;
  the source-support extension is in the separate `verified` output directory.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.audit_range_prediction_attribution \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --previous results/dranppr_motion_time_structure_verified_20261004 \
  --output results/dranppr_range_prediction_attribution_20261005
```

### 3. Analyze And Verify

- [x] Inspect ALL groups; explain D1/R5 and D4/R2 without ignoring improvements.
  Differentiate concentrated mean failures, widespread covariance shrinkage,
  and mean/covariance interaction. No new sensor/model claims from score swaps.
- [x] Write `docs/dranppr_range_prediction_attribution_report_20261005.md` and
  `docs/dranppr_blind_spot_register_20261005.md`.
- [x] Run previous 62 native / 171 Python regressions plus new tests. Store full
  commands, JUnit XML, logs, source snapshots and hashes. Repeat outputs exactly.
- [x] Read back archive SHA256SUMS; check production remains clean. No PCG/MPI
  research runs. Keep existing D2 high-noise negative result explicit.

Focused command:

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q tests/test_range_prediction_attribution.py \
  tests/test_motion_time_structure.py tests/test_network_noise_transfer.py
```
