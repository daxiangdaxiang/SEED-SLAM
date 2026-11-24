# Complete Benchmark-Suffix Prediction Audit

> Execution: inline in the existing research worktree; no delegation, commit,
> push or production changes. This implements the next audit approved by the
> user's continuation of the 2026-10-27 report.

**Goal:** Test whether frozen noise calibration's predictive benefit survives
complete temporal coverage of the existing 120-second D1/D4 benchmark.

**Architecture:** Reuse the native causal initializer, central LM and square-root
QR predictor unchanged. Generate disjoint (checkpoint, endpoint] windows on the
existing five-second grid, including the first partial window after the warm
prefix. Replay only the past, solve nominal and frozen joint-QR from identical
past initialization, predict with odometry only, and account for every suffix
observation as scored or unsupported. Do not claim whole raw-sequence coverage.

**Tech stack:** Python/NumPy/SciPy, pinned GTSAM native binaries, pytest.

## Frozen Protocol

- Discovery data: D1/D4, original stream and warm-prefix hashes. Stop at 120 s.
- Start at the actual warm cutoff, then use multiples of 5 s up to 120 s.
- Arms: nominal and the original first-connected-prefix joint-QR five-vector.
  No refit, sweep, mean correction, variance floor, new gate or outlier removal.
- Prefix factors retain nominal weights. LM maximum 100 iterations. Native
  replay is `joint_cached_one --lazy-refresh`, exactly as in the source audit.
- All forecasts exclude future observation assimilation. New observations may
  enter the graph only at the NEXT checkpoint. Known barcode identities remain.
- Score marginal 2D NLL, joint 95% coverage, range/bearing prediction RMSE,
  per-robot/window/30-second-band summaries, and equal-frame/equal-window NLL.
  These are descriptive correlated measurements, not independent trials.
- Unsupported local object copies are counted and distinguished from objects
  absent everywhere. Never infer a likelihood for an unmodelled new object.
- Replay the six old windows per arm as numerical controls. Require unchanged
  support, means/covariance/NLL within 1e-8; archive differences.
- One BLAS thread; 240 s per worker, 180 s per native subprocess, total outer
  watchdog 2400 s. Preserve failures without retuning or selecting checkpoints.
- No GT, state-accuracy or online latency claim in this audit. Raw sequences
  extend well beyond 120 s and remain outside this bounded experiment.

## Tasks

- [x] Add failing tests in `tests/test_rolling_prediction.py`: contiguous
  boundary assignment, invalid schedules, unsupported-copy classification,
  omission/duplicate rejection, and balanced descriptive scoring.
- [x] Implement `experiments/audit_rolling_prediction.py` using the existing
  `prepare`, `apply_scales`, `forecast`, `parity`, `moments`, and hash helpers.
- [x] Run focused tests and existing forecast/native-preparation regressions.
- [x] Execute `python3 -m experiments.audit_rolling_prediction --source
  results/dranppr_network_noise_transfer_qr_20261003 --output
  results/dranppr_rolling_prediction_20261028` with the declared limits.
- [x] Verify complete event accounting, matched arm support, old-window parity,
  source hashes and native dependency hashes. Rerun native/Python tests.
- [x] Archive commands, source snapshots, summaries, report and SHA256SUMS;
  update blind-spot register with scope and negative results retained.

## Self-Review

The audit changes evaluation coverage only. It does not add a new estimator.
The first partial window uses the original nominal training solve for BOTH
state arms but each arm's declared future process/observation noise law.
At later checkpoints both arms use the same nominal causal replay initialization.
Unsupported events and empty robot/window groups remain explicit. An aggregate
is publishable only when all planned workers complete and all observations are
accounted for exactly once. The next dataset confirmation remains a separate
gate; D1/D4 have already been inspected repeatedly.
