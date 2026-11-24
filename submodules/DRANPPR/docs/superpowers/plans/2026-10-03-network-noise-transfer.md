# Network Noise Transfer and Forecast Audit

> **For agentic workers:** Use executing-plans, inline. No delegation, commit,
> push, production change, or new adaptive policy in this diagnostic.

**Goal:** Test the frozen five-component calibration recipe on D1 and D4,
including observations not used to produce the predicting state.

**Architecture:** Reuse native joint LM and physical Jacobian export. Fit the
same joint-R, joint-QR and split-QR fixed-linear REML models once at the first
connected common past. Keep nominal prefix factors unchanged. Compare against
nominal at 30, 60, 90 and 120 seconds; forecast known-object observations in the
next five seconds after the first three checkpoints using only odometry, never
assimilating observations inside the forecast block. The checkpoint states are
batch-refined past-only diagnostics, NOT a new online estimator.

**Tech Stack:** Existing C++/GTSAM binaries, NumPy/SciPy sparse linear algebra,
pytest, immutable JSON/CSV manifests.

## Frozen Protocol

- Datasets: D1 and D4; chain 1-2-3-4-5; declared barcode identities.
- Training: first fully connected frame, same causal rule as D2. If D1 does
  not connect by 30 seconds, report the unmet gate rather than move the fit
  window according to quality. One 120-second cold replay may discover the
  event time, but a separate prefix-only replay must produce calibration input.
- Model: existing five Qx/Qy/Qyaw/Rbearing/Rrange multipliers shared across
  robots; same zero log initialization and [-16,10] numerical bounds.
- Arms: nominal, joint_r, joint_qr, split_qr. No Q-only extension in this stage.
- No injected noise, oracle noise variance, new prior, range term, covariance
  floor, robust kernel, fit gate or per-dataset parameter sweep.
- Same nominal state initialization per checkpoint across all arms. GN/LM
  endpoint agreement at 120 seconds; 100 iteration cap, 600-second watchdog.
- Predictions: plug-in Gaussian covariance from native physical factors, one
  fixed global gauge, full pose/object cross-covariance, conditional on the
  fitted scales. Integrate future process covariance; score only objects known
  to the predicting robot at checkpoint. Record excluded-new-object counts.
- Predictions are marginal two-dimensional polar NLL/energy, correlated across
  time and objects. Do not treat their sum as joint evidence or independent
  samples. Five-second forecasts are not per-frame online runtime benchmarks.
- Ground truth only in a separate evaluation phase after all solves/predictions.
- Preserve unsuccessful/negative outputs. Production DBA-DRANPP stays clean.

## Tasks

- [x] Add failing tests in `tests/test_network_noise_transfer.py`: sparse/dense
  native-Jacobian parity, physical polar Jacobian parity, propagation covariance,
  forecast independence from future measurements, exact-support truncation.
- [x] Implement `experiments/network_noise_prediction.py`: sparse assembly,
  selected covariance solve with residual verification, SE2 propagation and
  marginal polar forecast. No state fitting and no truth import.
- [x] Implement `experiments/audit_network_noise_transfer.py`: bounded sequential
  runner, reuse fitting and factor weighting, record all commands/hashes and
  elapsed times, evaluate only after producer processes terminate.
- [x] Run focused tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q
  tests/test_network_noise_transfer.py tests/test_network_noise_support.py`.
- [x] Execute the frozen matrix; compare terminal GN/LM states and costs,
  prediction normalization and state quality without selecting new parameters.
- [x] Run regression tests and archive source, binaries, hashes, report and
  verification. Write `docs/dranppr_network_noise_transfer_report_20261003.md`.

## Interpretation Gate

Improved state error alone does not establish calibration. Compare prediction
NLL and 95% ellipse coverage (nominal chi-square-2 threshold 5.991464547), inspect
per-robot and per-window heterogeneity, and disclose conditional uncertainty.
Do not claim universal adaptation, outlier robustness, equal-byte speedup or
independent-pipeline superiority from this experiment.

## Execution Notes

- The original normal-equation covariance check stopped at D1/t90/joint-R.
  The attempt is retained with `ABORTED.md`. A failing two-column regression
  exposed normal-equation covariance precision loss. Replaced this audit-only
  covariance computation with dense square-root QR of the native sparse
  Jacobian. No estimator, calibration setting, forecast time or noise floor
  changed. The complete matrix was rerun in a separate `_qr_` archive.
- Completed 59 producer subprocesses, 32 checkpoint LM solves, eight terminal
  GN controls and 24 forecasts. All six fitted models are interior and have
  full component rank. Such rank is not a physical-calibration certificate.
- Both datasets improve terminal and checkpoint state metrics and mean future
  marginal NLL. D1/R5 and D4/R2 worsen NLL, and coverage is heterogeneous.
- Added a separately labelled post-selected mean/covariance score-swap and
  robot/window moment analysis. This runs no estimator or fitting procedure.
- Fresh native build, 62 C++ tests and 147 Python tests pass. Existing exact
  support/prefix truncation tests are included in the regression suite.
- Worktree remains on the research branch; no production change, commit or push.
