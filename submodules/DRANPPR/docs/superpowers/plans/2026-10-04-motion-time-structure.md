# Motion Error Time Structure Audit

> Execute inline in the existing research worktree; no subagents, production
> changes, commits, pushes, noise fitting, or new estimator mechanisms.

**Goal:** Test the approved next question from the block-sensitivity report:
whether the motion error behind D1/R5 is consistent with the current independent,
zero-mean process-noise model, rather than assuming smooth cumulative drift is
proof of correlated noise.

**Architecture:** First verify archived increments against raw causal velocity
integration without GT. Then use references only in a labelled posthoc audit:
deduplicated forecast-frame motion responses and nonoverlapping, fixed-time yaw
increments over the held-out suffix. No reference-derived correction enters a
state estimator. Use the existing frozen Q values, not newly fitted ones.

**Tech Stack:** Existing exact unicycle increments, pose composition, reference
interpolator and forecast propagation; NumPy; pytest; checksummed JSON artifacts.

## Frozen Design

- Source: `results/dranppr_network_noise_transfer_qr_20261003`; verify all hashes.
- D1 and D4, robots 1--5. Reuse the same joint-QR 30/60/90 s checkpoints and
  observation support; also audit every pose event in those forecast windows.
- Validate raw command hashes. Independently reintegrate each archived pose
  increment through elapsed 120 s, holding the last arrived command. Duplicate
  timestamps use the repository's existing same-timestamp average. If parity
  fails, diagnose the input convention
  before interpreting statistical results; never silently change the stream.
- Reset each forecast trajectory to its reference at that robot's actual last
  checkpoint state timestamp, not the global checkpoint timestamp. Record age
  and actual propagation duration, and keep zero-support robot/windows visible.
- Decompose yaw error into starting-state error and subsequently integrated
  motion error. Compare the latter with the EXISTING yaw process variance
  `.05^2 * q_yaw * elapsed_time`, including stale-start intervals.
- Verify every matched observation's previous bearing-motion attribution.
  Separate bearing change from propagated heading error and translation geometry.
  Do not count multiple objects in one frame as independent motion errors.
- The held-out suffix starts at the first integer elapsed second at/after the
  training cutoff and ends at 120 s: D1 [28,120], D4 [16,120]. All robots use
  the same uniform boundaries. No GT-based segment selection.
- Compute raw-command minus reference yaw increments over 1 s, then sum into
  disjoint 2/5/10 s blocks. Report unused trailing seconds. These durations are
  fixed audit resolutions, not optimized or deployed algorithm parameters.
- Report signed mean rate, total/centred energy, and lag-1/2/5 correlations of
  ONE-second increments. For each horizon report observed squared-error versus
  the fixed white-noise variance. Do not test correlation on cumulative errors.
- For unequal-duration event increments, decompose whitened energy exactly into
  the constant-rate projection and orthogonal scatter; this descriptive rate
  is NOT a fitted bias model or an online correction.
- Reference differencing/interpolation can itself cause temporal dependence;
  no independent p-values, actuator-bias identification, or confidence claim.
  Correlation alone is insufficient: compare its amplitude with prescribed Q.
- No new nonlinear solves, PCG, MPI or performance comparison. Preserve D2's
  high-noise negative result and distinguish this audit from deployable learning.

## Tasks

- [x] Add red tests in `tests/test_motion_time_structure.py`: causal control
  integration/duplicate timestamps/no future influence; exact composition and
  rotation wrapping; unequal-duration energy identity; disjoint horizon blocks;
  no correlation statistic for constant/insufficient data; deduplicated forecast
  state timing; invariant reference-pose reset and exact heading decomposition.
- [x] Implement `experiments/audit_motion_time_structure.py`; reuse local helpers
  rather than writing a new motion engine or changing native factors.
- [x] Run the GT-free source/integration audit, then the fixed reference-assisted
  analysis for all robots. Inspect the preidentified failure and counterexamples.
- [x] Run prior regression suites plus focused tests, archive raw hashes, code,
  detailed rows and report at
  `docs/dranppr_motion_time_structure_report_20261004.md`.

Focused command:

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q tests/test_motion_time_structure.py \
  tests/test_predictive_error_sources.py tests/test_network_noise_transfer.py
```

Prefer direct increment checks to adding an AR/bias model or running a new
parameter sweep. A negative motion explanation is a useful outcome: do not force
this mechanism to explain the entire range/bearing predictive NLL failure.

## Protocol Correction Before Reference Analysis

The initial audit assumed that duplicate raw timestamps use the last row. It
stopped at the GT-free parity check. Two D1/R4 intervals differ under that
convention (maximum yaw difference .00085999918 rad); all D4 intervals match.
Inspection found differing commands at exactly equal timestamps. The repository
already averages duplicates in `_merge_duplicate_odometry`. Reuse that existing
canonicalization, test it explicitly, and rerun to a new `canonical` output.
Do not alter old input streams, numerical tolerances, or noise parameters.

## Posthoc Extension

Motion energy alone cannot explain adaptive-versus-nominal prediction quality.
After the fixed audit, decompose the original 2D Gaussian NLL into marginal and
conditional bearing/range scores, retaining cross covariance and checking BOTH
conditioning orders. This uses the same 533 observations and original two-arm
forecasts, no GT and no fitting. Report all robots, range innovation RMSE, and
total predictive range variance. Label this explicitly posthoc; it is an exact
score identity, not causal channel attribution or independent validation.

Tests first failed on missing `split_score`; implement it in
`experiments/analyze_motion_time_structure.py` and test both conditioning
orders and covariance coupling. The full-fixture regression also runs this
analyzer. Do not conflate 1,066 scored arm rows with independent observations.
