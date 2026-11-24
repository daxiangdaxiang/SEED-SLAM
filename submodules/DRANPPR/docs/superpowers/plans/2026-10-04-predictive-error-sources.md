# Predictive Error Source Audit

> **For agentic workers:** Use executing-plans inline. Keep the research
> branch; no subagents, commit, push, production change or new estimator gate.

**Goal:** Explain the previously observed D1/R5 and D4/R2 forecast failures
without selecting new noise parameters or confusing state error with noise.

**Architecture:** Reuse immutable D1/D4 nominal and joint-QR checkpoint states
and forecasts. First compute GT-free repeated co-observation distance evidence;
then, in a separately labelled posthoc phase, substitute reference poses/maps
and telescope the prediction error into reference-measurement, map, motion and
checkpoint-state terms. Retain interaction/cross terms and ordering caveats.

**Tech Stack:** Existing MRCLAM reference loader and NumPy; pytest. No solver
replay or dense covariance recomputation is needed.

## Protocol

- Source: `results/dranppr_network_noise_transfer_qr_20261003`, hash checked.
- D1/D4, all five robots, nominal and joint-QR, windows (30,35], (60,65],
  (90,95], exactly the same 533 observation identities as before.
- No model fit, new floor, robust loss, parameter sweep or online correction.
- Pose-free statistic: distance between two simultaneously observed static
  objects, minus its last previous value on that robot. All past observations
  can supply the reference; current pairs require both observations in the
  original forecast support. Report support and empty cells explicitly.
- First-order pair variance uses the existing nominal R or frozen learned R;
  past prefix observations retain nominal R, matching the earlier protocol.
  Repeated pairs/frames are correlated; no calibrated p-values are claimed.
- Reference-assisted diagnosis runs only after the GT-free evidence has been
  computed and saved. Use each checkpoint's archived single-global alignment,
  never robot-wise alignment or a future-trajectory alignment.
- Compare estimated pose/map, reference pose only, reference map only, both
  reference. Reference errors include sensor, preprocessing and GT interpolation
  error; they are not pure sensor-noise measurements.
- Reset only each checkpoint's last pose to its reference, then propagate the
  SAME arrived odometry. This distinguishes initial state error from subsequent
  motion error without changing an estimator or generating a scored solution.
- Telescoping order is reference measurement -> map -> motion -> checkpoint
  state. Also report the reversed pose/map attribution to expose interaction.
  Component variances do not add; retain cross moments and reconstruction error.
- Descriptive per-robot/window/source analysis only. No significance claim,
  selection of a new default, or claim that an oracle replacement is deployable.

## Tasks

- [x] Add failing tests in `tests/test_predictive_error_sources.py` for exact
  decomposition, circular bearing differences, frame invariance, finite-
  difference distance variance, chronology/support, and deterministic fixtures.
- [x] Implement `experiments/audit_predictive_error_sources.py` using
  `polar_model`, `propagate`, existing reference/CSV readers and archived states.
  Verify original forecast means and innovations before interpreting diagnosis.
- [x] Execute one bounded audit with the fixed source and preserve every
  prediction/component and all GT-free pair evidence in a new result directory.
- [x] Inspect all robots/windows, particularly the two preidentified failures;
  distinguish confirmed observations from inferred causes and unresolved ones.
- [x] Run focused plus previous regressions; record commands/hashes and report
  in `docs/dranppr_predictive_error_sources_report_20261004.md`.

## Execution Notes

- Completed one 13.81-second audit: 533 distinct observations, 1066 arm rows,
  and 175 distinct repeated-pair comparisons. No estimator or parameter change.
- Added post-selected subject/pair-tail stratification with an additional
  red-before-green test. Preserve the original source snapshot separately.
- Exact raw-row parity checked for all 533 measurements, confirming that this
  causal stream does not use static benchmark median preprocessing.
- Final-code replay took 13.92 seconds and exactly reproduced all four
  result JSONs, including the post-selected stratification.
- Focused tests: 17; regression tests: 62 native and 156 Python. No new PCG
  optimization and no MPI ranks. The D2 high-noise negative result remains open.

## Falsifiable Questions

1. Does replacing the pose or map remove the observed mean error? If not,
   reference-consistent measurement error remains a competing explanation.
2. Does error remain after resetting the checkpoint pose but keeping odometry?
   If so, unfinished initial optimization cannot by itself explain it.
3. Do state-free distances fluctuate more than the frozen R model predicts?
   If so, enlarging state covariance alone cannot explain all evidence.
4. Are persistent terms cancelling? If so, lower total residual does not
   establish correctness of the separate model components.
