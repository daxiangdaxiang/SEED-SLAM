# Noise Calibration Block Sensitivity Plan

> Execute inline on the existing research branch. No subagents, production
> changes, commits, pushes, new estimator parameters, or default changes.

**Goal:** Test the previously proposed sensitivity of the existing five noise
scales to correlated training-observation blocks, then assess the resulting
forecasts on exactly the prior future support.

**Architecture:** Reuse the hash-verified D1/D4 native physical linearizations.
Fit classical REML after dropping observation rows by declared subject or by
one of three chronological blocks of complete frames. Only calibration changes;
the full physical graph is restored for each checkpoint solve and forecast.

**Tech Stack:** Existing NumPy/SciPy contrast model and native LM executable;
pytest; JSON provenance. Large reproducible solver exports go to `/tmp`.

## Fixed Design And Alternatives

The user's continuation approves the preceding error-source report's proposed
block-deletion sensitivity experiment. Prefer exact object/block refits to a
new robust loss or new robot-wise noise family: they isolate data influence
without changing the model. Exhaustive single-frame refits would require 449
additional calibrations; this bounded study instead removes complete frames
in three ordered blocks. It does not claim exhaustive single-frame influence.

- D1 and D4, same first connected prefixes, no GT loaded by this experiment.
- Full fit control, every observed declared subject (14 D1, 13 D4), and three
  disjoint chronological frame blocks per dataset: 35 fits in total.
- A frame is `(robot, pose index)`, ordered by timestamp then robot/index.
  Split this ordered list into three near-equal frame-count blocks. Three is
  a fixed audit design choice, not a tuned or deployed algorithm parameter.
- Keep all odometry and the original nonlinear chart. Subject deletion removes
  that label across its already-declared network clusters. Do not introduce new
  cross-topology identity constraints. Rebuild columns and contrasts; report
  rank, nullity, residual DOF, Fisher spectrum and parameter-boundary outcomes.
- Five shared variance multipliers, initial log scales zero, same numerical
  bounds [-16,10] and optimizer. No prior, floor, kernel or fit retry.
- A retained chart was obtained using the full prefix. This is fixed-chart
  influence analysis, not fully cross-fitted nonlinear inference. Removed
  blocks can also remove geometric support, not just suspect measurements.
- Native central LM cap 100, same archived nominal initialization and complete
  physical graph at 30/60/90 s. Prefix covariances remain nominal; only suffix
  covariances use the new fit. Do not discard any physical factor during solving.
- Forecast the original windows (30,35], (60,65], (90,95] with unchanged QR
  covariance code and five learned scales. No future observation assimilation.
- Evaluate all successful fits, no winner selection or default change. Raw REML
  NLLs from different contrast spaces are NOT model-comparison scores. Compare
  future marginal NLL and coverage on matched identities; correlated samples
  prohibit independent significance or confidence claims.
- Bound each fit/solver/forecast worker by 180 s. Sequential workers, one BLAS
  thread. Preserve failures and report their denominators; no retuning.
- Keep full scratch evidence under `/tmp/dranppr_noise_block_sensitivity_20261004`;
  persist fits, predictions, states, summaries, logs, input/source hashes and
  final code under `results/dranppr_noise_block_sensitivity_20261004`. Do not
  duplicate large per-checkpoint linearization exports in the repository.

## Tasks

- [x] Add failing tests in `tests/test_noise_block_sensitivity.py`: native
  factor/package identity parity including fixed root, complete and disjoint
  frame partitions, all-cluster subject deletion, odometry preservation,
  input immutability, strict metadata rejection, fixed-linear fit invariance,
  and forecast identity matching with per-robot summaries.
- [x] Implement `experiments/audit_noise_block_sensitivity.py` using
  `assemble`, `components`, `ContrastVarianceModel`, `apply_scales`, `forecast`
  and `prediction_summary`; no shared estimator modifications.
- [x] Run focused tests. Run full-fit parity controls and the complete frozen
  35-fit/105-checkpoint protocol, with all failures retained.
- [x] Summarize scale sensitivity, rank/support changes and future effects for
  all deletions, including D1/R5 and D4/R2/R4. Distinguish sensitivity from a
  deployable robust solution and preserve the unresolved D2 negative result.
- [x] Run prior regression suites plus new tests; verify production unchanged,
  checksum input/output provenance, and write
  `docs/dranppr_noise_block_sensitivity_report_20261004.md`.

Focused command:

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q tests/test_noise_block_sensitivity.py \
  tests/test_network_noise_support.py tests/test_network_noise_transfer.py
```

No new adaptive-backend efficacy or online/distributed runtime claim is allowed
from this centralized statistical diagnostic. Delete-fit forecast success is
not evidence that the deleted object is actually wrong.

## Execution Notes

- All 35 fits and 105 checkpoint/forecast workers completed without failures,
  retries or parameter-boundary estimates; total producer time 630.16 s.
- All full-fit controls exactly reproduce prior forecast means/covariances.
- Post-selected mean/covariance swaps use the existing tested `rescore` helper
  for every arm/robot, not only selected examples. No model is selected.
- Maximum raw LM gradient .002961 motivated a same-start GN verification for
  ALL 105 endpoints, not just large-gradient cases. All returned; maximum
  position difference 3.213e-6 m and cost difference 4.284e-10. No predictor
  changed. This extra diagnostic cost 28.68 s and is not part of the primary
  predeclared protocol.
- Focused tests 16, full regressions 62 native / 161 Python. Production remains
  unchanged. No PCG/MPI in the research experiment; D2 negative result remains.
