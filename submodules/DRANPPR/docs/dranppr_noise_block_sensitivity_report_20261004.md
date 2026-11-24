# Block Sensitivity of the Existing Five Noise Scales

## Main Result

Calibration is sensitive to which object/time blocks provide evidence, but
removing influential data does not consistently repair predictive failures.
All 35 fits succeed, retain five structurally identifiable variance components,
and stay away from numerical bounds. Nevertheless, important robot-wise
failures persist. Optimizer success and structural rank are not sufficient
certificates of predictive reliability under this stochastic model.

The positive result should also be preserved: ALL 35 arms still have better
dataset-average future marginal NLL than the corresponding nominal fixed-weight
control. The unresolved issue is heterogeneous reliability, not absence of any
calibration benefit. There is no new default or proposed deletion algorithm.

## Purpose And Protocol

The preceding error-source audit found motion propagation effects, conditional
observation discrepancies, and state/map compensation. This experiment tests
whether a small set of calibration observations dominates the existing noise
fit, without adding parameters, kernels, rejection gates, or priors.

- Source: `results/dranppr_network_noise_transfer_qr_20261003`. All 360 source
  checksums and 11 linked-library hashes were verified.
- D1 and D4, same first connected training prefixes and original nominal
  physical linearization. Barcode identities and communication topology are
  unchanged. No GT is loaded anywhere in this new experiment.
- D1: full control + 14 subject deletions + 3 frame-block deletions = 18 fits.
  D4: full control + 13 subject deletions + 3 frame-block deletions = 17 fits.
- Subject deletion removes that declared label's observations across existing
  clusters, without creating new cross-topology associations. All odometry stays.
- Frame blocks are chronological thirds of complete `(robot, pose index)`
  observation frames, sorted by timestamp with deterministic ties. D1 blocks
  remove 181/148/112 observations; D4 blocks remove 72/77/60 observations.
  Three is a frozen audit partition, not a tuned/deployed algorithm parameter.
  This is not an exhaustive leave-single-frame-out study.
- Rebuild the retained Jacobian's columns and residual contrasts. The nonlinear
  chart stays fixed at the original full-prefix solution, so this is NOT fully
  cross-fitted nonlinear estimation or a causal effect of deleting an object.
- Same five shared variance components, zero log initialization, numerical
  bounds [-16,10], and existing classical REML optimizer. No retry or tuning.
- Restore ALL physical factors when solving the 30/60/90 s graphs. Every arm
  uses the same nominal initialization and LM cap 100. Prefix weights remain
  nominal; only suffix weights change. Deletion affects calibration only.
- Forecast the same (30,35], (60,65], (90,95] windows, without assimilating
  observations inside each forecast window. All 35 arms use identical relevant
  support: 251 observations in D1 and 282 in D4, 533 distinct observations total.
  Repeated arms yield 9312 scored rows, not 9312 independent measurements.
- Evaluate every arm, with no winner selection. Compare marginal future NLL
  on matched observations, not raw training REML objectives from different
  contrast spaces. The dependent samples do not support independent-sample
  significance tests or interpreting deletion ranges as confidence intervals.

All 35 fits and 105 checkpoint/forecast workers completed. Sequential producer
wall times sum to 630.16 s: 178.32 s fitting and 451.83 s solving/forecasting.
This dense centralized audit is not an online or communication-efficiency result.

## What Is Being Estimated?

For a fixed physical Jacobian A and residual vector b, choose orthonormal
contrasts C with C' A = 0. The existing fit uses

    z = C' b,
    V(q) = C' diag(v(q)) C,
    L(q) = 0.5 * [z' V(q)^-1 z + log det V(q) + constant].

The five positive entries of q scale longitudinal, lateral and yaw process
variance, then bearing and range measurement variance. Contrasts eliminate
linear state changes; they do not eliminate nonlinear model bias or establish
independent zero-mean sensor errors. Block removal changes both the data and
the available geometry. A changed q does not by itself identify bad data.

## Parameter Sensitivity

Ratios below are deleted-fit variance / full-fit variance, not standard
deviation ratios. Ranges summarize ALL subject deletions in each dataset.

| Dataset | Q longitudinal | Q lateral | Q yaw | R bearing | R range |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 | .664--2.264 | .396--1.336 | .904--1.033 | .820--1.227 | .800--1.027 |
| D4 | .991--2.136 | .326--1.528 | .599--1.240 | .170--1.068 | .618--1.042 |

D1's observation scales are relatively stable under individual subject
removal, while some process scales vary more. D4's bearing scale changes much
more: removing subject 15's 25 observations reduces it to 17.04% of the full
fit, without creating an additional state null direction.

Time-block deletion is more disruptive. Removing D4's earliest frame block
gives Q longitudinal 5.456 times and R range .04626 times the full fit. This
case also gains six unobservable state directions, so it cannot be interpreted
as a clean outlier-removal experiment.

Eight deletion variants have additional state null directions:

| Dataset | Subject deletions (nullity) | Frame-block deletions (nullity) |
| --- | --- | --- |
| D1 | 12 (1), 13 (1) | earliest (1), latest (4) |
| D4 | 12 (2), 13 (2) | earliest (6), latest (4) |

State nullity and variance-component identifiability are different quantities.
The contrast estimator handles the retained state gauge directions; all fits
still have variance-component rank five. This does not restore the lost state
geometry or guarantee the model's predictive calibration.

## Future Prediction: Gains And Trade-Offs

NLL is averaged over the same future observations and is lower-is-better;
negative values are valid for densities in fixed radian/metre units. Coverage
is for each predictor's nominal 95% ellipse, not a new tuned threshold.

| Dataset / deletion family | Improved mean NLL | Worsened | Numerically unchanged |
| --- | ---: | ---: | ---: |
| D1 subjects | 4/14 | 10/14 | 0/14 |
| D1 frame blocks | 0/3 | 3/3 | 0/3 |
| D4 subjects | 5/13 | 6/13 | 2/13 |
| D4 frame blocks | 1/3 | 2/3 | 0/3 |

The unchanged category uses an absolute NLL tolerance of 1e-8 solely to avoid
counting numerical roundoff as improvement. It is not an estimator gate.

### D1/R5 Is Not Repaired By Any Tested Deletion

Nominal fixed-weight R5 NLL is -1.130776; full joint-QR is -.987708. Across all
17 deletions, R5 NLL ranges from -1.033008 to -.686674. Even the retrospectively
best deletion remains worse than nominal; no deletion is promoted based on
that retrospective ranking.

Dataset-average gains can mask this failure. Removing subject 15 improves
D1 mean NLL from -2.023980 to -2.047648, but worsens R5 from -.987708 to
-.881992 and its coverage from 84.34% to 81.93%.

### D4 Improvements Are Not Uniform

| Calibration data | Dataset NLL | Dataset coverage | R2 NLL | R4 NLL | R4 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | -1.717643 | 91.49% | -1.360273 | -1.405697 | 84.09% |
| Remove subject 14 | -1.823630 | 90.78% | -1.765806 | -1.568850 | 85.23% |
| Remove subject 15 | -1.664922 | 89.72% | -1.792903 | -.992638 | 79.55% |
| Remove earliest frame block | -1.665757 | 89.36% | -2.242177 | -.913846 | 77.27% |
| Remove middle frame block | -1.824926 | 91.13% | -1.790276 | -1.637841 | 87.50% |

Subject 14 and middle-block removal are genuine descriptive improvements in
average predictive score, but do not establish a selection rule or calibrated
tails. Subject 15 and earliest-block removal strongly improve R2 while harming
R4. One shared noise vector is redistributing state and predictive errors,
not uniformly identifying which robot's observations are reliable.

An explicitly post-selected mean/covariance swap helps interpret this effect.
For D4/R2 after subject-15 removal, using the FULL predictive mean with the
new covariance gives NLL -1.252806, worse than full -1.360273. Using the NEW
mean with the full covariance gives -1.793611, close to the actual -1.792903.
The benefit is primarily associated with a shifted prediction mean in this
algebraic comparison, not simply more appropriate uncertainty magnitude.
For R4, both individual swaps worsen the score. These swaps are not coherent
new estimators or strict causal effects; they change no fitted parameters.

No GT state metrics were computed here. Better observation prediction may
include state/map compensation and does not prove lower trajectory or object
error. The previous end-to-end D1/D4 gains and unresolved D2 high-noise negative
result remain under their original protocols.

## Why Observation Count Is Not Calibration Evidence

D4 subjects 9 and 19 each have only one training observation. Removing either
changes rows from 847 to 845 and state columns/rank from 475 to 473, but leaves
372 residual-contrast degrees of freedom and the five fitted scales unchanged
to numerical precision.

The two components of a single polar observation can be absorbed by the two
unknown coordinates of its otherwise unsupported landmark. It supplies no
redundant residual contrast to calibrate noise. Counting it as another effective
calibration sample would be misleading. This illustrates the role of geometric
redundancy without asserting that every repeated observation is independent.

## Numerical Checks

The unmodified full-fit controls exactly reproduce archived parameter vectors,
forecast means and covariances. Maximum QR covariance-solve relative residual
across all 105 forecasts is 4.14e-15.

Some LM endpoints have nonzero raw gradient norms, up to .002961. To avoid
mistaking incomplete optimization for data sensitivity, a post-selected
verification ran native GN from the SAME original initialization for EVERY
one of the 105 cases, with cap 100 and no changed objective or predictor.
All controls completed in a summed 28.68 s. Largest LM/GN differences:

- State position: 3.213e-6 m.
- State yaw: 7.198e-7 rad.
- Physical objective: 4.284e-10.

This supports numerical endpoint consistency, not a global-optimality proof.
No GN solution replaces an LM prediction. All research solves are centralized
LM/GN: no experiment PCG iterations or MPI ranks; CCI is unchanged.

## Research Decision

Do not add a high-influence deletion gate, robot-specific variance family, or
extra kernel on the basis of these results. Sensitivity is real, but deletion
mixes information loss, different operating conditions and model mismatch.
The currently tested remedies do not uniformly fix the existing failures.

The next bounded question should target the motion error already isolated in
D1/R5: does its time evolution support the independent, zero-mean process-noise
assumption, or a persistent/time-correlated discrepancy? Test this before
adding parameters. For D4, retain the distinction between an observation mean
offset and random scale; a single-object sample cannot identify a robot-wide
bias. Neither question is answered by a successful five-parameter optimizer.

Adaptive noise robustness should mean using geometrically supported evidence
to assign credible conditional uncertainty without absorbing systematic errors
into arbitrary weights. This stage diagnoses limits of the current recipe; it
does NOT implement that complete adaptive backend or establish superiority over
riMESA or all baselines.

## Artifacts And Reproduction

Code: `experiments/audit_noise_block_sensitivity.py`.
Tests: `tests/test_noise_block_sensitivity.py` (five red-before-green tests).
Persistent archive: `results/dranppr_noise_block_sensitivity_20261004`.
Full scratch exports: `/tmp/dranppr_noise_block_sensitivity_20261004`.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m experiments.audit_noise_block_sensitivity run \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output /tmp/dranppr_noise_block_sensitivity_20261004
```

The output directory must not already exist. Every worker command, return code
and time is recorded in `manifest.json`; GN controls in `solver_verification.json`.
The post-selected `mean_covariance_swaps.json` uses the existing `rescore` helper
on every dataset/arm/robot, not just the highlighted cases. `sensitivity_summary`
is descriptive aggregation of `analysis.json` and forecast/solver summaries.

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q tests/test_noise_block_sensitivity.py \
  tests/test_network_noise_support.py tests/test_network_noise_transfer.py
```

Focused tests: 16 passed. Fresh regression suites: 62 C++ and 161 Python passed;
full commands/environment and JUnit files are in `verification_commands.json`.
No production code, solver default, commit, or remote repository was changed.
Large reproducible per-checkpoint linearization exports remain in scratch;
their hashes are listed in `scratch_only_files.json`. Other input packages,
fits, states, forecasts, logs and summaries are archived with `SHA256SUMS`.
