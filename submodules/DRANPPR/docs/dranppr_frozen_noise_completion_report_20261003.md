# Frozen Completion: Noise Calibration Is Not Outlier Robustness

Date: 2026-10-03. Research branch only; production DBA-DRANPP unchanged.

## Purpose

The preceding added-variance audit found better causal trajectory RMSE on one
Gaussian injection, but worse final trajectory and object estimates. This stage
asks whether that discrepancy is primarily unfinished online optimization, a
different local basin, or the changed statistical objective. It also tests
whether the causal improvement survives two additional fixed noise realizations.

This is an attribution experiment, not an implemented adaptive-noise algorithm
or a comparison establishing superiority over riMESA.

## Fixed Protocol

- MR.CLAM D2, five robots, chain 1-2-3-4-5, original first 120 seconds. The
  identical nominal bootstrap ends at elapsed 21.877000093460083 seconds.
- 1,222 suffix frames, 1,469 final poses, 59 object copies and 3,538 factors:
  1,464 odometry plus 2,074 bearing-range observations. Identities are supplied,
  not estimated. Noise injection affects 1,719 suffix observations only.
- Three fixed Gaussian bearing seeds: 20261003, 20261004, 20261005; injected
  standard deviation 0.15 rad. The first seed and original endpoints reuse the
  prior hash-verified archive. Four new online runs cover the other two seeds.
- Each data case has nominal and added-variance objectives. The latter sets
  suffix bearing variance to `0.05^2 + 0.15^2 = 0.025 rad^2`. Range sigma remains
  0.1 m and odometry sigma remains `[0.1,0.1,0.05]*sqrt(dt)`. Prefix factors are
  unchanged. Thus suffix bearing information is reduced by exactly ten times.
- For each of four data cases, run LM on each objective from BOTH online
  endpoints and GN on each objective from its own endpoint. Also run routed
  tight-PCG GN on both objectives for seed 20261003. Total: 26 frozen solves,
  max 100 nonlinear iterations and 600 seconds per process, no retries/tuning.
- A new diagnostic v2 package explicitly carries every factor's diagonal
  sigma. Export does not align robots, average shared copies, add measurements,
  modify timestamps or change the root. Weighted frozen packages are rejected
  as online warm starts to prevent silent covariance resetting.
- Ground truth is loaded only after all estimators terminate. All trajectory
  and object metrics use one global SE2 alignment per complete state. Causal
  estimates are separately checked with one alignment fixed from the shared
  bootstrap, without fitting to future states.

The known added variance is an oracle ONLY for the injected component, not for
the complete real-data noise distribution. Applying it to original data is
deliberate overinflation, not an oracle. Own weighted losses cannot be compared
as accuracy scores across objectives. Frozen completion uses future information
relative to earlier causal outputs and is not an online speed improvement.

## Results

Arrows below mean nominal objective -> added-variance objective, not optimization
progress within one objective. Units are meters.

| Data | Online causal trajectory | Online final trajectory | Completed final trajectory | Completed object-copy RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.29667 -> 0.36326 | 0.31167 -> 0.41370 | 0.31001 -> 0.41131 | 0.23951 -> 0.29547 |
| Gaussian seed 20261003 | 0.40223 -> 0.38616 | 0.34509 -> 0.42431 | 0.34462 -> 0.42205 | 0.26762 -> 0.30652 |
| Gaussian seed 20261004 | 0.34278 -> 0.37248 | 0.31612 -> 0.41349 | 0.31360 -> 0.41035 | 0.24626 -> 0.30582 |
| Gaussian seed 20261005 | 0.32094 -> 0.35988 | 0.32316 -> 0.41247 | 0.32122 -> 0.41041 | 0.26279 -> 0.32495 |

Completed rows use LM from each objective's own endpoint; all other completions
agree to the numerical precision reported below. Across the three Gaussian
seeds, the arithmetic mean of completed trajectory RMSE rises from 0.32648 to
0.41427 m (+26.89%); object-copy RMSE rises from 0.25889 to 0.31243 m (+20.68%).
These are descriptive means of three runs on ONE sequence, not independent
dataset evidence or a statistical significance claim.

The earlier causal gain is not consistent across seeds: -4.00%, +8.66%, +12.13%.
With the common-prefix alignment, causal RMSE changes are 0.45860 -> 0.43914,
0.39745 -> 0.40164, and 0.37978 -> 0.40689 m. The sign pattern is unchanged;
its magnitude is alignment-sensitive. Do not retain the previous one-seed
causal improvement as a general positive result.

### Completion And Solver Checks

- All 30 new estimator processes (4 online + 26 frozen) terminate with exit 0;
  no iteration-cap retry or covariance tuning occurs.
- Matched initial frozen cost reproduces online physical cost with maximum
  absolute difference `6.37e-12`. Factor counts, exact root and shared-state
  consensus are preserved.
- Completing each own objective reduces its loss by only 0.0090%--0.1115%.
  Final trajectory RMSE changes by at most 3.15 mm; object RMSE by at most
  5.79 mm. This is far smaller than the between-objective accuracy gap.
- Across alternate starts and solvers for the SAME objective, maximum final
  cost difference is `1.48e-10`, maximum XY difference `7.94e-6 m`, maximum
  yaw difference `5.60e-7 rad`. These are in the same fixed gauge, no alignment.
- LM final gradient norms range from `5.14e-5` to `4.89e-4`. GN norms range
  from `1.14e-6` to `1.81e-5`. GN/routed termination is `no_decreasing_trial`,
  NOT a gradient-tolerance certificate. Local numerical agreement strongly
  supports adequate completion for this attribution, not global optimality.
- Routed nominal/added runs use 501/484 TOTAL PCG iterations across 13/12
  nonlinear attempts, respectively. Maximum verified PCG relative residuals
  are `9.43e-11` and `8.77e-11`; application bytes are 4,631,976/4,456,800.
  They match centralized solutions. MPI ranks: 0, in-process topology routing.

**Supported conclusion:** for these graphs and initializations, the final-error
regression is not mainly online iteration shortage or selection of different
tested local basins. It survives solving the changed objective accurately.
This does not prove that covariance adaptation generally fails, nor that a
fully correct generative model would be worse in expected estimation error.

## Reference-Error Diagnosis

After all solves, compare each arrived observation to interpolated reference
poses and landmark coordinates. All 2,074 observations are supported. This
posthoc check never supplies calibration parameters to the estimator.

| D2 observations | Bearing mean / centered SD (rad) | Range mean / centered SD (m) |
| --- | ---: | ---: |
| Original prefix, n=355 | -0.00700 / 0.02299 | -0.06259 / 0.18727 |
| Original suffix, n=1719 | -0.00827 / 0.01592 | -0.02875 / 0.16733 |
| Gaussian suffix seed 20261003 | -0.00917 / 0.14932 | unchanged |
| Gaussian suffix seed 20261004 | -0.00885 / 0.15013 | unchanged |
| Gaussian suffix seed 20261005 | -0.00633 / 0.14667 | unchanged |

The injected bearing dispersion is close to its expected 0.15 rad scale, but
the retained range model is not a uniformly calibrated zero-mean 0.1 m model.
For example, Robot 5 suffix range errors have mean -0.09635 m and centered SD
0.23728 m, versus Robot 1 mean 0.00075 m and SD 0.10598 m. Preprocessing,
interpolation and reference errors are included; these are NOT pure independent
sensor-noise samples or proof of a unique bias mechanism.

**Mechanistic interpretation, not isolated causal proof:** lowering bearing
information tenfold also increases the relative authority of unchanged range
and odometry factors. Correcting only the injected component can expose other
model errors rather than improve the joint state. The original-data inflation
control already produces a large accuracy regression. Joint relative weighting,
bias and dependence deserve examination; more outer iterations do not repair
an inadequate model. This stage does not isolate which remaining component is
the dominant cause and must not use the reference-error table to tune a winner.

## What Adaptive Noise Can Actually Do

### Calibration Versus Robustness

In a Gaussian model, covariance calibration learns how much independent
information observations carry. A conventional joint likelihood is

```text
min_(x,theta) 1/2 sum_e [r_e(x)^T R_e(theta)^(-1) r_e(x) + log det R_e(theta)].
```

It rebalances relative information; it does not remove physical sensor noise.
Uniformly scaling every factor (with the same exact gauge and no differently
weighted prior) leaves the exact least-squares point minimizer unchanged.
Only changing relative directions/sources, correlations or distributional
assumptions changes the statistical compromise.

Learning Gaussian variance alone does not yield outlier resistance. For scalar
observations `[0,0,0,10]`, jointly estimating a common Gaussian mean and variance
still gives mean 2.5 and variance 18.75. The erroneous value remains influential;
the model merely explains part of it as increased spread. Robust modeling must
address contamination/heavy tails or identity error, not just one common scale.
Covariance learning can be part of a robust model, but the concepts are not
equivalent. [Khosoussi and Shames](https://arxiv.org/html/2502.04584v2), Section VII,
explicitly distinguish their joint Gaussian covariance estimator from an
outlier-robust extension.

### Residual Attribution Is The Hard Part

At an online estimate, a large residual can arise from sensor noise, systematic
bias, wrong identity, uncertainty in the predicted state, or unfinished/stale
distributed inference. It is not directly a noise sample. In a simple linear
Gaussian fit, fitted residuals are projections of noise onto the residual
subspace; fitting consumes degrees of freedom. With an independent predictive
measurement and a local linearization, innovation covariance is approximately
`J P J^T + R`, not `R` alone. Dependence adds cross terms.

Thus adaptation must distinguish discrepancy reducible by improving the state
from discrepancy supported as observation unreliability. Otherwise it can
downweight correct constraints precisely when they are needed for recovery.
Enough geometric redundancy and identifiable parameter groups are essential.
No method can infer separate state drift, bias and variance from data that
provide observationally equivalent explanations.

### Statistical Normalization Is Necessary, Not Sufficient

Without the log-determinant term, covariance can increase indefinitely to make
weighted residual cost look better. With that term, unrestricted per-factor
covariance can still collapse on exactly fitted directions. A justified model
family, sufficient support, and declared prior/constraints or uncertainty in
the calibration are required. They are modeling assumptions, not evidence that
all hyperparameters have disappeared. See the joint-estimation reference above
for its singular-covariance diagnosis and constrained/MAP formulations.

### Distributed Information Is Not Automatically Independent

Repeatedly relayed or shared-state information must not be counted as fresh
independent evidence. Noise changes must remain consistent with the factors and
linearizations used by each participant; coordination costs must be counted.
Communication lag cannot simply be renamed sensor noise. riMESA's own
[communication experiments](https://arxiv.org/html/2603.01178v1#S7.SS4) show that
low-information scenarios can improve substantially with more communication.
That is a different issue from estimating unknown physical noise covariance.

## Research Decision

Do not promote scalar suffix variance inflation or the previous one-seed causal
gain as a robust algorithm. The meaningful target remains reliable relative
information allocation under coupled, finite-budget distributed inference.

The next focused gate is a GT-free, structurally restricted whole-model
calibration with chronological predictive validation, retaining the earlier
D4 gain and D1 failure as mandatory transfer controls. Treat directional Q,
observation R, bias/dependence, and parameter uncertainty as competing
explanations, not mechanisms to add all at once. A proposed online update must
separate state/coordination error from measurement error and preserve fixed-model
accuracy when no independent evidence supports a noise change. Classical
Gaussian ML/REML or a robust kernel alone is not the novelty claim.

## Implementation And Verification

Files changed are limited to native diagnostic package parsing, its replay
schema guard, three experiment helpers and focused tests. Native optimizer,
production code, CCI and D-CCI remain unchanged. Branch is retained as-is; no
commit, push or merge.

```sh
cmake --build /tmp/dranppr_backend_build --target \
  native-joint-audit native-joint-replay native-conditional-audit -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_frozen_noise_completion \
  --output /tmp/reproduce_frozen_noise_completion \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.analyze_frozen_noise_completion \
  --results /tmp/reproduce_frozen_noise_completion \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

Verification: 62 C++ tests and 132 Python regression tests pass. Exact commands
and XML/logs are in `test_manifest.json`. The focused file has 13 tests, including
explicit covariance scaling, single-bearing analytic loss, malformed covariance,
weighted bootstrap rejection, exact terminal loss reconstruction, shared-state
validation and reference-angle wrapping. An initial pytest plugin auto-load
failed in unrelated user-installed Dash/Jinja dependencies; tests run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, without modifying that environment.

Archive: `results/dranppr_frozen_noise_completion_20261003/`. It contains all
inputs, terminal packages, binaries, source snapshots, process commands,
outcomes, full states, covariance-preserving objective checks, posthoc metrics,
reference diagnostics, and checksums. No timing superiority is claimed: online
timings are single runs and unrelated host workloads may be concurrent.
