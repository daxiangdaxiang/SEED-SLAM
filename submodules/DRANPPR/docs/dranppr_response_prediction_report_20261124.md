# Low-Order Error-Calibrated Response Prediction

Date: 2026-11-24. Research DRANPPR only. No production default, native backend,
physical objective, sensor weight, CCI, commit, or push changed.

## Decision

The cheaper architecture is useful, but scalar error calibration is NOT an
established overall optimizer improvement. Do not promote it to the default.
Calibrated prediction beats hold in 74/135 conditions, ties 27 and loses 34.
Its raw linear ablation wins 75, ties 27 and loses 33. Calibration improves
45/60 nonideal MRCLAM conditions versus linear, but loses all 36 SE3 conditions
and all 12 scalar controls. This is a tradeoff, not uniform evidence for or
against prediction-error feedback in general.

Fixed-horizon Python service time is about 77% lower than prior multi-order
risk prediction, but still 3.26x hold. Measured replay to the SAME accuracy
remains 1.85--2.11x slower than hold in the three tested SE3 network conditions.
Iteration/byte savings do not establish elapsed-time acceleration.

## Minimal Method

Each receiver keeps actual neighbor state separate from its response predictor.
For the fixed normalized quadratic, the contribution is z_ij=B_ij*x_j. Cache
B_ij*x_j on receipt for hold too, avoiding an unfair caching advantage.

On each accepted packet, validate the OLD slope at the new packet's SOURCE
time before refitting. Let v be that slope and u=(z_new-z_old)/(t_new-t_old).
In the existing inverse-majorizer metric W=M_i^-1, update

```
S_pp <- rho*S_pp + v' W v
S_pd <- rho*S_pd + v' W u
gamma = clip(S_pd/S_pp, 0, 1)
```

Use gamma=0 without validated nonzero support. This is discounted retrospective
scalar velocity regression, not a future uncertainty bound. Fit one OLS slope
with an intercept to L=8 actual response samples, only on packet arrival.
Between messages use z_hat(t)=z_last+gamma*(t-t_last)*v. The ablation uses
gamma=1 with the same slope. Predictions never enter history or overwrite the
actual cache. Reset clears all learned statistics.

The same L=8 fixes rho=(L-1)/(L+1)=7/9, without another tuning parameter.
This is not parameter independence. Packet slope work is O(L*d), plus metric
products; between-message prediction is O(d), without QR or a linear solve.
It is a bounded-window packet-driven fit, not a constant-memory Kalman filter.

Important: this gamma is learned on packet-interval average velocities. It is
NOT a horizon-dependent error model or a convergence guarantee. Transferring
calibration to longer unseen horizons requires local trend persistence.

## Protocol

- Five archived MRCLAM quadratics D1/D2/D3/D4/D9, three synthetic six-DoF SE3
  pose/object quadratics, one scalar SPD control.
- Five networks: ideal, delay, heterogeneous compute, bounded burst loss, and
  finite outage; seeds 31/32/33; 2000 ticks; size 8 everywhere; three arms.
- All 405 runs completed. Same majorizer, sin(k+1) start, physical models,
  weights, owner updates and message opportunities. Legacy secant/poly2/risk3
  comparisons use their sealed identical-model archives, not retuned reruns.
- Development: 27 runs, D1/D4/SE3_31, three networks, seed31, 600 ticks. No
  algorithm or endpoint tuning followed this smoke.
- Conditions are correlated, not independent datasets. Fixed charts and
  pre-distributed reduced rows; setup bytes excluded; known delay bounds;
  uncongested network; burst loss applies to source links, not relay hops.
- No nonlinear online SLAM, sensor-noise robustness, ATE, or superiority over
  production DRAN/riMESA is established by these experiments.

## Objective Results

These measure final relative quadratic objective gap, NOT ATE.

| Candidate / reference | Better | Tied | Worse |
| --- | ---: | ---: | ---: |
| Calibrated / hold | 74 | 27 | 34 |
| Linear / hold | 75 | 27 | 33 |
| Calibrated / linear | 45 | 27 | 63 |
| Calibrated / prior risk3 | 27 | 27 | 81 |
| Calibrated / secant | 34 | 27 | 74 |
| Linear / prior risk3 | 56 | 27 | 52 |

The older risk3 versus hold result was 93/27/15. The new method is cheaper but
has a narrower positive range. All ties above are ideal-network controls.
Each following row contains 12 nonideal conditions, calibrated versus hold:

| Model | Better / worse | Median gap reduction |
| --- | ---: | ---: |
| D1 | 0 / 12 | -2.34% |
| D2 | 1 / 11 | -1.64% |
| D3 | 9 / 3 | 3.09% |
| D4 | 12 / 0 | 11.68% |
| D9 | 4 / 8 | -3.05% |
| SE3_31 | 12 / 0 | 77.83% |
| SE3_32 | 12 / 0 | 77.59% |
| SE3_33 | 12 / 0 | 77.65% |
| Scalar SPD | 12 / 0 | 27.23% |

Versus linear, calibration has median 1.41% lower gap over the 60 MRCLAM
conditions, but median 8.37% higher gap over the 36 SE3 conditions. Lower
aggregate forecast error does not imply a better optimization trajectory.

Both hold and calibrated reach gap<1e-6 in 27 nonideal physical pairs.
Calibrated saves 6.84--9.19% application bytes, median8.51%; linear saves
median 9.09%. All eligible pairs are SE3 delay/heterogeneous/loss cases. The
other 69 physical nonideal pairs are NOT credited with a speedup. Calibrated
uses median 0.51% MORE bytes than its linear ablation at this target.

## Compute

Fixed-horizon timing: D1/D4/SE3_31 x delay/heterogeneous, 600 ticks, three
shuffled-order repeats, one BLAS thread. Take per-condition median times, then
the median ratio over six conditions:

| Time ratio | Median |
| --- | ---: |
| Calibrated / risk3 | 0.227 |
| Linear / risk3 | 0.195 |
| Calibrated / hold | 3.261 |
| Linear / hold | 2.811 |
| Calibrated / secant | 2.121 |

Same-accuracy replay: SE3_31/seed31, first archived gap<1e-6 crossing, three
shuffled-order repeats. Counts below include tick zero:

| Network | Hold ticks / seconds | Calibrated ticks / seconds | Time ratio |
| --- | ---: | ---: | ---: |
| Delay | 853 / 0.1017 | 781 / 0.2142 | 2.11 |
| Heterogeneous | 1317 / 0.1333 | 1225 / 0.2786 | 2.09 |
| Loss | 1284 / 0.1428 | 1169 / 0.2648 | 1.85 |

These are Python simulation service times including posthoc diagnostics, not
parallel robot CPU or native DRAN runtime. Target stopping is known offline,
not a deployment stopping protocol. All 45 target replays reproduce their
archived state prefixes exactly. All 90 fixed-horizon timed endpoints agree
with their archived 600-tick endpoints.

## Forecast And Stress Controls

The common-noise synthetic audit has 270 cells x 200 trials: coordinate
histories with a clean future reference, NOT sensor-noise SLAM data. For
quadratic history, SD 0.05 and horizon 16, mean squared forecast errors are:

| Hold | Secant | Risk3 | Linear | Calibrated |
| ---: | ---: | ---: | ---: | ---: |
| 0.162796 | 2.656824 | 0.112621 | 0.026943 | 0.120411 |

Calibration suppresses useful trend here. Good short-interval calibration
cannot be assumed to transfer to long horizons. This alone does not isolate
every cause of optimizer regressions.

There is a positive adaptation control: after an initially unobservable stop,
both linear predictors initially have horizon-4 error 0.008. After four actual
post-stop observations, error is 0.001311 for linear and 0.000292 for calibrated.
The latter suppresses an obsolete trend faster, but cannot anticipate the
unobserved event. Both become exact after eight post-change observations in
this noiseless control.

All 40 long-delay/outage stress runs completed, with identical prefix budgets
and zero declared age-bound violations. Delay 12--20 ticks; outage 200--600.
Calibrated versus hold gap reductions:

| Model | Long delay | Long outage |
| --- | ---: | ---: |
| D1 | -3.38% | -13.41% |
| D4 | 20.45% | -16.81% |
| SE3_31 | 90.71% | 2.89% |
| Scalar SPD, complete topology | 31.76% | 3.67% |

Finite runs do not establish stability. Long outages expose regressions rather
than supporting a blanket robustness claim.

## Exact-Neighbor Diagnostic

An explicitly NONDEPLOYABLE control grants current neighbor states for free,
eliminating prediction error with the same metric and activation calendar.
For D1/seed31 its final gap relative to hold is 1.009x (delay), 1.097x
(heterogeneous), and 1.025x (outage). Even exact gradients do not improve these
finite-horizon endpoints under the unchanged metric.

A second oracle also replaces delay-conservative damping with the fresh-
information metric. Its gap is 0.462x, 0.311x, 0.369x hold respectively. Neither
oracle is a valid distributed algorithm or a communication comparison.

This isolates two issues: removing gradient error changes iteration dynamics
and does not imply finite-budget dominance; retaining worst-case-delay damping
after successful compensation can leave progress unused. It does NOT justify
removing damping without error control. The earlier unsafe-step counterexample
still applies.

## Verification And Next Decision

- 86 focused Python tests pass, including 15 response/control tests.
- 135 hold trajectories exactly match both prior reference archives.
- 405 independent endpoint checks; 1215 every-prefix budget comparisons.
- Reconstructed inexact-gradient identities have zero positive normalized
  bound violation. This verifies algebra, not predictor convergence.
- Sealed source/input/output SHA256 ledgers; all runs terminal.
- No C++ change/rebuild or new PCG iterations; MPI ranks 0; CCI unchanged.

Retain low-cost response prediction as an opt-in research baseline. Do not call
scalar gain calibration a validated core contribution. Next investigate whether
realized errors at ACTUAL consumed prediction horizons can support less
conservative local updates while preserving stability control, rather than
increasing order or blindly enlarging steps.

Artifacts under `results/`: `dranppr_response_development_20261124` (27 smoke
runs), `dranppr_response_final_20261124` (405 main runs),
`dranppr_response_review_20261124` (independent review),
`dranppr_response_controls_20261124` (forecast/stress/oracle/timing),
`dranppr_response_target_time_20261124` (45 same-accuracy timed replays).

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q tests/test_response_prediction.py tests/test_neighbor_prediction.py \
  tests/test_history_forecast_controls.py tests/test_async_interface.py \
  tests/test_async_interface_audit.py tests/test_review_async_interface.py \
  tests/test_network_noise_support.py tests/test_native_rigid_subspace.py \
  tests/test_native_lazy_refresh.py tests/test_native_joint_audit.py tests/test_native_interface_support.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_neighbor_prediction \
  --output results/NEW_response --predictions hold response_linear response_calibrated \
  --ticks 2000 --parity-root results/dranppr_async_directional_20261123
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.review_response_prediction \
  --root results/NEW_response --prior results/dranppr_async_directional_20261123 \
  --archive results/dranppr_history_final_20261123 --output results/NEW_response_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_response_controls \
  --root results/NEW_response --output results/NEW_response_controls
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_response_controls \
  --root results/NEW_response --output results/NEW_response_target_time \
  --target-only-archive results/dranppr_history_final_20261123 --repetitions 3
```
