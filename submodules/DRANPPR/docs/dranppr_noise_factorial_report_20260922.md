# Native Noise Attribution and Transfer Audit

## Question and Scope

Does the selected Dataset 4 Robot 3 improvement require joint noise learning,
or does correcting the odometry model explain it? Does the unchanged procedure
transfer to other robots and sequences?

These experiments use independent local persistent iSAM2 references, not a
distributed DRAN++ implementation. There are no relationship edits, communication
events, or robust losses in these arms. Ground truth is used only by the separate
post-run evaluator. Classical REML is a diagnostic, not the proposed novelty.

## Fixed Procedure

All methods use the same native timestamp-atomic frames. Fit on seconds 0--30,
freeze the fit, and score known-landmark observations before assimilating their
whole future frame on seconds 30--120. Future odometry and observation factors
receive the selected model; old factors are not reweighted.

- `fixed`: nominal odometry and bearing/range noise.
- `fitted_residual`: ordinary past residual bearing and distance-dependent range
  fitting, with nominal odometry.
- `q_only`: retain only the jointly learned longitudinal/lateral/angular
  odometry variance multipliers; restore nominal observation noise.
- `r_only`: retain only the same jointly learned bearing and range model;
  restore nominal odometry.
- `reml_diagonal_q`: apply both parts of the same joint fit.

Q-only and R-only are interventions on one joint fit, not separately optimized
restricted models. Each arm independently pays its own fitting cost. No
ground-truth score is used to select a covariance or a per-sequence threshold.
The statistical family and 30-second calibration horizon remain assumptions,
not an assertion of parameter-free estimation.

## Dataset 4: All Five Robots

There are 1,751 paired future observations and 1,302 evaluated future pose frames.
Trajectory and object errors use an independent rigid alignment per robot and
arm; these are not globally aligned distributed trajectory metrics.

| Method | Predictive NLL | Final trajectory RMSE (m) | Causal trajectory RMSE (m) | Object-copy RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| Fixed | -2.43403 | 0.22275 | 0.22595 | 0.92346 |
| Fitted residual | -3.06008 | 0.22253 | 0.21342 | 0.93068 |
| Q-only | -2.46157 | 0.16053 | 0.17161 | 0.84160 |
| R-only | -3.18780 | 0.20128 | 0.19318 | 0.83449 |
| Joint Q/R | -3.24921 | 0.16221 | 0.16159 | 0.80577 |

Q-only improves final and causal trajectory RMSE for every robot against fixed.
Joint Q/R also improves both trajectory measures for every robot against fixed,
but not all individual object errors: Robot 1 changes from 0.33913 to 0.34807 m
and Robot 5 from 0.48279 to 0.48363 m. Joint fitting is not uniformly best:
Q-only has slightly better aggregate final trajectory RMSE.

For the original selected Robot 3 case, Q-only reaches 0.06930 m final trajectory
RMSE versus joint 0.07851 m and fixed 0.20036 m. Joint improves causal trajectory
RMSE to 0.14373 m versus Q-only 0.14982 m, and object RMSE to 0.12957 m versus
0.13605 m. Therefore the earlier large final trajectory improvement should not
be attributed exclusively to sophisticated observation-noise fitting. The
odometry model accounts for a substantial part of that improvement.

R-only improves aggregate likelihood much more than Q-only, but has worse
trajectory RMSE. A better measurement predictive score is not equivalent to
uniformly better state estimates. Use both to evaluate the modeling claim.

## Dataset 1: Transfer Gate Fails

The same procedure, with no retuning, gives 1,429 paired predictions and 1,075
future pose frames. Dataset 1 was not used to choose the diagonal-Q repair.

| Method | Predictive NLL | Final trajectory RMSE (m) | Causal trajectory RMSE (m) | Object-copy RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| Fixed | -2.50923 | 0.24138 | 0.31749 | 0.80231 |
| Fitted residual | -2.86997 | 0.26856 | 0.33945 | 0.88427 |
| Q-only | -2.43528 | 0.24508 | 0.36724 | 1.15658 |
| R-only | 43.41169 | 0.33936 | 0.42132 | 0.78035 |
| Joint Q/R | 92.17170 | 0.25031 | 0.32803 | 0.93277 |

All adaptive arms worsen both aggregate trajectory measures against fixed.
The huge likelihood failure is concentrated in Robot 5: mean NLL is 274.85777
for joint and 132.99536 for R-only, versus fixed -2.47758. Robot 3 also has
worse joint likelihood (-2.03790 versus fixed -2.44888), so Robot 5 is not the
only transfer limitation. Further broad deployment of this plug-in model is
not justified. Datasets 2, 3 and 9 have not been run with this candidate.

Robot 5 has only 19 warmup observations of three identities; 18 observations
support the calibration. Its observations cover just seconds 3.701--10.887
within the nominal 30-second warmup. Four of six fitted variance components
reach the numerical boundary: yaw odometry, bearing, range intercept and range
slope. Bearing variance is `2.813e-10 rad^2`; this is not convincing evidence
that the physical sensor is nearly noiseless.

### Parameter Uncertainty Is Not State Uncertainty

A post-selected profile refits all nuisance components at each fixed component
value, using two initialization points. On Robot 5, raising the bearing
multiplier from `1.125e-7` to `0.04979` increases training NLL by only `1.00882`.
That is a variance increase of about 442,000 times, with little likelihood
separation on the available warmup. The selected maximum-likelihood point is
therefore far more precise than the data support. Restoring nominal bearing
variance does cost 9.07791 nats; the diagnostic does not imply that nominal is
the statistically preferred value either.

These are likelihood profiles inside a fixed-linear Gaussian family, not
confidence intervals with established coverage. Tiny Fisher eigenvalues in
log-variance coordinates alone would be insufficient evidence: derivatives
also vanish as a component approaches zero. The finite profile comparison is
the stronger diagnostic here. Temporal dependence, geometry change and model
misspecification may still contribute to the future failure.

The current predictor integrates state uncertainty conditional on one fitted
noise parameter vector, `p(e | theta_hat, past)`. It does not account for
uncertainty in `theta_hat`. A future parameter-marginalized predictor would
instead use `integral p(e | theta, past) p(theta | past) d theta`, or an explicitly
justified likelihood-based uncertainty set. This is a research direction, not
an implemented fix or a new claim of novelty; priors or confidence levels must
be declared rather than disguised as parameter-free adaptation.

### Two Different Causes of Large Innovations

Independent GT-assisted diagnostics are joined by immutable raw observation
row IDs. They are not inputs available to the estimator.

- Row 408 at 91.512 seconds: joint range innovation is `-0.43913 m`; the actual
  observation's GT-assisted range error is `-0.40051 m`. A substantial part of
  the innovation is a real measurement error, not simply unfinished inference.
- Row 203 at 71.185 seconds: joint range innovation is `+0.24631 m`, whereas the
  GT-assisted measurement error is only `+0.00543 m`. Here the state/map
  prediction is the principal discrepancy. Fixed gives `-0.00632 m` innovation.

At row 203, a past-only nonlinear refinement of the same frozen weighted graph
reduces joint cost from `2.054e7` to `1.940e7` and the gradient norm from
`1.261e8` to `6.09`, but the innovation remains `+0.23258 m`. This is not a
global-optimality certificate; it shows this bounded extra solve does not repair
the statistical failure. Dense-QR prediction also reconstructs the sparse
prediction before this intervention. Neither example establishes a wrong
identity: this experiment uses the supplied landmark identities and makes no
association edits.

## Numerical and Runtime Audit

Selected covariance is obtained by solving the sparse augmented system
`[I A; A^T 0]`, retaining all requested pose/object cross terms and exactly
fixing the root. Normal equations and an artificial gauge prior are not used.
This still refactorizes the whole current local model at every scored frame.

Against the saved dense-QR implementation, 3,502 matched fixed/residual events
have maximum NLL difference `3.45e-13`; 1,008 matched Robot 3 fixed/joint events
have maximum difference `1.89e-11`. All 14 field-level comparisons pass, including
cross covariance. There is no innovation change.

This run spends about 48 seconds per arm in prediction for all five robots,
plus 5--6 seconds in state updates and up to 3 seconds in calibration. Maximum
simulated serial backlog is 3.75--4.19 seconds. Earlier dense runs spent about
100 seconds per arm in prediction, but these are separate host runs rather than
a controlled speed benchmark. This is neither constant-time inference nor a
real-time distributed-backend result.

A second implementation exports the native sparse GTSAM Jacobian instead of
first materializing a dense one. On all 8,755 Dataset 4 arm/event combinations,
all seven checked numerical fields reproduce the first sparse implementation
exactly. Prediction time is 23.84--24.27 seconds per arm; fixed backlog drops
to 0.151 seconds and joint backlog to 3.045 seconds, now dominated by calibration.
On 2,450 Dataset 1 Robot 5 arm/events, maximum NLL change is `6.04e-13`.
This does not make the failed covariance fit valid. GTSAM's sparse exporter
omits entries at or below `1e-12`; unit tests and native parity explicitly check
that this introduces no material change on these problems.

## Open Gates

1. Do not promote this model or sweep a variance floor after the Dataset 1
   failure. Preserve Dataset 4 gains and Dataset 1 regressions as paired gates
   for the next statistically justified candidate.
2. Separate weakly supported covariance estimates from confirmed sensor
   precision. Test parameter uncertainty and temporal predictive support before
   accepting a near-zero plug-in covariance. The current fixed 30-second batch
   calibration is not a complete continually adapting online algorithm.
3. Keep the ordinary residual baseline. Do not infer superiority to official
   riMESA, AEROS or covariance-learning methods from these internal controls.
4. Integrate only a validated model into a persistent distributed DRAN++ graph.
   The current experiments neither address wrong identities nor establish
   distributed consensus, message cost, or long-history incremental scaling.

## Artifacts

`results/dranppr_noise_factorial_20260922/dataset4/` contains producer/evaluator
manifests, per-event predictions, causal/final estimates, per-robot errors and
the dense-reference parity table. The adjacent `source/` directory preserves
the executed producer and its modeling dependencies before further optimization.

Dataset 1, the native-sparse parity replays, selected nonlinear checkpoints and
the parameter-profile audit are preserved under the same result root. Profile
and checkpoint experiments are post-selected diagnostics, not extra independent
validation sequences. The frozen-calibration reconstruction helper was extended
to replay future Q updates exactly once; this changes the post-hoc audit helper,
not the producer of the reported native results.

Reproduction uses `experiments/audit_noise_factorial.py` with
`--stream /tmp/dranppr_causal_streams_v1/dataset4`, then the separate
`experiments/analyze_online_noise.py` evaluator. Both accept a fresh `--output`
directory. Numerical work uses one BLAS/OpenMP thread.
