# D1/D4 Frozen Noise Calibration Transfer and Forecast Audit

## Main Finding

The unchanged five-component noise family and fitting recipe improve both
checkpoint state quality and mean future predictive likelihood on D1 and D4.
This is stronger evidence than lower training residual or a single final RMSE.
It is NOT evidence of uniform robot-wise calibration, outlier robustness,
automatic tracking of changing noise, or deployed distributed adaptation.

The most important remaining failure is heterogeneous predictive reliability:
D1/R5 and D4/R2 have worse mean predictive likelihood after calibration, even
though dataset means improve. D4's overall nominal 95% ellipse covers only
91.49% of the evaluated future observations after joint Q/R calibration.

## Frozen Protocol

Worktree: `projects/seed_slam/DRANPPR`; production `DBA-DRANPP` unchanged.
Primary evidence: `results/dranppr_network_noise_transfer_qr_20261003`.
Plan: `docs/superpowers/plans/2026-10-03-network-noise-transfer.md`.

- MRCLAM D1 and D4, robots 1--5, fixed chain 1--2--3--4--5, declared barcode
  identities. These are previously examined sequences, not unseen blind tests.
- First fully connected past defines training, without GT or quality selection:
  D1 27.153000116348267 s, 311 poses / 306 odometry / 441 observations;
  D4 15.534000158309937 s, 148 / 143 / 209. D1's discovery replay is followed
  by a separate prefix-only replay whose causal rows match exactly. Its final
  120-second discovery state never initializes a calibration or forecast.
- Complete this common past in native LM, export its actual physical Jacobian,
  then fit the same classical fixed-linear REML used in the D2 audit. Five
  variance multipliers: longitudinal, lateral, yaw process noise, bearing and
  range observation noise. Shared across robots. Zero log initialization,
  numerical bounds [-16,10], no new prior/floor/kernel/distance term.
- Compare nominal, joint-R, joint-QR and split-QR. Splitting only separates
  object columns in the SAME network-derived nonlinear chart. This is NOT a
  fully independent local estimator and does not remove its network history.
- Apply fitted scales only after training cutoff; keep prefix covariances
  nominal. No injected noise or oracle-added variance in this stage.
- At 30, 60, 90 and 120 seconds, every arm starts from the SAME nominal replay
  endpoint at that time and receives only that arrived physical graph. Native
  LM cap 100; separate native GN control at 120 s. One fixed global gauge.
- Forecast (30,35], (60,65], (90,95] s using each past-only batch state, future
  odometry, and its process uncertainty. NONE of the observations in a forecast
  window is assimilated, including earlier observations in that same window.
- Only locally known objects at the checkpoint are scored: D1 251 observations
  (16 new-object observations excluded), D4 282 (one excluded). All arms have
  exactly the same predictive support. Prediction is not GT-conditioned.
- GT is loaded in a separate evaluation command after all producer processes
  and predictions finish. Each state metric uses one global SE2 alignment, not
  a separate alignment for each robot. Objects count observed copies.

This is a bounded, centralized statistical audit, not an online runtime or
equal-communication comparison. The stochastic family is intentionally kept
simple so its failures remain visible.

## Predictive Quantity

At a checkpoint, native whitened Jacobian A gives conditional state covariance
P = (A' A)^-1 after removing the one fixed root. The implementation uses a QR
square-root solve, not explicit inversion or normal-equation factorization.
It retains full selected pose/object cross-covariance and exact shared points.

Propagate each robot's last pose through the arriving odometry only, with
right-tangent transport and integrated learned process covariance. For a future
polar observation, compute

    innovation = observed bearing/range - predicted bearing/range
    S = J P_propagated J' + R
    energy = innovation' S^-1 innovation
    NLL = 0.5 * (energy + log det S + 2 log(2 pi)).

Bearing innovations are wrapped. The Gaussian density is a local polar
approximation, conditional on fitted parameters; it does NOT integrate their
uncertainty. Ideally calibrated marginal 2D energy has mean 2 and a 95% ellipse
threshold of 5.99146. Those references are descriptive here, not independent
sample tests: observations share state, odometry and object errors. Mean NLL
is a marginal predictive score, not the joint evidence of 533 independent
observations. Negative NLL is valid for densities in fixed radian/metre units.

## Calibration

All six fits report success, no component hits a numerical bound, and their
variance component Fisher matrices have full structural rank.

| Dataset / arm | Q x | Q y | Q yaw | R bearing | R range | Contrast DOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 joint-R | 1 | 1 | 1 | .002113 | .609737 | 834 |
| D1 joint-QR | .180267 | .244547 | .875178 | .002630 | .648464 | 834 |
| D1 split-QR | .167836 | .175888 | .800327 | .003157 | .448242 | 816 |
| D4 joint-R | 1 | 1 | 1 | .002328 | .769284 | 372 |
| D4 joint-QR | .097856 | .576593 | 1.062142 | .002224 | .878803 | 372 |
| D4 split-QR | .180773 | .344685 | 1.176104 | .002082 | .651159 | 362 |

These are variance multipliers, not standard deviations. Joint-QR bearing
standard deviations are about .00256 rad (D1) and .00236 rad (D4). They are
conditional fitted model parameters, not demonstrated physical sensor accuracy.

D1 joint A is 1800 x 966, rank 966; split A is 1800 x 996, rank 984.
D4 joint A is 847 x 475, rank 475; split A is 847 x 497, rank 485. The split
models retain 12 relative-gauge null directions. Sharing contributes 18 and 10
additional residual-contrast dimensions, not independent sensor samples.
Raw joint and split REML NLL refer to different contrast spaces and must not be
used as a direct model-selection comparison. Their forecasts below share the
same observation space and CAN be compared descriptively.

## State Quality

Terminal 120-second metrics, metres; each cell is trajectory / object-copy RMSE.
D1 contains 1414 poses and 61 object copies (3341 physical factors); D4 contains
1710 poses and 68 object copies (4031 factors).

| Method | D1 | D4 |
| --- | ---: | ---: |
| Nominal fixed weights | .266524 / .179354 | .285677 / .184901 |
| Joint-R | .182924 / .119728 | .207047 / .175461 |
| Joint-QR | **.156369 / .119308** | **.186795 / .160961** |
| Split-QR | .156750 / .121542 | .188679 / .167446 |

Joint-QR reduces trajectory RMSE by 41.33% and 34.61%, object-copy RMSE by
33.48% and 12.95%, relative to this nominal control. Both metrics improve at
ALL four declared checkpoints, not only at the chosen final horizon:

| Dataset / time | Nominal trajectory | Joint-QR trajectory | Nominal object | Joint-QR object |
| --- | ---: | ---: | ---: | ---: |
| D1 / 30 | .249596 | .215241 | .273693 | .245371 |
| D1 / 60 | .285568 | .229520 | .292669 | .239738 |
| D1 / 90 | .294764 | .171107 | .228388 | .122356 |
| D1 / 120 | .266524 | .156369 | .179354 | .119308 |
| D4 / 30 | .264911 | .198489 | .334125 | .249109 |
| D4 / 60 | .250950 | .155641 | .265449 | .176448 |
| D4 / 90 | .306760 | .207488 | .207607 | .181261 |
| D4 / 120 | .285677 | .186795 | .184901 | .160961 |

This does not prove that the previous local online D1 failure has been solved:
that experiment used independent local graphs, different calibration windows,
six components and per-robot evaluation. The current controlled contrasts show
the five-component recipe transfers to the native JOINT graph without tuning.
They do not isolate which of all those cross-experiment differences mattered.

## Future Predictive Quality

Lower mean NLL is better. Coverage refers to the declared 95% polar ellipse.

| Dataset / arm | Mean NLL | Mean energy | Coverage |
| --- | ---: | ---: | ---: |
| D1 nominal | -1.576927 | .735203 | 97.61% |
| D1 joint-R | -1.761936 | .998512 | 96.81% |
| D1 joint-QR | **-2.023980** | 1.794216 | 94.02% |
| D1 split-QR | -1.982934 | 2.240937 | 92.83% |
| D4 nominal | -1.466104 | .786936 | 98.94% |
| D4 joint-R | -1.570276 | 1.061054 | 97.87% |
| D4 joint-QR | -1.717643 | 1.848097 | 91.49% |
| D4 split-QR | **-1.740122** | 1.904179 | 90.78% |

Joint-QR improves mean NLL in all six dataset/window pairs relative to nominal.
Its mean energy is closer to 2, but D4's tail is under-covered. The split arm
has slightly better D4 predictive NLL despite worse final state metrics. Thus
network support is useful additional information, not a guarantee that joint
calibration wins every criterion on every sequence.

### Failures Hidden By The Dataset Mean

- **D1/R5:** 83 forecasts. Joint-QR NLL worsens from -1.13078 to -.98771;
  coverage drops from 92.77% to 84.34%. Its mean energy is 3.953 while R1/R2/R4
  have .534/.497/.252. The fitted model is simultaneously conservative for
  some robots and overconfident for another. R5's range-innovation centred SD
  is .1934 m versus average predicted marginal SD .1143 m.
- **D4/R2:** 21 forecasts. NLL worsens from -1.64470 to -1.36027, although
  coverage remains 100%. Range innovation has mean +.1676 m and centred SD
  .0678 m; this is a coherent offset in this small forecast sample, not simply
  larger zero-mean fluctuation.
- **D4/R4:** NLL improves, but coverage is only 84.09% (88 forecasts). Better
  average scoring does not certify reliable tail probabilities.

These are prediction errors, combining motion/state/map/model effects. They
cannot be identified as pure sensor variance or sensor bias from this audit
alone. No per-robot variance setting or rejection gate was added to hide them.

### Post-Selected Attribution

After seeing these results, rescore the same observations by swapping predicted
means and covariances. This is a numerical attribution, not a new estimator,
refitting step, or predeclared model selection. Observation identities and
values are checked before swapping.

| Predicted mean / covariance | D1 mean NLL | D4 mean NLL |
| --- | ---: | ---: |
| Nominal / nominal | -1.576927 | -1.466104 |
| Nominal / joint-QR | -1.972708 | -1.692971 |
| Joint-QR / nominal | -1.602898 | -1.469286 |
| Joint-QR / joint-QR | -2.023980 | -1.717643 |

Most of the aggregate score gain can be reproduced by changing predictive
covariance while holding the nominal mean fixed. The mean also helps, but much
less in these short forecast windows. This supports better aggregate uncertainty
assignment, not merely an improved fitted trajectory. The attribution is
conditional and not an additive causal decomposition.

## Numerical Verification

The first attempt (`dranppr_network_noise_transfer_20261003`) stopped at D1/t90/
joint-R because a normal-equation selected solve residual was 1.36754e-8,
above its 1e-8 check. No threshold was loosened. A deterministic ill-conditioned
two-column test exposed 8.89e-5 relative covariance error even when a normal
equation residual passed. The predictor was changed to QR, and the ENTIRE
matrix was rerun in the separate `_qr_` archive without statistical changes.
The original source, failing checkpoint input/state/linearization, manifest and
`ABORTED.md` are preserved in `docs/archive/network_noise_transfer_abort_20261003.tar.gz`.
The redundant partial run directory was removed during the user-requested
2026-10-30 cleanup; the complete `_qr_` archive remains unchanged.

- 59 producer subprocesses exit zero: one discovery, one causal prefix replay,
  one preparation, two training solves, six fits, eight nominal checkpoint
  replays, 32 checkpoint LM solves and eight terminal GN controls.
- 24 QR forecasts complete. Maximum square-root solve relative residual:
  2.22579e-15. Analytic polar Jacobians agree with actual native factor export
  on deterministic tests; selected covariance agrees with a dense reference.
- Maximum training export loss/gradient discrepancies: 2.84218e-14 and
  2.86711e-14. Maximum terminal GN/LM positional difference: 3.83873e-7 m;
  maximum cost difference: 1.90994e-11. This rules out a material completion
  difference for these tested endpoints, not all nonlinear local minima.
- Checkpoint LM uses 5--16 iterations. Maximum reported posthoc gradient norm
  is 3.49955e-4. Neither the LM return code nor GN's stopping status is a global
  optimality certificate.
- Build succeeds; 62 C++ and 147 Python regression tests pass. Commands and
  JUnit XML are archived in `verification_commands.json` and matching logs.
- Serial subprocess wall times total about 95 s; 24 QR forecasts total about
  67 s, .32--10.82 s each. The dense diagnostic is NOT an online predictor or
  a communication/latency improvement. No prediction-speed claim is made.

The nominal initialization replays use the existing deliberately INEXACT
one-PCG-step lazy mode. Its maximum true linear residual need not be small:

| Replay | Total PCG iterations | Maximum relative residual |
| --- | ---: | ---: |
| D1 / 30 | 28 | 1.04495 |
| D1 / 60 | 331 | 1.73662 |
| D1 / 90 | 700 | 1.86924 |
| D1 / 120 | 1103 | 2.05345 |
| D4 / 30 | 260 | 1.68745 |
| D4 / 60 | 623 | 2.40762 |
| D4 / 90 | 1091 | 2.40762 |
| D4 / 120 | 1562 | 2.40762 |

These are initialization processes, not the completed states used for scoring.
All statistical arms subsequently receive native LM completion, with separate
GN checks above. MPI ranks: 0. Centralized chordal initialization and D-CCI are
unchanged.

## Interpretation And Next Gate

The useful distinction is now empirical, not just conceptual: adapting the
relative trust in process and observation factors can improve state quality
AND forecast distribution quality without dataset-wise tuning. But a single
shared noise vector still trades reliability between robots, and an offset is
not equivalent to larger zero-mean variance. Merely inflating residual scales
or adding a general robust kernel would not resolve that distinction.

Before another adaptive mechanism, diagnose D1/R5 and D4/R2 with the SAME
forecast protocol: separate persistent conditional-mean error from fluctuation
and from uncertain state/motion. Any extension must improve these conditional
failures without discarding the current good cases or adding per-case gates.
Do not infer whether heteroscedasticity, bias, unmodelled correlations, finite
linearization or parameter uncertainty dominates before such an isolation.

Retain the earlier D2 high-Gaussian-noise counterexample: learned calibration
has not beaten the old fixed weights there. This stage used original data,
not a fresh high-noise, outlier or changing-noise transfer experiment. No claim
of complete adaptive robustness or superiority over riMESA/all baselines is
justified. Production remains unchanged.

## Reproduction

From `projects/seed_slam/DRANPPR`, use a NEW output folder:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_network_noise_transfer run --output NEW_OUTPUT
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_network_noise_transfer evaluate --output NEW_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.analyze_network_noise_transfer --output NEW_OUTPUT
cmake --build /tmp/dranppr_backend_build --target native-joint-audit native-joint-replay testIncrementalLocalModel -j2
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_network_noise_transfer.py tests/test_network_noise_support.py tests/test_frozen_noise_completion.py
```

The archive includes input streams, immutable native binaries, linked-library
hashes, source snapshots, all fitted values, all checkpoint states, individual
forecasts, endpoint controls, evaluation and separately labelled attribution.
