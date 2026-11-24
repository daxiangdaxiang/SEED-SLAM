# Predictive Error Sources: Motion, Observations, and Compensation

## Main Finding

The D1/R5 and D4/R2 prediction failures have different structures. Increasing
one global or robot-wise variance cannot explain all of them. D1/R5's bearing
error is mainly associated with motion propagation during the forecast window,
while its range discrepancy is concentrated in a few object/frame conditions.
D4/R2 has only single-object support in these windows and retains a positive
range discrepancy even with reference poses and landmarks. Map and trajectory
errors also compensate for one another, hiding their separate magnitudes.

This is a diagnostic result, NOT a new robust estimator or a new accuracy win.
No state solver, covariance fit, robust kernel, threshold, prior, or production
code was changed. The previous D2 high-noise negative result remains unresolved.

## Frozen Protocol

- Research worktree: `projects/seed_slam/DRANPPR`; production `DBA-DRANPP`
  remains unchanged and clean.
- Source: `results/dranppr_network_noise_transfer_qr_20261003`; all 360 archived
  checksums verified before the audit. Use nominal and joint-QR only.
- D1 and D4, all five robots, identical forecast windows (30,35], (60,65],
  (90,95] seconds. Exactly 533 distinct observations, or 1066 rows across the
  two arms. No prediction is re-fitted or re-selected using ground truth (GT).
- First save GT-free repeated co-observation distance evidence. Only afterward
  load references for explicitly posthoc error decomposition.
- Reuse each checkpoint's past-only, single-global alignment. Never align each
  robot independently or fit alignment using the future forecast trajectory.
- Subject-level stratification and tail concentration were added after seeing
  the primary audit. They are explicitly post-selected descriptive analyses.
- Every scored observation was checked against its raw numeric measurement
  row: timestamp, decoded identity, range and bearing agree exactly (533/533).
  This causal stream does not apply the static benchmark's median preprocessing.

The primary audit took 13.81 seconds, without solver replay or covariance
recomputation. Artifacts are in
`results/dranppr_predictive_error_sources_20261004`.

## Two Complementary Checks

### State-Free Observation Consistency

Convert two simultaneous polar observations of static objects a and b into
local Cartesian points. Their distance

    d_ab(t) = || p_a(t) - p_b(t) ||

does not depend on the robot's rigid pose. Compare it with the last earlier
co-observation of that same pair on that robot. Both current observations must
belong to the original forecast support; the earlier pair can come from any
available past. No future data or state estimate enters this statistic.

Normalize its squared difference by first-order observation variance, using
the existing frozen R model and keeping prefix observations nominal. There are
175 distinct pair comparisons: 113 in D1 and 62 in D4. They cover 254/533 current
observations (47.7%), not the whole graph. Nominal and joint-QR have identical
raw differences; only their variance normalization differs.

Pairs sharing a frame, object or earlier measurement are dependent. The scalar
reference energy of one is descriptive, not a calibrated hypothesis test.
This check detects some pose-independent inconsistency but cannot identify
which endpoint, frame, or object caused it. It can miss persistent errors
that preserve pair distances. Empty support is reported, not treated as success.

### Reference-Assisted Telescoping

Let h(x,m) predict a polar observation. Hats denote estimated pose/map, stars
denote references, and x0* denotes the checkpoint reference pose propagated
through exactly the same later odometry as the original forecast. Then

    y - h(x_hat,m_hat)
      = [y - h(x*,m*)]                     measurement/reference
      + [h(x*,m*) - h(x*,m_hat)]            map
      + [h(x*,m_hat) - h(x0*,m_hat)]        motion propagation
      + [h(x0*,m_hat) - h(x_hat,m_hat)]     checkpoint state.

This is an exact telescoping identity, with bearing differences wrapped. It
is NOT a decomposition into independent random variables or strict causal
effects. Attribution depends on substitution order and reference alignment.
The artifact also reverses pose/map substitution and preserves the full range
cross-moment matrix, including terms that cancel. Component variances must not
be added as if independent. Reference discrepancies include measurement-model,
reference and interpolation errors, not pure sensor noise alone.

The reconstructed original forecast differs from its archive by at most
4.44e-15; telescoping reconstruction error is at most 5.64e-17. Oracle
substitutions are never passed to an estimator or presented as deployable output.

## D1/R5: Motion Error and Localized Range Tails

There are 83 predictions: none after 30 s, 17 after 60 s and 66 after 90 s.
Joint-QR component RMSEs are:

| Term | Bearing (rad) | Range (m) |
| --- | ---: | ---: |
| Original innovation | .082831 | .194817 |
| Measurement/reference | .009526 | .158268 |
| Map | .009082 | .144367 |
| Motion propagation | .080220 | .016666 |
| Checkpoint state | .008219 | .065417 |

Bearing error is mainly associated with propagation within the withheld
window, not just an inaccurate starting state or large bearing measurement
noise. The motion term's mean is .063055 rad. A directional error during one
correlated motion segment does NOT prove a persistent actuator bias.

The official dataset describes its odometry records as forward/angular
velocity commands, and its range/bearing observations as monocular barcode
measurements. Thus command-to-motion mismatch is a plausible interpretation
to test, not an identified cause from this audit alone.
[Official MRCLAM dataset description](https://asrl.utias.utoronto.ca/datasets/mrclam/).

Range errors are not homogeneous across objects. With reference poses/maps,
subject 7 has mean error -.486809 m (5 observations), whereas subject 9 has
+.049482 m (25) and subject 17 has +.043627 m (26). These small, dependent
conditional samples do not establish a sensor-wide bias law.

The GT-free check provides complementary evidence: R5 has 62 repeated pair
comparisons covering 65 current observations. Distance-difference centred SD
is .173154 m. Under joint-QR, mean normalized squared difference is 2.016,
but the median is only .05693; the largest seven comparisons account for
81.03% of the total normalized energy. The largest absolute distance change
is .654727 m. This is concentrated inconsistency, not evidence that all R5
observations need the same variance increase. Shared pair endpoints prevent
interpreting those seven comparisons as seven independent outliers.

## D4/R2: Single-Object Support and Error Compensation

All 21 forecasts concern subject 7: 11 after 60 s and 10 after 90 s. There
are no usable simultaneous-object pairs for this robot in these windows.

| Range diagnostic | Mean (m) | Centred SD (m) | RMSE (m) |
| --- | ---: | ---: | ---: |
| Original innovation | +.167564 | .067811 | .180765 |
| Reference pose and map | +.133402 | .044849 | .140739 |

The remaining reference discrepancy means that state/map error alone cannot
explain the predictions. However, this robot/object sample cannot distinguish
robot calibration, viewing conditions, object-specific effects and reference
model error. It does not justify a new robot-wide bias parameter. Other past
network observations may offer additional support; no claim of whole-system
unobservability is made.

Signed mean contributions explicitly show cancellation:

    +.133402 measurement/reference
    +.353758 map
    +.003260 motion
    -.322855 checkpoint state
    = +.167564 total range innovation.

Reversed pose/map substitution gives map +.351729 m and checkpoint -.320793 m:
similar cancellation in this chosen alignment, not an order-free causal result.
Replacing only the pose by its reference increases range RMSE to .494182 m;
replacing both pose and map gives .140739 m. The first replacement breaks a
compensation between errors. It does NOT imply that reference poses are worse
or constitute a comparison between usable algorithms.

The reference discrepancy also changes across windows: mean +.167268 m after
60 s versus +.096149 m after 90 s. A constant bias law is not established.

## Additional Failure and Limits

D4/R4 must not be hidden by focusing on the two initial counterexamples. Its
88 predictions have range RMSE .173650 m, while reference pose/map residuals
have RMSE .222658 m: compensation again makes the total look smaller. Subject
11 has mean reference range error -.502949 m with centred SD .002802 m (6
observations); subject 17 has -.004687 m and .006758 m (20 observations).
One homogeneous model per robot would still mix these different conditions.

All five robots, both datasets and all declared windows are retained in the
artifact, including empty cells. The examples above are diagnostic, not blind
replication or independent significance tests. Their conclusions do not identify
a particular camera extrinsic error, prove stationary bias, or provide automatic
labels for good and bad factors.

## Consequences For Adaptive Robustness

A prediction error mixes imperfect state, motion model discrepancy, observation
error, and interactions. Estimating their sum as sensor variance can make a
model more confident in the wrong direction or unnecessarily weaken good data.
Variance calibration addresses stochastic scale within a specified model;
it does not automatically handle biased means, conditional tails, correlated
errors or insufficient evidence. Robust weighting alone also need not tell these
causes apart.

The current evidence supports keeping the five-component model as a controlled
reference, not adding robot/object-specific parameters immediately. The next
bounded test should measure sensitivity of those EXISTING fits to removal of
whole observation frames or object-support blocks. Correlated rows should not
be treated as independent replication. Re-evaluate on the SAME future support
without selecting a winner using GT. This would test whether a few conditions
dominate calibration before introducing a more complex likelihood or adapting
additional parameters. It is proposed here, not implemented in this stage.

The earlier D1/D4 state-quality gains remain valid under their stated protocol,
but are not new results of this audit. D2's high-noise failure against old fixed
weights remains open. No universal robust performance, deployed online noise
tracking, distributed calibration, or superiority over riMESA is claimed.

## Verification And Reproduction

Primary invocation (output must not already exist):

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m experiments.audit_predictive_error_sources \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_predictive_error_sources_20261004 \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

Focused suite:

```sh
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q tests/test_predictive_error_sources.py \
  tests/test_network_noise_transfer.py tests/test_observation_pair_evidence.py
```

- Focused: 17 passed, including 6 new diagnostic tests developed red-before-green.
- Existing native suite: 62 passed. Python regression suite: 156 passed.
  Full commands/environment and JUnit files are archived in
  `verification_commands.json`, `native_tests.xml` and `python_tests.xml`.
- A second deterministic audit with the final source took 13.92 seconds. Its
  four result JSONs exactly match the primary audit and saved stratification.
- No new optimization or PCG run; MPI ranks: 0. CCI and production code unchanged.
- `source/` preserves the primary-run source before post-selected stratification;
  `verification_source/` preserves the final code, tests, plan and this report.
- `verification.json` records archive/parity checks. `SHA256SUMS` covers all
  final artifacts except itself, with a read-back verification.
