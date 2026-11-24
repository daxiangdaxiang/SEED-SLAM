# Network Evidence For Noise Calibration: A Bounded Diagnostic

Date: 2026-10-03. Research branch only; no production, commit or push changes.

## Question

Does shared-object network structure provide useful evidence for calibrating
relative physical information, beyond fitting each robot's map independently?
Does improvement require joint Q/R feedback, or mainly the odometry component?

Here Q denotes odometry covariance and R bearing/range covariance. This work
reuses classical fixed-linear restricted likelihood (REML); neither REML nor
joint covariance estimation is claimed as new. The new result is a controlled
diagnosis of the present native multi-robot graph, not an online adaptive method.

## Protocol

Use the existing D2 common past, ending at elapsed 21.877000093460083 seconds.
It has 247 poses, 242 odometry factors and 355 bearing-range factors. Complete
this prefix under nominal noise with native LM. Export the physical whitened
Jacobian and residual at that common solution, excluding only robot 1's fixed
root. No consensus penalties, artificial gauge priors, GT, or future factors
enter the calibration.

Fit one set of five multipliers shared by all robots: longitudinal/lateral/yaw
odometry variance, bearing variance, range variance. All start at one. No
distance term, new prior, variance floor, robust kernel or ATE-selected threshold
is added. The inherited numerical log bounds are [-16,10]; all current estimates
are interior, but interior estimates are not evidence of correct calibration.

For whitened model `b = A delta + epsilon`, form contrasts `z = C^T b`, where
`C^T A = 0`, and fit the existing normalized Gaussian contrast likelihood with
covariance `C^T diag(D q + base) C`. Contrasts eliminate arbitrary changes within
this fixed Jacobian's state subspace, not nonlinear error, bias or contamination.

Primary arms:

1. Nominal Q/R, without learning.
2. Joint R-only: fit bearing/range multipliers, Q remains nominal.
3. Joint Q/R: fit all five using shared object columns.
4. Split Q/R: fit all five after splitting each robot's object columns.

The split control retains the SAME factors, states and nonlinear linearization;
it relaxes shared-object equality in the linear model. It is NOT a complete
independent-robot pipeline, and does not remove all historical network influence.
Its raw contrast NLL is not directly comparable to joint NLL because the
contrast dimensions differ. All final state solves retain network object sharing.

Freeze fits and apply only to suffix factors through 120 seconds; prefix
covariances stay unchanged. Each arm starts from the same previously archived
nominal online endpoint. Use original data and the three fixed Gaussian bearing
streams from the previous stage. On injected streams add the KNOWN 0.0225 rad^2
after applying the learned base-bearing variance. This supplied information is
an oracle for the injected component, not automatic detection of noise change.

Run LM for 4 data cases x 4 arms, plus GN on original and seed 20261003: 24
terminal solves. Fit once on the common past; no refitting across noise seeds.
All solves cap at 100 iterations/600 seconds; each fitting subprocess also has
a 600-second watchdog. GT is loaded only after every estimator terminates.

## Calibration And Network Support

| Calibration | Q variance multipliers (x,y,yaw) | Bearing multiplier | Range multiplier | Contrast DOF |
| --- | --- | ---: | ---: | ---: |
| Joint R-only | 1, 1, 1 | 0.002425 | 0.61176 | 658 |
| Joint Q/R | 0.20569, 0.31277, 1.32006 | 0.002255 | 0.76161 | 658 |
| Split Q/R | 0.29176, 0.47315, 1.28041 | 0.002494 | 0.64925 | 646 |

There are 1,436 physical scalar residuals. Joint A has 778 columns and rank 778;
split A has 802 columns and rank 790, including 12 independent-map gauge null
directions. Network sharing supplies 12 additional residual-contrast dimensions,
not new independent sensor samples. All fitted variance-component models have
full structural rank. Joint Q/R log-coordinate Fisher eigenvalues range from
9.40 to 143.08 within the assumed fixed-linear model.

The fitted bearing standard deviations are only 0.00237--0.00250 rad. The prior
posthoc reference check found original prefix bearing centered SD 0.02299 rad.
These are different quantities: reference errors include bias/reference error,
whereas the fit assigns portions of discrepancy to unknown states and other
noise components. Do NOT call the smaller number verified sensor precision.
Full rank and an interior solution establish neither a correct noise family nor
out-of-sample uncertainty calibration.

## State Results

All metrics below use one global SE2 alignment per final state. Units are meters.
Gaussian columns are arithmetic means over three fixed seeds on ONE sequence,
not independent datasets or a significance claim.

| Weighting | Original trajectory / object RMSE | Gaussian mean trajectory / object RMSE |
| --- | ---: | ---: |
| Original fixed weights, ignore injection (previous archive) | 0.31001 / 0.23951 | 0.32648 / 0.25889 |
| Nominal base + known injection variance | 0.31001 / 0.23951 | 0.41427 / 0.31243 |
| Joint R-only + known injection variance | 0.18424 / 0.17427 | 0.48012 / 0.33826 |
| Joint Q/R + known injection variance | 0.16187 / 0.17341 | 0.36303 / 0.30465 |
| Split Q/R + known injection variance | 0.17079 / 0.17411 | 0.41625 / 0.32837 |

On original data the injected variance is zero. The uncorrected fixed-weight
row is deliberately retained: it outperforms all new calibrated arms on the
Gaussian conditions. Comparing only to the weaker corrected-noise control would
give a misleading impression of superiority over the current fixed baseline.

Detailed Gaussian final trajectory / object results:

| Seed | Nominal + injection | Joint R-only | Joint Q/R | Split Q/R |
| --- | ---: | ---: | ---: | ---: |
| 20261003 | .42205 / .30652 | .48911 / .33074 | .36620 / .29786 | .42129 / .31828 |
| 20261004 | .41035 / .30582 | .47670 / .33192 | .36058 / .29398 | .41349 / .32026 |
| 20261005 | .41041 / .32495 | .47456 / .35213 | .36232 / .32210 | .41398 / .34656 |

Joint Q/R improves original trajectory RMSE by 47.79% over nominal. Under
matched supplied injection variance, it improves Gaussian mean trajectory RMSE
by 12.37% over nominal and 12.79% over split calibration; object improvements
are 2.49% and 7.22%. These results support a useful role for network constraints
in this calibration experiment. They do not establish online adaptation,
cross-dataset transfer, fault resistance, or superiority over robust baselines.

## Post-Selected Q-Only Intervention

After reading the primary results, retain only the three Q multipliers from each
Q/R fit and restore R multipliers to one. Add the same known injection variance.
No refitting is performed. This is NOT the constrained optimizer that learns Q
while holding R fixed, and it is not an independently preregistered validation.

Eight further LM and four GN solves give:

| Intervention | Original trajectory / object RMSE | Gaussian mean trajectory / object RMSE |
| --- | ---: | ---: |
| Apply joint-fitted Q only | .25549 / .23044 | .33358 / .28820 |
| Apply split-fitted Q only | .27498 / .23312 | .36328 / .29827 |

Under injected noise, keeping only the network-fitted Q improves the trajectory
more than applying the complete Q/R fit (.33358 vs .36303 m). On original data,
the complete Q/R fit is much better (.16187 vs .25549 m). Therefore the two
components help differently across conditions; an across-the-board adaptive
covariance success claim is unsupported. Q-only still trails original fixed
weights on Gaussian mean trajectory and object metrics (.32648/.25889 m).

This is evidence against attributing the Gaussian improvement exclusively to
observation-noise learning. Directional state-evolution weighting matters. It
does not identify a unique physical noise cause: changing Q also changes the
state fitted to observations, and the Q-only values were originally fit jointly.

## Numerical And Implementation Checks

All 40 new processes exit zero: one prefix solve, three fits, 24 primary terminal
solves and 12 intervention solves. No covariance setting was swept against GT.
Four nominal-control completions reproduce the previous archive exactly in
states and final cost. Training export satisfies physical cost and global
gradient parity; fixed/shared key semantics are checked by deterministic tests.

GN/LM maximum state discrepancy is 4.66e-6 m in primary tests and 4.14e-6 m in
Q-only tests; maximum respective cost discrepancies are 4.98e-10 and 1.93e-10.
All primary LM runs stop in 6--18 iterations. Largest primary terminal gradient
norm is 0.001056; GN stops with no further decreasing trial, not a formal
gradient or global-optimality certificate. These differences are negligible
relative to the reported centimeter-scale changes.

No routed/PCG experiment is added this stage: terminal solves and dense REML are
centralized diagnostics. PCG iterations: not applicable; MPI ranks: 0. Production
optimizer, communication logic, CCI and D-CCI are unchanged. No distributed
calibration runtime/traffic or real-time claim is made. Individual fitting
processes take roughly 3--7.4 seconds on this host, not an online latency result.

Verification: 62 C++ and 139 Python tests pass, including 4 pre-existing REML
tests and 3 new native-export/suffix-weight/group-partition tests. Default native
audit state output remains byte-identical when optional linear export is enabled.
The only C++ modification adds that post-solve diagnostic export.

## Interpretation And Next Gate

The coherent direction is **network-supported relative information calibration**:
shared objects restrict how each robot can explain discrepancies through its
own trajectory/map. That can improve the weights learned from the same physical
data. It is not evidence that every mismatch is sensor noise or that all network
information is independent. It also does not prove a new statistical estimator.
Existing [joint state/noise estimation research](https://arxiv.org/html/2502.04584v2)
already treats covariance learning and notes its degeneracy and outlier limits.

Do not deploy these fits yet. The pooled five-component family assumes common
noise statistics across robots and ignores bias, temporal dependence, range
dependence and calibration uncertainty. It uses one nominal nonlinear chart,
not a jointly converged nonlinear covariance fit. The split control is a
fixed-chart relaxation rather than an independent end-to-end algorithm. Fitting
and final evaluation are batch operations, and injected variance is supplied.

Next, freeze this small model family and retain the established D1 failure and
D4 positive cases as transfer controls. Validate prediction before assimilation,
not just fitted-state RMSE; do not equate the tiny fitted bearing variance with
reliable uncertainty. The present Q-only intervention motivates examining the
state-prediction model first, but is not an ATE-selected replacement default.
Only after those checks should a compact distributed calibration mechanism be
considered. Adding more gates or priors merely to hide a failure is not the
intended research contribution.

## Artifacts And Reproduction

Primary archive: `results/dranppr_network_noise_support_20261003/`.
Post-selected interventions: `results/dranppr_network_noise_q_only_20261003/`.
Each retains inputs, fits, native binary, state outputs, commands/status, source
provenance and checksums. Initial executed runner and later intervention runner
versions are both retained rather than overwriting experiment provenance.

```sh
cmake --build /tmp/dranppr_backend_build --target native-joint-audit -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_network_noise_support run \
  --output /tmp/reproduce_network_noise \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_network_noise_support intervene \
  --source /tmp/reproduce_network_noise --output /tmp/reproduce_network_noise_q_only \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

Exact regression commands, XML results and logs are recorded in the primary
archive's `test_manifest.json`. Research branch remains
`research/dranppr-evidence-online`; packaged DBA-DRANPP remains clean.
