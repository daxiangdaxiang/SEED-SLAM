# Motion Time Structure And Predictive Score Attribution

## Main Result

The most important result is a correction to the failure interpretation:
**motion dominates D1/R5's bearing residual, but bearing is NOT the channel
responsible for its adaptive-versus-nominal predictive score regression.**
Bearing scores improve for every robot in D1 and D4. The D1/R5 and D4/R2 total
score regressions come from range, whose degradation outweighs bearing gains.
Both robots also have worse range prediction means and smaller predicted range
variances after adaptation. This does not identify a causal remedy, but rules
out treating the largest physical residual as automatically the source of the
relative performance loss.

The frozen motion audit finds sustained signed heading discrepancies, but no
large positive short-lag increment correlation in the targeted D1/R5 case.
Its mean squared error normalized by the EXISTING yaw variance stays below one
at all four audited horizons. Increasing process noise or adding an AR/bias
state is therefore not justified as the immediate repair of this score failure.
No estimator, noise scale, robust kernel, or gate was changed.

## Protocol And Scope

- Reuse the checksummed network-noise transfer archive (360 files) and previous
  error-source archive (25 files). D1/D4, all five robots, original joint-QR
  checkpoints at elapsed 30/60/90 s and five-second forecast windows.
- GT-free input check: raw velocity commands reproduce all 3,114 archived SE2
  increments through 120 s exactly: 1,409 in D1 and 1,705 in D4.
- Reference-assisted motion analysis: reset to the reference pose at each
  robot's actual last state timestamp, then propagate only unchanged odometry.
  Do not reset at the global checkpoint when the robot state is older.
- Audit 391 distinct forecast pose frames. The 533 observations occupy 386
  frames; repeated objects in one frame are not independent motion samples.
  D1/R5 has 83 observations on 41 frames, not 83 independent trajectories.
  Five robot/windows have no observations; four also have no forecast frames.
- Separately integrate raw commands on uniform, disjoint one-second intervals:
  D1 [28,120] s, D4 [16,120] s, all robots; 980 intervals total. These begin
  after their respective calibration prefixes. No GT-based segment selection.
- Sum into disjoint 2/5/10-second blocks. D1 drops 0/2/2 trailing seconds at
  those resolutions; D4 drops 0/4/4. Blocks at DIFFERENT resolutions reuse
  data and are not independent experiments.
- GT is used only for this labelled diagnostic, never for calibration or state
  feedback. No new nonlinear solve, parameter search, or benchmark comparison.
- After inspecting the motion audit, add an explicitly POSTHOC score-channel
  diagnostic on all original nominal/joint-QR forecasts. It uses no reference
  states or landmarks, verifies identical support, and fits nothing. It is
  not a preregistered independent confirmation.

The official dataset describes odometry as logged forward/angular velocity
commands, making command-to-motion mismatch plausible. This alone does not
identify actuator bias. Reference interpolation, endpoint error and timing
also affect increment diagnostics.
[Official MRCLAM description](https://asrl.utias.utoronto.ca/datasets/mrclam/).

## Motion Model Check

Let e_k be integrated command yaw minus the reference yaw change over duration
dt_k. The current white process model assigns variance s * dt_k, where
s = .05^2 * q_yaw and q_yaw is the already fitted, frozen scale. We report

    energy(T) = average[ (sum of increments over T)^2 / (s * T) ].

One is the model's expected squared-error ratio under its assumptions, not a
per-run acceptance threshold. These finite, dependent sequences do not support
an independent-sample significance claim. In particular, energy below one does
NOT establish unbiasedness, correct tails, or correct joint covariance.

To separate signed drift from scatter descriptively, define b = sum(e)/sum(dt):

    sum e_k^2/(s*dt_k)
      = (sum e_k)^2/(s*sum dt_k)
        + sum (e_k-b*dt_k)^2/(s*dt_k).

This is an exact weighted projection identity. The reported b is NOT an online
estimate, a stationary actuator parameter, or a correction passed to DRANPPR.

| Dataset/robot | Mean rate (deg/s) | Lag-1 correlation | Energy 1 s | 2 s | 5 s | 10 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D1/R1 | +.23431 | -.0699 | .4001 | .3622 | .3276 | .3785 |
| D1/R2 | +.13498 | +.0020 | 1.1445 | 1.2055 | 1.2407 | .8884 |
| D1/R3 | -.32463 | -.0136 | .8085 | .7752 | .7362 | .7126 |
| D1/R4 | -.23845 | -.3025 | 1.4368 | .6635 | .5730 | .2732 |
| D1/R5 | +.68962 | -.1553 | .8032 | .6151 | .5819 | .8393 |
| D4/R1 | +.21122 | +.0955 | .7053 | .7372 | .6436 | .3523 |
| D4/R2 | -.75974 | -.2980 | .7016 | .4969 | .6276 | .9064 |
| D4/R3 | +.52654 | -.0008 | .8156 | .6912 | .9661 | 1.1233 |
| D4/R4 | +.30234 | -.0995 | 1.4485 | 1.3400 | 1.1796 | 1.1900 |
| D4/R5 | +.15050 | +.0364 | .4114 | .4865 | .2923 | .3202 |

Correlations are computed on ONE-second errors, not accumulated trajectories.
D1/R5 lag-2/5 correlations are -.0298/+.0227. All nine complete ten-second
blocks have positive net error; all ten D4/R2 blocks have negative net error.
The constant-rate projection represents 81.37% and 70.99% of their respective
ten-second squared-error energies, despite mean energies .8393 and .9064.
Thus directional accumulation can coexist with an adequate or conservative
short-horizon variance magnitude. D1/R5's first/second-half mean rates differ
(.01696/.00712 rad/s), so these data do not establish a constant bias law.

This is not evidence that the white model is universally adequate: D4/R4 is
above one at every horizon, and particular forecast windows have larger
errors. Reference differencing shares endpoints and may itself induce negative
correlation. No p-values or identified correlation-time parameters are claimed.

## What Explains D1/R5 Bearing Error?

For each original forecast observation, hold its archived landmark estimate
fixed. The previous motion-bearing term splits exactly (modulo angle wrapping)
into propagated heading error and a position-geometry remainder. This is an
algebraic decomposition, not independent physical error sources.

| D1/R5 term, 83 observations | Mean (rad) | RMSE (rad) |
| --- | ---: | ---: |
| Motion bearing discrepancy | .063055 | .080220 |
| Heading contribution | .059484 | .076073 |
| Position-geometry contribution | .003571 | .005399 |

The cross term is nonzero; do not add component RMSEs or interpret their
squares as independent variance contributions. The previous motion-bearing
attribution is reproduced exactly on all 533 observations.

At the 90-second checkpoint, R5's initial yaw error is -.000154 rad and its
state age is .172 s. Over the next 26 pose frames, motion yaw RMSE is .08515
rad; mean normalized cumulative energy is 1.37347. At the last frame,
propagation lasts 5.10 s, yaw discrepancy is .07176 rad, and the existing
process-only standard deviation is .10563 rad. Thus an inaccurate starting
heading is not the main explanation of THIS window, but its error magnitude
also does not establish a grossly undersized Q. Other windows differ.

## Which Channel Actually Regresses?

The original score is the two-dimensional marginal Gaussian NLL of each
future range/bearing observation. It is not a joint likelihood across time.
For innovation (b,r) and covariance S, preserve the off-diagonal covariance:

    NLL(b,r) = NLL(b) + NLL(r | b)
             = NLL(r) + NLL(b | r),
    r | b: residual = r - S_rb/S_bb * b,
           variance = S_rr - S_rb^2/S_bb.

Both orders are evaluated by the exact Gaussian chain rule. The two marginal
NLLs cannot simply be added unless covariance coupling vanishes. Scores are
in unchanged radian/metre coordinates; differences below compare SAME-channel,
matched observations, not the absolute magnitudes of unlike-unit densities.

All values below are joint-QR minus nominal; negative means improvement.
Each row uses all available observations for that robot, not a selected window.

| Dataset/robot | Observations | Joint NLL change | Bearing marginal | Range given bearing |
| --- | ---: | ---: | ---: | ---: |
| D1/R1 | 68 | -.763756 | -.385603 | -.378153 |
| D1/R2 | 34 | -.826115 | -.306269 | -.519846 |
| D1/R3 | 46 | -.564174 | -.248942 | -.315232 |
| D1/R4 | 20 | -.905480 | -.452368 | -.453112 |
| D1/R5 | 83 | +.143069 | -.308434 | +.451503 |
| D4/R1 | 36 | -.391528 | -.034873 | -.356655 |
| D4/R2 | 21 | +.284424 | -.195715 | +.480139 |
| D4/R3 | 70 | -.323650 | -.236993 | -.086657 |
| D4/R4 | 88 | -.097761 | -.128567 | +.030806 |
| D4/R5 | 67 | -.470949 | -.270207 | -.200741 |

Reverse conditioning preserves the interpretation. D1/R5 range marginal
NLL worsens by .467521 and bearing-given-range improves by .324453. For
D4/R2 these changes are +.495280 and -.210856. These are statistical score
decompositions, NOT causal contributions of separately changed algorithm knobs.

| Case | Range innovation RMSE, nominal -> joint (m) | Mean predicted range variance (m^2) |
| --- | ---: | ---: |
| D1/R5 | .174215 -> .194817 | .040990 -> .013218 |
| D4/R2 | .123477 -> .180765 | .041744 -> .013442 |

These are future observation errors, NOT GT trajectory errors. Variance here
includes state and process uncertainty as well as measurement noise; the table
does not isolate R_range. A worse mean and tighter covariance are both warning
signs. Average variance alone does not determine NLL; the exact per-observation
scores above establish the regression.

## Research Decision

1. Do not add a yaw bias/colored-noise state merely because bearing residuals
   look large or cumulative trajectories look smooth. That would target a
   channel whose comparative predictive score already improves in every robot.
   Motion model fidelity remains a separate longer-horizon question.
2. Preserve the positive result: existing five-scale adaptation improves
   dataset-average predictive scores, and previously improved D1/D4 state
   estimates. This audit does not erase those results or rerun state estimation.
3. Prioritize conditional RANGE reliability. Earlier diagnostics found object-
   dependent reference discrepancies and state/map compensation. The next
   bounded test should separate changes in prediction mean from covariance on
   the existing held-out range observations, grouped by object and frame,
   without introducing GT-derived corrections or choosing a best subgroup.
4. A qualitatively stronger adaptive method must distinguish uncertainty in a
   zero-mean model from repeatable conditional mean error and sparse gross
   inconsistency. A single learned variance can conceal these distinctions.
   Whether they can be identified from available GT-free redundancy remains
   an open experimental question, not a demonstrated new algorithm.

No new online-convergence, robustness, or riMESA superiority claim follows.
D2's high-noise negative result remains unresolved. No new default is proposed.

## Verification And Reproduction

- Maximum motion decomposition/process-yaw variance discrepancy: 1.11e-15.
  Previous bearing-motion and original NLL scores: exact parity (0.0 maximum).
- Motion rows: 391; bearing rows: 533; uniform increments: 980; score rows:
  1,066 (two arms on the SAME 533 observations), not independent replicates.
- Primary motion audit: 27.94 s wall time. This includes raw parsing/hashing
  and reference diagnostics; it is not a deployed algorithm runtime.
- Tests cover causality, exact integration, duplicate canonicalization, unequal
  duration energy identities, disjoint blocks, actual state age, both Gaussian
  conditioning orders, covariance coupling, and a full archived-fixture run.
- Native regression: 62 tests; Python regression: 171 tests, zero skipped.
  Full commands and logs are archived in `verification_commands.json`.
  No new optimizer solve, PCG iteration or MPI rank; CCI is unchanged.
- The final full-fixture rerun exactly reproduces all seven numerical JSON
  artifacts, including both posthoc score and motion summaries.
- Production `DBA-DRANPP` remains clean. Research changes only; no commit/push.

Two incomplete development outputs are retained with `ABORTED.md`: the initial
last-duplicate-row assumption failed raw parity because extraction averages
same-time commands; the corrected-input attempt exposed a variable-shadowing
bug in the auditor. Regression tests caught both before the final run. Neither
was an estimator failure, and no input data or tolerance was changed to pass.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.audit_motion_time_structure \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --attribution results/dranppr_predictive_error_sources_20261004 \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --output /tmp/dranppr_motion_time_structure_repeat

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.analyze_motion_time_structure \
  --output /tmp/dranppr_motion_time_structure_repeat

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q tests/test_motion_time_structure.py \
  tests/test_predictive_error_sources.py tests/test_network_noise_transfer.py
```

Primary archive: `results/dranppr_motion_time_structure_verified_20261004`.
`source/` is the primary audit snapshot; `verification_source/` contains final
tests, the posthoc analyzer, completed plan and report. Raw source hashes and
upstream archive hashes are in `manifest.json`; archive integrity is recorded
in `SHA256SUMS` and checked by readback before reporting completion.
