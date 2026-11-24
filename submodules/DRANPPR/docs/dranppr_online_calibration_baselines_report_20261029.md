# Online Calibration Against Native DRAN and riMESA

Date: 2026-10-29. Research-only Pose2/Point2 prototype. No production changes.

## Decision

The old first-prefix Q/R calibration transfers to a persistent causal backend:
it improves final trajectory and object-copy RMSE on both D1 and D4, versus
fixed-noise native DRAN, paper-iMESA and repaired riMESA in this protocol.
However, D1's as-published causal trajectory becomes substantially WORSE.
Therefore final smoothing accuracy is not a sufficient success criterion for
the proposed robust-online method. Do not promote this candidate to production.

A separate six-case diagnostic propagates uncertainty in the fitted scales.
It softens several bad predictive scores but does NOT repair the observed
failures. These results do not justify arbitrary covariance inflation or extra
robot/object bias states.

## Fixed Protocol

- MRCLAM D1/D4, five robots, chain communication, known barcode identities.
  Same archived events and first-connected-past primal states. Replay ends at
  120 s, not the end of the approximately 1491/1390 s raw missions.
- Our baseline is native `joint_cached_one --lazy-refresh`, nominal Q/R. This
  is the research 2D backend, NOT packaged production 3D DRAN++.
- Candidate changes only suffix variances to the previously fitted shared
  five-vector: three process channels, bearing and range. Prefix remains
  nominal. No new fits, thresholds, robust kernels or optimization budget.
- Learned parameters are externally supplied and frozen. Their estimation was
  centralized on the first connected prefix, not continuously distributed.
- Pinned riMESA `ffddc21dad530ba0b0e57c42495146b410581fc4`;
  riSAM `c6fa05eaece103efd7e44e640bda8df6eff30e8d` with the previously disclosed
  Newton-endpoint cancellation and merit-consistency numerical repairs.
  Its robust kernel is unchanged. Label this repaired riMESA, not unchanged
  official upstream. Paper-iMESA is a Gaussian consensus control.
- Baselines start with the same primal values and physical prefix graph, but
  cold dual/kernel/internal optimizer states. All methods use the same gauge.
- Two baseline schedules: one full communication sweep per arriving frame;
  and a causal cumulative byte ceiling equal to the smaller of the TWO native
  arms' cumulative application-byte ledgers at each frame. Credits never use
  future frames. Byte ceilings are not exact equal-byte or equal-time runs.
- Sequential estimators, one BLAS/OMP thread, 600 s timeout per estimator.
  All 12 runs succeed. GT is loaded only after all estimators terminate.
- State evaluation uses one global SE(2) registration across all robots, never
  independent per-robot alignment. Object error counts observed robot copies.
  Causal error uses first-published suffix poses, never future revised states.

## Fresh Comparison

All state errors are meters, lower is better. `Service` is measured cumulative
native update/coordination/solve time, not end-to-end deployment latency.
Application MB is decimal, includes recorded optimizer/control messages but
not transport framing or calibration/bootstrap communication.

| Data | Method / schedule | Final trajectory | Final objects | Causal trajectory | Service (s) | MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D1 | Native nominal | 0.266416 | 0.180559 | 0.251488 | 6.266 | 29.116 |
| D1 | Native calibrated | 0.157835 | 0.113145 | 0.492764 | 6.174 | 29.135 |
| D1 | Paper-iMESA / one | 0.303615 | 0.270060 | 0.396938 | 3.052 | 3.714 |
| D1 | Paper-iMESA / bytes | 0.302832 | 0.267800 | 0.390840 | 17.423 | 29.115 |
| D1 | Repaired riMESA / one | 0.375762 | 0.498560 | 0.523592 | 19.544 | 3.714 |
| D1 | Repaired riMESA / bytes | 0.366174 | 0.489836 | 0.495361 | 53.371 | 29.115 |
| D4 | Native nominal | 0.285680 | 0.184796 | 0.271750 | 9.596 | 50.555 |
| D4 | Native calibrated | 0.186919 | 0.161053 | 0.201437 | 9.522 | 50.564 |
| D4 | Paper-iMESA / one | 0.281216 | 0.254721 | 0.290774 | 4.282 | 5.796 |
| D4 | Paper-iMESA / bytes | 0.280844 | 0.255369 | 0.290196 | 29.296 | 50.555 |
| D4 | Repaired riMESA / one | 0.371779 | 0.539241 | 0.408952 | 28.227 | 5.796 |
| D4 | Repaired riMESA / bytes | 0.371428 | 0.540944 | 0.411905 | 88.744 | 50.555 |

Against native nominal, candidate final trajectory improves 40.76%/34.57%,
objects 37.34%/12.85% on D1/D4. Causal trajectory worsens 95.94% on D1, improves
25.87% on D4. D1 is not a marginal or metric-rounding regression.

Native calibrated uses 7.84/8.72 times the one-sweep baseline communication.
It must not be advertised as dominating one-sweep riMESA at equal bytes.
The byte-ceiling rows retain the final-state advantage with almost the same
suffix application traffic. Candidate uses slightly more than the common
ceiling because that ceiling is the smaller native arm (0.068%/0.017% extra).
One-sweep paper-iMESA is faster and uses less communication than native DRAN.
The deliberately byte-filling schedule is NOT the only runtime baseline.

The archived scale-fit subprocesses alone cost 13.613 s (D1) and 1.776 s (D4)
in the old environment, excluding other prefix preparation. Those costs are
NOT in this suffix service table; their distributed communication is not
implemented/accounted. Adding old fit time to new service time is not a fresh
end-to-end timing measurement. Current process wall times, including I/O, are
also retained in `manifest.json`. Timings are one realization, not repeated
independent performance trials.

## What D1 Reveals

The online state is allowed to revise historical poses. A trajectory evaluated
at 120 s therefore differs from the trajectory actually delivered to a robot
at each measurement time. Calibration can improve the former and harm the latter.

Post-selected checks do not refit or retune anything:

- All five D1 robot groups have worse causal errors under the same all-robot
  causal alignment; R5 rises from 0.199359 to 0.581304 m. Its largest causal
  position error is 1.626598 m, versus 0.522353 m for nominal.
- D1 causal RMSE under a fixed COMMON PREFIX alignment also worsens:
  0.320923 -> 0.763417 m. Thus the main sign is not an artifact of choosing a
  separate best global alignment for each arm.
- Under candidate's single terminal alignment, the SAME 1103 suffix poses have
  RMSE 0.652360 m when first published and 0.143010 m after terminal revision.
  The largest change of a previously published pose is 2.105485 m. This directly
  demonstrates temporal inconsistency between published and revised estimates.
- Candidate's 30--60 s causal RMSE is 0.616815 m versus nominal 0.332662 m;
  60--90 s is 0.422236 versus 0.209877 m. Group diagnostics reuse a global
  alignment; they are not independently aligned favorable windows.
- D4's causal gain under best global alignment remains, but under the common
  prefix alignment it is tiny: 0.290854 -> 0.289938 m. Gauge/alignment convention
  matters for interpreting practical online delivery.

This does NOT yet identify whether D1 is dominated by early physical
underconstraint, finite-work optimizer tracking, inherited nominal-prefix error,
or changed statistical weights. One PCG interface iteration per frame is an
explicit inexact budget, not convergence: maximum relative residuals are
2.053/2.343 for D1 nominal/calibrated and 2.408/3.965 for D4. Small unit-test
linear-solve residuals must not be substituted for those online measurements.

Native shared objects agree by construction (zero copy disagreement).
Repaired riMESA terminal maximum copy discrepancies are 1.685/1.637 m on D1
and 3.378/3.403 m on D4 for one/bytes schedules. Gaussian paper-iMESA also
retains 0.565--0.620 m discrepancies. These are adapter/protocol outcomes, not
a theorem that all robust consensus methods are inconsistent. No cost ranking
is made across different kernels, weights or inconsistent shared copies.

## Separate Parameter-Uncertainty Diagnostic

Six declared discovery cases: D1/D4 at 30, 50, 85 s, forecast horizon 5 s.
These are selected batch checkpoints, not additional online comparison arms.
No GT or future observations enter parameter fitting or checkpoint optimization.

For log variances theta, compute expected REML Fisher information from the
ORIGINAL first-prefix contrasts, invert it without a floor/prior, and use
ten equal positive cubature points in log-scale space. At every point, re-solve
the same past-only nonlinear graph, then recompute conditional future means and
covariances. Average predictive DENSITIES, not NLLs. Between-mean variance is
retained. No point is selected by its result.

This is local asymptotic sampling-uncertainty sensitivity, NOT an exact Bayesian
posterior, misspecification-robust sandwich, or validated online uncertainty
algorithm. Calibration/state dependence and cubature accuracy remain limitations.
We do not label a moment-matched Gaussian ellipse as a calibrated mixture region.
We do not manufacture a Gaussian predictive covariance for robust riMESA.

| Data / start (s) | Nominal NLL | Plug-in calibrated NLL | Mixture NLL |
| --- | ---: | ---: | ---: |
| D1 / 30 | -1.889472 | -2.636300 | -2.633419 |
| D1 / 50 | 1.303858 | 7.751236 | 6.607390 |
| D1 / 85 | -1.644383 | -2.167020 | -2.165928 |
| D4 / 30 | -1.616870 | -1.975191 | -1.990039 |
| D4 / 50 | -1.425635 | 0.282175 | 0.133975 |
| D4 / 85 | -1.386331 | -1.083512 | -1.137616 |

The three selected cases that lost to nominal STILL lose. D1/R5/subject6 at
50 s retains 0.674461 m mean signed range error, with mean predicted SD only
0.114444 m (plug-in: 0.675274 and 0.113361 m). This particular parameter-uncertainty
correction cannot explain away the discrepancy. It does not exclude other
uncertainty methods or prove a unique physical bias. Changing the conditional
mean/dependence model remains a separate question from uncertainty in five scales.

## Verification and Implementation

- New opt-in `--suffix-variance-multipliers` in research `joint_replay.cpp`.
  Explicit ones preserve default state/control outputs byte-for-byte. Tests
  cover insertion loss, unchanged prefix, future-data causality and invalid input.
- Restored pinned riMESA/riSAM dependencies under research `build/`.
  All ten archived code files of each baseline source snapshot match restored
  sources exactly. riSAM status discloses only the two old numerical repairs.
- Fresh D4 nominal, paper-iMESA-one and repaired-riMESA-one final AND causal
  state files reproduce historical archives byte-for-byte (six comparisons).
- Common physical graph/factor support, gauge, arrived-frame ledgers and
  per-frame byte ceilings verified. Independent reconstructed terminal graphs
  match native physical objectives within 9.24e-16 relative error. These four
  one-iteration batch audits are not included in estimator quality or timing.
- Uncertainty: all 60 nonlinear solves finish, at most nine LM iterations.
  4510 component predictions independently checked. Maximum weight error
  1.11e-16, score reconstruction error 1.71e-13, mixture-score error 3.55e-15,
  QR residual 3.06e-15. Maximum posthoc gradient 3.70e-4 is not a global optimum
  certificate. Fisher spectra/correlations reproduce the original fit.
- Selected Python regression: 187 passed, zero skips. Native C++: 62 passed.
  Schur residual 3.25e-16, matrix-free product error 3.53e-17. Unit-fixture
  PCG: six iterations, residual 1.98e-16; 2/3/5/10 robots: 12/16/23/39 iterations.
  MPI ranks: zero. No CCI/D-CCI change or new MPI validation.
- Production `DBA-DRANPP` remains clean. No agents, commits or pushes.

Artifacts: `results/dranppr_online_calibration_baselines_20261029` and
`results/dranppr_noise_parameter_uncertainty_20261029`. Both retain inputs,
commands, numerical outputs, failed hypotheses, source snapshots and SHA256SUMS.
Exact commands and test selections are in the comparison's validation manifest.

## Next Gate

Before another mechanism, separate two causes on the SAME arriving graph:
does a stronger past-only solve at the failing time already correct the state,
or do only subsequent observations correct it? The former motivates a
statistically consistent finite-work update; the latter requires observability
or model adequacy, not more optimizer iterations. Compare both against native
nominal and repaired riMESA, and retain first-published errors as a primary metric.

Still open: untouched sequences, the retained D2 high-noise negative, outliers
and wrong identities, continuous decentralized noise learning, bounded memory
and latency, actual network execution, and ObjectPose3 transfer. This round
closes a fresh baseline comparison and one uncertainty diagnostic, not the full
robust-online claim.
