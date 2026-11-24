# Native Causal Replay: Cost, Tracking Lag, and Limits

Date: 2026-09-22. This completes the chronological local-replay diagnostic
(native design gate 6), not distributed gate 7 or the overall DRANPPR goal.
The result is a negative performance result with a useful cause-separation case.
The packaged DBA-DRANPP repository and legacy optimizer remain unchanged.

## Matched Controls

Eight serial replay runs use MRCLAM D1/D4/D9 for 120 seconds and D9 for 300
seconds, with all five robots independent. The two methods receive identical
timestamp-atomic frames, physical factors, declared barcode identities, and
fixed Gaussian covariance. Both use full-chain Pose2 odometry Jacobians.
Neither sees ground truth, future observations, or another robot's state.

- Native: persistent keyed local model, one full Gauss-Newton/Schur step per
  frame, at most 30 backtracks requiring true current-prefix cost decrease.
- iSAM2-exact: QR, relinearization threshold 1e-4, skip 1, wildfire 1e-3.
- Same initialization rule: previous own estimate plus arrived odometry, and
  first own polar sighting for a new landmark. Numerical starts can subsequently
  differ because the preceding estimates differ.
- Local root fixed exactly, by fixed-column elimination or constrained prior.
  No finite artificial prior is added to the physical loss.
- Bearing/range sigmas `(0.05 rad, 0.1 m)`; odometry sigmas
  `(0.1, 0.1, 0.05)*sqrt(dt)`. These are controls, not calibrated noise estimates.

This compares practical update policies, not equal flop or wall-time budgets.
Latency and quality are reported together. One step per frame is not bounded
incremental complexity, and neither arm here is full distributed riMESA.

## Implementation and Causality

`experiments/native_local/replay.cpp` adds optional `native-causal-replay` with
one persistent model per robot. Same-timestamp packets are grouped across
interleaved robot input. Empty-observation pose frames are not dropped. Source
IDs, parent indices, durations, finite data and observation support are checked.
Invalid streams exit unsuccessfully rather than silently skipping observations.

`experiments/audit_native_replay.py` runs solvers serially, records input/source/
binary/GTSAM hashes and failures, then loads GT only after all solvers exit.
Six subprocess contracts verify exact motion, timestamp atomicity, unchanged
past outputs after future mutation, pose-only frames, cost descent and malformed
input rejection. A separate test prevents double-counting nested timing fields.

Instrumentation separates preparation, insertion, optimization, extraction,
linearization, contribution updates, export, private factorization and interface
solve. Snapshot/linearization/contribution timers are nested in update; private
factorization is nested in response setup. They must not all be summed.
Frame service includes initialization but excludes JSON parsing, output CSV and
posthoc scoring; full solver-process wall time is also archived. Serial backlog
and independent per-robot backlog are distinct simulated service ledgers, not
measured network latency. Both methods materialize historical values each frame.

## Replay Results

All errors below use the existing posthoc, independently aligned local evaluation.
The causal trajectory contains each current pose as it was first output, without
later smoothing. Its alignment still uses evaluation GT, not an online gauge fix.

| Prefix | Final trajectory RMSE m, native / iSAM2 | Service work s, native / iSAM2 | Per-frame p99 ms, native / iSAM2 |
| --- | --- | --- | --- |
| D1, 120 s | 0.336198 / 0.336197 | 4.994 / 1.461 | 8.652 / 4.203 |
| D4, 120 s | 0.289545 / 0.289544 | 7.312 / 2.454 | 10.245 / 4.860 |
| D9, 120 s | 0.150499 / 0.150499 | 6.854 / 1.014 | 8.754 / 3.402 |

The short-prefix precision is essentially equal, but native costs 2.98--6.76
times more service work. The frozen-model correctness result is therefore not
an online efficiency improvement.

| D9, 300 s, all five robots | Native | iSAM2-exact |
| --- | ---: | ---: |
| Physical Gaussian loss | 3596.626 | 4413.151 |
| Final trajectory RMSE m | 0.809533 | 0.776570 |
| Causal trajectory RMSE m | 1.111420 | 1.103253 |
| Final object-copy RMSE m | 1.262388 | 1.265950 |
| Service work s | 60.197 | 9.613 |
| Full solver-process wall s | 60.499 | 9.895 |
| Per-frame p99 ms | 33.351 | 13.462 |
| Maximum serial backlog s | 0.098015 | 0.022796 |
| Maximum independent robot backlog s | 0.057473 | 0.019316 |

Both process 4541 frames / 9599 packets. The native loss is 18.5% lower, yet
trajectory error is worse and work is 6.26 times greater. It is not a Pareto
improvement. No run failed or timed out. All 9456 accepted native updates across
the four native runs decreased their own current-prefix physical loss; these
include overlapping D9 prefixes and are not independent statistical trials.

GT support exclusions are retained: 437 poses in D9's all-time evaluations.
The short D9 prefix also exposes a metric caveat: robot 1's estimated path spans
only about 0.036 by 0.042 m. Its position-only trajectory alignment gives
0.011 m trajectory RMSE but 0.882 rad yaw RMSE and 4.666 m object RMSE. This is
consistent with weak rotation determination from a nearly stationary path;
it is not evidence of precise map estimation. The evaluation convention was
not changed to improve reported numbers. No short-prefix object-accuracy claim
should be made without checking alignment conditioning.

## Actual Computational Bottleneck

The 120-second native runs spend 53.3--54.2% of frame work publishing the
accepted state and updating historical factors. Private factorization uses only
2.5--3.0%. D9 at 300 seconds confirms the result:

| Component | Share of total frame work |
| --- | ---: |
| Accepted-state publication, including nested counters | 50.45% |
| Publication snapshot, nested | 5.29% |
| Publication linearization, nested | 12.80% |
| Publication contribution updates, nested | 20.13% |
| New-frame insertion | 13.00% |
| Schur response setup, including factorization | 7.74% |
| Private factorization, nested | 2.08% |
| Sparse system export | 5.26% |
| Retraction and physical-cost evaluation | 6.10% |
| Interface solve | 3.26% |

Robot 5's mean service time grows from 2.472 ms in the first frame quartile to
28.685 ms in the last; history grows to 1034 values. Its iSAM2 control grows
from 0.225 to 6.925 ms. These are within-run growth diagnostics, not a fitted
asymptotic complexity theorem. Timing is one serial run per case, not replicated
hardware benchmarking.

The native model preserves factor caches, but the optimizer moves almost all
bases, invalidates almost all historical factors, copies O(history) model state,
and freshly factors the private system. Accelerating LDLT alone cannot repair
this bottleneck. Conversely, merely reusing Jacobians while silently changing
their coordinate charts would invalidate the normal model.

## Residual Cause: A Concrete Tracking-Lag Case

D9 robot 1 dominates the 300-second difference; the other four robots' final
trajectory RMSEs agree within 3e-6 m. We froze robot 1's identical 1012 poses
and 1097 observations, then completed the same physical problem from each
online terminal state. This adds ten diagnostic runs using the existing five
derivative/solver controls, not extra work hidden in online timing.

| Quantity | Native terminal state | iSAM2-exact terminal state |
| --- | ---: | ---: |
| Robot-1 physical loss before completion | 1344.486906 | 2161.012337 |
| Exact physical gradient norm before completion | 1794.617149 | 2628.534883 |
| Direct nonlinear directional derivative | 1794.617149 | 2628.534884 |
| Loss after exact Schur completion | 683.763994 | 683.763994 |
| Exact gradient norm after completion | 1.22e-6 | 9.03e-6 |
| Trajectory RMSE m before completion | 1.185870 | 1.080439 |
| Trajectory RMSE m after completion | 1.028760 | 1.028760 |
| Object RMSE m after completion | 1.997691 | 1.997691 |

Both completions reach the same state within 2.28e-7 m maximum positional
difference. Exact full elimination and exact batch LM agree in physical cost.
This supports unresolved optimization lag, rather than two distinct terminal
basins, as the explanation for this particular online discrepancy. It is not
a global-optimum certificate. Native completion returns at the floating-point
cost floor (`no_decreasing_trial`), not the strict 1e-8 gradient threshold.

Crucially, 49.1% and 68.4% of the terminal physical loss can still be removed
without changing covariance or identity. Large raw residuals at those states
must not automatically trigger sensor-noise inflation or relation rejection.
The remaining stationary loss and approximately 1 m trajectory error do not
identify their cause: model mismatch, measurement noise, geometric weakness and
wrong relations are not distinguished by this experiment. Declared identities
remain fixed, so it provides no wrong-association detection result.

## Next Decision

Retain this implementation as a correctness and cost-attribution reference;
do not promote it to the online default or add a larger fixed iteration budget.
The next change should address two measured issues together:

1. Preserve historical information in a declared local response, updating only
   the affected dependency closure instead of rebuilding the entire history.
   Explicitly account for invalidation after covariance or identity changes.
   Any condensed-history approximation must expose its error and loss of
   relinearization freedom, not disguise frozen history as exact inference.
2. Distinguish reducible optimization error from measurement evidence before
   adapting noise. A consistent local descent response provides evidence of
   state error; it does not by itself certify a sensor or identity failure.
   Validate scheduling from this evidence against fixed per-frame work at
   equal elapsed-compute budgets, including bursts and causal output quality.

The current native replay gate is passed. Efficient incremental elimination,
causal relative-frame initialization, topology-local coordination, identifiable
statistical adaptation, held-out data and matched communication comparisons
remain open. The full research objective remains active.

## Verification and Reproduction

The optional targets use the pinned no-TBB GTSAM commit
`03c263fe08b02c90a8861d36f77422765d39506d` and retain the tested Eigen ABI flags.

```sh
cmake --build /tmp/dranppr_backend_build --target native-causal-replay native-local-audit testIncrementalLocalModel -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit python3 -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py tests/test_native_causal_replay.py tests/test_native_replay_profile.py
```

22 native C++ tests, 15 Python tests, and the six existing Schur/product/block/
owner-PCG regressions pass. Python plugin autoload must be disabled because an
unrelated installed Dash/Jinja/MarkupSafe combination fails during pytest startup;
no production dependency was changed. Native replay uses sparse LDLT, not PCG.
Legacy PCG tests pass but do not print iteration counts. MPI was not exercised
(0 ranks); centralized CCI and D-CCI behavior are unchanged.

Full commands and sources are pinned in the manifests. Large traces remain at:

- `/tmp/dranppr_native_causal_120_20260922`
- `/tmp/dranppr_native_causal_300_20260922`
- `/tmp/dranppr_native_causal_d9r1_completion_20260922`

Small tables, profiles, manifests, test XML and source snapshots are retained
under `results/dranppr_native_causal_20260922/`. Build artifacts remain in `/tmp`
because `/home` has only about 80 MB free. No prior failures or negative results
were removed, and no commit or upstream push was performed.
