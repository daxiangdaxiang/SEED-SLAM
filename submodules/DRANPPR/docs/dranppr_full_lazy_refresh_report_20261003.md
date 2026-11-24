# Full Online-Suffix Validation of Descent-Triggered Refresh

Date: 2026-10-03. Research branch only; production DBA-DRANPP unchanged.

## 1. Decision

The full previously audited online windows support retaining the simple normal
solver plus lazy historical refresh as the next research candidate. It reduces
service time by 3.24-3.52 times with sub-millimeter terminal RMSE changes. On D9,
the virtual serial queue no longer develops the tens-of-seconds backlog seen
in the reference. This is a meaningful improvement in timely state delivery,
not just a favorable endpoint accuracy result.

Do not promote a new production default yet. This does not establish robustness
to unknown noise/outliers, bounded per-frame work, asynchronous deployment, or
novelty of lazy Gauss-Newton itself. No estimator code or parameter was changed
in this experiment. The executable is byte-identical to the preceding short
audit: SHA256 `2d7110d9d485eeba672312c9b14cdb9340cdc68c67d2e7c67e7e107ba50390a0`.

## 2. Hypothesis and Protocol

Hypothesis: a full-step physical-descent failure is a useful event for refreshing
old models, whereas a tiny pose displacement alone triggers much unnecessary
historical work. The preceding short tests are documented in
`dranppr_lazy_refresh_report_20261003.md`.

Both arms now use the same normal-equation cached implementation:

- Reference: `joint_cached_one`, existing displacement threshold `1e-4`.
- Candidate: the same mode with `--lazy-refresh`, no added refresh threshold.
- One interface PCG iteration per arrived frame; the existing physical-cost
  backtracking remains unchanged. A failed full step requests relinearization
  next iteration, not an extra same-frame interface solve.
- Same common-past states, arrived data, known barcode identities, fixed physical
  Gaussian covariances, exact root gauge and five-robot chain topology.
- Sequential ABBA for each dataset, single-thread BLAS/OpenMP, 600 s per-process
  watchdog. All eight estimator processes completed. Repeats are timing controls,
  not independent initializations or additional accuracy samples.
- Ground truth used only after estimator completion, with one network-wide SE(2)
  alignment. Object RMSE counts local copies, not unique physical landmarks.
- Same information and iteration caps, not strictly identical communication or
  compute budgets. Every backtracking reduction is counted.

| Dataset | Warm cutoff / stream cutoff (s) | New frames | Prefix / final factors | Final supported poses / object copies |
| --- | ---: | ---: | ---: | ---: |
| D4 | 15.534 / 120 | 1562 | 352 / 4031 | 1710 / 68 |
| D9 | 131.723 / 300 | 2514 | 4222 / 9594 | 4104 / 73 |

These are the complete previously audited windows, not the entire raw MRCLAM
missions. D9 excludes the same 437 unsupported historical GT timestamps in all
arms. All causal suffix poses support measurement- and delivery-time evaluation.
This is a native Pose2/Point2 backend test, not DBA association or ObjectPose3.

## 3. Primary Results

Times include native insertion, coordination and solver service, but exclude
JSON input, trace serialization/file I/O, and posthoc evaluation. Setup is
recorded separately and included in the virtual queue. No network latency or
robot-parallel execution is simulated.

| Dataset | Reference service, repeats (s) | Candidate service, repeats (s) | Speedup | Application bytes, reference -> candidate |
| --- | ---: | ---: | ---: | ---: |
| D4 | 35.612 / 36.831 | 10.605 / 10.477 | 3.36 / 3.52 | 50,469,704 -> 50,555,272 |
| D9 | 219.727 / 213.495 | 64.679 / 65.798 | 3.40 / 3.24 | 90,297,496 -> 90,522,840 |

Communication increases by 0.170% and 0.250%. There is no claim of exact
equal-byte dominance. Both repeats produce identical final/causal states,
decisions, iteration counts, refresh counts and communication, excluding timing.

| Dataset | Physical loss, reference -> candidate | Terminal trajectory RMSE (m) | Object-copy RMSE (m) | Causal trajectory RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| D4 | 1230.453239 -> 1230.453371 | 0.285604 -> 0.285680 | 0.184824 -> 0.184796 | 0.272675 -> 0.271750 |
| D9 | 3280.434094 -> 3280.539303 | 0.401418 -> 0.401393 | 0.229960 -> 0.230121 | 0.508970 -> 0.507303 |

Terminal trajectory differences are +0.076 mm and -0.025 mm; object differences
are -0.028 mm and +0.161 mm. These tiny differences should not be marketed as
statistically significant accuracy gains. The improvement is computation and
timely delivery at approximately maintained state quality. Candidate physical
loss is slightly higher, by 0.0000108% and 0.003207%.

| Dataset | Max serial state age, reference repeats (s) | Candidate repeats (s) | Delivery RMSE, reference repeats (m) | Candidate repeats (m) |
| --- | ---: | ---: | ---: | ---: |
| D4 | 0.172 / 0.184 | 0.073 / 0.071 | 0.273251 / 0.273280 | 0.271887 / 0.271885 |
| D9 | 55.433 / 53.370 | 0.337 / 0.333 | 2.163113 / 2.020372 | 0.507423 / 0.507421 |

Delivery RMSE compares an unextrapolated timestamped estimate with GT at its
virtual completion time, using the same alignment as causal measurement-time
RMSE. It is not ordinary ATE and not a deployed real-time guarantee. Under this
explicit serial model, D9 no longer loses most of its useful state accuracy to
queueing delay. Average state age drops from 15.01 to 0.0334 s in the first pair.

## 4. Mechanism and Recovery

| Dataset | Optimizer historical factor relinearizations, reference -> candidate | Reused local responses | Requested network refreshes | Non-updating frames |
| --- | ---: | ---: | ---: | ---: |
| D4 | 2,950,995 -> 64,747 | 6082 / 7810 | 33 / 1562 | 4 / 1562 |
| D9 | 16,043,213 -> 572,174 | 9643 / 12570 | 89 / 2514 | 12 / 2514 |

These historical optimizer relinearizations decrease 97.81% and 96.43%. The
counter excludes insertion and identity-chart coordination updates. No
conditional-elimination machinery is used in either arm. This isolates the
benefit to refresh policy and reuse of unchanged responses.

All 33 D4 and 89 D9 refresh requests are followed by a refreshed, accepted full
step. All 4 and 12 non-updating frames are isolated; the longest stalled run is
one frame. All runs end without a pending refresh. New data also arrive next
frame, so this is observed recovery, not same-graph causal attribution by itself.
The preceding unchanged-graph nonlinear unit test provides the separate
same-graph mechanism control.

Longest intervals without a requested network refresh are 162 and 215 frames.
New factors and identity-chart coordination may still update models within
those intervals. These are not indefinitely frozen graph experiments.

Every accepted optimizer step decreases its current physical loss. Graph
insertion/coordination occur before the step, so this is not a monotonicity claim
over the raw growing-graph cost sequence. Positive descent does not certify
an accurate gradient or a globally correct solution. The minimum observed
full-step actual/predicted gain ratio reaches -17.393 on D9, confirming that
occasional stale-model failures must not be ignored.

## 5. Temporal Check and Remaining Workload

Each row below uses the same whole-suffix causal alignment; there is no
per-quarter realignment. Quarters are equal-count frame partitions.

| Dataset | Quarter | Causal RMSE, reference -> candidate (m) | Median service, reference -> candidate (ms) |
| --- | ---: | ---: | ---: |
| D4 | 1 | 0.304746 -> 0.301199 | 7.63 -> 2.22 |
| D4 | 2 | 0.219710 -> 0.220424 | 16.58 -> 4.66 |
| D4 | 3 | 0.314957 -> 0.315388 | 29.11 -> 7.60 |
| D4 | 4 | 0.239116 -> 0.238169 | 37.14 -> 10.60 |
| D9 | 1 | 0.701813 -> 0.700744 | 64.61 -> 13.80 |
| D9 | 2 | 0.537568 -> 0.535251 | 74.10 -> 20.35 |
| D9 | 3 | 0.430637 -> 0.428663 | 94.37 -> 24.94 |
| D9 | 4 | 0.263601 -> 0.261522 | 113.53 -> 32.70 |

There is no observed growing accuracy penalty in these windows, but service
still grows with history. This is not a bounded-memory or constant-work online
solver. Full recovery, cost evaluation and response/model maintenance remain.

### The Worst Frame Is Not Necessarily a Refresh Frame

Peak service worsens despite lower averages: D4 first-pair maximum changes
46.03 -> 64.17 ms; D9 changes 197.94 -> 261.68 ms. On D9 frame 2438 (elapsed
294.756 s), 239.23 of 261.68 ms is evaluation/collectives, and only 7.20 ms is
model work. `refresh_rounds=0`, `refresh_requests=1`, and `accepted=0`.

The stale direction exhausts 30 backtracking trials BEFORE the deferred refresh.
The byte ledger independently confirms 12,992 extra optimizer bytes relative to
the same reference frame: 29 additional two-scalar reductions, each costing
`8 directed hops * (40 header + 16 payload) = 448 bytes`. Frame 2403 has the
same 29-extra-reduction pattern. D4 peak frame 1257 accepts only after 28 extra
trial reductions and spends 57.21 of 64.17 ms in evaluation/collectives.
Its accepted decrease is only 9.09e-13 at a loss of 1127.32, near floating-point
roundoff. Counting this as an accepted step is not evidence of useful progress
or stationarity; the next-iteration refresh request is retained regardless.

Thus the next tail-latency issue is repeated evaluation of an unreliable
direction, not simply slow relinearization. This attribution follows the trace
and message accounting; no new gate, backtracking cap or threshold was added.
Any future bounded-work change must be tested against lost progress and delayed
recovery, rather than silently truncating this failure evidence.

## 6. Historical Baseline Context

Existing iMESA/riMESA comparisons are reused only after verifying exact event
and warm-start hashes, matching arrived frames/factor counts and initial
physical likelihood. They were NOT rerun this turn. Historical timings and
virtual delivery errors should not be presented as newly paired measurements.

The old byte-ceiling arms remain below the new candidate's cumulative bytes
at every frame. Terminal slack is 1,000,120 bytes on D4 and 1,797,360 bytes on
D9. Their schedules still follow the old credit sequence, not a newly filled
candidate budget. One-sweep baselines use much less communication.

| Dataset | Method / historical schedule | MB | Terminal trajectory / object RMSE (m) | Causal trajectory RMSE (m) |
| --- | --- | ---: | ---: | ---: |
| D4 | Current normal lazy | 50.555 | 0.285680 / 0.184796 | 0.271750 |
| D4 | Paper iMESA, one sweep | 5.796 | 0.281216 / 0.254721 | 0.290774 |
| D4 | Paper iMESA, old byte ceiling | 49.555 | 0.281234 / 0.255591 | 0.290106 |
| D4 | Repaired riMESA, old byte ceiling | 49.555 | 0.368693 / 0.536705 | 0.411337 |
| D4 | Gaussian riMESA ablation, old byte ceiling | 49.555 | 0.281146 / 0.255356 | 0.291349 |
| D9 | Current normal lazy | 90.523 | 0.401393 / 0.230121 | 0.507303 |
| D9 | Paper iMESA, one sweep | 9.780 | 0.493795 / 0.418158 | 1.175964 |
| D9 | Paper iMESA, old byte ceiling | 88.725 | 0.495113 / 0.417262 | 1.207279 |
| D9 | Repaired riMESA, old byte ceiling | 88.725 | 1.746472 / 2.806287 | 3.191046 |
| D9 | Gaussian riMESA ablation, old byte ceiling | 88.725 | 0.494515 / 0.418057 | 0.919652 |

All archived one-sweep/byte-ceiling arms, including omitted rows, remain in
`temporal_and_context.json`. D4 still does NOT beat Gaussian baselines on
terminal trajectory RMSE; it has better object/causal quality here. D9 improves
these state metrics, but this is not a universal all-baseline result. Infeasible
disagreeing-copy physical losses must not be ranked as feasible lower values
of the exact-consensus objective.

## 7. Research Interpretation and Next Gate

The useful qualitative change is separating an estimator update from a full
historical model rebuild. An old model can remain computationally useful even
when its coordinate base has moved; expensive refresh is triggered by failed
physical progress, not by a length/angle threshold. This preserves precision
without adapting measurement noise merely to hide optimization lag.

This does not yet solve the user's robust-noise question. Known associations,
fixed Gaussian covariances and the same two development sequences cannot prove
adaptation to changing noise, outliers, or incorrect relations. Lazy refresh
and selective relinearization themselves are established optimization ideas,
not a sufficient standalone novelty claim.

Retain the normal lazy path as the simpler online research reference, while
keeping production unchanged. Next validate unchanged settings on independent
data/noise conditions and explicitly separate stale-model failure from genuine
measurement contamination. Audit failed-direction backtracking as a bounded-work
question; do not add another cache layer or tune away these failures. Unknown
frame bootstrap, ObjectPose3, bandwidth limits and asynchronous execution remain
separate unvalidated deployment dimensions.

## 8. Verification and Reproduction

Artifacts: `results/dranppr_full_lazy_refresh_20261003/`, with per-case manifests,
four estimator outputs each, binary/source/input snapshots, and
`temporal_and_context.json`. Posthoc analyzer and tests are separately copied.
No C++ estimator, CCI/D-CCI, production repository, upstream baseline, or noise
configuration changed. No commit or push was made.

One-step PCG counts are exactly 1562/2514 per run. Reference/candidate maximum
verified relative residuals are D4 3.11307/2.40762 and D9 4.24264/4.27505.
These are inexact steps, not PCG convergence claims. MPI: 0 ranks; execution is
a single-process topology-routed simulator. The root remains fixed and shared
copy disagreement stays below the audit tolerance of 1e-10 m.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_lazy_refresh \
  --dataset 4 --duration 120 --normal-only --timeout 600 \
  --output NEW_D4_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_lazy_refresh \
  --dataset 9 --duration 300 --normal-only --timeout 600 \
  --output NEW_D9_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.analyze_full_lazy_refresh \
  --results results/dranppr_full_lazy_refresh_20261003 \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw --output NEW_POSTHOC_JSON
```

Final verification: 62 C++ tests and 90 Python tests pass. The test and checksum
manifests record the commands and outcomes.
