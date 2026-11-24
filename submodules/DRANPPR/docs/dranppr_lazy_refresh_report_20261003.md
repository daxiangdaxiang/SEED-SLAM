# Descent-Triggered Historical Refresh: Controlled Online Audit

Date: 2026-10-03. Research-only DRANPPR prototype; production DBA-DRANPP unchanged.

## 1. Conclusion

Historical relinearization need not follow every appreciable coordinate change.
A parameter-free, rejection-triggered refresh policy reduced measured serial
service time by 62-69% against the existing normal-equation cached replay on four
bounded MRCLAM suffixes. Extended-window terminal trajectory RMSE changed by
less than 0.2 mm, while object-copy RMSE worsened by 0.7 mm on D4 and 5.3 mm on D9.
This is a promising computation/accuracy tradeoff, not uniform accuracy dominance.

The main gain comes from refreshing less often, not from conditional elimination:
the normal-equation implementation under the same lazy policy was faster than
the conditional implementation in all four single control runs and produced the
same states, decisions and bytes to numerical precision. Do not promote the more
complex conditional implementation on these results.

## 2. Minimal Policy

Opt-in flag: `--lazy-refresh`. Supported only with cached, connected common-past
replay. No production default, factor model, physical covariance, gauge prior,
CCI implementation, or association logic was changed.

1. Insert arrived measurements; reuse historical bases and linearized factors.
2. Solve the same interface problem with the same per-frame PCG cap.
3. Evaluate the full proposed step against the true arrived-graph physical loss.
4. If its predicted decrease is nonpositive/nonfinite, or the full step fails
   true physical descent, request a refresh for the next solver iteration.
5. Keep the existing backtracking in the current iteration. Publish only a
   physically decreasing candidate. On the next iteration, refresh all changed
   free-variable bases, then use the same solve budget.

The request is derived from scalar reductions already routed over the network.
There is no extra uncounted broadcast, same-frame retry, fixed refresh period,
new noise threshold, or model-agreement tuning parameter. The existing maximum
30 backtracks and factor-of-two step contraction are unchanged. Fixed noise
parameters and the interface budget still exist; this is not a parameter-free
SLAM system.

When a final frame leaves a refresh pending, `refresh_pending` reports it rather
than adding uncounted drain iterations. None of these four candidates ended
with a pending refresh. A pending refresh only runs when the solver is invoked
again; the replay does not invent another sensor frame to trigger it.

New factors are still linearized in the persistent common variable charts.
They are NOT silently linearized at all-current estimates. Common chart
staleness at the interface between new and historical factors remains a risk.

## 3. Relation to Inexact Optimization

For the fixed graph at one iteration, let `m(d)` denote the cached quadratic,
`x = theta.retract(d)`, and `y = theta.retract(d+s)`. Define

\[
P=m(d)-m(d+s),\qquad A=F(x)-F(y),
\]
\[
E=[F(y)-m(d+s)]-[F(x)-m(d)].
\]

Exactly, `A = P - E`. With `P > 0`, rejection of physical descent (`A <= 0`)
means the model discrepancy has consumed the whole predicted benefit (`E >= P`).
This gives a meaningful refresh event without comparing coordinate movement to
a dataset-specific length or angle threshold.

This is a one-step algebraic identity, not a convergence theorem. A positive
but tiny `A` can coexist with a poor local model; acceptance does not certify
gradient accuracy, stationarity, identity correctness, or low GT error.
Full-step failure can also occur with a fresh Gauss-Newton model, so the event
does not uniquely diagnose stale Jacobians. It is a conservative refresh rule.

This is a lazy/inexact Gauss-Newton diagnostic. Selective relinearization,
incremental elimination and actual-versus-predicted decrease are established
ideas. This experiment does not establish methodological novelty merely by
combining or renaming them. The useful research result is identifying and
removing an unnecessary refresh coupling in the current distributed response
implementation without changing its physical estimation objective.

## 4. Protocol

- Native Pose2/Point2 MRCLAM replay with known object identities; not ObjectPose3.
- Same frozen common-past states, causal suffix measurements, physical Gaussian
  noise, root gauge and chain topology `1:2, 2:3, 3:4, 4:5` in every arm.
- One interface PCG iteration per arrived frame, with exact private recovery for
  that inexact interface step. Not a converged interface solve.
- Sequential ABBA: normal cached reference, conditional lazy candidate,
  conditional lazy candidate, normal cached reference.
- Additional single controls: conditional ordinary refresh and normal lazy
  refresh. These isolate refresh policy from linear algebra implementation.
- GT loaded only after every estimator in a case had terminated. Terminal
  trajectories use one common SE(2) alignment across robots; object errors use
  that same alignment and count local object copies, not unique identities.
- D9 has 437 historical poses outside GT support, excluded identically for all
  methods. All suffix causal poses used here have GT support.
- No injected outliers, covariance estimation, adaptive robust kernel, bandwidth
  restriction, packet loss, asynchronous delivery or additional robots.
- Timing excludes JSON input and trace serialization/file I/O. Virtual serial
  backlog assumes zero network latency; it is not measured distributed runtime.

| Case | Warm cutoff (s) | Stream cutoff (s) | Suffix frames | Physical factors, prefix -> final |
| --- | ---: | ---: | ---: | ---: |
| D4 short | 15.5340001583 | 20 | 84 | 352 -> 549 |
| D4 extended | 15.5340001583 | 40 | 389 | 352 -> 1258 |
| D9 short | 131.7230000496 | 136 | 75 | 4222 -> 4404 |
| D9 extended | 131.7230000496 | 150 | 261 | 4222 -> 4825 |

Extended runs contain their short counterparts. Neither these nested windows
nor repeated timing runs are independent statistical samples.

## 5. Results

Reference/candidate below mean normal cached ordinary refresh / conditional
cached lazy refresh. Bytes include control, interface and backtracking traffic.

| Case | Reference service, two runs (s) | Candidate service, two runs (s) | Reduction | Bytes, reference -> candidate |
| --- | ---: | ---: | ---: | ---: |
| D4 short | 0.366845 / 0.368304 | 0.127293 / 0.126719 | 65.30-65.59% | 1,335,528 -> 1,337,320 |
| D4 extended | 2.933498 / 2.932328 | 0.927829 / 0.919919 | 68.37-68.63% | 8,387,208 -> 8,394,376 |
| D9 short | 3.312951 / 3.328508 | 1.259568 / 1.245866 | 61.98-62.57% | 1,030,392 -> 1,046,968 |
| D9 extended | 12.581552 / 12.359342 | 4.638672 / 4.594947 | 62.82-63.13% | 4,204,816 -> 4,255,888 |

Communication increases by 0.134%, 0.085%, 1.609%, and 1.215%, respectively.
Thus these are equal-iteration-cap, same-information comparisons, NOT exactly
matched-byte comparisons. No claim of equal-communication dominance is made.

| Case | Physical loss, reference -> candidate | Terminal trajectory RMSE (m) | Object-copy RMSE (m) | Causal suffix trajectory RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| D4 short | 148.270858 -> 148.671402 | 0.285268 -> 0.282431 | 0.314502 -> 0.308716 | 0.328284 -> 0.324694 |
| D4 extended | 359.796715 -> 359.804255 | 0.256514 -> 0.256614 | 0.309698 -> 0.310404 | 0.255505 -> 0.255503 |
| D9 short | 208.106958 -> 208.256133 | 0.465376 -> 0.462215 | 1.132606 -> 1.138793 | 0.572165 -> 0.570072 |
| D9 extended | 566.538929 -> 567.287706 | 0.504260 -> 0.504420 | 1.151627 -> 1.156900 | 0.494990 -> 0.491701 |

Every accepted per-frame step lowers its current physical loss. The graph grows
between frames, so these checks do not assert that the raw loss sequence is
monotone over time. The candidate terminal loss is slightly higher in all cases.

| Case | Historical factor relinearizations in optimizer, reference -> candidate | Network refresh rounds | Longest span without requested network refresh | Conditional reuse among visited conditionals |
| --- | ---: | ---: | ---: | ---: |
| D4 short | 37,172 -> 798 | 2 / 84 | 58 frames | 76.53% |
| D4 extended | 302,915 -> 8,593 | 11 / 389 | 106 frames | 83.66% |
| D9 short | 319,091 -> 21,532 | 5 / 75 | 29 frames | 69.86% |
| D9 extended | 1,177,527 -> 67,741 | 15 / 261 | 43 frames | 75.29% |

Historical optimizer relinearization decreases 93.25-97.85%. This counter does
not include new-factor insertion or identity-chart coordination linearization.
A no-network-refresh span does not mean no factors or variables were added or
updated. Entire local responses can also be reused without visiting conditionals.

The one-step PCG count equals the frame count in every arm. Maximum verified
relative interface residuals (reference -> candidate) are 1.21859 -> 1.20599,
3.11307 -> 2.40762, 2.08483 -> 1.74654 and 3.94949 -> 3.22042. They can exceed
one: these are explicitly inexact steps, not residual-converged solutions.
MPI was not used; this is a single-process topology-routed simulator.

### Implementation Control

Normal lazy service times were 0.121766, 0.895558, 1.037701 and 3.884447 s in
the four cases above. Conditional lazy was slower in each control, despite
substantial conditional reuse. Full private recovery and model evaluation
still have costs that reuse alone does not remove. The single control timings
are indicative, not a definitive independently replicated implementation ranking.

Ordinary normal/conditional and lazy normal/conditional final/causal states
agree within 8.17e-13 across all cases. Their frame decisions, PCG counts, bytes,
refresh keys and linearization counts match exactly. The two short normal
references also match archived pre-change states and solver decisions exactly.

## 6. Failure and Recovery Evidence

D4 extended accepted a decreasing backtracked or full step in all 389 frames.
It requested 11 network refreshes. D9 extended requested 15 refreshes and
accepted steps in 258/261 frames. At frames 57, 93 and 212, no backtracked step
decreased the loss; the current state was retained. Each following frame
refreshed and accepted a full step without another refresh request.

All 26 refresh requests across the two extended windows were followed by an
accepted full step. New data also arrive at the next frame, so this observed
recovery is not by itself a same-graph causal proof. A deterministic C++
nonlinear-factor test separately verifies deferred refresh and recovery on an
unchanged graph, without adding a second interface iteration to either call.

The median actual/predicted full-step gain is near one (0.995-1.002), but the
minimum is negative in every lazy case, down to -2.631 on D9 extended. This
directly rules out indefinite freezing. Physical backtracking prevents accepting
these bad full steps, but costs additional reductions and sometimes a stalled
frame. End-of-window `refresh_pending` is false in all four candidates.

Virtual maximum serial backlog (first ABBA pair) changes from 0.02160 to
0.01737 s on D4 extended and 0.19654 to 0.13049 s on D9 extended. This does not
establish bounded latency on a full mission or in a network deployment.

## 7. What Remains Unproved

1. Long-horizon stability: extended windows are only about 24.5 and 18.3 seconds
   after the warm cutoff, not the prior full 1562/2514-frame audits.
2. Robustness to high/unknown noise or outliers: covariances remain fixed and
   identities known. Lower computation alone is not a robust-estimation result.
3. First-order model validity: small accepted decrease can hide stale gradients.
   A rejection trigger is not an adaptive inexact-Newton forcing condition.
4. Bounded incremental work: true cost, recovery and some metadata still scan
   historical state; triggered network refresh touches long histories.
5. Local/asynchronous operation: a component-wide decision and scalar reduction
   remain. This is not a fully asynchronous event-triggered distributed solver.
6. Generalization: two nested-window datasets, one topology and one common-past
   initial state each; no new iMESA/riMESA or all-baseline comparison here.
7. Precision and stopping: no true-gradient stopping test was added. Numerical
   near-stationarity can cause a refresh request; low ATE is not a stopping test.

## 8. Next Gate

Keep the opt-in diagnostic and the original default. First rerun the previously
archived full online suffixes with the simpler normal lazy path and unchanged
noise/iteration settings. Count failed frames, pending refreshes, all bytes and
serial latency, not just terminal ATE. Only if that remains stable should we
spend effort on a local model-error criterion or robust noise adaptation.
Do not add multiple thresholds or a new conditional-elimination layer to
compensate for a failure in this minimal policy.

## 9. Reproduction and Verification

Artifacts: `results/dranppr_lazy_refresh_20261003/{d4_short,d9_short,d4_extended,d9_extended}`.
Each case contains six completed runs, commands, copied input streams/common
past, executable, source snapshot, CMake cache, byte ledgers and `analysis.json`.
Root verification/checksum files retain the test outputs and linked-library hash.

```sh
cmake --build /tmp/dranppr_backend_build --target native-joint-replay native-joint-audit testIncrementalLocalModel -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_native_lazy_refresh.py tests/test_audit_lazy_refresh.py \
  tests/test_native_chart_replay.py tests/test_audit_native_chart_refresh.py \
  tests/test_native_matched_online.py tests/test_native_conditional_replay.py \
  tests/test_audit_conditional_replay.py tests/test_native_conditional_audit.py \
  tests/test_native_joint_audit.py tests/test_native_joint_replay.py \
  tests/test_native_interface_support.py tests/test_audit_native_online.py

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_lazy_refresh \
  --dataset 9 --duration 150 --output /tmp/dranppr_lazy_d9_reproduction \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

Verification: 62 C++ tests and 85 Python tests pass. All 24 experiment processes
terminated successfully. Default-path baseline parity, physical descent, gauge,
shared-object consensus, causal input/outputs, refresh transitions and byte
ledgers were checked. No production change, commit, push or CCI modification.
