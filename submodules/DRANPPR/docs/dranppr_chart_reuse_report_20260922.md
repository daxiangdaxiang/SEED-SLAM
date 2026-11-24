# Declared-Chart Reuse: A Controlled Online Improvement

Date: 2026-09-22. Scope: independent local native Gaussian replay, not a complete
robust or distributed DRANPPR result. The previous turn was progress: it located
historical model publication as the main cost and identified a real optimization
lag case. This follow-up changes the native optimizer's reuse behavior and tests
the hypothesis without changing physical noise, identity or measurements.

## Decision and Mathematical Contract

The original diagnostic made every accepted current estimate a new linearization
base. Nearly every historical factor was then invalidated, even for tiny changes.
The new `optimizeNativeCachedLocal` separates current estimates x from declared
chart bases theta. The existing incidence map refreshes every factor touching a
changed base, including old observations. Other factors retain their bases.

For the cached quadratic and current chart displacement:

```text
m_theta(d) = c + g^T d + 0.5 d^T H d,
d0 = Local(theta, x),
H step = -(g + H d0),
x_trial = Retract(theta, d0 + alpha step).
```

The `H d0` term is essential. Solving for another absolute displacement from
theta and adding it to x would double-count an old increment. Partial chart
updates must not mix a Jacobian from one base with a gradient from another.
Only finite steps with positive model decrease and strict physical-cost decrease
are accepted. Publication updates x, not every chart base.

This is established selective-relinearization machinery, not our innovation.
The motivation is consistent with [iSAM2's fluid relinearization and incremental
ordering](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.html). We retain a fresh
sparse private factorization and a dense small interface solve to isolate the
effect of chart reuse. No Bayes-tree reuse, marginalization or bounded-time
inference is claimed.

The replay uses the same `1e-4` infinity-norm chart threshold already fixed in
the iSAM2 control. It was not swept or retuned per sequence. It is still a
numerical tolerance involving pose translation/angle coordinates, not a learned
noise parameter or a parameter-free method. Its broader sensitivity remains
untested. Cached model stationarity is explicitly reported as
`cached_model_gradient_small`, not true nonlinear stationarity.

## Test and Experiment Contract

The previous full-relinearization mode remains available as `native`; the new
mode is `native_cached`. The exact-Jacobian `isam2_exact` arm is unchanged. All
share the timestamp-atomic stream, native polar likelihood, odometry covariance,
declared identities, initialization rule and independent local gauges. Current
state extraction never overwrites x with stale theta.

There are 18 serial runs: three modes on D1/D4/D9 120-second prefixes and D9
300 seconds, followed by unchanged-binary D2/D3 120-second transfer checks.
D2/D3 were not used to select this reuse tolerance, but they were examined in
earlier baseline research; these are transfer checks, not untouched final test
sets. No run failed or timed out. No GT is opened until estimator processes exit.
No observation deletion, robust kernel, covariance fitting or identity change
was introduced.

All modes receive the full same stream; they do not have equal computation
budgets. Timing is one serial run per case, not replicated benchmarking. Compare
within-run controls below rather than mixing timing from earlier reports.
Frame service includes initialization but excludes parsing, CSV and posthoc GT;
full process wall times are also recorded. No network transport is simulated.

## Results

| Prefix | Full native work s | Reuse work s | Reduction | iSAM2-exact work s | Recomputed factor fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1, 120 s | 4.817 | 3.572 | 25.9% | 1.380 | 38.9% |
| D4, 120 s | 6.958 | 5.553 | 20.2% | 2.327 | 47.7% |
| D9, 120 s | 6.513 | 4.003 | 38.5% | 0.963 | 11.7% |
| D2, 120 s, transfer | 5.202 | 4.124 | 20.7% | 1.855 | 50.1% |
| D3, 120 s, transfer | 7.707 | 5.767 | 25.2% | 2.122 | 38.3% |
| D9, 300 s | 49.671 | 33.710 | 32.1% | 8.436 | 19.0% |

The recomputed fraction is `sum(refresh factors)/sum(refresh factors + reused
factors)` at cache maintenance after insertion. It counts factor visits across
frames, not unique observations; newly inserted factors have their separately
recorded initial linearization. On D9 300 s, 850,682 visits are recomputed and
3,623,659 reused.

Across all six cases, the maximum final point/pose positional difference from
full native is 3.47e-5 m. Aggregate trajectory RMSE changes by at most 1.46e-6 m;
aggregate causal trajectory RMSE changes by at most 5.23e-6 m. These small
differences support precision preservation at the tested tolerance, not universal
equivalence or an improvement in accuracy.

| D9, 300 s | Full native | Reuse | iSAM2-exact |
| --- | ---: | ---: | ---: |
| Physical Gaussian loss | 3596.625561 | 3596.646989 | 4413.150993 |
| Final trajectory RMSE m | 0.8095327 | 0.8095331 | 0.7765700 |
| Causal trajectory RMSE m | 1.1114203 | 1.1114151 | 1.1032525 |
| Object-copy RMSE m | 1.2623880 | 1.2623904 | 1.2659502 |
| Per-frame p99 ms | 26.307 | 22.474 | 11.168 |
| Maximum serial backlog s | 0.071428 | 0.048472 | 0.017364 |

The slight physical-loss increase is retained rather than rounded into equality.
Every accepted reuse step decreases its own current-prefix true loss. This does
not imply that its final loss is lower than another solver's, or that loss and
GT error have the same ordering.

The new path improves the native prototype but still costs 2.22--4.16 times
iSAM2's service work, and does not fix D9 robot 1's optimization lag. It is not
an end-to-end superiority result over iSAM2, riSAM or riMESA. The D9 short-prefix
alignment weakness documented in the previous report remains applicable.

## What Remains Expensive

D9 300-second reuse profile, percentage of frame work:

| Component | Share |
| --- | ---: |
| Chart checking and refresh, including nested counters | 27.36% |
| Refresh snapshot, nested | 5.57% |
| Refresh linearization, nested | 3.96% |
| Refresh normal-contribution update, nested | 6.86% |
| New-frame insertion | 15.25% |
| Sparse full-system export and gradient shift | 15.48% |
| Schur response setup, including factorization | 13.58% |
| Private factorization, nested | 3.92% |
| Retraction and physical-cost evaluation | 9.23% |
| Dense interface solve | 5.46% |
| Current-estimate publication | 0.71% |

These are not all additive: nested counters must not be counted twice.
The hypothesis is supported: retaining charts eliminates redundant historical
factor work. However, the implementation still copies full graph metadata,
checks all values, exports the full sparse matrix, builds a fresh response and
evaluates the historical physical loss. It has not achieved true incremental
elimination or bounded per-frame work.

Next work should reuse the reduced local response and its actual dependency
closure rather than continue tuning this threshold. The pinned GTSAM source
contains Gaussian Bayes-tree/partial-elimination primitives, but its simple
`GaussianISAM::update` is append-oriented; replacement/removal and interface
ordering cannot be assumed to work without a dedicated contract. Reuse an
established elimination engine where suitable, not reimplement an entire
nonlinear baseline and rename it DRANPPR. DRAN's topology-local interface solve
and causal gauge initialization still need to be connected and evaluated.

For residual attribution, separate two approximations: stale solver state and
stale linearization. A small cached gradient can result from the latter. Neither
is sufficient evidence for sensor variance or identity rejection. The frozen
physical completion from the previous report remains the evidence that much of
D9 robot 1's terminal loss was state-reducible. Efficient execution should free
work for those hard updates, not automatically turn leftover loss into noise.

## Verification and Artifacts

27 native C++ tests, 16 Python tests and six legacy Schur/owner-PCG regressions
pass (49 total). Added tests cover nonzero displacement, zero-threshold parity,
partial nonlinear pose-chart refresh against an independent full correction,
covariance replacement, factor removal, fixed gauge, malformed estimates and
rejected-trial state preservation. All three replay modes pass timestamp and
future-mutation causality checks. Full native's final and causal state CSVs are
byte-identical to the previous D1/D4/D9 runs, including D9 300 seconds.

Fresh test XML reports native fixed-gauge relative step error 1.13e-16, full
linear residual 3.25e-16 and matrix-free product error 3.53e-17. The new partial
chart correction passes its 1e-9 absolute tangent tolerance. The cached solver
uses LDLT, not PCG; legacy PCG iteration counts are not printed. MPI was not run
(0 ranks), and CCI/D-CCI and packaged DBA-DRANPP remain unchanged.

```sh
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel native-causal-replay native-local-audit -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit python3 -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py tests/test_native_causal_replay.py tests/test_native_replay_profile.py
```

Raw replay roots, with source/binary/dependency/input hashes and exact arguments:

- `/tmp/dranppr_chart_reuse_120_20260922`
- `/tmp/dranppr_chart_reuse_transfer_120_20260922`
- `/tmp/dranppr_chart_reuse_300_20260922`

Compact tables, profiles, manifests, source snapshots and test XML are retained
under `results/dranppr_chart_reuse_20260922/`. The GTSAM commit remains
`03c263fe08b02c90a8861d36f77422765d39506d`. No commit/push or default-method
replacement was performed. The robust, adaptive, distributed research goal
remains active; this local reuse improvement is only one prerequisite.
