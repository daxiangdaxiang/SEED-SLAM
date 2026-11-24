# Horizon-Matched Error Control For Asynchronous Updates

Date: 2026-11-24. Research DRANPPR only. Production DBA/DRAN++, CCI,
physical objectives, noise weights, native C++ solvers, and git remotes unchanged.

## Question And Decision Boundary

The previous scalar-gain experiment failed to deliver an overall gain. A
forecast can improve an approximate gradient without producing a better
finite-work solution if the optimizer still applies worst-delay damping.
This experiment changes that coupling, not the prediction order: estimate the
error at the actually consumed forecast horizon, and use that error to set
local update strength.

The result distinguishes three claims: lower objective at equal bytes,
lower measured compute to the same objective, and robustness to harder
communication. These are not interchangeable. In particular, the long-delay
stress tests already refute promoting the predictive variant as a robust
default, regardless of its ordinary-network results. The main matrix gives
108 wins and 27 ideal-network ties for BOTH adaptive-update arms versus
baseline. This supports further investigation of error-aware action, while
the no-prediction ablation has the stronger tested robustness record.

## One Local Update Rule

The unchanged fixed normalized physical model is

```
q(y) = .5 y' B y + c' y,
g_i = B_ii y_i + c_i + sum_j B_ij y_j.
```

The receiver forecasts the neighbor's contribution z_ij=B_ij*y_j, rather than
pretending its predicted pose is a received true pose. It uses the previous
packet-driven first-order OLS fit with L=8 samples. No higher order is added.

Let M0_i=B_ii+sum_j sqrt(B_ij B_ij') be the existing zero-delay block majorizer,
and P0_i=M0_i^-1. The delayed baseline uses
MD_i=B_ii+sum_j (1+D_ij+D_ji)*sqrt(B_ij B_ij'), with known age bounds D.
For a supported empirical error estimate epsilon_i, use

```
alpha_i = max(0, 1-epsilon_i / sqrt(ghat_i' P0_i ghat_i)),
s_i = -alpha_i P0_i ghat_i.
```

A zero gradient gives zero step. If any required nonzero age has insufficient
calibration data, use the original delayed-metric HOLD update. The predictor
continues to generate shadow forecasts during this fallback, but these do not
enter the actual state cache or the chosen update.

In plain terms: if the predicted gradient is much larger than its estimated
error, act on it; if error is comparable to the signal, reduce the action or
pause; if the error model is unsupported, retain the baseline rule. Crucially,
the non-predictive ablation applies exactly this rule to held neighbor values.

### Where Epsilon Comes From

At each local read, record the predicted contribution for source-time target
t-1, its actual age h, and the bound available before the future label arrives.
Later, a received packet can validate a forecast at its exact source timestamp.
Consecutive source optimizer revisions also certify the constant-state interval
between those revisions. A revision gap does NOT permit interpolation. Unknown
labels are discarded and counted; obsolete packets never overwrite caches.

For each neighbor and exact integer age, retain the last L=8 validated norms
||z_actual-z_forecast||_P0. Their maximum is epsilon_ij, requiring all eight
samples; epsilon_i=sum_j epsilon_ij. Age zero is exact. Pending forecasts expire
according to the supplied finite delay contract. No true current neighbor state
or global objective is exposed to the actors.

This avoids testing an old one-step error against a new long-horizon prediction.
It does NOT turn a historical maximum into a guaranteed future bound. L=8 is an
inherited shared memory length, not a claim of parameter independence. Exact-age
tables become sparse under long or irregular delays. Source revisions must count
optimizer state changes, not just transmitted packets; this contract is explicit
in the simulator and not yet implemented in a native online protocol.

### Conditional Mathematical Interpretation

The step is the closed-form minimizer of

```
ghat_i' s + .5 ||s||_M0_i^2 + epsilon_i ||s||_M0_i.
```

If ||ghat_i-g_i||_P0_i <= epsilon_i actually holds, norm duality bounds the
linear-model error. Since blockdiag(M0_i) majorizes B, simultaneous supported
updates satisfying these bounds obey the corresponding summed quadratic descent
bound, with contribution -.5*max(||ghat_i||_P0_i-epsilon_i,0)^2 per active block.
The offline audit reconstructs both the updates and this conditional inequality.

This is a standard norm-penalized inexact-gradient construction, not a new
convergence theorem. The empirical envelopes do not guarantee its premise.
Mixed supported/fallback steps do not automatically inherit a common Lyapunov
function. A high average empirical coverage is not sufficient for stability.
Relevant theoretical context includes [locally chosen asynchronous stepsizes](https://arxiv.org/abs/2203.11307)
and [relative inexact-gradient analysis](https://arxiv.org/abs/2506.17145);
neither validates the learned envelope used here.

## Frozen Protocol And Ablations

| Arm | Neighbor contribution | Update |
| --- | --- | --- |
| baseline | Last received value | Original delayed metric |
| prediction_only | First-order forecast | Original delayed metric |
| budget_hold | Last received value | Empirical-error-controlled fresh metric |
| budget_predict | First-order forecast | Same error-controlled fresh metric |
| unchecked_predict | First-order forecast | Fresh metric without error control |

All arms keep the same physical quadratic, initialization, packet calendar,
payload, and local update opportunities. No new messages or global reductions
are introduced. Zero steps still count as update opportunities and still send
the same heartbeat. Initial row/chart distribution is excluded from bytes.

Main protocol: five archived MRCLAM Schur models D1/D2/D3/D4/D9, three synthetic
six-DoF SE3 pose/object models, and one scalar coupled-SPD control; five network
conditions; three network seeds; 2000 ticks; five arms. These are correlated
conditions on nine fixed models, NOT 135 independent real-world datasets.
The previous hold and first-order state traces must match bit-for-bit. All arms
must match the baseline bit-for-bit on the ideal network.

Development used D1/D4/SE3_31, three nonideal networks, seed31, 600 ticks:
budget_predict beat baseline and budget_hold in all nine conditions. The
algorithm was then frozen. Implementation-only fixes addressed JSON scalar
serialization, matrix memory-order parity, stale planned-step reuse, and
summarizing a selected subset of arms. None changed an experimental parameter
based on endpoint performance. Failed driver output was removed only after its
arrays matched the complete replacement; negative scientific results remain.

## Main Results

All 675 runs and their independent review completed. Every-prefix bytes and
update opportunities match. All 270 original hold/linear runs retain archived
bitwise state parity; ideal-network equality holds across all five arms.
The maximum normalized independent update discrepancy is 5.424e-14; the
conditional descent checks pass their 1e-12 tolerance. These numerical checks
do not convert empirical envelopes into true bounds.

| Candidate / reference | Better | Tied | Worse |
| --- | ---: | ---: | ---: |
| prediction_only / baseline | 75 | 27 | 33 |
| budget_hold / baseline | 108 | 27 | 0 |
| budget_predict / baseline | 108 | 27 | 0 |
| budget_predict / prediction_only | 108 | 27 | 0 |
| budget_predict / budget_hold | 75 | 60 | 0 |
| budget_predict / previous risk3 | 108 | 27 | 0 |
| unchecked_predict / baseline | 82 | 27 | 26 |

All 27 ideal cases tie. The additional 33 ties between the adaptive variants
are numerical-floor comparisons, using an absolute 1e-20 gap tolerance. Do not
present negligible differences at that floor as practically significant.
The unchecked variant can catastrophically diverge even in this main matrix.

Median reduction in final relative quadratic gap over the twelve nonideal
conditions of each real-data-derived model:

| Model | budget_hold / baseline | budget_predict / baseline |
| --- | ---: | ---: |
| D1 | 24.78% | 40.66% |
| D2 | 67.29% | 89.15% |
| D3 | 76.69% | 92.45% |
| D4 | 95.29% | 99.54% |
| D9 | 75.11% | 87.01% |

All 36 nonideal synthetic SE3 conditions and twelve SPD controls improve as
well. SE3 gaps frequently approach numerical zero; this is not a comparable
percentage reduction in trajectory error. The earlier D1 negative result for
prediction alone is reversed only after changing the update rule.

Attribution: adaptive update strength already accounts for a large gain with
no extrapolation. Forecasting provides additional ordinary-network gains, but
its benefit must not be attributed to prediction alone or assumed robust under
longer delays.

The corrected nonideal actual-error coverage is 96.97% for budget_hold and
97.26% for budget_predict. Restricted to moving supported steps it is 96.12%
and 96.61%. Each arm takes 93,854 fallback opportunities. Of 672,652 supported
opportunities, hold has 341,692 zero steps and prediction has 181,364. Pauses
are part of the method and are counted, not removed from the budget. Even the
ordinary-network coverage is below 100%; there is no certified global guarantee.

## Stress And Transfer

The frozen size-8 rule was tested on D1/D4/SE3_31/SPD, seed34, with link delays
12--20 ticks or a 400-tick outage (eight conditions, forty runs):

| Candidate / baseline | Better | Tied | Worse |
| --- | ---: | ---: | ---: |
| prediction_only | 5 | 0 | 3 |
| budget_hold | 8 | 0 | 0 |
| budget_predict | 5 | 0 | 3 |
| unchecked_predict | 3 | 0 | 5 |

The three predictive failures are catastrophic, not small statistical losses:

| Long-delay model | Baseline gap | budget_hold gap | budget_predict gap |
| --- | ---: | ---: | ---: |
| D1 | 2.1205e-5 | 1.7023e-5 | 2.6556e21 |
| D4 | 1.1458e-3 | 6.5129e-4 | 9.7872e20 |
| Coupled SPD | 1.6377e-3 | 1.0696e-4 | 1.6078e103 |

All remain finite in floating point and finish the requested horizon; that is
NOT convergence. A run-complete flag must not be used as a success flag.

The actual-current-error coverage among moving supported steps is 22.31%,
20.77%, and 12.67% for these three predictive failures, versus 97.09%, 98.04%,
and 99.36% for budget_hold. For D1, predictive coverage falls from 100% in the
first 500 ticks to 25.61%, 13.87%, 15.06% in successive quarters, while the gap
grows from 1.77e-5 to 0.0822, 3.64e11, 2.66e21.

Interpretation: the evidence is consistent with closed-loop amplification
outpacing delayed error labels. A forecast changes the optimization trajectory,
so the next errors need not have the distribution of the historical ones.
Large historical error eventually causes zero steps, but stale bounds can
permit damaging updates first. This is not repaired merely by a better fit to
past data or by counting high coverage on zero steps.

A separate random normalized initialization, seed42 with network seed34,
D1/D4/SE3_31 and three ordinary networks, 800 ticks (36 runs), gives nine wins
for budget_hold versus baseline, nine for budget_predict versus baseline,
and nine for budget_predict versus budget_hold. This is a limited transfer
check, not a new dataset or evidence against the retained long-delay failures.

### Memory Sensitivity, Not Retuning

L=4 and L=16 were tested on the two retained long-delay D1/SPD cases. Both
predictive size-4 cases diverge; both size-16 cases remain stable and improve
the baseline. In the SPD case the size-4/8/16 gaps are 6.49e171, 1.61e103,
and 1.73e-4. Therefore robustness is materially memory-dependent. Increasing
memory also changes the slope fit, support warmup and empirical envelope;
this is not an isolated causal test of one of those components.

budget_hold improves the baseline at all three tested lengths in those two
cases, but its performance still varies. The default stays L=8; do not select
L=16 from these known failures and call the method automatically adaptive.

## Measured Cost

Timing was measured after the other runs terminated, with one BLAS/OpenMP
thread, three shuffled-order repeats, and exact archived prefix state/byte
parity. The target is the baseline's 2000-tick final gap, fixed without looking
at candidate endpoints. This is an offline oracle stopping experiment, not a
deployed stopping policy. It includes Python simulation and diagnostics, not
native distributed CPU/network elapsed time.
All eighteen candidate traces stay below this threshold after their first
crossing through the archived 2000-tick horizon; longer-term behavior is not
inferred from this finite check.

| Model / network | hold time / baseline | predict time / baseline | hold bytes saved | predict bytes saved |
| --- | ---: | ---: | ---: | ---: |
| D1 delay | 0.087 | 0.161 | 95.86% | 95.96% |
| D1 heterogeneous | 0.116 | 0.229 | 94.53% | 94.22% |
| D1 outage | 0.156 | 0.268 | 92.53% | 93.08% |
| D4 delay | 1.046 | 1.286 | 53.38% | 69.36% |
| D4 heterogeneous | 0.794 | 1.268 | 65.40% | 70.40% |
| D4 outage | 0.574 | 0.899 | 73.11% | 76.99% |
| SE3_31 delay | 0.800 | 1.004 | 55.35% | 64.30% |
| SE3_31 heterogeneous | 0.623 | 0.882 | 64.63% | 67.73% |
| SE3_31 outage | 0.065 | 0.100 | 96.50% | 96.50% |

Here hold/predict mean budget_hold/budget_predict. budget_hold is faster in
eight of nine conditions, with median time ratio 0.574 and median byte saving
73.11%. budget_predict is faster than baseline in six, but is slower than
budget_hold in all nine. Thus prediction buys some additional byte efficiency
but is not the most efficient measured implementation. The 0.4% SE3 delay
difference is essentially timing parity, not a material regression claim.

Important control: D1 and SE3 reach the coarse outage endpoint before the
blackout begins. Those early crossings are not evidence about outage recovery.
Requiring an eligible tick after the blackout ends moves both to 301 ticks.
Measured post-outage time ratios (hold/predict) are D1 0.285/0.511,
D4 0.568/0.890, SE3_31 0.238/0.348. This includes the blackout interval; it
does not assert that every outstanding packet has arrived at tick 301.

At the independently fixed relative-gap target 1e-6, SE3_31 delay needs
856/368/302 ticks (baseline/hold/predict), with hold/predict time ratios
0.762/0.992. Heterogeneous activation needs 1319/477/449 ticks, with ratios
0.635/0.932. In the outage case the baseline does NOT attain this target within
2000 ticks; hold/predict attain it in 503/441. No censored baseline timing is
invented for that condition. Short Python timings and baseline-relative
thresholds must not be generalized to native online wall time or all targets.

## Scope And Open Risks

- The useful direction is error-aware local action, not blindly higher order.
  Its non-predictive ablation must remain a primary comparison, not be hidden.
- The predictive variant is rejected as a robust default by the long-delay
  counterexamples. The non-predictive variant also lacks a guaranteed error
  envelope and a mixed-mode convergence proof, even if tested endpoints improve.
- Immutable physical quadratics are not nonlinear online SLAM. Fresh charts,
  graph growth, resets across epochs, and manifold tangent transport remain open.
- Delay contracts and source clocks are supplied. Unknown/unbounded delays,
  clock disagreement, congested links, and arbitrary multihop packet loss are
  not covered. Late labels are selected by delivery, not missing at random.
- Historical maximum errors assume some persistence. They are neither a
  posterior covariance nor a distribution-free confidence statement.
- Sensor-noise calibration, ATE, native online SE3, and superiority to public
  DRAN++ or riMESA are not tested by this fixed-objective comparison.
- The original native backend and its safety logic are unchanged. Do not
  promote this experiment by loosening production safeguards.

## Artifacts And Reproduction

All paths below are under `results/`. Every completed directory is sealed with
`SHA256SUMS`, manifests, raw traces, and source snapshots:

- `dranppr_error_budget_development_20261124`: 45 development runs.
- `dranppr_error_budget_final_20261124`: main matrix.
- `dranppr_error_budget_stress_20261124`: 40 stress runs, including divergence.
- `dranppr_error_budget_random_initial_20261124`: 36 initial-state controls.
- `dranppr_error_budget_memory4_20261124` and `...memory16_20261124`: six runs each.
- Corresponding `...review...` directories: independent updates, endpoints,
  budget parity, current error coverage and target crossings.
- `dranppr_error_budget_timing_20261124`: 81 measured endpoint-target replays.
- `dranppr_error_budget_recovery_timing_20261124`: 27 post-blackout replays.
- `dranppr_error_budget_fixed_target_timing_20261124`: fixed-1e-6 timing controls,
  explicitly retaining conditions where the baseline does not attain the target.

Final verification checked all fourteen result/review/timing ledgers. There are
808 experiment runs and 135 timing records, of which three are explicitly
censored rather than measured. All task processes terminated. Negative stress
results are retained, and no production configuration was promoted.

Run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```sh
python3 -m experiments.audit_error_budget --output results/NEW_MAIN
python3 -m experiments.review_error_budget --root results/NEW_MAIN \
  --history results/dranppr_history_final_20261123 --output results/NEW_REVIEW
python3 -m experiments.audit_error_budget --output results/NEW_STRESS \
  --cases d1 d4 se3_31 coupled_spd --scenarios delay outage --seeds 34 --stress
python3 -m experiments.benchmark_error_budget --root results/NEW_MAIN \
  --output results/NEW_TIMING
```

The offline coverage review uses an operand-scaled floating-point error bound:
the cancellation in B*y+c can make a tolerance based only on the resulting
gradient falsely reject exact ideal-network predictions. Original raw audit
counts remain unchanged; the review reports corrected coverage and separates
moving from zero steps. No learned envelope or actor decision is modified.

Focused regression validation: 100 tests passed, including red/green checks
for exact-horizon labels, missing revisions, no label self-coverage, bounded
memory, reset, robust-step algebra, matrix-order parity, causality, selected-arm
summary handling, cancellation-aware coverage, and modified-step detection.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q tests/test_prediction_error_budget.py \
  tests/test_review_error_budget.py tests/test_response_prediction.py \
  tests/test_neighbor_prediction.py tests/test_history_forecast_controls.py \
  tests/test_async_interface.py tests/test_async_interface_audit.py \
  tests/test_review_async_interface.py tests/test_network_noise_support.py \
  tests/test_native_rigid_subspace.py tests/test_native_lazy_refresh.py \
  tests/test_native_joint_audit.py tests/test_native_interface_support.py
```

Plugin autoload is disabled because an unrelated installed Dash/Flask plugin
has a Jinja2/MarkupSafe import conflict. No environment dependencies were changed.
No C++ or CCI implementation changed; PCG iterations and MPI ranks are not
applicable to this Python experiment (MPI processes launched: zero).
