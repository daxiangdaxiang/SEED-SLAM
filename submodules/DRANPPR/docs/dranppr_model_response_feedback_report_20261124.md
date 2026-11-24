# Corrected Local-Model Response: Stable Stress Runs, Negative Byte Efficiency

Date: 2026-11-24. Research-only fixed-quadratic experiment.

**Decision: do not release this candidate as the new backend.** The implementation
and experiment are complete, but the parent research/release milestone is not.
Continuous model correction helps its own frozen-model ablation, and the three
previous long-delay divergences are not reproduced. However, it loses every
nontrivial equal-byte comparison against the existing adaptive hold method.
No native C++ default, DBA behavior, CCI weighting, or GitHub content was changed.

## Hypothesis And Implementation

Instead of fitting a velocity to past states, predict the sender's finite
optimization response with its own local quadratic. A sender communicates its
actual post-update state x_s, held-cache gradient g_s, and last step policy.
The gradient is computed using only received neighbor states, not forecasts or
the current global state. The receiver keeps this model separate from its
actual state cache.

With A=B_jj, inverse step metric P, and gain a, freeze boundary forcing b_s:

```
b_s = g_s - A*x_s
x_(k+1) = x_k - a*P*(A*x_k + b_s)
```

For A=L*L^T and L^T*P*L=U*diag(lambda)*U^T, precompute W=L^-T*U and evaluate

```
xhat(s+n) = x_s - W*diag(1-(1-a*lambda)^n)*W^T*g_s.
```

The existing majorizers give eigenvalues in (0,1]. Under a frozen model, this
response saturates instead of increasing linearly forever with packet age.
The unit tests compare it against explicit repeated updates for dimensions
1, 2, and 6, including zero gains and long horizons. n counts source update
opportunities, not arbitrary wall-clock ticks.

An arriving packet updates the state anchor and the forcing. Its signed
forcing innovation is

```
Delta_b = g_new - g_old - A*(x_new - x_old).
```

This removes the sender's own state change from the observed gradient change.
It corrects the sender-model forcing known at the packet stamp. It does NOT
reveal the unknown current forcing after that stamp. This is not classical
compression error feedback with a known, conserved compression residual.

The frozen ablation initializes forcing from its first packet and never changes
that forcing again; it still receives actual states and step policies with
identical payloads. The strong comparator is adaptive hold, not this deliberately
stale-model ablation. Both predictive modes reuse the existing causal exact-age
error memory, L=8, and the unsupported-age directional hold fallback. No
per-dataset tuning, polynomial-order search, or extra merit gate was added.

## Protocol

- Stress: D1, D4, synthetic SE3_31 quadratic, and coupled-SPD; delay12--20 and
  outage200--600; seed34; five arms; 2000 ticks: 40 runs.
- Ordinary controls: the same four models; ideal, delay, heterogeneous, loss,
  and outage; independent network seed35; five arms; 2000 ticks: 100 runs.
- Arms: directional hold, adaptive hold, adaptive linear prediction, corrected
  model response, and frozen-forcing model response.
- All 140 runs completed the horizon. This means finite completed simulations,
  not a convergence theorem. Conditions are correlated, not 28 independent
  datasets. The prior deterministic initial state is retained.
- A state packet costs 40+8*d application bytes. Model packets add 8*d gradient
  bytes and 16 policy/gain bytes at every hop. The incremental static descriptor
  costs (48+24*d*d) bytes per dependency per hop. Its setup delivery is assumed;
  latency/retry is not simulated. Common offline row/chart setup stays excluded.
- Equal-byte endpoints use the last completed state within the common byte
  budget; no interpolation or future state is used. Equal-tick comparisons are
  reported separately and must not substitute for equal-byte evidence.

## Results

Counts below are win/tie/loss for corrected model response. Ties use the existing
absolute gap floor 1e-20 and relative tolerance 1e-8, rather than ranking
numerical-zero SE3 differences.

| Reference | Stress, equal ticks | Stress, equal bytes | Ordinary, equal ticks | Ordinary, equal bytes |
| --- | ---: | ---: | ---: | ---: |
| Original directional hold | 8/0/0 | 6/0/2 | 16/4/0 | 16/2/2 |
| Adaptive hold | 6/1/1 | **0/0/8** | 12/8/0 | **0/2/18** |
| Adaptive linear prediction | 3/1/4 | 3/0/5 | 0/8/12 | 0/2/18 |
| Frozen-forcing model | 8/0/0 | 8/0/0 | 16/4/0 | 16/4/0 |

Continuous correction beats frozen forcing on all 24 nonideal conditions and
ties all four ideal conditions. This shows that stale model forcing matters;
it does not establish that transmitting this model is worth its cost.

### Previous Long-Delay Failures

Relative objective gaps at 2000 ticks, not ATE or localization errors:

| Model | Previous linear predictor | Corrected model response | Adaptive hold |
| --- | ---: | ---: | ---: |
| D1 | 2.65559e21 | 1.72658e-5 | 1.70228e-5 |
| D4 | 9.78721e20 | 5.62482e-4 | 6.51289e-4 |
| Coupled SPD | 1.60783e103 | 7.64172e-5 | 1.06961e-4 |

The new response avoids these observed divergences. However, hold already avoids
them, and on D1 it is slightly better even before charging the new payload.
Moving-step current-error coverage is 95.30%, 95.37%, and 95.76% for the new
response in these cases, versus 97.09%, 98.04%, and 99.36% for hold. Retrospective
error envelopes remain imperfect; neither stable runs nor the conditional
descent identity proves that the coupled predictive algorithm always converges.

### What Causes The Equal-Byte Failure?

The candidate uses 1.501--1.841 times hold's application bytes at the same
opportunity horizon. The extra static setup accounts for only 0.05--0.93% of
candidate bytes. A separate accounting counterfactual removes that setup charge
while leaving trajectories unchanged: the aggregate result remains **0 wins,
2 ties, 26 losses against adaptive hold**. Thus this is not primarily an
initialization-accounting penalty; the recurring gradient payload is too costly
for the finite-work progress it buys.

In ordinary networks the linear predictor is also faster in objective progress
at equal opportunities where the result is above the numerical floor. The model
response is conservative about unobserved boundary changes. Saturation helps
the stress behavior but does not supply a free acceleration mechanism.

Recorded service times are single Python-run observations, not controlled
same-accuracy runtime evidence. No native acceleration claim is made. A native
port or broad parameter sweep is not justified by this failed byte-efficiency
gate, so neither was performed.

## Independent Verification

- 119 focused regression tests pass. Tests cover copied packet payloads,
  delayed/reordered messages, invalid model data, reset, source opportunity
  counts, ideal bitwise parity, causal horizon prefixes, and byte alignment.
- A traffic-only replay independently reconstructs every prefix's bytes and
  update counts without using `Packet.wire_bytes` or `Network.send`.
- Sender-gradient tests replay deliveries and reconstruct each packet's gradient
  using the actual receiver cache, ruling out free access to current global data.
- The 24 legacy stress runs (three old arms, eight conditions) exactly match
  archived states and communication counters bit for bit.
- Independent objective and update checks pass for all 140 runs. Maximum
  normalized update discrepancy: 2.6103e-14. Maximum conditional-bound numerical
  violation: 1.3933e-32. The bound is checked only when its current-error premise
  actually holds; it is not an unconditional stability certificate.
- Both experiment archives and both separate review archives are SHA256 sealed.

## Reproduce

From the research repository; use new output paths on rerun:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_model_response --output results/dranppr_model_response_stress_20261124 --stress
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_model_response --output results/dranppr_model_response_controls_20261124 --scenarios ideal delay heterogeneous loss outage --seeds 35
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.review_model_response --root results/dranppr_model_response_stress_20261124 --parity-root results/dranppr_error_budget_stress_20261124 --output results/dranppr_model_response_stress_review_20261124
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.review_model_response --root results/dranppr_model_response_controls_20261124 --output results/dranppr_model_response_controls_review_20261124
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_quadratic_response.py tests/test_prediction_error_budget.py tests/test_review_error_budget.py tests/test_response_prediction.py tests/test_neighbor_prediction.py tests/test_history_forecast_controls.py tests/test_async_interface.py tests/test_async_interface_audit.py tests/test_review_async_interface.py tests/test_network_noise_support.py tests/test_native_rigid_subspace.py tests/test_native_lazy_refresh.py tests/test_native_joint_audit.py tests/test_native_interface_support.py
```

## Release Decision And Remaining Work

The parent plan's first acceptance gate fails on unit-communication benefit
against the strongest current comparator. Do not promote the weaker original-
baseline wins, the frozen-model ablation, or fixed-chart SE3 as a full solution.
No new riMESA comparison, native asynchronous nonlinear SE3 implementation,
unknown-delay guarantee, or graph/chart-change recovery has been established.

Retain the small optional model module, regression tests and sealed negative
evidence as diagnostics. Do not replace the production backend, delete useful
historical evidence, synchronize the dirty research fork, tag, push, or publish
a release. Subsequent research must reduce the recurring information cost or
obtain a larger correction benefit without sacrificing the tested stability.
Merely transmitting the same full gradient less often with a tuned interval is
not established here as an adaptive or generally effective solution.
