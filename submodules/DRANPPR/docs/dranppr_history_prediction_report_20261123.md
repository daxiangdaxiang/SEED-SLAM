# History-Based Neighbor Prediction: Positive But Bounded Milestone

Date: 2026-11-23. Research DRANPPR branch only. No production default change,
commit, push, sensor reweighting, CCI change, or conversion to CORD.

## Decision

Received history is useful for delay compensation in the existing directional
interface optimizer. It is NOT enough to increase extrapolation order. The
adaptive forecast-risk candidate improves the final fixed-quadratic objective
gap in 93/135 matched conditions, ties 27 and loses 15. All 27 ties are the
ideal-network controls, whose states are exactly unchanged.

This is a positive fixed-model algorithm result, not end-to-end nonlinear SLAM
or a new DRAN/iMESA/riMESA comparison. It does not establish that adaptive higher
order is always faster than a two-sample predictor: the candidate beats secant
in 49 conditions, ties 27 and loses 59. It beats fixed quadratic history in
37, ties 27 and loses 71. More wins versus hold is NOT universal superiority
over another predictor. The useful result is a broader positive range and
control of high-order noise amplification, with remaining speed tradeoffs.

## Method

See [the method and conditional derivation](dranppr_history_prediction_method_20261123.md).

The receiver retains eight actual, timestamped messages, fits local polynomial
histories up to degree three, and predicts the fitted increment from the latest
received state. It selects order using one criterion:

`coupling-space residual variance x squared norm of absolute forecast weights`.

This accounts for both fit mismatch and horizon-dependent noise amplification.
Degree zero means hold; degree one estimates velocity; degrees two/three allow
changing velocity. The existing cross-Hessian block measures relevance to the
receiver gradient. No extra messages, manually weighted blend, kinetic state,
mass/damping parameter, or global merit check is introduced. Maximum degree
and memory length are still explicit structural choices. The criterion is a
risk proxy, not calibrated uncertainty or a guaranteed future-error bound.

## Protocol And Provenance

- Five archived MRCLAM physical models D1/D2/D3/D4/D9, three synthetic six-DoF
  SE3 pose/object Jacobians, and one scalar-block SPD control.
- Five networks: ideal, delay, heterogeneous compute periods, bounded burst
  loss, and finite outage; three network-calendar seeds; 2000 ticks.
- Four arms: hold, secant, eight-sample quadratic, adaptive risk up to cubic.
  Final primary total: 540 runs. Models, initialization, owner updates, sensor
  weights, directional metric and message opportunities are identical.
- A tick is a synthetic compute opportunity, not wall-clock time. Relay links
  remain reliable in the burst-loss model; queues/contention are absent. Valid
  delay bounds, fixed charts and pre-distributed reduced rows remain required.
- These Jacobians were exported near convergence. The common initial condition
  is the information-scaled perturbation sin(k+1); it does not use truth, but
  it tests linear solver dynamics, not a valid nonlinear step of that size.
- Development grid: D1/D4/SE3_31, seed 31, three nonideal networks, 600 ticks.
  Other five models were not used in that predictor-development grid. This is
  not a blinded external benchmark or statistical independence claim; archived
  data and some earlier control results were already available in this project.

## Primary Results

All entries below compare adaptive risk prediction with the SAME directional
optimizer using hold. Each row has 12 nonideal conditions. Percentages are
median reductions of the final relative quadratic gap, NOT ATE or RMSE.

| Model | Better / tied / worse | Median gap reduction |
| --- | ---: | ---: |
| MRCLAM D1 | 3 / 0 / 9 | -2.32% |
| MRCLAM D2 | 8 / 0 / 4 | 0.40% |
| MRCLAM D3 | 12 / 0 / 0 | 4.27% |
| MRCLAM D4 | 12 / 0 / 0 | 12.01% |
| MRCLAM D9 | 10 / 0 / 2 | 3.53% |
| SE3 model 31 | 12 / 0 / 0 | 77.93% |
| SE3 model 32 | 12 / 0 / 0 | 77.69% |
| SE3 model 33 | 12 / 0 / 0 | 77.76% |
| Scalar SPD control | 12 / 0 / 0 | 27.31% |

Across the 60 nonideal MRCLAM conditions, secant improves 32 and worsens 28;
quadratic history improves 43 and worsens 17; adaptive risk improves 45 and
worsens 15. In the three SE3 models all predictors improve all 36 conditions,
and simple secant is often slightly faster. Higher-order necessity in SE3 is
therefore NOT demonstrated.

On the five models outside the predictor-development grid, adaptive risk wins
54/60 nonideal comparisons versus 41/60 for secant and 50/60 for quadratic.
No per-model setting was changed. Do not interpret these correlated conditions
as 60 independent datasets or a significance test.

The final gradient-residual comparison for adaptive risk is also 93 wins,
27 ties and 15 losses, but this aggregate is not a proof that every gap win
improves every metric. Cumulative forecast-gradient error can get worse while
the endpoint improves. D9's median same-trajectory forecast-error ratio is
1.041, versus hold on that trajectory, despite its positive endpoint median.

### Same-Accuracy Communication

For the 27 nonideal physical-model pairs in which BOTH hold and adaptive risk
reach the previously used relative-gap target 1e-6 within the horizon, adaptive
risk uses 6.96--9.34% fewer application bytes, median 8.62%. All these qualifying
pairs are SE3 delay/heterogeneous/loss cases. The other 69 physical nonideal
pairs do not qualify and are not silently credited with a speedup. For example,
SE3_31/delay/seed31 reaches the target at tick 779 versus 852 for hold.

At the fixed endpoint, every candidate/reference pair has identical bytes and
local-update counts at EVERY prefix. The gains are not extra iterations or
extra velocity messages. Setup communication remains excluded in all arms.

### Compute Cost Is Not Yet A Win

The Python risk prototype has a median simulated-run service-time ratio of
13.39 versus hold over nonideal conditions. This includes offline diagnostics
and is not distributed robot CPU time, but it clearly precludes claiming a
wall-clock acceleration here. Repeated tiny QR fits have substantial Python
overhead. No claim that a native implementation will eliminate this cost is
made without measuring it. Current speed evidence is iteration/communication
efficiency, not end-to-end runtime.

## Noise And Unobservable-Change Controls

The separate forecast audit has five synthetic trajectories, three noise
levels, three horizons and six predictors, with 200 common-noise trials per
cell. These are coordinate-history tests, not sensor-noise SLAM experiments.

For a smooth quadratic trajectory, history noise SD 0.01 and horizon 16:

| Predictor | Mean squared forecast error |
| --- | ---: |
| Hold | 0.156533 |
| Two-sample secant | 0.109401 |
| History linear | 0.010881 |
| History quadratic | 0.024595 |
| History cubic | 0.987855 |
| Adaptive risk | 0.011849 |

Adaptive risk chooses linear in all 200 trials here. It is close to, but NOT
better than, the fixed linear oracle choice for this cell. At SD 0.05/horizon
16, its MSE is 0.112621 versus 0.614865 quadratic and 24.696376 cubic, but fixed
linear remains better at 0.042361. At SD 0.05/horizon 1 it is worse than hold
(0.005936 versus 0.005085). Thus adaptive selection mitigates rather than
eliminates prediction risk. Noiseless polynomial tests recover supported
curvature to numerical precision.

The stop/reversal controls share exactly the same past with a continuing
linear trajectory, then change after the last packet. A predictor cannot know
which future occurred; the adaptive predictor also fails these controls. This
is an information boundary, not a defect that more derivative orders can cure.

## Stability And Remaining Negative Cases

The complete-graph SPD control B=0.2I+0.8*11^T, five owners, uniform four-tick
delay, 1000-tick budget gives:

| Update rule | Hold gap | Adaptive risk gap |
| --- | ---: | ---: |
| Unguarded stale step | 6.94e61 | 2.86e201; stopped before horizon |
| Directional delay-controlled step | 3.79e-5 | 2.29e-5 |

Secant and quadratic also explode with the unguarded step. Prediction is NOT a
substitute for stable coupling control. The finite norm stop is an experiment
health check, not an optimization safeguard.

The 15 primary adaptive-risk regressions are concentrated in D1 (9), D2 (4),
and D9 (2). Largest worsening is 5.04% in D1/heterogeneous/seed31. D1's negative
median cannot be hidden by gains in the easier two-owner SE3 models.

Memory-size checks at 4/8/16 samples, D1/D4/SE3_31, seed32, delay/heterogeneous/
outage, retain six wins and three losses for each size. All three D1 conditions
remain worse. There is no per-dataset history sweep. This limited transfer
check supports a direction, not parameter independence across all datasets.

## Independent Checks

- 540 final metric checks; 135 hold traces exactly match the prior sealed
  directional archive; 405 candidate/hold prefix-budget checks.
- Reconstruct estimated gradient as -M*increment from saved states. Independent
  squared gradient-error totals match the simulation's recorded diagnostic.
- The conditional inexact-gradient bound in the method note holds numerically
  on all checked traces; maximum normalized positive violation is zero. This
  verifies the algebra/implementation, NOT the predictor's convergence.
- 71 focused Python tests pass, including 19 new prediction/review tests;
  67 existing native C++ tests pass. No new native solver or PCG iteration.
  MPI ranks: zero; not exercised. No CCI behavior changed.

## Artifacts And Attempts

- `results/dranppr_history_final_20261123`: canonical 540-run primary matrix.
- `results/dranppr_history_review_20261123`: independent review and eight
  complete-graph stability controls, including explicit divergent traces.
- `results/dranppr_history_noise_20261123`: synthetic forecast audit.
- `results/dranppr_history_window4_20261123` and `results/dranppr_history_window16_20261123`:
  36 memory-transfer runs.
- `history_development`, `history_shrink_development`, `history_risk_development`:
  45/27/27 development runs and frozen source. A coefficient-only shrinkage
  candidate failed its noise target and was removed from active code. It is
  retained as negative evidence in its source archive.
- `dranppr_history_primary_20261123` is interrupted and NOT final evidence.
  Its verifier wrongly demanded relative accuracy on residuals around 1e-16.
  A regression test and matrix-product forward-error tolerance fixed this
  numerical audit issue; the complete primary matrix was rerun separately.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_neighbor_prediction \
  --output results/NEW_history --predictions hold secant poly2 risk3 --ticks 2000 \
  --parity-root results/dranppr_async_directional_20261123
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.review_history_prediction \
  --root results/NEW_history --prior results/dranppr_async_directional_20261123 \
  --output results/NEW_history_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_history_forecast \
  --output results/NEW_history_noise
```

## Next Research Boundary

The positive conclusion is that receiver-side history can improve the existing
delay-controlled optimizer without extra state messages. The next step is NOT
blindly increasing order. It is controlling prediction-induced gradient error
under nonlinear model/chart changes, while reducing the measured compute
overhead and preserving original-objective correctness. Then perform native
online SE3 and matched DRAN/iMESA/riMESA reruns. Until then, this is an audited
research candidate rather than a production or TRO-level completion claim.
