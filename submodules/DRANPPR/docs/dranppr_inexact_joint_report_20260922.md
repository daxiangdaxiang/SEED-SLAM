# Causal Joint Tracking: Model Reuse and Finite Interface Work

## Question and Scope

The previous D9/300 replay approached its frozen nonlinear completion, yet
accumulated 53.22 s of virtual serial backlog. Does solving each interface system
to 1e-10 unnecessarily delay usable estimates? Can cached local models avoid
repeated work without changing the measurement model?

This experiment changes computation only. Raw polar observations, native exact
Pose2 odometry factors, noise scales, arrived-label identity rules, chain topology,
gauge initialization, and the physical nonlinear acceptance loss are unchanged.
It is not a new robust estimator or a complete online riMESA comparison.

## Four Prespecified Arms

| Mode | Declared local charts | Interface PCG per arrived frame |
| --- | --- | --- |
| `joint` | Fresh | Relative tolerance 1e-10, cap 500 |
| `joint_cached` | Reuse below existing 1e-4 displacement threshold | Same |
| `joint_one` | Fresh | At most one iteration |
| `joint_cached_one` | Reuse below 1e-4 | At most one iteration |

All arms take at most one nonlinear joint step per frame, with the same physical
cost backtracking. A capped PCG solve may have a large verified residual; it is
not mislabeled converged. Nonfinite values and nonpositive CG curvature remain
hard failures. The one-iteration cap is an extreme diagnostic, not an adaptive
rule. The 1e-4 chart threshold is inherited from the local/iSAM2 audit, not tuned
on these sequences; it still mixes the chosen pose coordinate units and is not
a parameter-free or statistically calibrated policy.

## Why Cached Models Need a Different Right-Hand Side

Let theta be the declared linearization chart and let the currently published
estimate be x = Retr_theta(d). The cached quadratic is

`q(d) = c + g^T d + 0.5 d^T H d`.

The gradient for a correction to the current displacement is `g + H d`, not g.
With private/boundary partition P/B, define

`S = H_BB - H_BP H_PP^-1 H_PB`,
`b = g_B - H_BP H_PP^-1 g_P`.

Eliminating the private correction gives exactly `b + S d_B`: dependence on d_P
cancels. The routed interface therefore solves

`S s_B = -(b + S d_B)`.

Private recovery uses the original immutable response at `q_B = d_B + s_B`,
then subtracts the full current displacement: `s = recover(q_B) - d`.
Every trial evaluates the actual nonlinear loss at `Retr_theta(d + alpha*s)`.
Retracting s at x would silently mix tangent charts and is not what we do.

Both shared Point2 bases and their current displacements must agree. Cached
identity messages contain base and current point, with the extra doubles counted.
New poses/objects are initialized from current estimates, not stale bases.
Gauge mergers transform current values and fully rebase transformed models.
Response reuse requires matching local revision, fixed keys, and private keys.
Network identity changes alone need not invalidate an unchanged local response.

## Relation to Established Theory

Inexact Newton methods control the linear residual relative to the nonlinear
equation residual; demanding excessive inner accuracy can waste work when the
local model itself is inaccurate. Eisenstat and Walker also explain why a
forcing rule needs safeguards. Their fixed-problem local convergence conditions
do not automatically apply to a graph that grows and changes gauge/identity.
Source: [Eisenstat and Walker, SISC 1996](https://users.wpi.edu/~walker/Papers/forcing_terms,SISC_17,1996,16-32.pdf).

Real-time iteration treats optimization as tracking a moving solution rather
than repeatedly completing an offline solve. Its contraction results require
regularity and controlled perturbations. A single SQP step in that literature
is not a single CG iteration; our extreme arm must not inherit its guarantees.
Source: [Diehl, Bock and Schloder, CDC-ECC 2005](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/1186.pdf).

The elementary interpretation here is conditional: if each update contracts
optimization error by rho < 1 and newly arriving data move the local optimum by
v_t, then the next tracking error is bounded by roughly rho times the previous
error plus a term in v_t. This is a triangle-inequality intuition, not a verified
bound for this implementation. One CG iteration need not uniformly contract
the reported Euclidean residual. Physical cost descent alone is also insufficient
to establish a uniform tracking bound.

## Implications for Residual Attribution

An observation residual can persist because the state has not caught up with
available information. Solver residual, nonlinear completion gap, and delivery
age provide evidence about this computational cause. They do not establish that
an observation is incorrect or that its sensor variance changed. Conversely,
small optimizer residual does not establish a correct identity or noise model.

Do not use between-frame gradient changes directly as a Taylor-model defect:
the objective has changed when new data arrive. Any future forcing rule must
separate same-graph model discrepancy from information newly added at that frame.

## Results and Verification

Results are recorded after all estimation/completion processes terminate in
`/tmp/dranppr_inexact_joint_20260922`. The manifest retains source/binary/library
hashes, stream hashes, commands, exits, and wall times. Ground truth is loaded
only during postprocessing. Final comparison uses a single global SE2 alignment;
causal comparison uses the same suffix starting after every successful arm has
connected. Virtual delivery uses that suffix's fixed alignment, without motion
extrapolation. Actual network latency and asynchronous deployment are excluded.

### Same Arrived Data, Different Computational Work

Communication is decimal application MB, including identity/chart coordination.
It is not a measured network throughput. Times are one serial replay per arm,
not confidence intervals or independent random seeds.

| Case | Mode | Final trajectory / object RMSE (m) | Causal tail RMSE (m) | Service (s) | Max virtual backlog (s) | Application MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D4/120 | joint | .285676 / .184899 | .277125 | 36.800 | .1745 | 588.373 |
| D4/120 | cached | .285678 / .184898 | .277124 | 30.277 | .1346 | 585.202 |
| D4/120 | one | .285604 / .184824 | .273002 | 32.433 | .1537 | 49.838 |
| D4/120 | cached + one | .285605 / .184824 | .273001 | 32.653 | .1445 | 50.758 |
| D9/300 | joint | .401063 / .228986 | .559460 | 226.225 | 54.7375 | 1028.681 |
| D9/300 | cached | .401063 / .228986 | .559461 | 167.477 | 12.3525 | 1026.727 |
| D9/300 | one | .401430 / .229976 | .500603 | 208.601 | 40.9067 | 90.233 |
| D9/300 | cached + one | .401431 / .229978 | .500604 | 216.694 | 48.4344 | 91.816 |

Every arm processes 1710/4541 D4/D9 frames. All connect at the same sensor times,
15.534/131.723 s. Causal tails contain the same 1563/2515 supported pose timestamps
with zero omitted tail poses. Final D9 evaluation still excludes 437 poses outside
ground-truth support, as in the previous report. All final identity-copy consensus
errors are zero. All per-frame fixed-graph steps satisfy physical cost descent;
this does not claim descent across changes in graph or feasible identity domain.

Final physical costs for fresh/cached are 1230.388544407/1230.388544165 on D4 and
3280.284013642/3280.284014165 on D9. Their frozen-completion gaps are below 2.7e-6
and 9.9e-5. One-step gaps are .063869/.063787 on D4 and .089216/.088623 on D9
(fresh/cached). All eight separate completions reach the same respective candidate
minimum; no global-optimality certificate is claimed.

### The Two Speed Ideas Do Not Add Together

| Case / mode | Repeated factor linearizations in joint steps | Reused responses / opportunities | Model + publication (s) | Interface (s) |
| --- | ---: | ---: | ---: | ---: |
| D4 fresh | 3,464,588 | 0 / 8,043 | 25.298 | 6.167 |
| D4 cached | 1,795,506 | 636 / 8,043 | 18.354 | 6.197 |
| D4 one | 3,464,588 | 0 / 8,043 | 25.620 | 1.811 |
| D4 cached + one | 2,961,733 | 0 / 8,043 | 25.505 | 1.879 |
| D9 fresh | 17,901,216 | 0 / 13,193 | 159.733 | 24.471 |
| D9 cached | 6,854,162 | 1,484 / 13,193 | 99.856 | 24.542 |
| D9 one | 17,901,216 | 0 / 13,193 | 160.255 | 8.133 |
| D9 cached + one | 16,453,874 | 0 / 13,193 | 167.235 | 8.410 |

These counts exclude insertion and gauge/identity coordination, which are timed
separately. Model time includes chart refresh in cached arms; publication includes
relinearization in fresh arms. Their sum is the comparable category.

1. Tight-solve chart reuse preserves accuracy and reduces service by 17.7% on D4
   and 26.0% on D9. D9 backlog falls 77.4%, but 12.35 s remains unacceptable as a
   real-time claim. Its p99 service remains .1215 s and maximum .1783 s.
2. One PCG iteration reduces total bytes by 91.5%/91.2% while preserving nearly
   the same terminal state. This is a strong communication ablation, not a CPU
   solution: local model updates dominate, so service improves only 11.9%/7.8%.
3. Combined caching and one-step solving fails as an additional speedup. Repeated
   state changes leave no unchanged response to reuse. Compared with tight cached
   solves, many more historical factors cross the same chart refresh threshold.
   On D9 the small reduction in relinearizations versus fresh one-step solves
   does not repay displacement/refresh bookkeeping. This supports an interaction
   between unfinished optimization and cache invalidation, not two independent
   speed benefits. No threshold sweep was used to hide it.
4. The slightly smaller causal error in one-step arms is an observed path effect,
   not proof that less accurate solves generally estimate better. Linear residuals
   can exceed one (max 3.29 on D4 and 5.56 on D9); the fixed-norm inexact Newton
   contraction condition is not established. Physical descent remains verified.

### Delivery-Age Check

All variants use the same supported tail timestamps. Each retains its own
measurement-tail alignment when comparing against truth at virtual delivery time.

| D9 mode | Measurement-time RMSE (m) | Stale pose at delivery RMSE (m) | Mean / p99 age (s) |
| --- | ---: | ---: | ---: |
| joint | .559460 | 2.083866 | 13.588 / 52.957 |
| cached | .559461 | .771674 | 2.207 / 11.090 |
| one | .500603 | 1.580657 | 8.290 / 39.318 |
| cached + one | .500604 | 1.897612 | 11.563 / 46.562 |

The ranking changes when delivery age matters. Caching is the strongest tested
CPU/delivery intervention even though one-step solving has slightly smaller
timestamped causal error and much lower byte counts. D4 delivery errors remain
.277714/.277611/.273509/.273508 m in the same mode order, with no large queue.
These figures describe a serial-service, no-extrapolation diagnostic, not an
online navigation benchmark or measured multi-robot deployment.

### Verification

Eight chronological replays and eight frozen LM completions exit 0. Native tests
pass 37/37, focused Python tests 69/69, and six legacy Point2/owner-PCG regressions
pass. Tests cover nonzero cached displacement, private recovery against a full
dense solve on 2/3/5/10 robots, inconsistent shared charts, finite-iteration algebra,
all four streaming modes, causality, delayed merging, and disconnected aliases.
Both real fresh-mode final CSVs are byte-identical to the previous causal run.
Their causal CSVs are also byte-identical. A delivery-analysis invocation made
before `evaluation.json` existed exited with a missing-file error; the final
invocation after pipeline completion exits 0, without rerunning any estimator.

Tight-arm PCG iteration totals (fresh/cached) are 59,881/59,543 on D4 and
101,959/101,606 on D9, with independently recomputed relative residuals <= 1e-10.
One-step arms use 1,673 and 2,747 iterations total, deliberately without claiming
linear convergence. CCI/D-CCI is untouched; MPI is not used (0 ranks).

Replay binary SHA256:
`216b1d139ab19526aa04a0b8577b930deb2088606a00575de11ce23e8ff7246f`.
Completion binary SHA256:
`316f28a95315fcbf03a8f0a60ee32c5b3fd92f12c113f64d75b1ea7491f54d59`.

## Decision and Remaining Gates

Retain tight cached solving as the accuracy-preserving reference and one-step
solving as the low-communication ablation. Do not promote the combined mode.
Do not respond by sweeping chart thresholds or adding robust gates.

The next computational experiment should account for TOTAL work: extra interface
work can make local states stable enough to avoid many later relinearizations.
An adaptive stopping rule must estimate same-graph model error and remaining
tracking error, rather than choosing an arbitrary fixed inner iteration count.
Its benefit must be tested with separate byte and service budgets, and against
a local incremental-elimination implementation, not only this replay reference.

Independent native-model riMESA/iMESA, held-out sequence transfer, asynchronous
revision handling, and cause-aware noise/relationship adaptation remain open.
The present result establishes a computational tradeoff and a concrete failure
of combining two plausible speedups, not the overall research goal.

## Reproduction

From `projects/seed_slam/DRANPPR` after the native build:

```sh
python3 -m experiments.audit_native_joint_replay \
  --binary /tmp/dranppr_backend_build/bin/native-joint-replay \
  --oracle /tmp/dranppr_backend_build/bin/native-joint-audit \
  --stream-root /tmp/dranppr_causal_streams_v1 \
  --case 4:120 --case 9:300 \
  --mode joint --mode joint_cached --mode joint_one --mode joint_cached_one \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --output /tmp/dranppr_inexact_joint_20260922

python3 -m experiments.analyze_joint_delivery \
  --results /tmp/dranppr_inexact_joint_20260922 \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --output /tmp/dranppr_inexact_joint_20260922/delivery.json
```
