# Same-Objective Interface Study: A Communication Milestone

Date: 2026-11-23. Research DRANPPR only. No production/default change or push.

## Decision

The original target of consistently improving online state accuracy through
collective-motion acceleration is NOT met. Its five-dataset experiment is
mixed, and the candidate is not recommended as a default.

A narrower, verified milestone IS met: the existing one-step online update
can be executed with **22.78--27.13% less application communication**, with
**byte-identical first-published and terminal state files in all ten conditions**.
There is no new numerical tuning parameter, covariance change, approximation
to the state increment, or change to physical backtracking. This is an
algebraic protocol improvement, not a new robust estimator or a major
theoretical contribution. Wall-time improvement is not established.

## Fixed Experiment Protocol

- MRCLAM D1/D2/D3/D4: declared 120-second prefixes; D9: 300 seconds.
- Same five-robot chain, common-past initialization, arrived events, fixed root,
  known identities, Gaussian factors, persistent local models and lazy refresh.
- Both nominal weights and the previously fitted, frozen suffix weights.
  Calibration is not rerun or modified. No new GT-based parameter selection.
- One nonlinear update opportunity per arriving frame. Native PCG control uses
  one interface iteration, not an accurately solved interface problem.
- Two stages: 20 native executions for coarse-space testing, then 20 for the
  equivalent fused protocol. All finish under the existing 600 s process cap.
- All 20 fresh control state CSV pairs reproduce the corresponding archived
  states exactly. Timing is serial single-process service, not deployed network
  latency. Startup/calibration and transport framing remain excluded.

## 1. Collective-Motion Candidate

For the immutable reduced physical model

`m(delta) = g' delta + 0.5 delta' S delta`,

let `p = -D^-1 g` be the original block-Jacobi direction and let `Z` contain
the two translations and one infinitesimal rotation of the current landmark
positions. Solve the four-dimensional projected model with `V=[Z,p]`:

`c = -(V' S V)^dagger V' g`, `delta = V c`.

Each robot evaluates its local projected contribution. Existing topology-routed
collectives sum the compact model. Private trajectories recover through the
original elimination; the fixed root is never part of the free vector. Numerical
rank handling uses scaled machine precision, not a dataset-specific threshold.

In exact arithmetic, this subspace contains the old one-step PCG line, so its
minimum cannot be worse on the SAME quadratic model. This does not imply
better nonlinear tracking across successive, different linearizations or lower
truth error. All acceptance tests still use the unchanged nonlinear loss.

Rigid-mode coarse spaces have established precedents in numerical solvers;
they are not, by themselves, a novelty claim. The structural-mechanics results
of [Jonsthovel et al., SIAM J. Sci. Comput. 2013](https://epubs.siam.org/doi/abs/10.1137/100803651)
motivate this design but do not prove a SLAM convergence or statistical result.
Our projected one-step method is not their full deflated-CG algorithm.

### Calibrated Results, Including Failures

Both error columns refer to first-published poses. The first uses the existing
single global suffix registration, the second uses ONE common registration
from the identical warm prefix, never re-estimated per arm. Neither is silently
substituted for the other.

| Data | Existing causal RMSE, old -> coarse (m) | Fixed-prefix-frame RMSE, old -> coarse (m) | Physical terminal cost, old -> coarse |
| --- | ---: | ---: | ---: |
| D1 | .492764 -> .403563 | .763417 -> .525032 | 1829.849 -> 1825.798 |
| D2 | .175197 -> .184025 | .232779 -> .189305 | 2306.118 -> 2298.478 |
| D3 | .139516 -> .137862 | .179802 -> .304928 | 2179.832 -> 2183.880 |
| D4 | .201437 -> .208858 | .289938 -> .256162 | 2705.755 -> 2705.554 |
| D9 | .608974 -> .642755 | .911663 -> .759047 | 40735.882 -> 40707.833 |

Across both weight settings, only 3/10 conditions improve the existing causal
RMSE, and 6/10 improve fixed-prefix-frame RMSE. D3 calibrated fixed-frame error
increases about 70%. Coarse correction thus does not close the online-accuracy
gap. Terminal trajectory changes are small, which alone is not success.

The D1 calibrated service time increases from 6.20 to 11.37 seconds and refresh
requests from 18 to 170. This is consistent with larger steps making the reused
linearization inadequate more often; it is not an isolated causal proof of all
the truth-error regressions. D9 calibrated service increases 67.35 -> 92.64 s.

The earlier near-rigid completion displacement was a useful direction hypothesis,
not a guarantee that accelerating that direction improves online accuracy.
Different registration conventions, statistical/model error, linearization
error and optimization lag must remain distinguished. No gating or damping
parameter was added to conceal these failures.

## 2. Equivalent One-Step Protocol

The original PCG interface already communicates the local direction `p_i` and
the scalar reductions that define `alpha`. For one iteration,

`delta_i = alpha p_i`, `S_i delta_i = alpha (S_i p_i)`.

With each local response immutable during the epoch, every observer can form
its final increment from the direction it already received. Every owner can
form the final product from the accumulated direction product. Therefore the
final Update/VerifyProduct vector exchange and one local operator application
per endpoint are redundant. Remove them without changing the original gradient,
direction, curvature reduction, alpha arithmetic or physical accept/reject step.

This requires no learned threshold and preserves the original update. It is
limited to one-step execution; incompatible/multi-iteration options are rejected.
The final linear-residual diagnostic uses the cached product, not a second
independent operator application. This is explicitly labeled in summary JSON.
Against the old fresh-product diagnostic, the maximum discrepancy across all
real-data frame traces is **8.66e-14**.

### Verified Milestone

The following table shows calibrated arms. Nominal arms have the same saved
byte counts, with slightly different total percentages because physical
backtracking traffic differs between noise settings.

| Data | Old MB | Fused MB | Reduction | State outputs |
| --- | ---: | ---: | ---: | --- |
| D1 | 29.135232 | 21.852320 | 25.00% | byte-identical |
| D2 | 24.143720 | 18.641048 | 22.79% | byte-identical |
| D3 | 25.103176 | 18.520376 | 26.22% | byte-identical |
| D4 | 50.563784 | 37.189752 | 26.45% | byte-identical |
| D9 | 90.973528 | 66.412936 | 27.00% | byte-identical |

All ten conditions have zero cumulative-byte budget exceedances at every
published frame. Independent review checks identical per-frame physical costs,
acceptance decisions, refresh decisions, relinearization counts and predictions,
not just matching endpoints. Application bytes include every routed hop and
the existing 40-byte application header.

Observed service changes range from -1.35% to +0.23%. These are small, unrepeated
timing differences, not evidence of a meaningful wall-time speedup. The exact
communication saving is the established benefit. Recovered compute budget or
traffic savings have not yet been converted into a more accurate update policy.

## Baseline and 3D Boundaries

iMESA/riMESA were NOT rerun at the new smaller byte ceiling. Their earlier
results remain in the October/November comparison archives. Since the fused
states exactly reproduce the old native states, its previous accuracy ranking
does not change; there is no new claim of beating all baselines or of an equal-
byte SOTA accuracy win. The baseline here is the current native online research
path, not an assertion that the production GitHub solver has been upgraded.

The geometric basis is checked with six physical Pose3 motion directions and
finite differences, plus a full-6-DoF pose/object quadratic with the root exactly
excluded and private pose recovery. The linear reconstruction identity is also
checked on that SE3 Schur operator. These are deterministic local-model tests,
NOT a working 3D distributed online replay or a real-3D accuracy benchmark.
Native persistent values and the routed interface remain Pose2/Point2.

The SE3 fused-product relative discrepancy is 4.19e-16; private-recovery
relative residual is 3.72e-16. The controlled projected quadratic decrease
is 4.77134 -> 6.00106. These numbers are not 3D trajectory accuracy results.

The old high-Gaussian-noise negative cases, small-variance boundary estimates,
outlier/alias handling and continuous decentralized calibration remain open.
An equivalent protocol cannot fix their estimation errors. No variance floor,
anchor, GNC schedule, new prior or confidence gate was introduced.

## Code and Verification

- `include/DPGO/NativeSubspace.h`: small projected solve and geometric bases.
- `src/NativeInterface.cpp`: opt-in coarse solve and equivalent fused protocol.
- `experiments/native_local/joint_step.h`, `joint_replay.cpp`: opt-in flags
  `--rigid-subspace` and `--fused-one-step`, mutually exclusive.
- `experiments/audit_rigid_subspace.py`: paired replay and separate GT analysis.
- `experiments/summarize_subspace.py`: independent state/physical-trace review.

242 Python regression tests and 67 C++ tests pass. All four selected native
targets build. Schur full residual: 3.25e-16; matrix-free product error:
3.53e-17. Existing routed PCG check: 6 iterations, residual 1.98e-16;
2/3/5/10-robot checks: 12/16/23/39 iterations. MPI ranks: zero, not run.
CCI/D-CCI behavior is unchanged. The experiment and review archives contain
source snapshots, commands, state outputs, summaries and SHA256 ledgers.

Reproduce a paired fused run, without evaluating GT during estimation:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_rigid_subspace run --source results/dranppr_online_calibration_baselines_20261029 --datasets 1 4 --candidate fused --output NEW_OUTPUT
```

Actual paired archives: `results/dranppr_rigid_{d1,d23,d4,d9}_20261123` and
`results/dranppr_fused_{d14,d23,d9}_20261123`. Review:
`results/dranppr_subspace_review_20261123`.

## Next Research Gate

Retain fused execution as a verified opt-in building block. Do not promote
coarse correction or frozen calibration as a generally safer estimator. The
next accuracy proposal must explain and repair the D3 fixed-frame regression
and D1/D9 causal calibration failures, while preserving nominal-noise controls.
The important open question is how much confidence noisy data justify in weak
collective directions, not merely how to move farther along them. Do not answer
it with a dataset-specific damping constant or a new collection of gates.
