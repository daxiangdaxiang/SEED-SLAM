# Native Network-Joining Transient and Baseline Contract Audit

## Purpose

Separate residuals caused by unfinished joint estimation after network joining
from a need to change the sensor model or identity. Also inspect prerequisites
for a matched native riMESA/iMESA comparison before writing another adapter.
This is an existing-code diagnostic, not a new algorithm or a SOTA ranking.

## Setting

Use the unchanged `joint_cached` replay with the existing chain 1-2-3-4-5,
declared barcodes, native polar/odometry factors, and noise parameters. Stop at
the first complete frame where the five robots become one component. The rule
is causal; the two cases are development cases, not held-out tests.

- D4: 15.534000158309937 seconds, 148 pose frames, 209 observations,
  143 odometry factors.
- D9: 131.72300004959106 seconds, 2027 pose frames, 2200 observations,
  2022 odometry factors.

Both causal output prefixes exactly match the corresponding rows in the
archived full-duration control. No truth was loaded. Package each terminal
graph with the existing frozen-joint preparation tool, preserving only robot
1's root gauge and leaving all other roots free. Preparation changes the
reported initial cost by less than 2.1e-12.

Run one routed GN update, one separate central GN update, and a separate
central LM completion on each identical package. No new observation, covariance
change, relation edit, or robust loss enters these six solves. All two replays,
two preparations and six solves exit zero. The one-step arms intentionally
report `iteration_limit`; they are not claimed converged.

## Results

| Case | Cost at join | After one routed update | LM completion | One-step cost reduction | PCG iterations | Verified relative residual | Routed application bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D4 | 70.239691410 | 70.230103800 | 70.230086641 | 0.01365% | 33 | 2.47e-12 | 107168 |
| D9 | 530.327499133 | 109.438504007 | 109.400444859 | 79.36398% | 35 | 1.11e-11 | 109192 |

Maximum routed-versus-central positional differences are 8.62e-14 m and
3.20e-11 m. Cost differences are 4.26e-14 and 2.94e-12. The fixed root stays
unchanged; other roots move up to .00910 m and .15421 m, respectively. Thus
the agreement is not obtained by fixing all robot coordinate frames.

In D9, the joining frame itself has already reduced the new joint cost from
1659.317263 to 530.327499. The additional frozen update is distinct from that
original online step. It closes 99.99096% of the gap to this particular LM
completion, but that completion is not a global-optimality certificate.
LM takes six iterations on D9 and five on D4; final physical gradient norms
are 3.19e-5 and 1.82e-5.

The additional routed updates cost about 0.107/0.109 MB of application traffic
(decimal MB), including existing logical packet headers. Do not describe one
outer update as one communication or as a cheap constant-cost operation.
Preparation/rosters and wire transport remain outside the frozen-solve ledger.
This is a single-process synchronous simulation, MPI 0 ranks. CCI is unchanged.

## Interpretation

This is a concrete natural event where substantial fixed-model residual is
removable by additional state/communication work. Immediate sensor-noise
inflation at the joining state could absorb an optimization transient. D4
provides a useful contrast: the same event type does not imply that extra
computation has substantial value.

The audit does not establish sensor correctness, identity correctness, improved
ATE, calibrated covariance, or an online scheduling policy. No ground-truth
evaluation was performed. It also does not show that an arbitrary large residual
has the same cause. A schedule based solely on 'a component just joined' would
still need justification; no new event gate is introduced here.

## Baseline Contract Findings

The pinned riMESA source is still clean at
`ffddc21dad530ba0b0e57c42495146b410581fc4`. Its public `RIMESA::update` accepts
new factors and values, but does not expose removal of an old root prior or an
atomic coordinate-frame change for an entire connected component. Underlying
riSAM has factor-removal and force-linearization operations; this is an adapter
compatibility issue, not a claim that riMESA cannot estimate relative frames.

Permanently adding one root prior per robot would change the physical problem.
A proposed first comparison instead gives every backend the same past-only
bootstrap state and prefix factors, then measures subsequent updates. It must
retain one global gauge and initialize ordinary riMESA neighbor state before
attempting ungauged local solves. This adapter is not yet implemented or tested.
Using a DRAN-derived start is also a declared conditioning choice; it does not
establish unbiased from-scratch end-to-end superiority.

riMESA's `CommunicationHandler::commSizeKB()` counts logical variable/key and
solution payload in decimal KB, without the 40-byte logical headers included
in our routed ledger. A future comparison must report both payload-only and
common-framing application bytes. Handshake counts cannot be equated with PCG
outer iterations. Initial declarations, startup updates, and completion work
must not become free work. Official and numerically repaired riMESA variants
must remain separately labelled.

The paper itself distinguishes measurement updates from communication and
reports sensitivity to communication quality. This supports auditing both,
not inferring backend quality from local riSAM alone:
[riMESA, Sections V and VII](https://arxiv.org/html/2603.01178v1).

## Next Steps

The proposed next step is a common-past-initialization backend control, followed
by a separately validated from-scratch experiment. The user has been asked to
choose this scope or prioritize full causal initialization first. Neither
option is silently treated as already implemented. In either case, compare
the same arrived factors at budget checkpoints; do not compare states with
different information simply because cumulative bytes or elapsed time match.

The statistical adaptation transfer failure remains unresolved. This audit
does not complete the broad DRANPPR research goal.

## Reproduction and Artifacts

Worktree: `projects/seed_slam/DRANPPR`. Artifacts:
`/tmp/dranppr_matched_bootstrap_d4_20260923` and
`/tmp/dranppr_matched_bootstrap_d9_20260923`. Each contains the replay output,
`frozen.json`, and `routed_one`, `central_one`, and `completion` outputs.

Example D9 commands (use a fresh output directory):

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/dranppr_backend_build/bin/native-joint-replay /tmp/dranppr_causal_streams_v1/dataset9/events.jsonl OUTPUT joint_cached 131.72300004959106 1:2 2:3 3:4 4:5
python3 -m experiments.prepare_native_joint --stream /tmp/dranppr_causal_streams_v1/dataset9 --snapshot OUTPUT/final_estimates.csv --duration 131.72300004959106 --links 1:2 2:3 3:4 4:5 --output OUTPUT/frozen.json
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/dranppr_backend_build/bin/native-joint-audit OUTPUT/frozen.json OUTPUT/routed_one routed 1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/dranppr_backend_build/bin/native-joint-audit OUTPUT/frozen.json OUTPUT/central_one central_gn 1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/dranppr_backend_build/bin/native-joint-audit OUTPUT/frozen.json OUTPUT/completion central_lm 100
```

For D4 substitute dataset4 and duration 15.534000158309937. Binary hashes:

- Replay: `a50a4e57ba772d6f2ca1f0f66355c1f1a8a561c6b877c5384e92bde2917c2bd1`.
- Audit: `4bc6384ae7b4490bbb4ecc29f28fc03552c410474436dfdba458092663a280c4`.

Package SHA256:

- D4: `23f3e3a9935ede843bd2e2607822a437e69dc66401c2562a915d727abe26b138`.
- D9: `1e47f5bb7eef1d028569bdb04a4f95737e3afc0d25594243e67a301f907f388c`.

Source and parameters are unchanged from the archived instrumented control;
package provenance also records event, snapshot and preparation-script hashes.
No new estimator code, baseline patch, commit, or push was made.
