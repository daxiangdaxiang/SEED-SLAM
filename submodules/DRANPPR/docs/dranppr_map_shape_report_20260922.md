# Map-Shape Disagreement: State Uncertainty or Model Inconsistency?

Date: 2026-09-22. Frozen independent-local diagnostics, not a new distributed
solver or a new adaptation policy. The network gauge/bootstrap representation
is still awaiting confirmation. This audit proceeds without deciding it.

## Purpose and Controls

The previous interface-support audit found approximately 2.05 m robot-1/2
rigid-fit RMS on D9 300 s after robot 1 completed its fixed Gaussian problem.
This turn asks whether the conditional state covariance explains that gap and
whether a neighbor-informed restart reaches a better state on unchanged data.

All five robots on D9 300 s and D4 120 s are completed from their existing native
replay snapshots. Five established controls are retained per start: pinned
Schur, exact-Jacobian Schur, exact full solve, pinned LM and exact LM. The state,
odometry/observation factors and declared barcode identities are unchanged.
The covariance model stays at the common experimental values: polar sigmas
(.05 rad, .1 m), odometry sigmas (.1, .1, .05)*sqrt(dt). These are NOT claimed
to be calibrated sensor specifications.

One fixed local root remains a coordinate convention for each independent map.
No inter-robot pose prior, joint-map equality, covariance floor, robust gate,
learned weight or new sensor parameter is introduced. The producer uses no GT;
the existing wrapper writes its separate GT evaluation only after each solver
process exits. Neither the pair-distance analysis nor the restart construction
uses those GT metrics. D9 reference-yaw anomalies remain a limitation of the
separate posthoc trajectory metrics.

## Uncertainty Computation

The frozen native audit now exports object means and the complete covariance
from the exact-Jacobian GN information matrix. With private poses eliminated,

```text
S = H_BB - H_BP H_PP^(-1) H_PB,
P_B = S^(-1).
```

This is conditional on the chosen Gaussian measurement model, fixed gauge and
one local linearization, not an exact nonlinear posterior or an uncertainty
estimate over noise parameters. Correct whitening, rank and gauge handling
are prerequisites for inverse-information covariance; see the primary
[Ceres covariance implementation documentation](https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/covariance.h).

The export is audit-only. It materializes the small local object information
matrix; it does not modify the matrix-free network solver or claim cheap online
covariance updates. Export time is logged separately from optimization time.
No statistical scale is fitted to make the covariance agree with the errors.

For objects a and b, define d = ||l_a-l_b|| and u = (l_a-l_b)/d. Its first-order
conditional variance is

```text
v_ab = u^T (P_aa + P_bb - P_ab - P_ba) u.
```

Distances do not require inter-robot frame alignment. For independently solved
robots i and j, use z = |d_i-d_j| / sqrt(v_i+v_j). We report descriptive z
statistics, NOT fault probabilities or a thresholded detector. Distances share
objects/factors and are strongly dependent. Covariance misspecification,
nonlinearity, other basins and parameter uncertainty are not accounted for by z.
Coincident points have an undefined distance derivative and are reported as such.

## Results

All ten original-start audits and four restart audits exit normally: 14 native
producer invocations, each with five solver controls, 70 total completions.
They use the same binary SHA-256
`c7d2035d641e8cc0a26353ef89646e863d743933b2bc7e0b7339638861d91708`.
The maximum exact-Schur final gradient norm is 6.62e-6. Some runs end at
`no_decreasing_trial`, not the strict 1e-8 stationarity tolerance. The largest
reported covariance inversion residual across all controls is 5.05e-12.

### Keeping Correlations Changes the Interpretation

| Prefix / edge | Shared object pairs | Median z, full covariance | Max z, full covariance | Median z, object-block-only covariance |
| --- | ---: | ---: | ---: | ---: |
| D9 / 1-2 | 91 | 3.736 | 18.269 | .608 |
| D9 / 2-3 | 105 | 1.874 | 5.829 | .252 |
| D9 / 3-4 | 105 | 2.475 | 8.454 | .599 |
| D9 / 4-5 | 91 | 2.348 | 10.965 | .526 |
| D4 / 1-2 | 28 | .704 | 3.966 | .312 |
| D4 / 2-3 | 45 | 1.349 | 3.412 | .282 |
| D4 / 3-4 | 105 | 1.510 | 10.345 | .292 |
| D4 / 4-5 | 105 | 2.681 | 8.553 | .697 |

Dropping object cross covariance inflates the estimated variance of D9 edge-12
distances by a median factor 44.27 (maximum 1252). A common translation or
rotation can move every point substantially while preserving map shape. Treating
those point displacements as independent assigns that uncertainty to the wrong
physical quantity. This result argues against independent point-covariance
scores as a replacement for relative-geometry information. It does not prove
that the full GN covariance is well calibrated, and dropping cross covariance
is not always conservative in general.

A concrete D9 pair is objects 16/20: robot 1 estimates separation 0.250813 m,
robot 2 estimates 3.814998 m. The combined conditional standard deviation is
0.195097 m, yielding z=18.269. Pair 12/17 has 0.837923 versus 6.113275 m and
z=17.982. These are post-selected examples, not new independent trials. Such
large discrepancies are poorly explained by the current first-order fixed-noise
uncertainty, but do not identify whether the wrong assumption is noise, mean
model, identity, static structure or local-Gaussian state approximation.

D4 also has substantial discrepancies. Therefore a large z is not a valid
standalone gate for identifying the particular bad D9 trajectory, deciding a
relation edit, or inflating a sensor covariance. D4 is a comparison prefix, not
a perfectly calibrated negative control.

### Rigid Scale and Restart Alternatives Do Not Repair the Case

Before the covariance calculation, a descriptive registration audit of the D9
robot-1/2 completed maps gave a Sim(2) scale 0.996203 and RMS 2.053385 m, versus
SE(2) RMS 2.053426 m. Permitting reflection did not change the fitted determinant
or residual. A single common scale/reflection explanation is therefore not
supported for this pair; no similarity transform was used by an estimator.

Next, align robot 2's completed map to robot 1 by ALL common declared objects.
Replace ONLY robot 1's common object initial coordinates. Its poses, fixed root,
keys, observations, noise and objective remain unchanged. The opposite-direction
control applies the negative of that same object-coordinate perturbation. There
is no fitted step size or anchor selection. The perturbation norms are 7.683 m
on D9 and 1.914 m on D4 across all changed object coordinates.

| Prefix | Original completed cost | Neighbor-seeded exact-Schur cost | Opposite-seeded exact-Schur cost | Maximum positional difference to original |
| --- | ---: | ---: | ---: | ---: |
| D9 / robot 1 | 683.763994 | 683.763994 | 683.763994 | 2.02e-7 m |
| D4 / robot 1 | 59.092959 | 59.092959 | 59.092959 | 2.85e-8 m |

D9 neighbor/opposite exact-Schur takes 34/25 accepted iterations; D4 takes 9/13.
Exact full and exact LM controls corroborate the returned costs. This is a
negative repair result: importing a different object initialization does not
remove the remaining map-shape mismatch in these attempts. It is not a global
minimum proof, a complete multistart search, or evidence that joint multi-robot
constraints would have no effect. Joint constraints would change the available
information; these restarts deliberately do not.

### Raw Pose-Free Evidence

Reuse the already-tested `distance_variance` routine from
`audit_observation_pair_evidence.py` on each raw same-robot/same-timestamp
co-observation. This produces 581 pair rows on D9 300 s and 814 on D4 120 s,
with source-row IDs preserved. It does not use any estimated state or GT.

For D9 robot 1, raw observations of objects 13/20 imply:

| Elapsed time s | Pair distance m | Nominal first-order std m |
| --- | ---: | ---: |
| 32.410 | 2.41966 | .19315 |
| 32.644 | 2.37028 | .19916 |
| 123.643 | 2.42165 | .19375 |
| 189.551 | 2.93598 | .14683 |
| 226.392 | 3.03054 | .13902 |

The early observations include ranges about 5.0 and 7.284 m; later they include
3.131/6.034 m and 1.318/4.2 m. Robot 2 has 37 same-frame observations of this pair
within the prefix; its first and last pair distances are 2.87918 and 3.09017 m.
The completed local maps estimate 2.45718 m and 2.98386 m respectively.

These raw within-frame distances cannot be changed by any trajectory or gauge
update. They show a condition-dependent tension in the static-identity plus
measurement model, independent of lower-solver convergence. They do NOT establish
which reading is wrong or a unique calibration law. The small, dependent sample
does not justify a significance test. The worst inferred pairs 16/20 and 12/17
have NO same-frame support on either robot within this prefix; absence of that
evidence must not be replaced by an automatic fault label.

The official dataset describes monocular barcode-derived range/bearing data
and logged commanded velocities rather than exact measured body motion. It
does not justify treating our nominal Q/R as known physical truth.
[UTIAS MRCLAM data description](https://www.stars.utias.utoronto.ca/datasets/mrclam/index.html).
Quantization, viewpoint bias, temporal dependence and motion-model mismatch are
possible mechanisms, not identified causes from this audit. No new calibration
parameters have been introduced on that basis.

## Decision for DRANPPR

1. Keep full relative-geometry information in the Schur response. A compact
   representation must preserve shared uncertainty modes relevant to its score;
   independent object marginals can explain away genuine shape disagreement.
2. Do not allocate indefinitely more local iterations or add another restart
   mechanism for this case. The tested finite-solve and restart interventions
   already return the same local solution; add actual neighbor physical
   constraints in the pending joint-state experiment rather than only an initial
   transform or copied starting map.
3. Keep measurement-model adaptation separate from optimization residuals.
   Raw co-observation invariants and independently supported predictions are
   useful additional evidence, but sparse coverage prevents them from becoming
   a mandatory gate. A shared condition-dependent model needs identifiable,
   chronological support before replacing the failed six-parameter plug-in fit.
4. No association edit or noise inflation is justified by these z values alone.
   The full cause-aware distributed online research goal remains open.

Relative invariants, covariance propagation and neighbor initialization are
established tools, not the proposed novelty. The unverified research objective
is how their evidence changes the appropriate state/model/relation response
within a finite-budget distributed loop. This turn narrows wrong responses;
it does not yet demonstrate that complete algorithm.

## Verification and Artifacts

34 native C++ checks, 30 Python checks and six legacy regressions pass (70 checks).
New tests verify analytic range covariance 1/150, full/Schur marginal parity,
shared-translation cancellation, gauge/order invariance, invalid covariances,
coincident points, and pose/key preservation in restart construction. All producer
processes terminate normally. Solver timings are diagnostic only; concurrent
compilation/tests occurred during parts of this audit, so this is not a clean
hardware timing comparison. MPI 0 ranks; CCI/D-CCI and packaged DRAN++ unchanged.

```sh
cmake --build /tmp/dranppr_backend_build --target native-local-audit testIncrementalLocalModel -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit python3 -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py tests/test_native_map_shape.py tests/test_native_causal_replay.py tests/test_native_replay_profile.py tests/test_native_interface_support.py tests/test_observation_pair_evidence.py
```

Full outputs: `/tmp/dranppr_map_shape_20260922`. `run_status.json` records ten
original-start wrapper commands; `reseed_run_status.json` records the two
two-start wrapper commands. Their manifests pin all 14 native invocations,
sources, input hashes and the binary. Shape analysis uses
`python3 -m experiments.audit_native_map_shape --map ROBOT=PATH ... --links 1:2 2:3 3:4 4:5 --output NEW_DIRECTORY`.
Use each robot's `native/exact_schur_object_covariance.json`. The `_shape_final`
outputs pin the final analysis source; their pair CSVs match the initial analysis
byte-for-byte. `neighbor_object_starts` generates both restart CSVs from one
completed state and its allowed neighbor covariance record; per-start provenance
is in `d{4,9}_starts/provenance.json`.

Raw pair CSVs and their provenance reuse the existing `distance_variance` function:
group in-prefix observations by `(robot,timestamp_s)`, enumerate distinct object
pairs within each group, and preserve the two input source rows. No new numerical
estimator is involved in that extraction. Small results and source snapshots are
archived in `results/dranppr_map_shape_20260922/`; large state/trace files remain
in `/tmp` because `/home` is nearly full. No commit, push or default promotion.
