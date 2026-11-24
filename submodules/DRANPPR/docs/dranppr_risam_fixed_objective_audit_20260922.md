# Native Fixed-Objective Failure Attribution

Date: 2026-09-22. Research worktree only. This is an independent local-robot
diagnostic, not a completed distributed DRANPPR or full riMESA comparison.

## Question and Controls

Dataset 9 Robot 5 has a natural, uncorrupted native-stream failure: with the
same declared polar/odometry noise, Gaussian iSAM2 gives final local trajectory
RMSE 0.46968 m, while repaired riSAM gives 2.16851 m. Are the remaining local
iterations responsible, or can the robust target itself prefer poorer states?

Freeze the identical 300-second prefix: 1,020 poses and 1,167 native polar
observations. Keep the root pose fixed to the same local identity gauge. Test
the two saved initial states against two batch targets, using the same LM
settings (100 iterations maximum, relative/absolute error tolerance 1e-10).
No new measurement, relationship edit, noise change, GT initialization or
per-sequence parameter tuning enters these four solves.

The targets are ordinary Gaussian observations and final-mu SIG observations;
odometry stays Gaussian in both. For squared whitened polar error s, final SIG
is rho(s) = c^2 s / [2(c^2+s)], c = 1.8997411558140616. It is implemented by
GTSAM's Geman--McClure loss. Direct checks against the pinned official SIG kernel
give maximum cost and weight differences exactly 0.0 on all before/after
factors. This is the fixed final target, not a reconstruction of every internal
graduation parameter at the last online update. No claim of global optimality
or calibrated probability for this bounded penalty is made.

## Results

All four native C++ processes and the posthoc evaluator exit zero. The evaluator
uses an independently fitted local SE(2) alignment on poses after 30 seconds:
956 supported poses, 29 unsupported poses excluded/reported, 14 landmark copies.
This is not a network-wide ATE. Timing below is batch LM time, not online latency.

| Start | Target | Target cost before -> after | Local trajectory RMSE (m) | Object RMSE (m) | LM iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian iSAM2 | Gaussian | 734.387 -> 734.387 | 0.46968 -> 0.46968 | 0.33587 -> 0.33587 | 0 |
| Gaussian iSAM2 | Final SIG | 670.496 -> 473.058 | 0.46968 -> 0.84491 | 0.33587 -> 1.26341 | 100 |
| riSAM | Gaussian | 90374.943 -> 1305.082 | 2.16851 -> 3.03902 | 4.16558 -> 4.39793 | 54 |
| riSAM | Final SIG | 414.661 -> 411.711 | 2.16851 -> 2.19777 | 4.16558 -> 4.20620 | 100 |

Compare costs only within the same target. The two SIG arms reach the iteration
cap; their remaining free-variable gradient norms are 0.00337 and 0.000598.
The Gaussian-from-riSAM gradient is 1.21e-5. Gaussian-from-iSAM2 accepts no LM
iteration and retains gradient 0.10587; do not call it an exact batch optimum.
The three nontrivial completions each take approximately 0.75--0.78 seconds.

Three bounded conclusions follow:

1. **More of the same local optimization is not sufficient here.** Completing
   from riSAM substantially reduces its gradient (745.216 -> 0.000598) without
   recovering the desired state.
2. **The fixed robust objective can reward degradation even from a good start.**
   Starting from the Gaussian state, SIG optimization nearly doubles trajectory
   error while lowering the target by 197.438. The worse riSAM-derived terminal
   state also has lower SIG cost (411.711) than the Gaussian-derived terminal
   state (473.058). This is an observed objective/state-quality mismatch, not
   proof that every robust optimum is poor or that its global minimum is known.
3. **Nonlinear history matters as well.** Switching the failed riSAM state back
   to Gaussian loss does not find the much better Gaussian solution. Both
   objective preference and distinct local basins are implicated; a single
   solver-versus-model explanation would be too strong.

## Where the Objective Gain Comes From

For Gaussian-start SIG completion, ordinary odometry cost falls
555.640 -> 240.049, while robust observation cost rises 114.856 -> 233.009.
Thus the total improvement comes from fitting odometry better while sacrificing
observation agreement. This decomposition is on unchanged measurements.

For descriptive stratification only, count final-SIG weights below 0.1. They
increase from 4 to 91 out of 1,167 observations. The 91 are spread across 12
landmark identities rather than a single declared relationship. Their time-bin
counts are 3, 41, 19, 15 and 13 in [120,150), [150,180), [180,210), [210,240)
and [240,270) seconds. These are largely after the previously observed causal
trajectory divergence near 162 seconds. The riSAM-start SIG arm has 152 low
weights before and after completion. The 0.1 value is not an algorithm gate or
a tuned estimator setting; it only describes the existing continuous weights.

This supports the narrower interpretation that robust downweighting can
protect a motion-consistent but observation-inconsistent state. It does not
by itself establish whether the physical cause is odometry covariance mismatch,
bias, inadequate motion integration, observation-model mismatch, or some mixture.
Dataset 9's reference-yaw issues still prohibit oracle fault labels inferred
from every truth-conditioned residual.

## Independent Observation Evidence and Its Coverage

For two landmarks observed at the same timestamp, their measured separation

```text
d_ab(t) = || r_a(t) u(b_a(t)) - r_b(t) u(b_b(t)) ||
```

does not depend on the robot pose or global coordinate frame. Compare this
quantity with the immediately preceding observation of the same pair, using
first-order variance propagation from the unchanged nominal polar covariance.
This is an elementary geometric diagnostic, not a new association algorithm.
It uses only earlier/current raw observations, no estimated state and no GT.
Final SIG weights are joined retrospectively only for stratified reporting.

Of 1,019 observation-bearing frames, 875 contain one observation, 140 contain
two and four contain three. There are 136 repeated-pair comparisons covering
263 distinct current observations. Eleven comparisons touch a low-weight
observation; they cover only 13 of the 91 low-weight observations. Their median
absolute separation change is 0.07766 m, median nominal normalized magnitude
0.384 and maximum 2.768. The remaining 125 comparisons have median change
0.02561 m and median normalized magnitude 0.106.

Some downweighted measurements therefore retain cross-view geometric agreement
independent of the current pose estimate. But **78 of 91 lack this particular
independent test**. Temporal correlation, shared observations and uncertain
nominal noise prevent interpreting these magnitudes as calibrated independent
p-values. Pair distances also cannot recover a trajectory or disambiguate every
identity. Do not turn this limited evidence into a binary all-inlier claim or a
new mandatory multi-view gate that discards single-view measurements.

## Numerical Revalidation

The earlier Python/system-GTSAM completion intermittently exited 139 after
printing results. The pinned C++ completion reproduces its riSAM-start SIG
cost and state metrics to numerical precision and exits normally. The earlier
Python shutdown bug is still unresolved, not silently declared fixed.

Fourteen new tests pass with exit zero: native completion contracts, exact
prefix/key/gauge checking, official SIG parity, JSON conversion, pair-distance
rigid invariance, finite-difference covariance and chronological prefix invariance.
The existing 36 native replay/local-reference/evaluator tests also pass in a
separate clean process. No production DRAN++ source, CCI/D-CCI or MPI code changed.

## Reproduction

The build configuration is shared with `docs/dranppr_native_risam_report_20260922.md`.

```sh
cmake --build /tmp/dranppr-native-risam-build --target native-risam-checkpoint -j2
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_CHECKPOINT_BINARY=/tmp/dranppr-native-risam-build/native-risam-checkpoint python3 -m pytest -q tests/test_native_risam_checkpoint.py tests/test_audit_native_checkpoint.py tests/test_observation_pair_evidence.py
python3 experiments/audit_native_checkpoint.py --binary /tmp/dranppr-native-risam-build/native-risam-checkpoint --stream /tmp/dranppr_causal_streams_v1/dataset9 --start isam2=/tmp/dranppr_native_risam_dataset9_300_20260922/dataset9_isam2/final_estimates.csv --start risam=/tmp/dranppr_native_risam_dataset9_300_20260922/dataset9_risam/final_estimates.csv --robot 5 --duration 300 --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw --output NEW_OUTPUT
python3 experiments/audit_observation_pair_evidence.py --events /tmp/dranppr_causal_streams_v1/dataset9/events.jsonl --weights NEW_OUTPUT/isam2_sig_final/after_weights.csv --robot 5 --duration 300 --output NEW_OUTPUT/pair_evidence
```

Canonical artifact: `results/dranppr_native_risam_bridge_20260922/native_completion_d9r5/`.
It includes the four full outputs, input hashes, command/exit records, binary and
dynamic-library hashes, source snapshot and pair diagnostic. Earlier failed
Python attempts remain alongside it.

## Decision

### Positive Controls Rule Out a Simple Failure Gate

The same binary, noise, loss and completion protocol was subsequently applied to
Robot 5 in Datasets 1 and 4, using their existing 120-second saved states. These
are positive development controls, not a held-out test or equal-duration ranking
against the 300-second Dataset 9 case. Eight additional native completions and
both evaluators exit normally; all twelve runs share the same binary hash.

| Dataset | Gaussian-start SIG trajectory RMSE (m) | Object RMSE (m) | Low-weight observations before -> after |
| --- | ---: | ---: | ---: |
| 1, 120 s | 0.34957 -> 0.19133 | 1.08972 -> 0.80707 | 9 -> 26 of 517 |
| 4, 120 s | 0.23692 -> 0.15263 | 0.48279 -> 0.36397 | 5 -> 23 of 608 |
| 9, 300 s | 0.46968 -> 0.84491 | 0.33587 -> 1.26341 | 4 -> 91 of 1167 |

In Dataset 1, the odometry cost drops 49.047 -> 20.278 while robust observation
cost increases 97.515 -> 102.211. This is the same *sign pattern* as the failed
Dataset 9 completion, but the state improves. Therefore neither an increase in
the number of low weights nor trading observation agreement for odometry
agreement is, by itself, a valid failure detector. The target decomposition
explains the optimization incentive; it does not identify the physical fault.

For both Dataset 1 and 4, Gaussian-start and riSAM-start SIG completions reach
nearly the same local result (trajectory RMSE differences 6.45e-7 and 5.56e-6 m).
Returning the riSAM states to the Gaussian target instead recovers approximately
the original, poorer Gaussian accuracy. Removing robustness universally is
therefore contradicted by these controls. The earlier Dataset 9 evidence for
distinct nonlinear basins does not automatically extend to these two prefixes.

The controls contain 311/374 poses and 517/608 observations. Their post-30-second
evaluations use 291/268 poses, with no unsupported poses, and 11/15 landmark
copies. Dataset 1's evaluation-only label map remains unchanged. Artifacts are
`native_completion_d1r5/` and `native_completion_d4r5/` beside the Dataset 9 audit.

### Next Implementation Boundary

Do not spend the next stage sweeping SIG thresholds or adding extra rejection
gates. The demonstrated failure is more specific: observation residuals can be
reduced in influence when the mismatch should instead prompt reconsideration
of state or the motion/measurement model, and recovery can become basin-dependent.
An admissible improvement must preserve low-noise efficiency, use evidence
independent of the challenged state where available, and admit uncertainty
where it is not available. The previous short-warmup plug-in noise fit failed
transfer and must not be reintroduced as a validated solution.

This attribution closes a diagnostic question, not the research objective.
Persistent native-factor DRAN++ updates, evidence-aware statistical response,
causal network gauge initialization and matched distributed budget experiments
are still required.

Source inspection confirms that the current fixed-size, coordinate-isotropic
`QuadraticProblem` cannot directly represent the native polar likelihood.
The proposed extension is a persistent native-factor local model connected to
DRAN's reduced interface response, not an incremental file reader around the
old isotropic objective. Its architecture, chart-consistency contract and gates
are proposed for user review in
`docs/superpowers/specs/2026-09-22-dranppr-native-incremental-model-design.md`
at the workspace root. No production backend implementation is claimed in this
follow-up. Six existing DRAN Schur/owner-PCG component tests pass; that does not
validate a native-factor integration that has not yet been written.
