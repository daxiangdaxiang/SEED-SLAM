# Frozen Joint Native Estimation: Real-Data Gate

## Decision

Actual shared-object joint estimation is now verified on MRCLAM D4 120 s and
D9 300 s. It changes trajectories and object geometry, not only initial frame
alignment. Routed Schur/PCG reaches the same fixed-Gaussian physical solution
as separately executed centralized GN and LM controls. No noise parameter,
robust loss, association threshold or rejection mechanism was introduced.

This is a correctness-first **frozen synchronous graph** experiment. It does
not complete online DRANPPR, prove robust cause attribution, or beat riMESA.
The existing research worktree remains experimental and the packaged backend
is unchanged. The common-frame representation is a reversible experimental
choice under the approved native-model plan, not a claimed explicit approval
of a deployed initialization protocol.

## Model And Execution Contract

Each robot retains original Pose2 odometry and polar object observations, with
the same fixed noise as the earlier native controls: odometry sigmas
`(.1,.1,.05)*sqrt(dt)` and bearing/range sigmas `(.05,.1)`. The objective is
one half of the sum of whitened squared physical residuals. Odometry uses the
previously verified complete Pose2 chart Jacobian. No pseudo-observation or
isotropic Cartesian replacement is used.

The supplied communication graph is `1-2-3-4-5`. Arrived independent estimates
are registered along supported neighbor edges without ground truth. For each
barcode identity, shared copies are merged only inside connected components of
its observer-induced communication graph. A missing observing intermediate robot
does not create an extra identity edge. D4 has 68 local object copies and 20
components; D9 has 73 copies and 15 components. Singleton objects remain valid.

All states enter a common component chart. Only robot 1 pose 0 is fixed exactly;
other root poses remain free. Common object initial values come from the lowest
ID observer, ensuring equal chart bases. This can make initial joint residuals
large: it is a projection of independently inconsistent maps onto the shared
identity model, not an optimized baseline.

Each nonlinear iteration:

1. Export the local native quadratic and eliminate private free poses.
2. Solve the shared Point2 system with topology-routed owner-PCG.
3. Recover private pose increments.
4. Tree-reduce physical cost, directional slope and curvature, with every hop
   counted. Evaluate common backtracking trials using another counted tree sum.
5. Publish only a decreasing joint physical state and freshly relinearize.

Numerical controls are fixed: at most 100 nonlinear iterations, 500 PCG steps,
PCG target relative residual `1e-10`, verified residual acceptance `1e-8`, and
30 step halvings. No per-dataset sweep. Exact shared Point2 increments preserve
consensus. The scalar reducer uses the existing interface transport, not an
uncounted central objective oracle. Full gradient and state-error diagnostics
are computed only after the solver exits.

The centralized controls use identical factors, initial values and one gauge.
GN solves the entire sparse tangent system; LM uses an exact constrained root
factor. The pinned GTSAM generic `rekey` does not update a BearingRangeFactor's
expression tree, so centralized polar factors are reconstructed with the new
keys and unchanged measurement/noise. A two-robot first-step parity test caught
this before any real-data result was accepted.

## Real-Data Results

All six final runs exit normally. Main metrics use **one global SE(2) alignment
for all robots**, fitted to GT-supported poses after all estimation. They use
the full arrived prefix, not the earlier post-30-second evaluation window.

| Prefix | State | Physical cost | Global trajectory RMSE m | Object-copy RMSE m |
| --- | --- | ---: | ---: | ---: |
| D4 120 s | Prepared initial state | 39971.874122 | .409744 | .729536 |
| D4 120 s | Routed joint | 1230.388542 | .285677 | .184901 |
| D4 120 s | Central GN | 1230.388542 | .285677 | .184901 |
| D4 120 s | Central LM | 1230.388542 | .285677 | .184901 |
| D9 300 s | Prepared initial state | 400238.334343 | .930431 | 1.957320 |
| D9 300 s | Routed joint | 3635.515115 | .482310 | .264407 |
| D9 300 s | Central GN | 3635.515115 | .482310 | .264407 |
| D9 300 s | Central LM | 3635.515115 | .482310 | .264407 |

D4 has 1710 supported poses and no omitted poses. D9 has 4541 estimated poses,
4104 GT-supported poses and 437 unsupported timestamps, explicitly excluded.
The known D9 ground-truth yaw issues are not reinterpreted as sensor fault labels.
Objects are copy-weighted, not uniformly weighted unique physical identities.

Maximum unaligned positional difference versus routed is `1.78e-8 m` for D4
central GN and `1.81e-7 m` for LM; D9 differences are `3.10e-8 m` and `5.60e-7 m`.
All modes have exactly zero shared-copy discrepancy and unchanged root gauge.
Routed final free joint gradient norms are `2.63e-7` and `2.10e-6`. These are
numerically converged candidates, not global-optimality certificates.

### Separating Shape Improvement From Registration

A secondary diagnostic gives **each robot its own posthoc alignment in every
arm**, intentionally removing relative-frame errors to inspect local shape.
It must not be mixed with the main global metrics above. Independent completed
states come from the previous exact-Schur frozen local controls on the same
factors, not from extra hidden work before the joint start.

| Prefix | State | Per-robot-aligned trajectory RMSE m | Object-copy RMSE m |
| --- | --- | ---: | ---: |
| D4 | Independent online snapshot | .289545 | .600412 |
| D4 | Independent fully completed | .289539 | .600389 |
| D4 | Joint, same per-robot diagnostic | .216583 | .480488 |
| D9 | Independent online snapshot | .809533 | 1.262390 |
| D9 | Independent fully completed | .761066 | 1.235874 |
| D9 | Joint, same per-robot diagnostic | .433686 | .379825 |

Thus the D9 benefit is not explained only by local convergence or coordinate
alignment. The previous neighbor-initialization-only intervention returned to
the same local optima; retaining actual shared constraints changes the solution.
This is evidence for useful cross-robot information in these cases, not a novel
property of Gaussian joint estimation or an equal-budget SOTA comparison.

### Why Not Every Residual Decreases

At the prepared D9 start, odometry cost is `2021.670` and observation cost is
`398216.664`. Joint estimation changes them to `2636.979` and `998.536`.
D4 odometry cost similarly rises `263.201 -> 291.231`. Joint consistency trades
off conflicting measurements; it does not promise each robot/factor gets better.
D9 robot 1 alone changes from odometry/observation costs `417.861/926.648` to
`795.203/199.520`. Interpreting the larger odometry residual as newly corrupted
odometry would be unjustified: the state and information set also changed.

Independent completed sums are lower still: D4 `942.284`, D9 `2935.903`. This is
expected because independent object copies have extra degrees of freedom. Their
smaller loss and worse state errors are **not** a contradiction. Objective values
across different feasible sets do not rank physical accuracy. Likewise, the huge
decrease from the prepared common-object start is not a fair baseline advantage.

## Cost And Communication

| Routed diagnostic | D4 120 s | D9 300 s |
| --- | ---: | ---: |
| Nonlinear rounds, including failed final trial | 16 | 20 |
| Accepted rounds | 15 | 19 |
| PCG iterations, total | 539 | 602 |
| PCG iterations per round | 21--39 | 30--31 |
| Max verified PCG relative residual | 9.43e-11 | 4.18e-11 |
| Application bytes | 7297264 | 10703104 |
| Numeric payload bytes | 2099424 | 3095424 |
| Cost/model scalar-collective bytes (included) | 33728 | 43392 |
| Routed hop messages | 129946 | 190192 |
| Serial simulator solve seconds | .652 | 2.250 |
| Central GN solve seconds | .843 | 2.669 |
| Central LM solve seconds | .151 | .533 |

Each logical message includes a 40-byte header. These are explicit simulator
application bytes, not measured sockets/MPI traffic. Offline bootstrap, identity
rosters, framing, retransmission and asynchronous revision management are not
included. Preparation takes about `.163/.174 s` and is separately logged.
Reported solve time includes fresh model work and updates but excludes initial
package parsing/model assembly (`.018/.044 s`) and posthoc GT evaluation.
Single serial runs provide diagnostic timings, not statistically established
speedups or distributed wall-time claims. Central LM is substantially faster.
No concurrent experiment/build was launched during the final solve batch.

The accurate-but-expensive completion contract spends many late rounds at the
floating-point objective floor. Treat its 7.30/10.70 MB as a complete ledger,
not the necessary cost of an eventual inexact online algorithm. Do not silently
truncate these curves or declare online communication efficiency.

## Boundaries And Next Gate

- D9 at 120 s is explicitly rejected by the frozen bootstrap: chain overlaps
  are `0/0/3/1`, insufficient for this two-point neighbor registration. This is
  not proof of global network unobservability. An online implementation must
  retain separate components until supported, not fabricate a transform.
- Labels are supplied observed barcode identities; incorrect-identity recovery
  is not tested here. Joint consistency can propagate erroneous associations.
- The same-frame geometry discrepancies from the earlier audit remain raw-data
  facts. A successful joint solve does not identify sensor bias, noise variance,
  odometry mismatch or wrong identity, nor calibrate the fixed covariance.
- Fresh full-history relinearization and factorization are used. This is not
  incremental elimination, stale-message tolerance or deadline-bounded updating.
- No official distributed riMESA comparison at matched likelihood, initialization,
  communication and wall time is yet established.

Next: retain this frozen solver as an oracle, implement causal component merging
and finite joint updates with reusable local responses, then compare state age,
per-event latency and accuracy at equal communication. Only after resolving
state-reducible residuals should persistent independent predictive evidence drive
noise/identity adaptation. Preserve no-evidence/insufficient-support cases instead
of adding a fitted rejection threshold. Broader held-out transfer remains open.

## Reproduction And Verification

Source entry points: `experiments/prepare_native_joint.py`,
`experiments/native_local/joint.cpp`, `experiments/audit_native_joint.py`, and
`experiments/evaluate_native_joint.py`. The optional CMake target is
`native-joint-audit`; it shares the pinned GTSAM-native ABI, not legacy AVX DPGO.
All commands run inside `projects/seed_slam/DRANPPR` except root result paths.

```sh
cmake --build /tmp/dranppr_backend_build --target native-joint-audit testIncrementalLocalModel -j2
python3 -m experiments.audit_native_joint --binary /tmp/dranppr_backend_build/bin/native-joint-audit --case d4=/tmp/dranppr_frozen_joint_20260922/d4_final.json --case d9=/tmp/dranppr_frozen_joint_20260922/d9_final.json --output /tmp/dranppr_frozen_joint_repeat
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit python3 -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py tests/test_native_map_shape.py tests/test_native_causal_replay.py tests/test_native_replay_profile.py tests/test_native_interface_support.py tests/test_observation_pair_evidence.py tests/test_native_joint_prepare.py tests/test_native_joint_audit.py tests/test_evaluate_native_joint.py tests/test_evaluate_incremental_reference.py
```

35 native tests, 44 Python checks and six focused legacy ManualObjectDran tests
pass (85 total). Contracts cover first-step/full optimum parity, physical descent,
one gauge, global-frame equivariance, exact shared copies, topology-restricted
identities, malformed input, missing bootstrap and single-global evaluation.
CCI/D-CCI is unchanged; MPI rank count is 0 (not run).

Final binary SHA256:
`438820d12b292cb0fe43a3efb414bf82aca3ce4ed2499ad112febc323ce05349`.
Full final records and source snapshots:
`/tmp/dranppr_frozen_joint_20260922/final_solves/`.
Main evaluations: `d4_final_evaluation.json`, `d9_final_evaluation.json` in the
parent directory. Earlier pre-ledger-refinement runs are retained but not counted
as independent trials. Root archive: `results/dranppr_frozen_joint_20260922/`.
