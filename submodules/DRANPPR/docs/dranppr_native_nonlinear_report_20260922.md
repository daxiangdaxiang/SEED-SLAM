# Native Nonlinear Response and Jacobian Audit

Date: 2026-09-22. This report advances the approved native-local design through
nonlinear gate 5. It does not establish online latency, distributed convergence,
robust superiority, or a new statistical method.

## Question and Controls

Does the persistent native model plus private/interface elimination actually
decrease the physical observation objective and reproduce a batch solution?
Does the previously observed approximate-Jacobian discrepancy matter on real data?

Thirty final runs: MRCLAM D1/D4/D9, robot 5, two archived starts (iSAM2 and riSAM),
five solver/derivative controls each:

- `pinned_schur`: native GTSAM approximate pose Jacobians and our Schur response.
- `exact_schur`: same likelihood, full pose-factor chain rule and Schur response.
- `exact_full`: same complete derivatives, all free variables eliminated together.
- `pinned_lm`: batch GTSAM LM with the pinned approximate derivatives.
- `exact_lm`: batch GTSAM LM with the full pose-factor chain rule.

D1/D4 use the existing **120-second** checkpoints; D9 uses **300 seconds**. The
initial plan incorrectly described all saved checkpoints as 300 seconds. The D1
300-second attempt was rejected before optimization because prefix/state keys
disagreed; its failure log remains at `/tmp/dranppr_native_nonlinear_d1_v1_20260922`.
The horizon correction matches the archived inputs, not an accuracy-driven choice.

All arms retain exactly the same raw polar observations, declared identities,
odometry, Gaussian covariance and initial state within a case. Bearing/range
sigmas are `(0.05 rad, 0.1 m)`; odometry sigmas are `(0.1, 0.1, 0.05)*sqrt(dt)`.
The first local pose is fixed exactly. No extra statistical prior, robust kernel,
variance floor, observation deletion or GT state enters the solvers. These are
independent local problems, not multiple roots fixed in a joint network frame.

All methods have a 100-outer-step cap. The native correctness loop halves the
step at most 30 times and accepts only finite physical loss decrease with positive
quadratic-model decrease. Its model-gradient tolerance is `1e-8`. LM uses
absolute/relative objective tolerances `1e-12`. These numerical stopping rules
are not identical computational budgets, so runtimes are diagnostic and no
equal-wall-time/communication superiority is claimed.

## Code Change

`NativeBetweenPose2Factor` preserves `BetweenFactor<Pose2>`'s residual and noise,
but composes GTSAM's existing `Between` and `Local` Jacobians completely.
No upstream macro, installed library or previous baseline is modified. It is
opt-in, retaining both derivative arms for audit.

`NativeLocalOptimizer` is a correctness-first Gauss-Newton/backtracking loop
around the persistent model. It uses a fresh sparse private factorization and a
small **dense interface solve**. Production `NativeSchurModel::apply` is still
matrix-free, but this diagnostic optimizer is not the distributed interface
solver and is not claimed as completed incremental inference.

The frozen-prefix executable and Python runner preserve manifests, source and
binary hashes, dependency hashes, per-step traces and final states. Evaluation
opens GT only after the estimator processes exit. D9 evaluation excludes and
reports 29 unsupported pose states for this robot/window; no GT extrapolation.

## Key Result: Model Stationarity Can Be False

For D9 initialized from riSAM, the pinned Schur loop returns
`model_gradient_small` with:

| Quantity | Value |
| --- | ---: |
| Approximate model gradient norm | 8.0100e-9 |
| Complete physical-objective gradient norm | 135.9929969 |
| Direct central finite difference along its unit gradient | 135.9929959 |
| Physical loss at this point | 1305.0824829 |

The finite difference uses the actual nonlinear graph error at retracted states,
not the linear model; step size is `1e-6`. It independently confirms a substantial
remaining descent direction. The pinned batch LM control reaches essentially
the same nonstationary point (loss `1305.0824851`, true gradient `135.9930225`).
Therefore this is not caused by our Schur elimination or insufficient PCG work.

With the complete chain rule, the Schur loss is `1275.7838074` and the physical
gradient norm is `1.106e-5`. Exact batch LM reaches the same loss within `6.96e-10`.
The same phenomenon is smaller but present on D1/D4. For example, D1 from riSAM
has pinned model gradient `2.236e-9` while its physical gradient is `1.8912`.

For residual Jacobian approximation `J_tilde`, this distinction is:

```text
g_model = J_tilde^T W r,
g_physical - g_model = (J - J_tilde)^T W r.
```

At nonzero residual these need not vanish together. A residual should not be
classified as irreducible sensor noise merely because an approximate model's
gradient is small. This is an established differentiation issue, not a proposed
new robustness principle or proof that all residual causes are identifiable.

## Physical Loss and State Quality

The table compares pinned versus complete Jacobians within the same native
Schur loop, not different robust objectives. RMSE is the established local
trajectory evaluation after 30 seconds; object error is object-copy RMSE.

| Data/start | Loss, pinned -> exact | Trajectory RMSE m, pinned -> exact | Object RMSE m, pinned -> exact |
| --- | --- | --- | --- |
| D1 / iSAM2 | 220.0041615 -> 219.9971706 | 0.349575 -> 0.349689 | 1.089724 -> 1.074274 |
| D1 / riSAM | 220.0041679 -> 219.9971706 | 0.349580 -> 0.349689 | 1.089738 -> 1.074274 |
| D4 / iSAM2 | 245.6811893 -> 245.6789785 | 0.236923 -> 0.236810 | 0.482787 -> 0.482594 |
| D4 / riSAM | 245.6811857 -> 245.6789785 | 0.236899 -> 0.236810 | 0.482767 -> 0.482594 |
| D9 / iSAM2 | 734.3871310 -> 733.6961720 | 0.469683 -> 0.458773 | 0.335869 -> 0.323750 |
| D9 / riSAM | 1305.0824829 -> 1275.7838074 | 3.039018 -> 3.090053 | 4.397934 -> 4.283176 |

Thus physical optimization improves in all six cases; trajectory accuracy does
not. D9's bad start remains in a much worse basin than the iSAM2 start even with
complete derivatives. Its trajectory RMSE worsens while object RMSE improves.
D1's small trajectory degradation is retained, not rounded into a positive win.
These results neither remove the need for robust modeling nor show superiority
over riSAM/riMESA as end-to-end algorithms. The riSAM state is an initial condition
here; all completions optimize a fixed Gaussian objective.

## Elimination, Nonlinear Checks and Cost

- All 342 accepted native steps in the final runs strictly decrease physical loss.
- Across six cases, maximum exact-Schur/exact-full final loss difference is
  `4.55e-13`, maximum per-state translation difference `8.56e-8 m`.
- Maximum relative full linear-system residual across native traces is `3.06e-12`.
- Exact Schur and exact LM differ by at most `1.14e-5 m` per state in the hardest
  case. This is final-state agreement, not a global optimum certificate.
- Exact Schur final physical gradients range from `8.61e-8` to `1.11e-5`.
  The loop reports `no_decreasing_trial` when floating-point objective decrease
  cannot be found. These statuses were not relabeled as certified convergence.
- The native loop is slower than exact batch LM in these local checks. D9 from
  riSAM takes approximately `0.926 s` versus `0.528 s`; D1 from iSAM2 takes
  `0.091 s` versus `0.014 s`. Snapshot copying, cache publication and repeated
  factorization still need work. No online speedup follows from these timings.

Verification: **22 C++ native tests, 8 Python audit/input tests, and 6 legacy
DRAN regressions pass**. Tests include full-chain finite differences, exact gauge,
small-graph Schur/full/LM agreement and unchanged state after every backtracking
trial fails. No MPI tests ran (0 ranks), and the new local solves use LDLT rather
than PCG. Legacy owner-PCG tests pass without printing iteration counts.

## Artifacts and Reproduction

Final datasets with finite-difference diagnostics:

- `/tmp/dranppr_native_nonlinear_d1_v3_20260922`
- `/tmp/dranppr_native_nonlinear_d4_v2_20260922`
- `/tmp/dranppr_native_nonlinear_d9_v2_20260922`
- `/tmp/dranppr_native_nonlinear_20260922/unit.xml` and `legacy.xml`

Earlier completed runs without the final finite-difference fields and the rejected
prefix attempt are retained. They are not counted as independent data/seed trials.
The pinned GTSAM commit remains `03c263fe08b02c90a8861d36f77422765d39506d`.
All native binaries link the pinned no-TBB library with compatible Eigen flags.
The packaged `projects/seed_slam/DBA-DRANPP` repository remains clean.

From the DRANPPR worktree, after the previous native CMake configuration:

```sh
cmake --build /tmp/dranppr_backend_build --target native-local-audit testIncrementalLocalModel -j2
ctest --test-dir /tmp/dranppr_backend_build -R native_local --output-on-failure
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit \
  python -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/audit_native_local.py \
  --binary /tmp/dranppr_backend_build/bin/native-local-audit \
  --stream /tmp/dranppr_causal_streams_v1/dataset9 --duration 300 \
  --start isam2=/tmp/dranppr_native_risam_dataset9_300_20260922/dataset9_isam2/final_estimates.csv \
  --start risam=/tmp/dranppr_native_risam_dataset9_300_20260922/dataset9_risam/final_estimates.csv \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --output /tmp/dranppr_native_nonlinear_d9_new_run
```

D1/D4 use duration 120 and starts from
`/tmp/dranppr_native_risam_serial_v2_20260922/datasetN_METHOD/final_estimates.csv`.
Every manifest records the exact invocation. Output directories must be new.

## Decision and Next Work

Use complete pose-factor derivatives for the native research path and offer the
same correction to comparison controls. It is a correctness baseline, not a
DRAN-only innovation. Keep the old pinned arm to expose the failure mechanism.

Next integrate chronological, timestamp-atomic arrival with bounded solve budgets
and explicit timing for insertion, relinearization, factorization and recovery.
First determine whether work is reusable in the native implementation before
introducing a history approximation. Distributed gauge/coordination, exact accepted
identity equalities, online backlog and evidence-aware noise adaptation remain
open. The D1 covariance-identifiability failure is not solved by a correct Jacobian.
