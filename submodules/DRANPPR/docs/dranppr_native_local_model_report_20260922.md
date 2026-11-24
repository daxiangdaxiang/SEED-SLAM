# DRANPPR Native Local Model: First Implementation

Date: 2026-09-22. Scope: approved design gates 1--4, not a completed online or
robust distributed optimizer. The packaged DBA-DRANPP repository is unchanged.

## Purpose

Test whether a persistent local model can retain the native Pose2/Point2
bearing/range likelihood, update only affected factor contributions, and expose
DRAN-style private/interface responses without rebuilding an isotropic chordal
pseudo-observation model or replacing DRAN by iSAM2.

This is enabling infrastructure. It adds no noise adaptation, robust kernel,
association gate, statistical rejection threshold, or tunable variance floor.

## Implementation

- `IncrementalLocalModel`: stable variable keys and factor IDs; factor cloning;
  transactional append, removal, replacement and declared-base relinearization.
  Every factor incident to a changed base is invalidated, including old factors.
- Cached keyed contributions use native whitened Jacobians: `H = A^T A`,
  `g = -A^T b`, `c = 0.5 b^T b`, where the native residual is `A delta - b`.
  Removing/replacing a factor subtracts the old contribution exactly once.
- `linearSystem`: sparse export, explicit key ordering and exact zero-increment
  fixed keys. Appending a pose does not move existing object offsets in the
  default stable order. Unused variables remain visible with zero information.
- `NativeSchurModel`: immutable revision, sparse private LDLT, matrix-free
  `S v = H_BB v - H_BP solve(H_PP, H_PB v)`, reduced gradient and recovery.
  No dense reduced matrix is formed by production code. Tests form it as an
  independent oracle. Numerically singular private systems are rejected, not
  repaired by an artificial prior. A valid private solve does not establish
  that the remaining interface is observable.
- `BUILD_DPGO_NATIVE_MODEL=OFF` by default. Only the optional native target and
  its test executable link GTSAM. Existing DPGO/DRAN behavior is unchanged.

The code uses native factors/Jacobians, not GTSAM's iSAM2 optimizer. Source
provenance beyond the stable caller-supplied factor ID still belongs to the
stream adapter; it is not a newly implemented metadata store.

## Setting and Reproduction

Research worktree: `projects/seed_slam/DRANPPR`, branch
`research/dranppr-evidence-online`, base `104a200`.
Pinned GTSAM commit: `03c263fe08b02c90a8861d36f77422765d39506d`.
Native build: system Eigen 3.3.7, GCC 11.5, Release, pinned no-TBB GTSAM.

From the workspace root:

```sh
cmake -S projects/seed_slam/DRANPPR -B /tmp/dranppr_backend_build \
  -DBUILD_DPGO_TESTS=ON -DBUILD_DPGO_NATIVE_MODEL=ON \
  -DGTSAM_DIR=/tmp/dranpp_noise_audit/gtsam-paper-build \
  -DGTSAM_INCLUDE_DIR=/tmp/dranpp_noise_audit/gtsam-paper
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel testDPGO -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel \
  --gtest_output=xml:/tmp/dranppr_native_local_20260922/tests.xml
ctest --test-dir /tmp/dranppr_backend_build -R native_local --output-on-failure
ctest --test-dir /tmp/dranppr_backend_build -R native_local --repeat until-fail:20 --output-on-failure
/tmp/dranppr_backend_build/bin/testDPGO \
  --gtest_filter='testDPGO.ManualObjectDranPoint2SchurMatchesDirectElimination:testDPGO.ManualObjectDranPoint2MatrixFreeProductMatchesDenseSchur:testDPGO.ManualObjectDranBlockTridiagonalBlocksMatchDenseLdlt:testDPGO.ManualObjectDranSparseCouplingAggregatesAndMatchesDense:testDPGO.ManualObjectDranPoint2OwnerPcgMatchesDenseFirstStepAndLedger:testDPGO.ManualObjectDranRoutedOwnerPcgMatchesDirectNumerically' \
  --gtest_output=xml:/tmp/dranppr_native_local_20260922/legacy_tests.xml
```

The tests are deterministic small-graph correctness checks, not dataset accuracy
or latency benchmarks. Builds and raw XML are under `/tmp` because `/home` is
nearly full. No MPI run was performed (0 ranks); the new private solve is LDLT,
not PCG. The legacy owner-PCG tests pass but do not print iteration counts.
No CCI/D-CCI implementation or initialization weighting was modified.

## Results

All 17 new tests pass: 12 model lifecycle/linearization tests and 5 Schur tests.
The 6 existing DRAN Schur, matrix-free, sparse/block solve and owner-routing
regressions also pass. Twenty separate CTest test processes exit normally;
these repeats test process stability, not statistical performance variation.

| Check | Measured relative error or residual |
| --- | ---: |
| Initial cached/fresh Hessian | 2.79e-17 |
| Initial cached/fresh gradient | 8.07e-17 |
| Final Hessian after 40 covariance/base edits | 1.35e-16 |
| Final gradient after those edits | 1.57e-16 |
| Recovered native Schur step versus full direct step | 1.13e-16 |
| Full tangent normal-equation residual | 3.25e-16 |
| Private recovery residual | 1.44e-16 |
| Matrix-free product versus explicit dense elimination | 3.53e-17 |

Every one of the 40 edit revisions is checked at 1e-10 relative H/g tolerance;
the table records the final revision, not an asserted maximum over all edits.
Tests also check the quadratic constant and a finite-difference derivative.
Appending one odometry factor linearizes 1 factor and reuses 5. Changing X1's
base linearizes its 3 incident factors and reuses the other 2. These counters
establish factor-cache reuse, not a total-runtime complexity claim.

Verified rejection/rollback cases include duplicate IDs, unknown keys/removals,
type changes, nonfinite states/linearizations, null factors/noise, nonpositive
noise scales, robust or constrained noise, malformed ordering and singular
private models. Native factors returned for inspection cannot mutate the stored
model through their keys. Exact fixed keys are tested without an anchor prior
in the native Schur fixture.

## Failed Checks and Their Interpretation

1. Initial native construction aborted with `double free or corruption (out)`.
   GDB located it in GTSAM Gaussian covariance construction, before model update.
   The repository globally enables `-march=native`; the pinned GTSAM does not.
   In this Eigen 3.3.7 build, setting `EIGEN_MAX_ALIGN_BYTES=16` alone did not
   fix allocation ABI and caused a static-initializer abort. Local header
   inspection showed `EIGEN_DEFAULT_ALIGN_BYTES` still follows the larger
   architecture alignment. Removing the inherited `-march=native` only from
   the two optional native targets resolved the observed failures. Default
   targets keep their original flags. Native consumers must match their GTSAM
   build's Eigen/compiler ABI; arbitrary dependency builds are not certified.
   `ldd` confirms the native executable uses only the pinned GTSAM and no TBB;
   legacy `libDPGO.so` has no GTSAM/TBB dependency. This is distinct from the
   earlier Python/TBB shutdown problem.

2. The first fresh oracle included an unused landmark in GTSAM's Jacobian
   ordering after deleting its final factors. GTSAM explicitly rejects such an
   ordering. The independent oracle now linearizes active factor keys and pads
   unused state columns with exact zeros, matching the declared model contract.

3. The first nonlinear directional-gradient assertion failed: native linear
   gradient sum `93.391492029655` versus physical finite difference
   `91.030645785750`, gap `2.360846243905` in the nonzero-residual fixture.
   Cache/fresh parity already held. Pinned `BetweenFactor.h` omits the local-chart
   chain Jacobian unless `GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR` is enabled;
   `PriorFactor` also uses an identity Jacobian approximation. We did not change
   the pinned dependency or silently enable a different baseline derivative.
   The gradient contract explicitly refers to the cached native quadratic.
   Its sign is checked by finite-differencing that quadratic; a separate native
   bearing/range-only test passes finite differences of the true nonlinear
   objective at 1e-5 absolute tolerance. The full-fixture discrepancy remains
   recorded in the XML, not reclassified as exact nonlinear differentiation.

## Limits and Next Experiment

This implementation passes the local design gates 1--4 only. It does not pass
nonlinear replay, online scaling or distributed deployment gates.

- Atomic rollback currently copies O(graph) metadata, values and keyed normal
  blocks. Only changed factor linearizations/contributions are recomputed.
- Each new Schur snapshot uses a fresh private factorization. Incremental
  elimination/factorization and bounded per-event work are not implemented.
- No native nonlinear step acceptance, chronological backend replay, network
  gauge initialization, message transport or covariance adaptation is connected.
- Before linking this component and legacy DRAN into one executable, use one
  compatible Eigen/compiler ABI throughout that process. Separately passing
  native and legacy test executables does not certify mixed-library linking.
- No claim of ATE improvement, riMESA superiority, convergence or new statistical
  identifiability follows from linear parity. The earlier D1 variance-adaptation
  failure is not solved by this infrastructure.

Next: use the same physical likelihood to test finite nonlinear local steps and
small-graph batch agreement. Track actual objective change separately from
the cached-model gradient and diagnose the native Jacobian approximation before
claiming nonlinear stationarity. Then connect timestamp-atomic causal replay and
measure insertion, relinearization, factorization, recovery and backlog separately.
Only afterward assess distributed interface integration and evidence adaptation.
