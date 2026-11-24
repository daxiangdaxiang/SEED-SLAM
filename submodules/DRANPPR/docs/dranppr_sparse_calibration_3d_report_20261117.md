# Sparse Calibration and Full SE3 Controls

Date: 2026-11-17. Research DRANPPR only. Production DBA-DRANPP is unchanged.
This round changes calibration linear algebra and adds a controlled 3D test;
it does not implement a full 3D distributed online backend.

## What the Online Changes Mean

A fixed-graph solver receives the complete graph and iterates toward its
solution. The research replay instead inserts newly arrived states and
measurements into an existing estimator, performs limited work, and publishes
a state before later observations arrive. It has a common-past bootstrap and
retains history; it is not a bounded-memory, constant-work or deployed real-time
system.

The persistent local factor model stores factor identities, nonlinear factors,
linearization bases, whitened Jacobians and their local quadratic contributions.
Unchanged factors retain their cached contributions. Adding, replacing or
removing a factor updates its contribution rather than rebuilding the model
from the first measurement. This is persistent model reuse, not a claim of
incremental sparse factorization or exact historical marginalization.

The key earlier improvement separates **updating the state** from **rebuilding
historical linearizations**. New factors are inserted immediately. Existing
models and local responses are reused; a failed full-step physical-descent
test requests historical refresh for the next iteration. Existing backtracking
still checks physical loss. It does not blindly freeze all old information.
The October full-suffix test reduced historical relinearizations by 97.81% on
D4 and 96.43% on D9, with 3.24--3.52x service speedups. Those are historical
results, not newly measured speedups in this round. Full cost checks, recovery
and growing history remain; tail latency and finite-work tracking are open.

Noise calibration is a separate change. Write the local observation model as
`y = A delta + epsilon`, with `V(q) = diag(base + D q)`. A residual can be large
because the state is wrong, not because the sensor is noisy. REML eliminates
the linear state-response subspace, then fits variance components from the
remaining redundant observations. The likelihood includes its log determinant,
so increasing variances everywhere is not a free way to reduce the score.
The fitted scales replace relative physical weights, `Sigma_g = q_g Sigma_g0`,
not merely robust downweights for the largest residuals. This removes linear
state nuisance parameters, not nonlinear errors, unknown biases or outliers.

In the tested real-data path, five shared scales are estimated centrally from
one connected past prefix and then frozen for the suffix. The prefix retains
nominal weights. Continuous or decentralized noise learning is NOT implemented.

## Exact Sparse Formulation

For gauge-fixed full-column-rank A and diagonal positive V, the objective is

`2 L(q) = logdet V + logdet(A' V^-1 A) - logdet(A'A)`
`         + min_delta (y-A delta)' V^-1 (y-A delta) + (m-n) log(2 pi)`.

The subtraction matches the old orthonormal error-contrast normalization.
Sparse QR of `V^-1/2 A` evaluates the determinant and weighted solve without
constructing normal equations or a dense contrast covariance. Exact deterministic
blocks provide leverage, score and Fisher information. Chunk size changes
memory use, not the statistical model. Column equilibration changes numerical
units only. The original log bounds [-16,10], L-BFGS-B tolerances and variance
family remain unchanged; there is no added prior, variance floor or gate.

The first sparse normal-equation implementation failed D3's optimizer line
search. That failed attempt is preserved, not relabelled successful because
its parameter values were close. A weak full-rank design with condition number
about 1e6 reproduced likelihood error. Sparse square-root elimination passes
that regression without loosening convergence tolerances. A small triangular
solve residual alone would not have justified accepting the old implementation.

The optional dependency is pinned `sparseqr==1.6.0`, a wrapper for the installed
SuiteSparseQR library. Dense mode remains the default and does not import it.
See [PySPQR's primary documentation](https://github.com/yig/PySPQR) and
`experiments/requirements-sparse-calibration.txt`. No stochastic trace estimator
or alternate approximate likelihood was introduced.

## Real-Data Calibration Results

Same archived physical Jacobians, source checksums, optimizer settings and
600 s per-command cap; one BLAS/OMP thread. Times include process setup and
artifact writing. Old times are historical, not freshly paired timing trials.

| Dataset | Old dense command (s) | New sparse command (s) | Independent dense-QR check (s) |
| --- | ---: | ---: | ---: |
| D2 | 7.341 | 1.873 | 0.721 |
| D3 | 544.712 | 30.367 | 23.952 |
| D9 | timeout at 600.016 | 66.929 | 143.239 |

All three new fits and all three independent checks terminate successfully.
Maximum log-scale difference from archived fitted vectors is 3.04e-14;
maximum archived NLL difference is 1.37e-12. D9's independent dense-QR NLL
and score errors are 1.82e-12 and 1.17e-12. The new procedure therefore
changes implementation, not the fitted statistical objective on these inputs.
Independent validation overhead is separate from the calibration command.

**Correction to the previous D9 diagnosis:** the old sealed archive contains a
successful `fit.json` written before timeout, plus a 193,213,452-byte truncated
`contrast.npz`. `unzip -l` fails because its central directory is absent. The
old code writes JSON before `np.savez_compressed`; its final stdout log is empty.
Thus the old command timed out during export of dense intermediates after a
fit had already been computed. It is not evidence that REML could not converge.
The reported workflow improvement includes avoiding dense construction/storage
and compression; no stage-level measurement isolates an optimizer-only speedup.
The old comparison correctly rejected a timed-out producer even if partial
output existed. Its unavailable D9 candidate is not retroactively a success.

## Fresh D9 Online Comparison

The new successful fit feeds the unchanged native suffix solver. All native
and baseline binaries remain byte-identical to the prior study. One-sweep and
fresh per-frame minimum-native-arm byte-ceiling iMESA/riMESA controls are run,
rather than reusing a nominal-only historical credit sequence. All six runs
finish under the unchanged 600 s per-process cap; GT evaluation begins only
after all terminate. This is the same five-robot chain, known identities,
common-past primal initialization and cold solver internals as October.

| Method | Final trajectory (m) | Object copies (m) | First-published trajectory (m) | Suffix service (s) | MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native fixed | .401393 | .230121 | .507303 | 60.774 | 90.523 |
| Native calibrated | .386044 | .188604 | .608974 | 68.884 | 90.974 |
| Paper-iMESA / one sweep | .493795 | .418158 | 1.175964 | 30.224 | 9.780 |
| Paper-iMESA / byte ceiling | .494341 | .417851 | 1.179365 | 200.510 | 90.519 |
| Repaired riMESA / one sweep | 1.579328 | 2.590361 | 3.299326 | 154.778 | 9.780 |
| Repaired riMESA / byte ceiling | 1.227665 | 1.642529 | 3.127440 | 513.043 | 90.519 |

Calibrated native improves final trajectory/object error by 3.82%/18.04% but
WORSENS first-published error by 20.04%, despite beating the tested iMESA and
riMESA arms in those three metrics. Service grows 13.34% and bytes grow .498%.
The candidate does not dominate fixed-native, so it is not promoted as default.
With this new result all five inspected sequences improve final trajectory and
object error, but only D2/D3/D4 improve first-published trajectory; D1/D9 regress.
This is not an all-sequence online improvement claim.

Service excludes calibration (66.929 s), prefix preparation, trace I/O and
evaluation. Timings are single realizations, not an equal-time comparison.
The byte-ceiling baselines receive the smaller cumulative native-arm budget at
each frame, not exactly equal bytes; one-sweep iMESA uses much less traffic.
These baseline arms retain nominal noise, so the comparison is not an
optimizer-only intervention. riMESA/riSAM are the same pinned numerical-repair
versions disclosed in the October report, not untouched official binaries.
Raw costs under different noise/kernel models are not ranked.

All methods score the same 4104 final poses, omit the same 437 unsupported GT
poses, and score all 2514 first-published suffix poses and 73 object copies.
One global network registration is used for each error convention, not one
registration per robot. Re-running the original fixed-native and one-sweep
baselines preserves their archived final/causal state files byte-for-byte.
Physical weight and common-prefix-frame diagnostics are audited separately.

The causal regression persists with the SAME fixed prefix registration:
.759267 -> .911663 m. All six nonempty 30 s bins worsen under both pooled
causal and fixed-prefix conventions. Yet the maximum pooled-causal position
error falls 3.215937 -> 1.753129 m. Mean and worst-case performance therefore
tell different stories; neither should replace the other. These bins are
posthoc correlated descriptions. D9 has not received a same-arrived-graph
completion intervention, so D1's diagnosed finite-work cause must not simply
be assumed for D9. Model mismatch versus optimization lag remains to be tested.

## Full SE3 Calibration Controls

This is a physical GTSAM SE(3) graph, not a lifted 2D trajectory or a random
matrix: two robots with 12 poses each, four six-DoF object states, nonzero roll,
pitch, yaw and translation in all axes. One root pose is exactly fixed.
Known identities connect the robots through shared objects. Exact full-chain
between-factor derivatives are checked at nonzero residual against central
finite differences; maximum error is 8.47e-10 and root displacement is zero.

Seeds 31--40 are fixed in advance for each of two regimes. Training and test
measurement noises use independent RNG streams on the same geometry. Training
is nominal nonlinear LM followed by the same sparse REML. Test nominal,
calibrated and oracle arms use byte-identical measurements and initial states;
all 60 C++ runs finish. Truth generates synthetic measurements and evaluates
errors, never fits the noise. Initial states are common perturbed-truth warm
starts: this is a controlled experiment, not GT-free initialization evidence.
The solves are centralized batch controls, not distributed online competition.

Four physical variance groups are declared: odometry rotation/translation and
object-observation rotation/translation. Nominal base standard deviations are
.03 rad/.1 m and .05 rad/.15 m. Correct-model regime uses scales [1,1,1,1];
mismatched regime uses [.1,4,9,.25]. These are independent Gaussian noises,
not outliers, arbitrary within-group anisotropy or changing online noise.

Mean per-seed RMSE below uses one fixed root frame, not per-robot alignment.
These are ten paired noise realizations of one geometry, not ten datasets.

| Regime / metric | Nominal | Calibrated | Known-scale oracle | Calibrated wins |
| --- | ---: | ---: | ---: | ---: |
| Mismatched / pose translation (m) | .212694 | .095369 | .091678 | 10/10 |
| Mismatched / pose rotation (rad) | .098808 | .036302 | .033057 | 10/10 |
| Mismatched / object translation (m) | .146114 | .065637 | .064083 | 9/10 |
| Mismatched / object rotation (rad) | .090998 | .061518 | .061386 | 9/10 |
| Correct model / pose translation (m) | .146571 | .150326 | .146571 | 2/10 |
| Correct model / pose rotation (rad) | .038748 | .038810 | .038748 | 5/10 |
| Correct model / object translation (m) | .118241 | .121851 | .118241 | 3/10 |
| Correct model / object rotation (rad) | .032998 | .032397 | .032998 | 6/10 |

Under noise mismatch, mean pose translation/rotation improve 55.16%/63.26% and
object translation/rotation improve 55.08%/32.40%. The oracle is a comparison
using true variances, not a deterministic lower bound on every realization's
truth error. Dense-contrast NLL/score parity errors stay below 2.28e-13/1.65e-12.

**Do not omit the failures.** When nominal noise is already correct, mean pose
and object translation worsen by 2.56% and 3.05%, about 3.75 and 3.61 mm. This
is consistent with variance-estimation uncertainty, not a proof of its sole
cause. In mismatched seeds 35 and 38, odometry rotation variance reaches the
existing lower log bound -16, despite a positive generating scale .1. Thus
full component rank, numerical success and improved state error do not certify
an accurate or safe uncertainty estimate. No floor, reject gate or tuned prior
was added to hide these cases. Nonlinear fixed-chart effects also remain.

## Decision and Remaining Work

1. Retain sparse square-root calibration as an opt-in implementation of the
   same likelihood. It removes a workflow bottleneck, not a statistical flaw.
2. Genuine 3D controls support usefulness under the declared noise mismatch,
   but disprove universal no-regression claims. Next distinguish weak finite
   noise evidence from fixed-chart bias before adding parameters or heuristics.
3. The persistent native interface still assumes Pose2/Point2. Migrating the
   online graph, retractions, six-DoF object interfaces, consensus/transport and
   physical-descent checks requires separate implementation and parity tests.
   The existence of an ObjectPose3 offline backend is not that migration.
4. Indoor_10_robots is a future real-3D validation target, not a result here.
   Its mixed measured/pseudo-reference provenance must be reported. Full 3D
   streaming tests must retain clean-noise controls as well as noisy ones.
5. D1 causal degradation and the older high-Gaussian-noise D2 negative remain
   open. More accurate likelihood evaluation does not repair finite-work
   tracking, misspecified means, time correlation or false associations.

## Reproduction and Evidence

```sh
env CFLAGS=-I/usr/include/suitesparse python3 -m pip install --user --no-deps sparseqr==1.6.0
cmake -S . -B build/native -DBUILD_DPGO_NATIVE_MODEL=ON -DBUILD_DPGO_TESTS=ON -DGTSAM_DIR="$PWD/build/research-deps/gtsam-imesa-build" -DGTSAM_INCLUDE_DIR="$PWD/build/research-deps/gtsam-imesa"
cmake --build build/native --target native-pose3-calibration-audit -j2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_sparse_calibration real --source results/dranppr_extended_calibration_20261030 --output NEW_CALIBRATION
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_sparse_calibration pose3 --output NEW_POSE3
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_online_calibration_baselines run --source NEW_CALIBRATION --datasets 9 --output NEW_ONLINE
python3 -m experiments.audit_online_calibration_baselines evaluate --output NEW_ONLINE --raw /home/jackal/zyx/dpgo_ICRA/datasets/MRCLAM/raw
python3 -m experiments.analyze_online_calibration_baselines --output NEW_ONLINE --raw /home/jackal/zyx/dpgo_ICRA/datasets/MRCLAM/raw
python3 -m experiments.analyze_pose3_calibration --source NEW_POSE3 --output NEW_REVIEW/pose3_summary.json
```

Archives: `dranppr_sparse_calibration_formal_20261117` (negative normal-equation
attempt), `dranppr_sparse_sqrt_calibration_20261117` (final calibration),
`dranppr_pose3_calibration_20261117` (all SE3 controls),
`dranppr_sparse_d9_online_20261117` (fresh online comparison), and
`dranppr_sparse_calibration_review_20261117` (analysis, final source/tests).
Each lives under research `results/`. Producer snapshots remain immutable;
final review sources record later provenance/report refinements separately.

Verification: the 25-file Python selection passes 220 tests, and the rebuilt
`testIncrementalLocalModel` passes all 62 C++ tests. Exact commands are saved in
the review archive's `verification_commands.txt`, with JUnit XML for both runs.
The weak-design likelihood test uses zero relative tolerance so its absolute
tolerance cannot be weakened by the assertion library's default relative term.
SE3 factor finite differences and exact input equality are included.

Existing Schur full-system residual is 3.24647e-16 and matrix-free product error
3.53484e-17. The small routed-PCG test uses six iterations with residual
1.97875e-16; 2/3/5/10-robot tests use 12/16/23/39 iterations. MPI ranks: zero
(single-process routed communication tests, no MPI run). Chordal/CCI/D-CCI
behavior is not modified. Six historical state files reproduce exactly, and
independent terminal physical-cost reconstruction errors are 0 and 1.79e-16
relative. Production git status remains clean. No commit or push is made.
