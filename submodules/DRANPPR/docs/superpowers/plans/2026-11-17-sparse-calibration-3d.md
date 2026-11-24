# Sparse Calibration and 3D Validation Plan

> Execute inline, continuing the approved sparse-equivalent calibration direction.
> Preserve the existing research worktree and all archives. No agents, commits,
> pushes or production changes. Use executing-plans and test-driven-development.

**Goal:** Remove the demonstrated dense calibration bottleneck without changing
the likelihood, and test the SAME mathematical implementation on genuine SE(3)
pose/object graphs before claiming any three-dimensional applicability.

**Architecture:** For full-column-rank, gauge-fixed A and diagonal V, evaluate
REML using sparse QR of V^-1/2 A, a weighted least-squares residual and log
determinants. Subtract logdet(A'A), also computed by QR, to match contrasts exactly.
Compute score and expected Fisher by deterministic blocked solves, not random
trace estimates. No variance floor, damping, new statistical parameter or
changed optimizer bounds. Dense contrasts remain the independent reference.

**Tech Stack:** NumPy/SciPy, SuiteSparseQR via sparseqr 1.6.0, pinned C++17
GTSAM, pytest. SuperLU factors the triangular R only, not normal equations.
Column equilibration addresses numerical units, not statistical weights. Reject
singular designs; do not silently anchor or regularize an unidentified graph.

## Tasks

- [x] Add failing `tests/test_sparse_variance_components.py`: dense/sparse NLL,
  analytic score/Fisher, state-shift and column-scale invariance, single-variance
  fit, invalid/rank-deficient designs and indistinguishable noise components.
- [x] Add `scripts/dranppr_sparse_variance_components.py` implementing the same
  `nll`, `gradient`, `fisher`, `fit` contract. Block size is a memory chunk only;
  test that it cannot alter the statistical result. No modification of old fits.
- [x] Add a sparse mode to the existing physical factor assembly and an opt-in
  sparse fit arm. Verify the assembly is exactly equal to the dense reference.
- [x] Check archived D2/D3 likelihoods, gradients and scales against dense
  references. Attempt D9 under the SAME 600 s per-process cap, retaining failure.
- [x] Complete D9 calibrated replay and rerun byte-ceiling baselines
  under the common native-arm ceiling; no hidden nominal-only credit substitution.
- [x] Add a small GTSAM SE(3) object-pose calibration diagnostic: exact full-chain
  pose derivatives checked by finite differences, one exact root gauge, 3D
  rotations/translations, separate training/test measurements and identical
  initial states for fixed/calibrated/oracle controls. Use four declared isotropic
  physical groups (odometry rotation/translation, object rotation/translation).
  Synthetic ground truth generates measurements and scores states, never fits
  noise. Do not describe a dimension-generic random matrix as a 3D SLAM test.
- [x] Test dense/sparse equivalence on these physical 3D Jacobians and report
  calibration/state quality over a fixed seed set, retaining neutral/negative
  results. This is NOT a complete 3D online backend: current NativeInterface is
  Point2-specific, and real ObjectPose3/online/association validation remains open.
- [x] Run focused regressions, preserve hashes/source/commands and write a
  report distinguishing exact algebra, numerical approximation, speed and
  statistical effectiveness. Keep finite-work tracking as a separate next step,
  not a second simultaneous solver change that confounds this experiment.

## Numerical Formulation

With D_k = diag(component k), P = V^-1 - V^-1 A H^-1 A' V^-1,
the log-scale score is q_k/2 [tr(P D_k) - (P y)' D_k (P y)],
and expected Fisher is q_k q_l/2 tr(P D_k P D_l).
Blocks of columns of P supply these exactly without retaining a dense P or
contrast covariance. H above is a mathematical identity, not an explicitly
formed matrix in the final implementation.

## Observed Numerical Revision

The first sparse normal-equation attempt passed D2/D9 but failed D3 L-BFGS-B
line search. It is retained in `dranppr_sparse_calibration_formal_20261117`, not
silently accepted because its fitted scales looked close. A weak-design
regression exposed likelihood cancellation; the same test passes with sparse
square-root QR. No optimizer tolerance or model parameter was changed.

Final fits finish in 1.873/30.367/66.929 s on D2/D3/D9. Independent dense QR
checks pass; these are numerical equivalence results, not fresh paired timing
trials against the October dense measurements. The SE3 controls also expose
nominal-noise regressions and two variance-boundary fits. Full 3D online
migration remains an explicit next phase, not a completed task in this plan.

Archive inspection corrects the old D9 failure attribution: `fit.json` already
contains a successful fit, but `contrast.npz` is truncated and lacks its ZIP
directory. The writer saves fit JSON before compressing contrasts. Thus the old
600 s failure is of the complete command, during artifact export, not proof
that REML failed to converge. Preserve its failed status in the original online
comparison. New elapsed times include removal of that unnecessary export; do
not describe the ratio as an isolated optimizer speedup.

All six D9 replays complete. Final trajectory/objects improve, but causal RMSE
worsens by 20.04%, also under fixed-prefix registration. No default promotion.
220 Python and 62 C++ regressions pass. Full 3D online migration, statistical
safety and the D1/D9 tracking/model question remain open beyond this round.
