# DRANPPR Research Entry Points

This worktree is the adaptive-noise/online Pose2-Point2 research line. It is not
the production DBA-DRAN++ repository. No continuously distributed noise learner
or high-noise robustness claim has yet been established.

## Current Validation Path

Latest horizon-matched error/update experiment:
[`docs/dranppr_horizon_error_budget_report_20261124.md`](../docs/dranppr_horizon_error_budget_report_20261124.md).
This separates first-order prediction from error-aware update strength. The
non-predictive adaptive-update ablation is a primary control. Long-delay
counterexamples reject promoting the predictive empirical-error rule as a
robust default; historical errors do not guarantee future error bounds.

Previous low-order/error-feedback validation:
[`docs/dranppr_response_prediction_report_20261124.md`](../docs/dranppr_response_prediction_report_20261124.md).
Packet-driven response prediction reduces overhead versus multi-order fitting,
but scalar error calibration does not improve overall optimization versus its
linear ablation. D1 remains negative, long outages expose further regressions,
and measured same-accuracy Python runtime is still worse than hold. An
exact-neighbor oracle identifies unchanged delay damping as another limit.
Do not promote this prototype to the production default.

Previous neighbor-prediction study:
[`docs/dranppr_history_prediction_report_20261123.md`](../docs/dranppr_history_prediction_report_20261123.md).
History-based forecast-risk selection improves 93/135 fixed-model conditions
over hold, ties 27 and loses 15, with identical prefix bytes/updates. It does
not dominate simpler predictors, costs more Python compute, and is not native
nonlinear online SLAM. Keep it as research, not a production default.

Latest communication study:
[`docs/dranppr_async_interface_report_20261123.md`](../docs/dranppr_async_interface_report_20261123.md).
The fixed-model directional delay metric improves 102/135 matched conditions
over scalar delay damping but loses nine and does not dominate synchronous or
unguarded asynchronous controls. This is not a nonlinear online default.

Latest results and remaining failures:
[`docs/dranppr_subspace_milestone_report_20261123.md`](../docs/dranppr_subspace_milestone_report_20261123.md).
The fused one-step protocol saves 22.78--27.13% communication with identical
state outputs in ten conditions; the collective-motion candidate has mixed
accuracy and is not recommended as a default. Neither fixes noise robustness.
Earlier tracking and finite-evidence diagnosis:
[`docs/dranppr_tracking_evidence_report_20261117.md`](../docs/dranppr_tracking_evidence_report_20261117.md).
The [sparse calibration/SE3 report](../docs/dranppr_sparse_calibration_3d_report_20261117.md)
contains the latest D9 baseline comparison and nonlinear SE3 controls.
The [October expanded comparison](../docs/dranppr_expanded_online_report_20261030.md)
retains the five-sequence context and the D1 finite-work failure.

1. `prepare_extended_calibration.py`: D2/D3/D9 archived common-past inputs and
   unchanged first-prefix five-component REML. Fits never load ground truth.
2. `audit_online_calibration_baselines.py`: native fixed/calibrated replay,
   pinned repaired riMESA and paper-iMESA; one-sweep and causal byte ceilings.
   `run` and `evaluate` are separate to keep GT out of estimation.
3. `audit_checkpoint_tracking.py`: frozen-arrived-graph LM/GN completion to
   distinguish finite-work tracking from information/model limitations; D1
   defaults preserved, D9 accepts explicit checkpoints. `geometry` describes
   collective versus deformation displacement without changing any estimates.
4. `audit_rolling_prediction.py`: complete declared 120-second D1/D4 predictive
   coverage; `analyze_rolling_prediction.py` independently checks inputs/scores.
5. `audit_noise_parameter_uncertainty.py`: bounded six-case diagnostic, NOT a
   production uncertainty module or replacement for the online comparison.
6. `audit_sparse_calibration.py`: exact sparse square-root REML parity on
   archived D2/D3/D9 inputs, and full SE3 pose/object synthetic controls.
   `analyze_pose3_calibration.py` includes both nominal-noise regressions and
   mismatched-noise gains. Install the optional pinned dependency from
   `requirements-sparse-calibration.txt`; dense mode remains unchanged.
   The SE3 diagnostic is centralized batch, NOT a 3D online backend.
7. `audit_pose3_evidence.py`: exact-linear independent-evidence controls on a
   physical SE3 Jacobian; test support fixed across 1x/4x calibration evidence.
   Known variances are explicit oracle-generative diagnostics, not estimator input.
8. `audit_rigid_subspace.py`: paired opt-in rigid/fused one-step native runs;
   separate `evaluate` and `fixed_frame` modes. `summarize_subspace.py` verifies
   exact fused state and physical-trace parity against fresh and archived controls.
9. `audit_async_interface.py`: immutable physical Schur models, actor-local
   updates and routed delay/loss calendars. `async_interface_model.py` supplies
   setup-only elimination; `async_interface.py` supplies the event simulation.
   `review_async_interface.py` checks the sealed archives, matched prefixes and
   the directional history-energy bound. Pre-distributed rows, known delay
   bounds and uncongested links are explicit restrictions.
10. `audit_neighbor_prediction.py`: received-history prediction with unchanged
    directional steps; `neighbor_prediction.py` contains the predictor and
    `review_history_prediction.py` checks independent metrics, budgets and
    inexact-gradient algebra. `audit_history_forecast.py` isolates noisy
    derivative estimation and unobservable post-message changes.
11. `review_response_prediction.py`: compares packet-driven linear and
    error-calibrated response modes against sealed legacy predictors.
    `audit_response_controls.py` checks noise, change recovery, long delays,
    explicitly nondeployable exact-neighbor oracles, and shuffled-order timing.
    `--target-only-archive` measures replay to a common accuracy target.
12. `prediction_error_budget.py`: causal exact-horizon error memory and a
    norm-penalized local step. `audit_error_budget.py` provides five matched
    arms, stress, memory and initial-state controls. `review_error_budget.py`
    reconstructs updates and distinguishes moving-step error coverage from
    zero-step coverage. `benchmark_error_budget.py` measures prefix replay to
    a baseline-defined accuracy, with offline oracle stopping explicitly labeled.
13. `quadratic_response.py`: optional frozen-boundary model response with
    packet-driven forcing correction. `audit_model_response.py` charges gradient
    and descriptor payloads; `review_model_response.py` independently replays
    traffic and checks legacy parity. This is a rejected release candidate:
    stable inspected stress runs, but 0 wins/2 ties/26 losses against adaptive
    hold at equal bytes. See the
    [model-response report](../docs/dranppr_model_response_feedback_report_20261124.md).

Core implementations are `native_local/joint_replay.cpp`, `joint.cpp`,
`joint_step.h`, `src/Native*`, and `src/Incremental*`. The current comparison
adapter is `native_matched/`. Statistical helpers are `network_noise_support.py`,
`network_noise_prediction.py`, `scripts/dranppr_variance_components.py`, and
its opt-in `scripts/dranppr_sparse_variance_components.py` implementation.
Use the combined `rimesa_endpoint_merit_diagnostic.patch` for the pinned riSAM
dependency. The endpoint-only predecessor is historical evidence, not another
supported baseline configuration.

## Historical Diagnostics

Other `audit_*` / `analyze_*` scripts are retained where they implement a
referenced diagnostic or supply shared helpers. In particular, rejected mean
corrections, synthetic stress cases, numerical probes and their regression tests
are not silently deleted because their algorithmic results are negative.

`dba/` belongs to the inherited DBA snapshot and is not this research branch's
active experiment path. It has not been modified or purged by the online study.
See `docs/cleanup_20261030.md` for the precise cleanup boundary.
