# MR.CLAM Native Multi-Robot Object Benchmark

This benchmark enforces the project's strict native-team rule: all trajectories
must come from distinct physical robots operating simultaneously in the same
experiment. Single-robot trajectory partitioning, sequential acquisition, and
concurrent replay of sequential bags are rejected.

The admitted sources are UTIAS MR.CLAM Datasets 1, 4, and 9. Each contains five physical
iRobot Create robots, synchronized clocks, wheel odometry, range-bearing
observations of shared cylindrical landmarks, and Vicon ground truth. Vicon is
loaded only by the evaluator and is never used to initialize or optimize the
common graph. Teammate observations are excluded; the benchmark graph contains
only robot odometry and observations of 15 static cylinders.

The local SlideSLAM parking-lot bags are not eligible for this benchmark. Their
original acquisition intervals are disjoint, so launching them concurrently
does not make them a native simultaneous multi-robot run.

## Pinned Source

Place the official archive and extracted directory at:

```text
datasets/MRCLAM/raw/MRCLAM1.zip
datasets/MRCLAM/raw/MRCLAM_Dataset1/
datasets/MRCLAM/raw/MRCLAM4.zip
datasets/MRCLAM/raw/MRSLAM_Dataset4/
datasets/MRCLAM/raw/MRCLAM9.zip
datasets/MRCLAM/raw/MRCLAM_Dataset9/
```

The pinned archive SHA-256 is:

```text
57c0166b0e2761680e83ffa67b4f58bad7187c41fd723345a4a45ebd50a6557e
81f33738c17a3b0ca3d168aab974a7d08777cce2c0ff728dfc25564c7c761a0a
013714987dc0f797853689a80146da1fee50478ad884491d0eeac5f6357602d4
```

Audit the source before any graph construction:

```sh
python -m benchmarks.mrclam.cli audit
```

The audit requires five distinct robot streams, a nonempty five-way sensor
overlap, connected shared-landmark incidence, the exact archive hash, and
manifest declarations that forbid partitioning and sequential replay.

## Common Graph

Materialize the frozen graph and the lossless LARPG, DRAN, and Distributed
Mapper adapters:

```sh
python -m benchmarks.mrclam.cli materialize
```

The default output is:

```text
datasets/MRCLAM/canonical/mrclam_dataset1_native5_no_perturb_v1/
```

Robot-local odometry is integrated on a common one-second keyframe grid.
Static-landmark observations are motion-compensated to the nearest keyframe and
duplicate robot/keyframe/landmark measurements are median-reduced. Their
range--bearing uncertainty, source count, and within-bin dispersion are
propagated to a per-factor Point2 scalar information value. Odometry uses the
same explicit translation/yaw Gaussian model for every method. All factors
remain quadratic; no robust kernel or residual-dependent weighting is used. The common
initialization registers robot-local landmark maps over the first 30 seconds
and never reads Vicon files. Planar SE(2) factors are embedded in SE(3), with a
small documented object-rotation nuisance regularizer.

No artificial pose, yaw, or landmark perturbation is added to this
ground-truth-free initialization. Repeat labels `0`, `1`, and `2` select
independent runtime repetitions only; their materialized initial states and
solver inputs are byte-identical.

MR.CLAM's cylinders are position-only landmarks. The original Distributed
Mapper object formulation instead requires a full SE(3) object pose and a full
relative pose from each robot observation. Its MR.CLAM adapter is therefore a
diagnostic model-mismatch ablation, not a valid equal-model baseline. LARPG and
Edge-wise DRAN++(Object) remain valid position-object comparisons. The added
iMESA(Point2) baseline natively keeps each cylinder as a heterogeneous global
Point2 variable and reaches consensus through the official iMESA biased-prior
implementation. See
`results/dataset1_distributed_mapper_root_cause_20260715.md` for the audit.

Materialization also writes
`adapters/dran/fixed_associations.csv`. It contains the reciprocal, one-hot
barcode association for every pair of robots that physically observed an
object. This file is fixed benchmark input; no association variable is learned
during a solver run.

## Full-Communication Run

Preview the exact commands without launching a solver:

```sh
python -m benchmarks.mrclam.cli run --dry-run
```

Run the pinned three repetitions and six checkpoints (the legacy `--seeds`
option name is retained for adapter compatibility):

```sh
python -m benchmarks.mrclam.cli run \
  --seeds 0 1 2 \
  --checkpoints 1 2 5 10 20 50 \
  --timeout 3600
```

LARPG must upload and broadcast every active robot-object block each round.
The current DRAN entry is Edge-wise DRAN++(Object Point2 shared-only), invoked through
`manual-object-dran-example` with `association_policy=manual`,
`association_mode=fixed_file_soft`, and
`association_lower_shared_latent_mode=direct_edge_dual`. Its upper Sinkhorn,
upper gradient/tracking, and bilevel association updates remain disabled. Frame
transport and gauge synchronization are retained because the lower backend must
compare edge endpoints in a common coordinate system. On the complete topology,
each logged exchange must contain exactly `15 * 5 * 4 = 300` Point2 blocks of
16 bytes. Direct-edge proximal targets are constructed deterministically at both
endpoints from the received shared state; no target or certificate message is
sent. Communication accounting validates but excludes the cache-initialization
exchange at CSV iteration zero and includes every requested lower iteration. Distributed Mapper
runs with zero stopping thresholds and must record nonzero rotation and pose
exchange in every requested round. Any omitted round or payload-count mismatch,
or any nonzero DRAN target/certificate payload, invalidates the run.

iMESA(Point2) inserts each complete robot-local factor graph once, then performs
one synchronous complete pair sweep per reported round. Every sweep contains all
ten robot pairs, 300 directional Point2 blocks, 300 global-key declarations,
and 40 directional messages. Its logical payload is therefore 7200 bytes per
round. There is no range, failure, wildfire, age, or innovation communication
filter. The public object position is the arithmetic mean of all observer
copies; the evaluator supplies only the closed-form nuisance yaw required by
the common SE(3) objective.

The pinned Edge-wise DRAN++(Object) options are equivalent to:

```sh
build/bin/manual-object-dran-example \
  --data_dir <seed_adapter> \
  --num_robots 5 --num_objects 15 --iters <rounds> \
  --topology complete --communication_policy all \
  --association_policy manual \
  --association_mode fixed_file_soft \
  --association_file <fixed_associations.csv> \
  --association_freeze_upper_variables true \
  --association_lower_shared_latent_mode direct_edge_dual \
  --association_lower_direct_edge_synchronous_shared_only true \
  --object_orientation_observed false \
  --object_communication_representation point2 \
  --local_solver reduced_rotation --local_max_iterations 3 \
  --reduced_rotation_preconditioner cholesky \
  --local_initial_max_iterations 0 --local_initial_rounds 0 \
  --object_interface_mode copy_baseline
```

The optional edge-wise residual-balanced penalty keeps all state at the edge
owner and adds no communication payload:

```sh
  --beta 0.2 \
  --association_lower_direct_edge_adaptive_beta true \
  --association_lower_direct_edge_beta_min 0.05 \
  --association_lower_direct_edge_beta_max 0.5 \
  --association_lower_direct_edge_beta_imbalance_ratio 5 \
  --association_lower_direct_edge_beta_update_factor 1.25 \
  --association_lower_direct_edge_beta_ema_decay 0.8 \
  --association_lower_direct_edge_beta_update_period 5
```

This is currently a diagnostic ablation rather than the default. It preserves
the physical edge multiplier when beta changes and uses a gauge-transported,
locally derived midpoint only to estimate the balance residual.

The two `local_initial_*` flags are a paired, disabled-by-default diagnostic
schedule. Setting them to `5` and `3`, for example, uses a five-step cap for
outer rounds 1--3 and the normal three-step cap afterward. The pinned result
keeps both at zero because the tested 5-to-3 schedules degraded convergence.

The runner reports the common external objective, gradient norm, gap to the
SE-Sync reference, cumulative logical payload, messages, and runtime.
One SE(2) alignment fitted from robot poses is reused for robot and object
physical metrics.

For the public objective, every DRAN trajectory and object copy is first
transported to robot 0's frame using the exported lower-backend transforms.
For each canonical object, the evaluator then takes the arithmetic mean of the
positions from all physical observers and evaluates that single fused position.
It never selects an owner copy. The projected chordal mean of copy rotations is
used only for the tiny point-object nuisance-rotation term. LARPG already has a
single center-owned object state, so no copy fusion is needed.

Dataset 1 has a consistent declared-identity anomaly between landmark subjects
11 and 17. The graph is left unchanged because all five robots use those source
identities consistently. A raw-data audit fixes the evaluation-only ground-truth
mapping as `11 -> 17` and `17 -> 11` before any solver is run. Every method uses
this same mapping; no estimate-dependent assignment is performed.

## Pinned Result

The current Point2 shared-only Edge-wise DRAN++ result, local-solver ablation,
and communication audit are documented in
[`results/dataset1_dranpp_point2_shared_only_20260715.md`](results/dataset1_dranpp_point2_shared_only_20260715.md).

The earlier perturbed iMESA(Point2) three-seed comparison is retained in
[`results/dataset1_imesa_point2_fullcomm_20260715.md`](results/dataset1_imesa_point2_fullcomm_20260715.md).

The corrected three-method result, including the object-aware Distributed
Mapper audit, is documented in
[`results/dataset1_object_dgs_v2_fullcomm_20260715.md`](results/dataset1_object_dgs_v2_fullcomm_20260715.md).
The earlier owner-copy/legacy-DRAN and legacy Distributed Mapper reports are
retained only as visibly superseded artifacts.

The complete-versus-ring DRAN/iMESA rebuttal experiment, frozen CSVs, LaTeX
subsection, and publication figures are packaged under
[`rebuttle_experiments/`](../../rebuttle_experiments/README.md). The ring is the
fixed cycle `0-1-2-3-4-0`; all five edges communicate every round, with no lazy
filtering or within-round multi-hop forwarding.

Official source: <https://asrl.utias.utoronto.ca/datasets/mrclam/>

Dataset paper: <https://doi.org/10.1177/0278364911398404>
