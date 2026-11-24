# DBA algorithm and code map

## Problem

For robot-local trajectory and object states `x` and a network identity
partition `P`, the released loop has the bilevel form

```text
minimize_P  U(P; x*(P), Sigma^-(P))
subject to x*(P) approximately solves the distributed DRAN++ lower problem.
```

`P` is not a fixed-cardinality assignment matrix. It is a partition of the
observed object copies. A component may contain at most one copy from each
robot, and singleton components are valid. The association cardinality is
therefore selected by the partition itself.

## Lower response

The lower problem fixes `P` and optimizes the robot/object pose graph with
DRAN++. In the released Point2 DBA configuration, cross-robot object-copy
constraints use the `exact_weighted` shared-latent mode. The lower run then
exports one prediction for every communication edge after removing all direct
association factors on that edge.

Around the current lower solution, the nonlinear objective is approximated by

```text
F(x + dx) ~= F(x) + g^T dx + 0.5 dx^T H dx.
```

After excluding an edge block `e`, the local response is predicted by solving

```text
dx^{-e} = -(H^{-e})^{-1} g^{-e}.
```

Only the object/interface marginal means and covariances required by the upper
level are exported. This is a finite local response, not a claim of exact
nonlinear conditional independence.

Relevant backend code:

- `include/DPGO/ManualObjectDran.h`: options, diagnostics and result schema.
- `src/ManualObjectDran.cpp`: Point2/ObjectPose3 DRAN++ execution and leave-edge
  response export.
- `examples/ManualObjectDranExample.cpp`: command-line interface and CSV output.
- `include/DPGO/BilevelAssociation.h` and
  `src/bilevel/BilevelAssociation.cpp`: association statistics and empirical
  track-information calibration.

## Upper objective

The upper layer scores one network partition rather than independent pairwise
matches. `scripts/dba_joint_cavity_state_predictive.py` constructs the joint
uncertainty-aware score from edge-held-out lower predictions. The score keeps
the full covariance among objects owned by the same robot and profiles shared
frame transforms over the topology folds.

The finite feasible edits are:

- `merge`: join compatible identity components;
- `move`: transfer one object copy between components;
- `detach`: turn one copy into a singleton.

`scripts/dba_collapsed_profile_partition.py` generates and evaluates these
edits on the fixed candidate-support graph. The selected edit batch must be
component-disjoint and must strictly reduce the same joint objective.

`scripts/dba_network_quotient_reference.py` provides canonical network labels,
routed sufficient-statistic aggregation and min-consensus. Min-consensus means
that neighboring robots repeatedly retain the best feasible edit identifier
and score they have seen; after propagation, all robots in the affected
network component agree on the same minimum without a central coordinator.

## Closed loop

`scripts/run_dba_fixed_point.py` is the main entry point. For every outer epoch
it performs:

```text
current partition
  -> fixed-association DRAN++ lower response
  -> edge-held-out predictive statistics
  -> distributed network partition edits
  -> next partition.
```

The loop stops when the canonical partition hash is unchanged. A safety limit
is reported as nonconvergence rather than silently accepted as a fixed point.
Ground truth is used only after optimization for metrics.

## Scope boundary

The backend also implements ObjectPose3 interface PCG, compact pose messages,
private block-Jacobi preconditioning and block-tridiagonal private solves. The
current DBA upper objective and released 05/07 protocol use Point2 translation
states. Extending the cross-fitted joint upper likelihood to orientation and
object symmetry is separate work and is not claimed by this snapshot.

