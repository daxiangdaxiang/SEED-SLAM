# Indoor/second ground truth

This directory is the single canonical location for the ten-robot trajectory
references and the 15 physical-object ground-truth poses used by the evaluator.

## Files

- `trajectory_<robot>.tum`: timestamped trajectory reference in TUM format.
- `object_ground_truth.json`: 15 object poses in the motion-capture global frame.

The directly measured VRPN trajectories are `dingo001`, `dingo002`,
`dingo003`, `jackal1`, `jackal2`, `ugv1`, `ugv2`, and `ugv3`. The `A2W`
reference is 99.18% measured VRPN LiDAR trajectory with map-assisted gap
repair. The `go2W` trajectory is a UGV2-map-assisted pseudo-reference. The
mixed ten-robot metric must therefore be described as a trajectory-reference
metric rather than pure motion-capture ground truth.

These files are evaluation-only and are not read by the DRAN optimizer.
