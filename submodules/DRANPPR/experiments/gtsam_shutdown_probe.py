"""Minimal dependency-lifecycle reproducer, independent of DRANPPR data/model.

The caller must check the process exit status after the final JSON is printed.
A successful calculation does not imply a successful process shutdown.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['import', 'error', 'linearize', 'optimize'],
                        default='linearize')
    parser.add_argument('--dump-maps', action='store_true')
    args = parser.parse_args()

    import gtsam
    import numpy as np

    result = dict(mode=args.mode, calculation_complete=False)
    if args.mode != 'import':
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.2, .2, .1]))
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(), noise))
        values = gtsam.Values()
        values.insert(1, gtsam.Pose2(.5, 0., .2))
        for index in range(2, 101):
            graph.add(gtsam.BetweenFactorPose2(
                index-1, index, gtsam.Pose2(2., 0., 0.), noise))
            values.insert(index, gtsam.Pose2(2*(index-1)+.3, .1, -.2))
        if args.mode == 'error':
            result['value'] = float(graph.error(values))
        elif args.mode == 'linearize':
            result['value'] = graph.linearize(values).size()
        else:
            estimate = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
            result['value'] = float(graph.error(estimate))

    if args.dump_maps:
        result['tbb_mappings_before_shutdown'] = [
            line for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libtbb' in line]
    result['calculation_complete'] = True
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
