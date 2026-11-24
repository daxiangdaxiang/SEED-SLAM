"""Frozen-data local completion audit, not an online estimator or GT-based repair."""
import argparse
import csv
from itertools import groupby
import json
from pathlib import Path
import sys
import time

import gtsam
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dranppr_incremental import IncrementalPoint2Robot
from dranppr_sparse_linearization import sparse_linearize
from audit_residual_evidence import sha256,write_csv


def robustify_observations(graph,shape):
    output = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
        f = graph.at(i)
        if isinstance(f,gtsam.BearingRangeFactor2D):
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.GemanMcClure.Create(shape),f.noiseModel())
            measured = f.measured()
            f = gtsam.BearingRangeFactor2D(*f.keys(),measured.bearing(),measured.range(),noise)
        output.push_back(f)
    return output


def gradient_norm(graph,values):
    a,r,keys = sparse_linearize(graph,values)
    g = np.asarray(a.T@r).ravel()
    offset = 0
    zero = values.zeroVectors()
    for key in keys:
        dim = zero.at(key).size
        if key == gtsam.symbol('x',0):
            g[offset:offset+dim] = 0
        offset += dim
    return float(np.linalg.norm(g))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--kernel',type=Path,required=True)
    parser.add_argument('--robot',type=int,required=True)
    parser.add_argument('--duration',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    metadata = json.loads((args.stream/'metadata.json').read_text())
    events = args.stream/'events.jsonl'
    assert sha256(events)==metadata['event_sha256']
    shapes = {float(r['shape']) for r in csv.DictReader(args.kernel.open()) if int(r['dimension'])==2}
    assert len(shapes)==1
    shape = shapes.pop()
    packets = [json.loads(line) for line in events.open()]
    first = packets[0]['timestamp_s']
    packets = [p for p in packets if p['timestamp_s']-first<=args.duration and p['robot']==args.robot]
    robot = IncrementalPoint2Robot(args.robot)
    for _,group in groupby(packets,key=lambda p:p['timestamp_s']):
        robot.push_batch(tuple(group))
    source_rows = [r for r in csv.DictReader(args.state.open()) if int(r['robot'])==args.robot]
    initial = gtsam.Values()
    for r in source_rows:
        if r['kind']=='pose':
            initial.insert(gtsam.symbol('x',int(r['index'])),gtsam.Pose2(*[float(r[k]) for k in ['x','y','yaw']]))
        else:
            initial.insert(gtsam.symbol('l',int(r['index'])),np.array([float(r['x']),float(r['y'])]))
    assert set(initial.keys())==robot.graph.keys, 'state and factor prefix disagree'
    np.testing.assert_allclose(initial.atPose2(gtsam.symbol('x',0)).matrix(),np.eye(3),atol=1e-10)
    graph = robustify_observations(robot.graph.factor_graph(),shape)
    physical = robustify_observations(robot.physical_graph(),shape)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-10)
    params.setAbsoluteErrorTol(1e-10)
    before = dict(cost=graph.error(initial),gradient_norm=gradient_norm(physical,initial))
    start = time.perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph,initial,params)
    final = optimizer.optimize()
    elapsed = time.perf_counter()-start
    after = dict(cost=graph.error(final),gradient_norm=gradient_norm(physical,final))
    assert after['cost']<=before['cost']+1e-8
    for label,values in [('before',initial),('after',final)]:
        rows = []
        for original in source_rows:
            row = dict(original)
            if row['kind']=='pose':
                p = values.atPose2(gtsam.symbol('x',int(row['index'])))
                row.update(x=p.x(),y=p.y(),yaw=p.theta())
            else:
                p = values.atPoint2(gtsam.symbol('l',int(row['index'])))
                row.update(x=p[0],y=p[1])
            rows.append(row)
        write_csv(args.output/f'{label}_estimates.csv',rows)
        weights = []
        for identifier,f in robot.physical_factor_items():
            if identifier[0]!='observation':continue
            r = f.unwhitenedError(values)
            r2 = 2*f.error(values)
            weights.append(dict(source_row=identifier[1],pose_index=gtsam.Symbol(f.keys()[0]).index(),
                                bearing_error=float(r[0]),range_error=float(r[1]),
                                mahalanobis2=r2,gm_weight=shape**4/(shape**2+r2)**2))
        write_csv(args.output/f'{label}_weights.csv',weights)
    summary = dict(robot=args.robot,dataset=metadata['dataset'],shape=shape,before=before,after=after,
                   lm_iterations=optimizer.iterations(),lm_seconds=elapsed,truth_used=False,
                   event_sha256=sha256(events),state_sha256=sha256(args.state),kernel_sha256=sha256(args.kernel),
                   scope='fixed final-mu SIG objective; batch completion, not online and not a global optimum certificate',
                   source_hashes={str(p):sha256(p) for p in [Path(__file__),Path(__file__).resolve().parents[1]/'scripts/dranppr_incremental.py',Path(__file__).resolve().parents[1]/'scripts/dranppr_sparse_linearization.py']})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
