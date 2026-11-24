"""GT-free frozen joint input; offline bootstrap, not online initialization."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.audit_native_checkpoint import sha256, typed_state
from experiments.audit_native_interface_support import analyze


def matrix(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def prepare(events, states, links, duration):
    prefix = []
    for event in events:
        if prefix and event['timestamp_s']-prefix[0]['timestamp_s'] > duration:
            break
        prefix.append(event)
    events, states = prefix, list(states)
    support = analyze(events, states, links, duration)
    robots = sorted({r for edge in links for r in edge})
    root = robots[0]
    adjacency = {r: {} for r in robots}
    for edge in support['edges']:
        if edge['registration']['status'] != 'estimated':
            continue
        a, b = edge['target'], edge['source']
        transform = matrix(edge['registration']['target_from_source'])
        adjacency[a][b] = transform
        adjacency[b][a] = np.linalg.inv(transform)
    charts, queue = {root: np.eye(3)}, [root]
    for a in queue:
        for b, transform in sorted(adjacency[a].items()):
            if b not in charts:
                charts[b] = charts[a] @ transform
                queue.append(b)
    if set(charts) != set(robots):
        raise ValueError('insufficient pairwise bootstrap support; not a network observability verdict')
    points = {(s['robot'], s['index']): s for s in states if s['kind'] == 'landmark'}
    clusters, lookup, initial_points = [], {}, {}
    for label in sorted({key[1] for key in points}):
        unseen = {r for r in robots if (r, label) in points}
        while unseen:
            owner = min(unseen)
            component = {owner}
            pending = [owner]
            unseen.remove(owner)
            for a in pending:
                for u, v in links:
                    b = v if u == a else u if v == a else None
                    if b in unseen:
                        component.add(b)
                        unseen.remove(b)
                        pending.append(b)
            cluster = len(clusters)
            clusters.append(dict(cluster=cluster, label=label, owner=owner, observers=sorted(component)))
            p = points[owner, label]
            initial_points[cluster] = (charts[owner] @ [p['x'], p['y'], 1.])[:2].tolist()
            for r in component:
                lookup[r, label] = cluster
    records = {r: dict(id=r, poses=[], objects=[], odometry=[], observations=[]) for r in robots}
    for state in states:
        r = state['robot']
        if state['kind'] == 'pose':
            value = charts[r] @ matrix([state['x'], state['y'], state['yaw']])
            records[r]['poses'].append(dict(index=state['index'], timestamp_s=state['timestamp_s'],
                value=[float(value[0, 2]), float(value[1, 2]), math.atan2(value[1, 0], value[0, 0])]))
        else:
            cluster = lookup[r, state['index']]
            records[r]['objects'].append(dict(cluster=cluster, label=state['index'], point=initial_points[cluster]))
    first = support['first_event_time_s']
    for event in events:
        if event['timestamp_s']-first > duration:
            break
        record = records[event['robot']]
        if event['kind'] == 'pose' and event['pose_index']:
            record['odometry'].append(dict(source=event['from_pose_index'], target=event['pose_index'],
                delta=event['delta'], dt=event['duration_s']))
        elif event['kind'] == 'observation':
            record['observations'].append(dict(pose=event['pose_index'],
                cluster=lookup[event['robot'], event['landmark_subject']],
                range=event['range_m'], bearing=event['bearing_rad'], source=event['source_row']))
    for record in records.values():
        record['poses'].sort(key=lambda p: p['index'])
        record['objects'].sort(key=lambda p: p['cluster'])
    return dict(schema='dranppr_frozen_joint_v1', truth_used=False, root_robot=root,
        links=sorted([sorted(edge) for edge in links]), robots=list(records.values()), clusters=clusters,
        support=support, duration_s=duration,
        scope='Frozen physical joint graph, one fixed root pose; all other roots free',
        bootstrap_scope='Offline neighbor registration and roster preparation; NOT included in solve communication ledger')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--links', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    event_file = args.stream/'events.jsonl'
    metadata = json.loads((args.stream/'metadata.json').read_text())
    if sha256(event_file) != metadata['event_sha256']:
        raise ValueError('event hash mismatch')
    with event_file.open() as stream:
        result = prepare((json.loads(line) for line in stream), typed_state(args.snapshot),
            [tuple(map(int, edge.split(':'))) for edge in args.links], args.duration)
    result['provenance'] = dict(dataset=metadata['dataset'], event_sha256=sha256(event_file),
        snapshot=str(args.snapshot), snapshot_sha256=sha256(args.snapshot), script_sha256=sha256(__file__))
    result['preparation_seconds'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), robots=len(result['robots']),
        clusters=len(result['clusters']), seconds=result['preparation_seconds'])))


if __name__ == '__main__':
    main()
