"""Audit arrived identity support and frozen pair geometry; not an initializer."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.mrclam.materialize import estimate_se2_registration, MRCLAMMaterializationError
from experiments.audit_native_checkpoint import typed_state, sha256


def analyze(events, states, links, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid duration')
    normalized = [tuple(sorted(edge)) for edge in links]
    if not normalized or len(set(normalized)) != len(normalized) or any(
            len(edge) != 2 or edge[0] == edge[1] or edge[0] < 0 for edge in normalized):
        raise ValueError('invalid topology')
    robots = sorted({r for edge in normalized for r in edge})
    seen = {r: set() for r in robots}
    ready = {edge: None for edge in normalized}
    expected, pose_times, born, sources = set(), {}, {}, set()
    first, previous = None, -math.inf
    for row in events:
        t = float(row['timestamp_s'])
        if not math.isfinite(t) or t < previous:
            raise ValueError('invalid event timestamp')
        if first is None:
            first = t
        if t-first > duration:
            break
        previous = t
        if float(row.get('available_time_s', t)) != t or not float(row['support_max_time_s']) <= t:
            raise ValueError('noncausal event')
        robot, index = row['robot'], row['pose_index']
        if robot not in seen:
            raise ValueError('event robot absent from topology')
        key = (robot, 'pose', index)
        if row['kind'] == 'pose':
            if key in expected:
                raise ValueError('duplicate pose')
            expected.add(key)
            pose_times[key] = t
            born.setdefault(robot, t-first)
        elif row['kind'] == 'observation':
            if key not in pose_times or pose_times[key] != t:
                raise ValueError('observation pose timestamp mismatch')
            source = (robot, row['source_row'])
            if source in sources:
                raise ValueError('duplicate observation')
            sources.add(source)
            landmark = row['landmark_subject']
            expected.add((robot, 'landmark', landmark))
            seen[robot].add(landmark)
            for edge in normalized:
                if ready[edge] is None and len(seen[edge[0]] & seen[edge[1]]) >= 2:
                    ready[edge] = t-first
        else:
            raise ValueError('unknown event kind')
    if first is None:
        raise ValueError('empty stream')
    received, points = set(), {r: {} for r in robots}
    for state in states:
        key = (state['robot'], state['kind'], state['index'])
        if key in received:
            raise ValueError('duplicate state')
        received.add(key)
        if key not in expected:
            raise ValueError('snapshot and prefix keys disagree')
        if not all(math.isfinite(state[k]) for k in ('x', 'y')):
            raise ValueError('nonfinite state')
        if state['kind'] == 'pose':
            if state['timestamp_s'] != pose_times[key] or not math.isfinite(state['yaw']):
                raise ValueError('snapshot pose timestamp or yaw mismatch')
        else:
            points[state['robot']][state['index']] = (state['x'], state['y'])
    if received != expected:
        raise ValueError('snapshot and prefix keys disagree')
    edges = []
    for edge in normalized:
        a, b = edge
        common = sorted(seen[a] & seen[b])
        registration = dict(status='insufficient_correspondences')
        if len(common) >= 2:
            target = np.array([points[a][key] for key in common])
            source = np.array([points[b][key] for key in common])
            registration = dict(status='degenerate_geometry',
                target_spread_rms_m=float(np.linalg.norm(target-target.mean(axis=0))/math.sqrt(len(common))),
                source_spread_rms_m=float(np.linalg.norm(source-source.mean(axis=0))/math.sqrt(len(common))))
            try:
                transform, rms = estimate_se2_registration(source, target)
                registration.update(status='estimated', rms_m=rms,
                    target_from_source=[transform.x, transform.y, transform.yaw])
            except MRCLAMMaterializationError as error:
                registration['reason'] = str(error)
        edges.append(dict(target=a, source=b, common_landmarks=common,
            first_two_shared_elapsed_s=ready[edge], registration=registration))
    return dict(truth_used=False, duration_s=duration, first_event_time_s=first,
        robot_first_pose_elapsed_s=born, poses=len(pose_times), observations=len(sources),
        landmark_copies=sum(map(len, seen.values())), edges=edges,
        all_links_two_labels_elapsed_s=max(ready.values()) if all(t is not None for t in ready.values()) else None,
        scope='Identity arrival counts plus frozen independent-estimate registrations; no initialized network or communication simulation',
        limitations='Two labels are not a geometry or uncertainty certificate. Pairwise support is not a necessary network observability condition. Registrations use ONLY the final supplied checkpoint, not earlier arrival states.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--links', nargs='+', required=True, help='robot:robot, one per allowed edge')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    links = [tuple(map(int, text.split(':'))) for text in args.links]
    events = args.stream/'events.jsonl'
    metadata = json.loads((args.stream/'metadata.json').read_text())
    if sha256(events) != metadata['event_sha256']:
        raise ValueError('event hash mismatch')
    with events.open() as stream:
        result = analyze((json.loads(line) for line in stream), typed_state(args.snapshot), links, args.duration)
    result['provenance'] = dict(dataset=metadata['dataset'], stream=str(events),
        snapshot=str(args.snapshot), event_sha256=metadata['event_sha256'],
        snapshot_sha256=sha256(args.snapshot), script_sha256=sha256(__file__),
        registration_source_sha256=sha256(Path(__file__).resolve().parents[1]/'benchmarks/mrclam/materialize.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
