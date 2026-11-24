"""Freeze a terminal replay graph without changing states, measurements or noise."""
import math


def prepare(events, states, warm, duration, variance):
    if warm.get('truth_used', False) or warm['schema'] != 'dranppr_frozen_joint_v1':
        raise ValueError('expected GT-free nominal warm prefix')
    if not math.isfinite(variance) or variance < 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid duration/variance')
    first = min(r['poses'][0]['timestamp_s'] for r in warm['robots'])
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in warm['robots'])
    if first+duration <= cutoff:
        raise ValueError('duration must extend prefix')
    records = {r['id']: dict(id=r['id'], poses=[], objects=[], odometry=[], observations=[])
               for r in warm['robots']}
    state = {}
    for row in states:
        key = (int(row['robot']), row['kind'], int(row['index']))
        values = [float(row[k]) for k in (['x', 'y', 'yaw'] if row['kind'] == 'pose' else ['x', 'y'])]
        if key in state or key[0] not in records or key[1] not in ('pose', 'landmark') or not all(map(math.isfinite, values)):
            raise ValueError('invalid/duplicate terminal state')
        state[key] = dict(value=values, timestamp_s=float(row['timestamp_s']) if row['kind'] == 'pose' else None)
    past = {}
    for r in warm['robots']:
        labels = {o['cluster']: o['label'] for o in r['objects']}
        past[r['id']] = dict(
            poses={p['index']: p['timestamp_s'] for p in r['poses']},
            odometry={o['target']: (o['source'], o['delta'], o['dt']) for o in r['odometry']},
            observations={o['source']: (o['pose'], labels[o['cluster']], o['range'], o['bearing'])
                          for o in r['observations']})
    seen, observed, source_ids = set(), {}, set()
    last = -math.inf
    for event in events:
        t, r, p = event['timestamp_s'], event['robot'], event['pose_index']
        if not math.isfinite(t) or t < last:
            raise ValueError('invalid event order')
        last = t
        if t-first > duration:
            break
        if r not in records or event['available_time_s'] != t or event['support_max_time_s'] > t:
            raise ValueError('invalid frame support')
        record = records[r]
        key = (r, 'pose', p)
        if key not in state or state[key]['timestamp_s'] != t:
            raise ValueError('missing/mismatched terminal pose state')
        if event['kind'] == 'pose':
            if key in seen or p != len(record['poses']):
                raise ValueError('duplicate/nonconsecutive frame')
            seen.add(key)
            record['poses'].append(dict(index=p, timestamp_s=t, value=state[key]['value']))
            if t <= cutoff and past[r]['poses'].pop(p, None) != t:
                raise ValueError('prefix pose mismatch')
            if p:
                odom = (event['from_pose_index'], event['delta'], event['duration_s'])
                if t <= cutoff and past[r]['odometry'].pop(p, None) != odom:
                    raise ValueError('prefix odometry mismatch')
                if odom[0] != p-1 or not math.isfinite(odom[2]) or odom[2] <= 0:
                    raise ValueError('invalid odometry')
                record['odometry'].append(dict(source=p-1, target=p, delta=odom[1], dt=odom[2],
                    sigmas=[s*math.sqrt(odom[2]) for s in [.1, .1, .05]]))
            elif event['delta'] is not None or event['from_pose_index'] is not None or event['duration_s'] != 0:
                raise ValueError('invalid root frame')
        elif event['kind'] == 'observation':
            label, source = event['landmark_subject'], event['source_row']
            if key not in seen or (r, source) in source_ids:
                raise ValueError('observation before pose or duplicate source')
            source_ids.add((r, source))
            obs = (p, label, event['range_m'], event['bearing_rad'])
            if t <= cutoff and past[r]['observations'].pop(source, None) != obs:
                raise ValueError('prefix observation mismatch')
            if not math.isfinite(obs[2]) or obs[2] <= 0 or not math.isfinite(obs[3]):
                raise ValueError('invalid observation')
            observed.setdefault(label, set()).add(r)
            record['observations'].append(dict(pose=p, label=label, source=source, range=obs[2],
                bearing=obs[3], sigmas=[math.sqrt(.05**2+(variance if t > cutoff else 0.)), .1]))
        else:
            raise ValueError('unknown event kind')
    if any(values for record in past.values() for values in record.values()):
        raise ValueError('incomplete prefix')
    links = warm['links']
    clusters, lookup = [], {}
    for label, observers in sorted(observed.items()):
        unseen = set(observers)
        while unseen:
            owner = min(unseen)
            unseen.remove(owner)
            component = [owner]
            for a in component:
                for u, v in links:
                    b = v if a == u else u if a == v else None
                    if b in unseen:
                        unseen.remove(b)
                        component.append(b)
            cluster = len(clusters)
            point = None
            for r in sorted(component):
                key = (r, 'landmark', label)
                if key not in state:
                    raise ValueError('missing terminal object state')
                value = state[key]['value']
                if point is not None and value != point:
                    raise ValueError('inconsistent shared terminal state; no averaging allowed')
                point = value
                seen.add(key)
                lookup[r, label] = cluster
                records[r]['objects'].append(dict(cluster=cluster, label=label, point=value))
            clusters.append(dict(cluster=cluster, label=label, owner=owner, observers=sorted(component)))
    if seen != set(state):
        raise ValueError('terminal state does not match frozen event support')
    for r, record in records.items():
        for obs in record['observations']:
            obs['cluster'] = lookup[r, obs.pop('label')]
    return dict(schema='dranppr_frozen_joint_v2', truth_used=False, root_robot=warm['root_robot'],
        links=links, clusters=clusters, robots=[records[r] for r in sorted(records)], duration_s=duration,
        suffix_added_bearing_variance=variance, warm_cutoff_s=cutoff,
        scope='Offline terminal completion; exact terminal states, per-factor sigmas, no registration or averaging')
