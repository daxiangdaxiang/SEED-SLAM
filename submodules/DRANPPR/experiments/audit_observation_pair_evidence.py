"""Pose-free co-observation consistency, not an outlier classifier or online repair."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def distance_variance(a, b):
    points = [(float(row['range_m'])*math.cos(float(row['bearing_rad'])),
               float(row['range_m'])*math.sin(float(row['bearing_rad']))) for row in [a, b]]
    dx, dy = points[0][0]-points[1][0], points[0][1]-points[1][1]
    distance = math.hypot(dx, dy)
    if not math.isfinite(distance) or distance <= 1e-12:
        raise ValueError('degenerate pair distance')
    wx, wy = dx/distance, dy/distance
    variance = 0.
    for row in [a, b]:
        r, angle = float(row['range_m']), float(row['bearing_rad'])
        c, s = math.cos(angle), math.sin(angle)
        # The second observation has opposite signs, which cancel when squared.
        variance += .1**2*(wx*c+wy*s)**2 + .05**2*r*r*(-wx*s+wy*c)**2
    return distance, variance


def pair_evidence(frames):
    past, rows = {}, []
    for frame in frames:
        for a, b in itertools.combinations(sorted(frame, key=lambda p:p['landmark_subject']), 2):
            key = (a['landmark_subject'], b['landmark_subject'])
            if key[0] == key[1]:
                raise ValueError('duplicate subject within a frame')
            distance, variance = distance_variance(a, b)
            t = a['timestamp_s']
            if t != b['timestamp_s']:
                raise ValueError('mixed timestamp frame')
            if key in past:
                previous, previous_variance, previous_time = past[key]
                if previous_time >= t:
                    raise ValueError('pair evidence must be strictly chronological')
                rows.append(dict(subject_a=key[0], subject_b=key[1], source_row_a=a['source_row'],
                                 source_row_b=b['source_row'], timestamp_s=t,
                                 previous_timestamp_s=previous_time, distance_m=distance,
                                 previous_distance_m=previous, difference_m=distance-previous,
                                 nominal_std_m=math.sqrt(variance+previous_variance),
                                 normalized_difference=(distance-previous)/math.sqrt(variance+previous_variance)))
            past[key] = (distance, variance, t)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--robot', type=int, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    packets, first = [], None
    with args.events.open() as stream:
        for line in stream:
            row = json.loads(line)
            if first is None:
                first = row['timestamp_s']
            if row['timestamp_s']-first > args.duration:
                break
            if row['robot'] == args.robot and row['kind'] == 'observation':
                packets.append(row)
    frames = [list(group) for _,group in itertools.groupby(packets, key=lambda row:row['timestamp_s'])]
    rows = pair_evidence(frames)
    # Final weights are joined only after all past-only evidence is computed.
    with args.weights.open() as stream:
        weights = {int(row['source_row']):float(row['gm_weight']) for row in csv.DictReader(stream)}
    supported = set()
    for row in rows:
        ids = [row['source_row_a'], row['source_row_b']]
        supported.update(ids)
        row['min_final_gm_weight'] = min(weights[i] for i in ids)
    if rows:
        with (args.output/'pairs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = {}
    for name, group in [('all', rows), ('low_weight', [r for r in rows if r['min_final_gm_weight']<.1]),
                        ('other', [r for r in rows if r['min_final_gm_weight']>=.1])]:
        groups[name] = dict(pairs=len(group),
                           median_abs_difference_m=statistics.median(abs(r['difference_m']) for r in group) if group else None,
                           median_abs_normalized_difference=statistics.median(abs(r['normalized_difference']) for r in group) if group else None,
                           max_abs_normalized_difference=max((abs(r['normalized_difference']) for r in group), default=None))
    summary = dict(observations=len(packets), observation_frames=len(frames),
                   multi_observation_frames=sum(len(frame)>1 for frame in frames),
                   supported_observations=len(supported),
                   low_weight_observations=sum(w<.1 for w in weights.values()),
                   supported_low_weight_observations=sum(weights[i]<.1 for i in supported), groups=groups,
                   arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.events,args.weights,Path(__file__)]},
                   truth_used=False, state_used_for_evidence=False,
                   caveat='Nominal first-order variance only; temporal/pair dependence prevents calibrated independent p-values. Weight 0.1 only stratifies final outputs; it is not an estimator gate. No evidence for single-view frames or first-seen pairs.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
