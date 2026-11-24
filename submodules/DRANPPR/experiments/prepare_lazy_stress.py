"""Fixed measurement-stress controls; never modify the common-past package."""
import argparse
import json
import math
from pathlib import Path
import random

from experiments.audit_conditional_replay import sha256


def perturb(lines, cutoff, stop, kind, seed):
    if not math.isfinite(cutoff) or not math.isfinite(stop) or stop <= cutoff:
        raise ValueError('invalid stress window')
    if kind not in ('gaussian_bearing', 'sparse_bearing'):
        raise ValueError('unknown stress condition')
    rng = random.Random(seed)
    output, changes, eligible = [], [], 0
    for index, line in enumerate(lines):
        event = json.loads(line)
        if cutoff < event['timestamp_s'] <= stop and event['kind'] == 'observation':
            eligible += 1
            delta = (rng.gauss(0., .15) if kind == 'gaussian_bearing' else
                     rng.choice([-.5, .5]) if rng.random() < .05 else 0.)
            if delta:
                old = event['bearing_rad']
                event['bearing_rad'] = math.atan2(math.sin(old+delta), math.cos(old+delta))
                changes.append(dict(line_index=index, robot=event['robot'],
                    source_row=event['source_row'], timestamp_s=event['timestamp_s'],
                    before_bearing_rad=old, after_bearing_rad=event['bearing_rad'],
                    added_bearing_rad=delta))
                line = json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n'
        output.append(line)
    return output, dict(kind=kind, seed=seed, cutoff_time_s=cutoff, stop_time_s=stop,
        eligible_observations=eligible, changes=changes,
        fixed_condition=(dict(additive_bearing_sd_rad=.15) if kind == 'gaussian_bearing' else
                         dict(probability=.05, absolute_bearing_offset_rad=.5)),
        truth_used=False, estimator_noise_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--kind', choices=['gaussian_bearing', 'sparse_bearing'], required=True)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = json.loads(args.warm_start.read_text())
    if package.get('truth_used', False):
        raise ValueError('truth-assisted package')
    first = min(r['poses'][0]['timestamp_s'] for r in package['robots'])
    cutoff = max(r['poses'][-1]['timestamp_s'] for r in package['robots'])
    original = args.events.read_text().splitlines(keepends=True)
    lines, manifest = perturb(original, cutoff, first+args.duration, args.kind, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'events.jsonl'
    target.write_text(''.join(lines))
    manifest.update(original_events_sha256=sha256(args.events),
        events_sha256=sha256(target), warm_start_sha256=sha256(args.warm_start),
        script_sha256=sha256(Path(__file__)),
        prefix_byte_identical=all(a == b for a, b in zip(original, lines)
                                 if json.loads(a)['timestamp_s'] <= cutoff))
    (args.output/'stress_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(kind=args.kind, eligible=manifest['eligible_observations'],
                         changed=len(manifest['changes']), prefix_byte_identical=manifest['prefix_byte_identical'])))


if __name__ == '__main__':
    main()
