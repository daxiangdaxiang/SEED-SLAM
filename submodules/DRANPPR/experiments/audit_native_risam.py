"""Serial native local-solver audit. Evaluation truth never enters the C++ replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from evaluate_incremental_reference import evaluate_rows, read_rows
from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.source import audit_dataset, load_manifest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_output(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--risam-source', type=Path, required=True)
    parser.add_argument('--risam-library', type=Path, required=True)
    parser.add_argument('--stream-root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', type=int, default=[4, 1])
    parser.add_argument('--modes', nargs='+', choices=['isam2','risam','risam_gn'],
                        default=['isam2','risam','risam_gn'])
    parser.add_argument('--duration', type=float, default=120.)
    parser.add_argument('--timeout', type=float, default=180.)
    args = parser.parse_args()
    if args.duration <= 30 or args.timeout <= 0:
        parser.error('require duration > 30 and positive timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = args.output/'source_snapshot'
    source.mkdir()
    source_files = [Path(__file__), root/'experiments/native_risam/replay.cpp',
                    root/'experiments/native_risam/CMakeLists.txt',
                    root/'experiments/evaluate_incremental_reference.py',
                    root/'tests/test_native_risam_replay.py']
    for path in source_files:
        shutil.copy2(path, source/path.name)
    patch = command_output(['git','-C',str(args.risam_source),'diff','--no-ext-diff'])
    (source/'linked_risam.patch').write_text(patch+'\n')
    metadata = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    binary_sha256=sha256(args.binary), library_sha256=sha256(args.risam_library),
                    risam_commit=command_output(['git','-C',str(args.risam_source),'rev-parse','HEAD']),
                    risam_dirty=command_output(['git','-C',str(args.risam_source),'status','--porcelain']),
                    dynamic_libraries=command_output(['ldd',str(args.binary)]),
                    source_hashes={p.name:sha256(p) for p in source.iterdir()},
                    scope='independent local native riSAM bridge; NOT full riMESA or DRANPPR',
                    measurement_model='native polar sigma=(0.05 rad,0.1 m); odometry sigma=(0.1,0.1,0.05)*sqrt(dt)',
                    gaussian_solver='QR, relinearize=1e-4, wildfire=1e-3',
                    risam_solver='upstream default local settings with explicitly archived numerical repairs',
                    risam_gn_solver='upstream GNC with Gaussian-control ISAM2 settings and no extended Dogleg',
                    identity='declared barcodes; no relationship edits or evaluation relabeling in estimator',
                    timing='serial runs; per-frame build/update/full-estimate extraction; parsing, CSV IO and posthoc evaluation excluded',
                    streams={})
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summaries, robot_rows, failures = [], [], []
    for dataset in args.datasets:
        stream = args.stream_root/f'dataset{dataset}'
        stream_meta = json.loads((stream/'metadata.json').read_text())
        assert sha256(stream/'events.jsonl') == stream_meta['event_sha256']
        metadata['streams'][dataset] = stream_meta
        (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        completed = []
        for mode in args.modes:
            tag = f'dataset{dataset}_{mode}'
            out = args.output/tag
            command = [str(args.binary),str(stream/'events.jsonl'),str(out),mode,str(args.duration)]
            start = time.monotonic()
            try:
                result = subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
                wall = time.monotonic()-start
                (args.output/f'{tag}.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    raise RuntimeError(f'exit {result.returncode}: {result.stderr[-1000:]}')
                completed.append((mode,out,wall))
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                failures.append(dict(dataset=dataset,mode=mode,command=command,error=str(error)))
                (args.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
            print(f'{tag}: completed={len(completed)}, failed={len(failures)}',flush=True)
        # No truth is loaded until all estimator processes for this stream exit.
        manifest = load_manifest(root/f'benchmarks/mrclam/manifests/dataset{dataset}.json')
        audit = audit_dataset(manifest,args.raw)
        truth = load_ground_truth(manifest,audit.dataset_directory)
        label_map = dict(manifest.evaluation.object_ground_truth_label_map)
        for mode,out,wall in completed:
            timing = json.loads((out/'summary.json').read_text())
            updates = read_rows(out/'updates.csv')
            ms = np.array([float(r['total_ms']) for r in updates])
            first = min(float(r['timestamp_s']) for r in updates)
            for period in ['all','after30']:
                for state in ['final','causal']:
                    rows = read_rows(out/f'{state}_estimates.csv')
                    if period=='after30':
                        rows = [r for r in rows if r['kind']=='landmark' or float(r['timestamp_s'])-first>30.]
                    aggregate, per_robot = evaluate_rows(rows,truth,label_map)
                    summaries.append(dict(dataset=dataset,mode=mode,period=period,state=state,
                                          **aggregate, total_work_s=timing['total_work_s'],
                                          max_serial_backlog_s=timing['max_serial_backlog_s'],
                                          update_p50_ms=float(np.quantile(ms,.5)),
                                          update_p95_ms=float(np.quantile(ms,.95)),
                                          update_p99_ms=float(np.quantile(ms,.99)),wall_s=wall,
                                          physical_gaussian_cost=timing['physical_gaussian_cost']))
                    robot_rows.extend(dict(dataset=dataset,mode=mode,period=period,state=state,**r) for r in per_robot)
        for name,rows in [('summary.csv',summaries),('robots.csv',robot_rows)]:
            if rows:
                with (args.output/name).open('w',newline='') as file:
                    writer = csv.DictWriter(file,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    if failures:
        raise SystemExit(f'{len(failures)} native replay failures; do not drop them from interpretation')


if __name__ == '__main__':
    main()
