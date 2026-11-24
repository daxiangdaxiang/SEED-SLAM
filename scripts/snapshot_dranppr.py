"""Create a new, content-addressed research snapshot without modifying its source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


DIRECTORIES = ('src', 'include', 'examples', 'cmake', 'benchmarks', 'scripts', 'experiments', 'tests', 'docs')
TOP_FILES = ('CMakeLists.txt', 'LICENSE', 'README.md', 'package.xml', '.gitignore')
SUFFIXES = {'.cpp', '.h', '.hpp', '.c', '.cmake', '.in', '.py', '.md', '.txt', '.json', '.sh', '.patch', '.g2o'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_files(source):
    paths = [source/name for name in TOP_FILES if (source/name).is_file()]
    for name in DIRECTORIES:
        for path in (source/name).rglob('*'):
            relative = path.relative_to(source)
            if any(part in ('__pycache__', '.pytest_cache', 'archive') or part.startswith('build')
                   for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError('source symlink requires explicit review: '+str(relative))
            if path.is_file() and path.suffix in SUFFIXES:
                paths.append(path)
    return sorted(paths)


def verify_archive(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if digest(root/name) != expected:
            raise ValueError('source archive changed: '+str(root/name))
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest['all_terminal']:
        raise ValueError('unfinished source archive')
    return manifest


def seal(root):
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n' for p in paths))


def seal_source(root):
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SOURCE_SHA256SUMS'
                   and not any(part in ('__pycache__', '.pytest_cache', 'results') or part.startswith('build')
                               for part in p.relative_to(root).parts))
    (root/'SOURCE_SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n' for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if target.exists():
        parser.error('target must not exist; no destructive refresh is supported')
    paths = source_files(source)
    models = source/'results/dranppr_response_final_20261124'
    manifest = verify_archive(models)
    target.mkdir(parents=True)
    for path in paths:
        destination = target/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    fixtures = target/'fixtures/interface_models'
    for case in manifest['cases']:
        folder = fixtures/case
        folder.mkdir(parents=True)
        for name in ('quadratic.npz', 'model.json'):
            shutil.copy2(models/case/name, folder/name)
    write_json(fixtures/'manifest.json', dict(all_terminal=True, cases=manifest['cases'],
        scope='Frozen physical quadratics and a synthetic SPD control; not raw sensor datasets',
        source_archive=models.name, source_ledger=digest(models/'SHA256SUMS')))
    seal(fixtures)
    references = target/'evidence/reference'
    archives = ('dranppr_model_response_stress_20261124', 'dranppr_model_response_controls_20261124',
                'dranppr_model_response_stress_review_20261124', 'dranppr_model_response_controls_review_20261124')
    for name in archives:
        root = source/'results'/name
        verify_archive(root)
        folder = references/name
        folder.mkdir(parents=True)
        for file in ('manifest.json', 'runs.json', 'summary.json', 'comparisons.json', 'review.json'):
            if (root/file).exists():
                shutil.copy2(root/file, folder/file)
        shutil.copy2(root/'SHA256SUMS', folder/'ORIGINAL_ARCHIVE_SHA256SUMS.txt')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(source), 'branch', '--show-current'], text=True).strip()
    provenance = dict(source_repository='DRANPPR', source_base_commit=commit, source_branch=branch,
        source_contains_uncommitted_research=True, source_file_count=len(paths),
        scope='Experimental source snapshot, not a stable backend replacement',
        source_files={str(p.relative_to(source)): digest(p) for p in paths},
        excluded='Generated builds/caches, raw historical results, docs/archive; selected evidence copied separately')
    write_json(target/'SNAPSHOT.json', provenance)
    print(json.dumps(dict(target=str(target), source_files=len(paths), fixture_cases=manifest['cases'])))


if __name__ == '__main__':
    main()
