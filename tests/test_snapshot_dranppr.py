import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1]/'scripts/snapshot_dranppr.py'
    assert path.is_file(), 'snapshot builder not implemented'
    spec = importlib.util.spec_from_file_location('snapshot_dranppr', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_selection_excludes_generated_and_archival_payloads(tmp_path):
    fn = module().source_files
    for name in ('src/test.cpp', 'tests/test_ok.py', 'experiments/model.py', 'cmake/DPGOConfig.cmake.in',
                 'CMakeLists.txt', 'README.md', 'LICENSE', 'results/run/log.json',
                 'build/bin/model', 'docs/archive/old.tar.gz',
                 'experiments/__pycache__/model.pyc', '.git/config'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    names = [str(p.relative_to(tmp_path)) for p in fn(tmp_path)]
    assert names == ['CMakeLists.txt', 'LICENSE', 'README.md', 'cmake/DPGOConfig.cmake.in', 'experiments/model.py',
                     'src/test.cpp', 'tests/test_ok.py']


def test_snapshot_rejects_external_symlinks(tmp_path):
    (tmp_path/'src').mkdir()
    (tmp_path/'src/leak.cpp').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='symlink'):
        module().source_files(tmp_path)
