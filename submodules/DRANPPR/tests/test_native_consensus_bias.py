import json
from pathlib import Path
import subprocess

import pytest


def test_pinned_gaussian_dual_decay_has_information_dependent_consensus_bias():
    binary = Path('/tmp/dranppr_matched_build/native-consensus-bias')
    assert binary.is_file(), 'build the pinned Gaussian consensus diagnostic'
    process = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    rows = json.loads(process.stdout)
    assert len(rows) == 4
    for row in rows:
        assert row['numerical_gap'] == pytest.approx(row['predicted_gap'], abs=1e-8)
        assert row['mean'] == pytest.approx(.5, abs=1e-10)
    decayed = [r for r in rows if r['dual_decay'] == .9]
    assert decayed[0]['sigma'] == 1.
    assert decayed[1]['sigma'] == .1
    assert decayed[1]['numerical_gap'] > 10*decayed[0]['numerical_gap']
