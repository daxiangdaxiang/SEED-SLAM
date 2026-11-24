import json

import pytest

from test_native_conditional_replay import run, rows


@pytest.mark.parametrize('suffix', ['', '_one'])
def test_chart_audit_is_read_only_and_accounts_for_all_refresh_factors(tmp_path,suffix):
    mode='joint_conditional_cached'+suffix
    process,baseline=run(tmp_path/'off',mode)
    assert process.returncode==0,process.stderr
    process,audited=run(tmp_path/'on',mode,audit=True)
    assert process.returncode==0,process.stderr
    for name in ['final_estimates.csv','causal_estimates.csv','control_messages.jsonl']:
        assert (baseline/name).read_bytes()==(audited/name).read_bytes()
    a,b=[json.loads((p/'summary.json').read_text()) for p in [baseline,audited]]
    assert b['chart_audit'] is True
    for field in ['physical_cost','factors_linearized','refreshed_keys','application_bytes','pcg_iterations']:
        assert a[field]==b[field]
    records=[json.loads(line) for line in (audited/'chart_audit.jsonl').read_text().splitlines()]
    assert records
    assert set(r['stage'] for r in records)<=set(['optimizer_refresh','identity_chart'])
    assert any(r['stage']=='optimizer_refresh' for r in records)
    updates=rows(audited/'updates.csv')
    for epoch,update in enumerate(updates):
        local=[r for r in records if r['epoch']==epoch and r['stage']=='optimizer_refresh']
        assert sum(r['touched_factors'] for r in local)==int(update['factors_linearized'])
        assert sum(r['requested_pose_keys']+r['requested_point_keys'] for r in local)==int(update['refreshed_keys'])
    for record in records:
        assert record['pose_origin']['alignment_rotation_source']=='pose_origin'
        assert record['pose_origin']['touched_factors']==record['touched_factors']
        assert record['touched_factors']==sum(record[k] for k in ['pose_only_factors','point_only_factors','both_factors'])
        for family in record['families'].values():
            assert family['gauge_invariance_max']<1e-9
            assert family['cached_residual_max']<1e-9
            assert sum(family['raw_error_bins'])==family['count']


def test_chart_audit_rejects_models_without_square_root_cache(tmp_path):
    p,_=run(tmp_path/'bad','joint_cached',audit=True)
    assert p.returncode!=0
    assert 'requires conditional' in p.stderr
