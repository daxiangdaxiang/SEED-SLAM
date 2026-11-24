import json
import pytest

from test_native_conditional_replay import rows, run
from test_native_joint_audit import package
from test_native_matched_online import stream


def test_subspace_is_causal_root_fixed_and_descending(tmp_path):
    data = package()
    events = stream(data)
    process, short = run(tmp_path/'short', 'joint_cached_one', events=events,
                         duration=3., lazy=True, extra=['--rigid-subspace'])
    assert process.returncode == 0, process.stderr
    for event in events:
        if event['timestamp_s'] > 3. and event['kind'] == 'observation':
            event['range_m'] += 10.
    process, long = run(tmp_path/'long', 'joint_cached_one', events=events,
                        lazy=True, extra=['--rigid-subspace'])
    assert process.returncode == 0, process.stderr
    for folder in (short, long):
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['interface_solver'] == 'rigid_subspace'
        assert summary['root_displacement'] == 0
        assert summary['consensus_max_m'] < 1e-12
        assert summary['pcg_iterations'] == 0
        assert summary['subspace_steps'] == summary['frames']
        assert summary['truth_used'] is False
        for row in rows(folder/'updates.csv'):
            assert float(row['cost_after']) <= float(row['cost_before'])+1e-10
    prefix = (short/'causal_estimates.csv').read_text().splitlines()
    assert prefix == (long/'causal_estimates.csv').read_text().splitlines()[:len(prefix)]


def test_subspace_rejects_unsupported_mode(tmp_path):
    process, _ = run(tmp_path/'bad', 'joint_cached', extra=['--rigid-subspace'])
    assert process.returncode != 0
    assert 'rigid subspace requires one-step common-past' in process.stderr


def test_budget_audit_checks_every_prefix_not_only_endpoint():
    from experiments.audit_rigid_subspace import budget_audit
    def records(costs):
        return [dict(robot=1, timestamp_s=k, optimizer_bytes=v, control_bytes=0)
                for k, v in enumerate(costs)]
    result = budget_audit(records([10,10]), records([11,1]))
    assert result['candidate_final_bytes'] == 12
    assert result['prefix_exceedances'] == 1
    assert result['max_prefix_excess_bytes'] == 1
    assert budget_audit(records([10,10]),records([5,5]))['prefix_exceedances'] == 0
    with pytest.raises(ValueError):
        budget_audit(records([10]),records([5,5]))


def test_fused_one_step_keeps_entire_causal_and_terminal_state(tmp_path):
    outputs=[]
    for name, extra in [('pcg',[]),('fused',['--fused-one-step'])]:
        process, folder=run(tmp_path/name,'joint_cached_one',lazy=True,extra=extra)
        assert process.returncode==0,process.stderr
        outputs.append(folder)
    for filename in ('final_estimates.csv','causal_estimates.csv'):
        assert (outputs[0]/filename).read_bytes()==(outputs[1]/filename).read_bytes()
    left,right=[json.loads((p/'summary.json').read_text()) for p in outputs]
    assert right['interface_solver']=='fused_one_step_pcg'
    assert right['application_bytes']<left['application_bytes']
    assert right['physical_cost']==left['physical_cost']
    assert right['refreshed_keys']==left['refreshed_keys']
