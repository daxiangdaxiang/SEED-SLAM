import csv
import json

import pytest

from experiments.audit_conditional_replay import compare_runs


def write_run(path, shift=0., accepted=1):
    path.mkdir()
    (path/'summary.json').write_text(json.dumps(dict(frames=1, physical_factors=3,
        prefix_physical_factors=2, physical_cost=1.+shift, application_bytes=100,
        service_seconds=2., setup_seconds=.1)))
    for filename in ['causal_estimates.csv', 'final_estimates.csv']:
        with (path/filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['robot','kind','index','x','y','yaw'])
            writer.writeheader()
            writer.writerow(dict(robot=1,kind='pose',index=0,x=shift,y=0.,yaw=0.))
    with (path/'updates.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['robot','timestamp_s','accepted','cost_before',
            'cost_after','pcg_iterations','optimizer_bytes','control_bytes','refreshed_keys','factors_linearized'])
        writer.writeheader()
        writer.writerow(dict(robot=1,timestamp_s=1.,accepted=accepted,cost_before=2.,cost_after=1.+shift,
            pcg_iterations=1,optimizer_bytes=60,control_bytes=40,refreshed_keys=2,factors_linearized=3))


def test_comparison_reports_numeric_and_discrete_differences(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    write_run(a)
    write_run(b, .01, 0)
    result = compare_runs(a,b)
    assert result['final_state_max_abs'] == pytest.approx(.01)
    assert result['causal_state_max_abs'] == pytest.approx(.01)
    assert result['cost_after_max_abs'] == pytest.approx(.01)
    assert result['frame_mismatches']['accepted'] == 1
    assert result['frame_mismatches']['optimizer_bytes'] == 0
    assert result['service_speedup'] == 1.
    assert result['graph_equal'] is True
    with (b/'causal_estimates.csv').open('a') as f:
        f.write('1,pose,1,0,0,0\n')
    with pytest.raises(ValueError, match='rows'):
        compare_runs(a,b)
