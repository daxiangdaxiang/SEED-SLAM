import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1]/'experiments/audit_native_checkpoint.py'
    assert path.is_file(), 'native completion harness not implemented'
    spec = importlib.util.spec_from_file_location('audit_native_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_to_typed_state_keeps_all_robot_rows(tmp_path):
    module = load_module()
    source = tmp_path/'states.csv'
    source.write_text('robot,kind,index,timestamp_s,x,y,yaw\n1,pose,0,100,0,0,0\n2,landmark,6,,2,3,\n')
    assert module.typed_state(source) == [
        dict(robot=1, kind='pose', index=0, timestamp_s=100., x=0., y=0., yaw=0.),
        dict(robot=2, kind='landmark', index=6, x=2., y=3.)]


@pytest.mark.parametrize('value', ['nan', 'inf'])
def test_rejects_nonfinite_state_before_launch(tmp_path, value):
    module = load_module()
    source = tmp_path/'states.csv'
    source.write_text(f'robot,kind,index,timestamp_s,x,y,yaw\n1,pose,0,100,{value},0,0\n')
    with pytest.raises(ValueError, match='nonfinite'):
        module.typed_state(source)
