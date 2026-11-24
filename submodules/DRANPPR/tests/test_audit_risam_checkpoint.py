import sys
from pathlib import Path
import gtsam
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from audit_risam_checkpoint import robustify_observations


def test_final_sig_objective_equals_geman_mcclure_and_preserves_odometry():
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    values.insert(0,gtsam.Pose2())
    values.insert(1,gtsam.Pose2(.8,0.,0.))
    values.insert(2,np.array([2.,.2]))
    odom = gtsam.BetweenFactorPose2(0,1,gtsam.Pose2(1.,0.,0.),
                                  gtsam.noiseModel.Isotropic.Sigma(3,.1))
    graph.push_back(odom)
    obs = gtsam.BearingRangeFactor2D(1,2,gtsam.Rot2(.1),1.,
                                   gtsam.noiseModel.Diagonal.Sigmas(np.array([.05,.1])))
    graph.push_back(obs)
    c = 1.899741155814062
    robust = robustify_observations(graph,c)
    r2 = 2*obs.error(values)
    expected = .5*c*c*r2/(c*c+r2)
    assert robust.at(0).error(values)==pytest.approx(odom.error(values),abs=1e-12)
    assert robust.at(1).error(values)==pytest.approx(expected,abs=1e-12)
    linear = robust.at(1).linearize(values)
    a,_ = linear.jacobian()
    plain_a,_ = obs.linearize(values).jacobian()
    assert np.sum(a*a)/np.sum(plain_a*plain_a)==pytest.approx(c**4/(c*c+r2)**2,abs=1e-12)
