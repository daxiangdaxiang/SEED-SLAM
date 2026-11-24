// Controlled SE(3) physical-likelihood audit, not a distributed online backend.
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/BetweenFactor.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

using Json = nlohmann::json;
using gtsam::Pose3;
using gtsam::Vector6;
using gtsam::Matrix6;
using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::L;

class ExactBetween : public gtsam::BetweenFactor<Pose3> {
 public:
  using gtsam::BetweenFactor<Pose3>::BetweenFactor;
  gtsam::Vector evaluateError(const Pose3& first, const Pose3& second,
      boost::optional<gtsam::Matrix&> h1 = boost::none,
      boost::optional<gtsam::Matrix&> h2 = boost::none) const override {
    const auto relative = gtsam::traits<Pose3>::Between(first, second, h1, h2);
    Matrix6 local;
    const auto residual = gtsam::traits<Pose3>::Local(measured(), relative,
        boost::none, (h1 || h2) ? &local : nullptr);
    if (h1) *h1 = local*(*h1);
    if (h2) *h2 = local*(*h2);
    return residual;
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<ExactBetween>(*this);
  }
};

struct Measurement {
  gtsam::Key from, to;
  Pose3 value;
  int group;
};

struct Dataset {
  gtsam::Values truth, initial;
  std::vector<Measurement> measurements;
};

Vector6 sigmas(int group, const gtsam::Vector4& q) {
  Vector6 result;
  const double rotation = group == 0 ? .03 : .05;
  const double translation = group == 0 ? .1 : .15;
  result.head<3>().setConstant(rotation*std::sqrt(q[group]));
  result.tail<3>().setConstant(translation*std::sqrt(q[group+1]));
  return result;
}

Dataset generate(unsigned seed, const gtsam::Vector4& q) {
  Dataset data;
  std::mt19937 random(seed);
  std::normal_distribution<double> normal;
  auto draw = [&]() {
    Vector6 v;
    for (int j=0; j<6; ++j) v[j] = normal(random);
    return v;
  };
  for (int robot=0; robot<2; ++robot) for (int k=0; k<12; ++k) {
    const double t = k+.4*robot;
    data.truth.insert(X(12*robot+k), Pose3(gtsam::Rot3::RzRyRx(
        .12*std::sin(.25*t), .08*t, .1*t),
        gtsam::Point3(.25*t+.6*robot,1.2*std::sin(.4*t)+.8*robot,.2*t+.3*robot)));
  }
  for (int j=0; j<4; ++j) data.truth.insert(L(j), Pose3(
      gtsam::Rot3::RzRyRx(.2*j,-.15*j,.3*j), gtsam::Point3(1.+.6*j,j%2 ? .8 : -.8,.5+.5*j)));
  auto add = [&](gtsam::Key from, gtsam::Key to, int group) {
    const auto relative = data.truth.at<Pose3>(from).between(data.truth.at<Pose3>(to));
    data.measurements.push_back({from,to,relative.retract(sigmas(group,q).cwiseProduct(draw())),group});
  };
  for (int robot=0; robot<2; ++robot) for (int k=0; k<12; ++k) {
    const auto key = X(12*robot+k);
    if (k) add(X(12*robot+k-1),key,0);
    for (int j=0; j<4; ++j) add(key,L(j),2);
  }
  for (auto key : data.truth.keys()) {
    Vector6 perturb = draw();
    perturb.head<3>() *= .04;
    perturb.tail<3>() *= .15;
    data.initial.insert(key, key == X(0) ? data.truth.at<Pose3>(key) :
                        data.truth.at<Pose3>(key).retract(perturb));
  }
  return data;
}

gtsam::NonlinearFactorGraph graphFor(const Dataset& data, const gtsam::Vector4& q) {
  gtsam::NonlinearFactorGraph graph;
  for (const auto& m : data.measurements) graph.emplace_shared<ExactBetween>(
      m.from,m.to,m.value,gtsam::noiseModel::Diagonal::Sigmas(sigmas(m.group,q)));
  graph.emplace_shared<gtsam::NonlinearEquality<Pose3>>(X(0),data.initial.at<Pose3>(X(0)));
  return graph;
}

gtsam::Values optimize(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial) {
  gtsam::LevenbergMarquardtParams params;
  params.maxIterations = 100;
  params.relativeErrorTol = params.absoluteErrorTol = 1e-12;
  return gtsam::LevenbergMarquardtOptimizer(graph,initial,params).optimize();
}

double checkDerivatives(const Dataset& data) {
  double maximum = 0.;
  for (const auto& m : data.measurements) {
    ExactBetween factor(m.from,m.to,m.value,gtsam::noiseModel::Unit::Create(6));
    const auto& a = data.initial.at<Pose3>(m.from);
    const auto& b = data.initial.at<Pose3>(m.to);
    gtsam::Matrix first, second;
    factor.evaluateError(a,b,first,second);
    for (int j=0; j<6; ++j) {
      Vector6 step = Vector6::Zero(); step[j] = 1e-6;
      const gtsam::Vector da = (factor.evaluateError(a.retract(step),b)-
          factor.evaluateError(a.retract(-step),b))/(2e-6);
      const gtsam::Vector db = (factor.evaluateError(a,b.retract(step))-
          factor.evaluateError(a,b.retract(-step)))/(2e-6);
      maximum = std::max({maximum,(da-first.col(j)).norm(),(db-second.col(j)).norm()});
    }
  }
  return maximum;
}

Json linearization(const Dataset& data, const gtsam::Values& state) {
  const auto graph = graphFor(data,gtsam::Vector4::Ones());
  std::map<gtsam::Key,int> offsets;
  int n = 0;
  for (auto key : state.keys()) if (key != X(0)) { offsets[key] = n; n += 6; }
  Json rows = Json::array(), values = Json::array(), groups = Json::array();
  for (std::size_t i=0; i<data.measurements.size(); ++i) {
    const auto linear = graph[i]->linearize(state);
    const auto ab = linear->jacobian();
    gtsam::Matrix full = gtsam::Matrix::Zero(6,n);
    int column = 0;
    for (auto key : linear->keys()) {
      if (key != X(0)) full.middleCols(offsets.at(key),6) = ab.first.middleCols(column,6);
      column += 6;
    }
    for (int r=0; r<6; ++r) {
      Json row = Json::array();
      for (int c=0; c<n; ++c) row.push_back(full(r,c));
      rows.push_back(row); values.push_back(ab.second[r]);
      groups.push_back(data.measurements[i].group+(r>=3));
    }
  }
  return Json{{"schema","dranppr_pose3_physical_calibration_v1"},
              {"a",rows},{"b",values},{"groups",groups},{"truth_used_in_fit",false}};
}

Json quality(const Dataset& data, const gtsam::Values& state) {
  double poseTranslation=0.,poseRotation=0.,objectTranslation=0.,objectRotation=0.;
  int poses=0,objects=0;
  for (auto key : state.keys()) {
    const auto& truth = data.truth.at<Pose3>(key);
    const auto& value = state.at<Pose3>(key);
    const double position = (truth.translation()-value.translation()).squaredNorm();
    const double rotation = gtsam::Rot3::Logmap(truth.rotation().between(value.rotation())).squaredNorm();
    if (gtsam::Symbol(key).chr() == 'x') { poseTranslation+=position; poseRotation+=rotation; ++poses; }
    else { objectTranslation+=position; objectRotation+=rotation; ++objects; }
  }
  return Json{{"pose_translation_rmse_m",std::sqrt(poseTranslation/poses)},
      {"pose_rotation_rmse_rad",std::sqrt(poseRotation/poses)},
      {"object_translation_rmse_m",std::sqrt(objectTranslation/objects)},
      {"object_rotation_rmse_rad",std::sqrt(objectRotation/objects)}};
}

Json physicalInput(const Dataset& data) {
  auto matrix = [](const Pose3& pose) {
    Json rows = Json::array();
    const auto value = pose.matrix();
    for (int r=0; r<4; ++r) {
      Json row = Json::array();
      for (int c=0; c<4; ++c) row.push_back(value(r,c));
      rows.push_back(row);
    }
    return rows;
  };
  Json initial = Json::array(), measurements = Json::array();
  for (auto key : data.initial.keys()) initial.push_back(
      {{"key",key},{"pose",matrix(data.initial.at<Pose3>(key))}});
  for (const auto& m : data.measurements) measurements.push_back(
      {{"from",m.from},{"to",m.to},{"value",matrix(m.value)},{"group",m.group}});
  return Json{{"initial",initial},{"measurements",measurements}};
}

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 8) throw std::invalid_argument("output seed nominal|heterogeneous [four variance scales]");
    const std::filesystem::path output(argv[1]);
    if (std::filesystem::exists(output)) throw std::invalid_argument("output exists");
    const std::string regime(argv[3]);
    if (regime != "nominal" && regime != "heterogeneous") throw std::invalid_argument("invalid regime");
    gtsam::Vector4 truth = gtsam::Vector4::Ones(), used = truth;
    if (regime == "heterogeneous") truth << .1,4.,9.,.25;
    if (argc == 8) for (int k=0; k<4; ++k) used[k] = std::stod(argv[4+k]);
    if (!used.allFinite() || (used.array() <= 0.).any()) throw std::invalid_argument("invalid variances");
    const unsigned seed = std::stoul(argv[2]);
    const auto training = generate(seed,truth), testing = generate(seed+1000000,truth);
    const auto trainGraph = graphFor(training,gtsam::Vector4::Ones()), testGraph = graphFor(testing,used);
    const auto trainState = optimize(trainGraph,training.initial), testState = optimize(testGraph,testing.initial);
    const double derivative = std::max(checkDerivatives(training),checkDerivatives(testing));
    if (derivative > 1e-6) throw std::runtime_error("SE3 derivative audit failed");
    const double root = testState.at<Pose3>(X(0)).localCoordinates(testing.initial.at<Pose3>(X(0))).norm();
    if (root > 1e-12) throw std::runtime_error("root gauge changed");
    gtsam::Vector3 tmin = gtsam::Vector3::Constant(1e30), tmax = -tmin;
    gtsam::Vector3 rmin = tmin, rmax = tmax;
    for (auto key : testing.truth.keys()) {
      const auto& value = testing.truth.at<Pose3>(key);
      const auto rotation = gtsam::Rot3::Logmap(value.rotation());
      tmin = tmin.cwiseMin(value.translation()); tmax = tmax.cwiseMax(value.translation());
      rmin = rmin.cwiseMin(rotation); rmax = rmax.cwiseMax(rotation);
    }
    Json summary{{"seed",seed},{"regime",regime},{"quality",quality(testing,testState)},
      {"jacobian_max_error",derivative},{"root_displacement",root},
      {"training_test_measurements_differ",training.measurements[0].value.localCoordinates(testing.measurements[0].value).norm()>0.},
      {"pose_tangent_dimension",6},{"object_tangent_dimension",6},
      {"training_test_noise_independent",true},{"noise_fit_uses_truth",false},
      {"scope","Synthetic full SE3 batch control, known identities, common perturbed truth initialization; not online or real data"}};
    for (int k=0; k<3; ++k) {
      summary["truth_translation_extent"].push_back((tmax-tmin)[k]);
      summary["truth_rotation_extent"].push_back((rmax-rmin)[k]);
    }
    for (int k=0; k<4; ++k) { summary["generating_scales"].push_back(truth[k]); summary["used_scales"].push_back(used[k]); }
    std::filesystem::create_directories(output);
    std::ofstream(output/"training.json") << linearization(training,trainState).dump();
    std::ofstream(output/"test_input.json") << physicalInput(testing).dump();
    std::ofstream(output/"summary.json") << summary.dump(2) << '\n';
    std::cout << summary.dump() << '\n';
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
