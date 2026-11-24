#include <DPGO/NativeBetweenPose2Factor.h>
#include <DPGO/NativeLocalOptimizer.h>
#include <DPGO/NativeSchurModel.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>
#include <Eigen/Cholesky>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using Json = nlohmann::json;
using namespace DPGO;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double number(const Json& row, const std::string& key) {
  require(row.at(key).is_number(), "nonnumeric " + key);
  const double value = row.at(key).get<double>();
  require(std::isfinite(value), "nonfinite " + key);
  return value;
}

int identifier(const Json& row, const std::string& key) {
  require(row.at(key).is_number_integer(), "noninteger " + key);
  const int value = row.at(key).get<int>();
  require(value >= 0, "negative " + key);
  return value;
}

double gradientNorm(const gtsam::NonlinearFactorGraph& graph,
                    const gtsam::Values& values, gtsam::Key root) {
  double squared = 0.;
  for (const auto& item : graph.linearize(values)->gradientAtZero()) {
    if (item.first != root) squared += item.second.squaredNorm();
  }
  return std::sqrt(squared);
}

double directionalDifference(const gtsam::NonlinearFactorGraph& graph,
                             const gtsam::Values& values, gtsam::Key root) {
  auto direction = graph.linearize(values)->gradientAtZero();
  direction.at(root).setZero();
  const double norm = direction.norm();
  if (norm == 0.) return 0.;
  direction *= 1./norm;
  const double epsilon = 1e-6;
  return (graph.error(values.retract(epsilon*direction))-
          graph.error(values.retract(-epsilon*direction)))/(2*epsilon);
}

Json objectCovariance(const NativeModelUpdate& factors, const gtsam::Values& values,
                      const std::vector<gtsam::Key>& privateKeys, gtsam::Key root) {
  auto input = factors;
  input.newValues = values;
  IncrementalLocalModel model;
  model.update(input);
  NativeSchurModel response(model.linearSystem({}, {root}), privateKeys);
  const auto size = response.gradient().size();
  const gtsam::Matrix identity = gtsam::Matrix::Identity(size, size);
  const gtsam::Matrix information = response.apply(identity);
  gtsam::Matrix covariance(size, size);
  double inverseResidual = 0.;
  if (size > 0) {
    Eigen::LDLT<gtsam::Matrix> factor(information);
    require(factor.info() == Eigen::Success && factor.vectorD().allFinite() &&
            (factor.vectorD().array() > 0.).all(), "object information is not positive definite");
    covariance = factor.solve(identity);
    require(covariance.allFinite(), "object covariance is nonfinite");
    inverseResidual = (information*covariance-identity).norm()/identity.norm();
    require(inverseResidual <= 1e-8, "object covariance inverse residual is too large");
  }
  Json result{{"truth_used", false}, {"fixed_root_index", gtsam::Symbol(root).index()},
      {"model", "Exact-Jacobian GN covariance conditional on fixed physical Gaussian noise and local root; not a calibrated posterior"},
      {"information_inverse_relative_residual", inverseResidual},
      {"object_ids", Json::array()}, {"means", Json::array()}, {"covariance", Json::array()}};
  for (const auto key : response.boundaryKeys()) {
    require(response.boundaryDimensions().at(key) == 2, "non-Point2 covariance boundary");
    const auto& point = values.at<gtsam::Point2>(key);
    result["object_ids"].push_back(gtsam::Symbol(key).index());
    result["means"].push_back({point.x(), point.y()});
  }
  for (Eigen::Index i = 0; i < size; ++i) {
    Json row = Json::array();
    for (Eigen::Index j = 0; j < size; ++j) row.push_back(covariance(i,j));
    result["covariance"].push_back(std::move(row));
  }
  return result;
}

int main(int argc, char** argv) {
  try {
    require(argc == 6, "usage: native-local-audit events.jsonl state.json output robot duration");
    const int robot = std::stoi(argv[4]);
    const double duration = std::stod(argv[5]);
    require(robot >= 0 && std::isfinite(duration) && duration > 0., "invalid robot/duration");
    const std::filesystem::path output(argv[3]);
    require(!std::filesystem::exists(output), "output already exists");
    std::ifstream stream(argv[1]), snapshot(argv[2]);
    require(stream.good() && snapshot.good(), "cannot read inputs");
    Json states;
    snapshot >> states;
    require(states.is_array(), "state must be an array");
    NativeModelUpdate pinned, exact;
    std::map<int, double> poseTimes;
    std::set<int> sourceRows;
    std::set<gtsam::Key> expectedKeys, receivedKeys;
    std::vector<gtsam::Key> privateKeys;
    const auto root = gtsam::Symbol('x', 0);
    const auto polar = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05, .1));
    double first = -1., previous = -1.;
    int lastPose = -1;
    std::string line;
    while (std::getline(stream, line)) {
      const auto packet = Json::parse(line);
      const double t = number(packet, "timestamp_s");
      if (first < 0.) first = t;
      require(t >= previous, "time decreased");
      previous = t;
      if (t-first > duration) break;
      if (identifier(packet, "robot") != robot) continue;
      require(number(packet, "available_time_s") == t &&
              number(packet, "support_max_time_s") <= t, "noncausal packet");
      const int index = identifier(packet, "pose_index");
      const auto key = gtsam::Symbol('x', index);
      if (packet.at("kind") == "pose") {
        require(index == lastPose+1, "nonconsecutive pose");
        if (index == 0) {
          require(packet.at("delta").is_null() && packet.at("from_pose_index").is_null(), "invalid root parent");
        } else {
          require(identifier(packet, "from_pose_index") == lastPose, "invalid parent");
          const double dt = number(packet, "duration_s");
          require(dt > 0. && std::abs(t-poseTimes.at(lastPose)-dt) < 1e-6, "invalid duration");
          const auto& delta = packet.at("delta");
          require(delta.is_array() && delta.size() == 3, "invalid odometry delta");
          const gtsam::Vector3 d(delta[0].get<double>(), delta[1].get<double>(), delta[2].get<double>());
          require(d.allFinite(), "nonfinite odometry");
          const auto measurement = gtsam::Pose2(d[0], d[1], d[2]);
          const auto noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector3(.1, .1, .05)*std::sqrt(dt)).eval());
          const NativeFactorId id = 2*static_cast<NativeFactorId>(index);
          pinned.addFactors[id] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
              gtsam::Symbol('x', lastPose), key, measurement, noise);
          exact.addFactors[id] = boost::make_shared<NativeBetweenPose2Factor>(
              gtsam::Symbol('x', lastPose), key, measurement, noise);
          privateKeys.push_back(key);
        }
        poseTimes[index] = t;
        expectedKeys.insert(key);
        lastPose = index;
      } else {
        require(packet.at("kind") == "observation", "unsupported packet kind");
        require(index == lastPose && poseTimes.at(index) == t, "observation timestamp mismatch");
        const int source = identifier(packet, "source_row");
        require(sourceRows.insert(source).second, "duplicate observation");
        const auto landmark = gtsam::Symbol('l', identifier(packet, "landmark_subject"));
        const double range = number(packet, "range_m"), bearing = number(packet, "bearing_rad");
        require(range > 0., "nonpositive range");
        const NativeFactorId id = 2*static_cast<NativeFactorId>(source)+1;
        pinned.addFactors[id] = boost::make_shared<Observation>(key, landmark, gtsam::Rot2(bearing), range, polar);
        exact.addFactors[id] = pinned.addFactors.at(id);
        expectedKeys.insert(landmark);
      }
    }
    require(lastPose >= 1, "need at least two poses");
    for (const auto& row : states) {
      if (identifier(row, "robot") != robot) continue;
      const int index = identifier(row, "index");
      const std::string kind = row.at("kind");
      require(kind == "pose" || kind == "landmark", "unsupported state kind");
      const auto key = gtsam::Symbol(kind == "pose" ? 'x' : 'l', index);
      require(receivedKeys.insert(key).second, "duplicate state keys");
      const double x = number(row, "x"), y = number(row, "y");
      if (kind == "pose") {
        require(poseTimes.count(index) && number(row, "timestamp_s") == poseTimes.at(index), "state timestamp mismatch");
        pinned.newValues.insert(key, gtsam::Pose2(x, y, number(row, "yaw")));
      } else pinned.newValues.insert(key, gtsam::Point2(x, y));
    }
    require(receivedKeys == expectedKeys, "state and prefix keys disagree");
    require(pinned.newValues.at<gtsam::Pose2>(root).equals(gtsam::Pose2(), 1e-10), "nonidentity root gauge");
    exact.newValues = pinned.newValues;
    gtsam::NonlinearFactorGraph pinnedGraph, exactGraph;
    for (const auto& factor : pinned.addFactors) pinnedGraph.push_back(factor.second);
    for (const auto& factor : exact.addFactors) exactGraph.push_back(factor.second);
    const double initialCost = pinnedGraph.error(pinned.newValues);
    require(std::isfinite(initialCost) && std::abs(initialCost-exactGraph.error(pinned.newValues)) < 1e-10,
            "physical objectives disagree");
    std::filesystem::create_directories(output);
    Json summary{{"truth_used", false}, {"poses", poseTimes.size()}, {"observations", sourceRows.size()},
                 {"initial_cost", initialCost}, {"initial_exact_gradient_norm", gradientNorm(exactGraph, exact.newValues, root)},
                 {"initial_directional_fd", directionalDifference(exactGraph, exact.newValues, root)},
                 {"scope", "Frozen local prefix; fixed Gaussian likelihood; not online or distributed"},
                 {"runs", Json::array()}};
    for (const std::string method : {"pinned_schur", "exact_schur", "exact_full", "pinned_lm", "exact_lm"}) {
      const bool complete = method.find("exact") == 0;
      const auto& input = complete ? exact : pinned;
      const auto& physical = complete ? exactGraph : pinnedGraph;
      gtsam::Values final;
      std::size_t iterations = 0;
      std::string status;
      const auto start = std::chrono::steady_clock::now();
      NativeLocalResult result;
      if (method.find("lm") != std::string::npos) {
        auto graph = physical;
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
            root, gtsam::Pose2(), gtsam::noiseModel::Constrained::All(3));
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = 100;
        params.absoluteErrorTol = params.relativeErrorTol = 1e-12;
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, input.newValues, params);
        final = optimizer.optimize();
        iterations = optimizer.iterations();
        status = "batch_optimizer_returned";
      } else {
        IncrementalLocalModel model;
        model.update(input);
        auto keys = privateKeys;
        if (method == "exact_full") keys = model.linearSystem({}, {root}).keys;
        result = optimizeNativeLocal(model, keys, {root});
        final = model.bases();
        iterations = result.trace.size();
        status = result.status;
      }
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      const double cost = pinnedGraph.error(final);
      require(std::isfinite(cost) && cost <= initialCost+1e-8, "physical objective increased");
      require(final.at<gtsam::Pose2>(root).equals(gtsam::Pose2(), 1e-10), "solver changed root gauge");
      const double exactGradient = gradientNorm(exactGraph, final, root);
      require(std::isfinite(exactGradient), "nonfinite exact gradient");
      Json run{{"method", method}, {"cost", cost}, {"exact_gradient_norm", exactGradient},
               {"physical_directional_fd", directionalDifference(exactGraph, final, root)},
               {"pinned_gradient_norm", gradientNorm(pinnedGraph, final, root)},
               {"iterations", iterations}, {"seconds", seconds}, {"status", status}};
      const auto covarianceStart = std::chrono::steady_clock::now();
      const auto covariance = objectCovariance(exact, final, privateKeys, root);
      run["covariance_seconds"] = std::chrono::duration<double>(
          std::chrono::steady_clock::now()-covarianceStart).count();
      run["covariance_inverse_relative_residual"] = covariance.at("information_inverse_relative_residual");
      std::ofstream covarianceFile(output/(method+"_object_covariance.json"));
      covarianceFile.exceptions(std::ios::failbit|std::ios::badbit);
      covarianceFile << covariance.dump(2) << '\n';
      covarianceFile.close();
      std::ofstream estimates(output/(method+"_estimates.csv")), trace(output/(method+"_trace.csv"));
      estimates.exceptions(std::ios::failbit|std::ios::badbit);
      trace.exceptions(std::ios::failbit|std::ios::badbit);
      estimates << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
      for (auto key : final.keys()) {
        const gtsam::Symbol symbol(key);
        if (symbol.chr() == 'x') {
          const auto& p = final.at<gtsam::Pose2>(key);
          estimates << robot << ",pose," << symbol.index() << ',' << poseTimes.at(symbol.index()) << ','
                    << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        } else {
          const auto& p = final.at<gtsam::Point2>(key);
          estimates << robot << ",landmark," << symbol.index() << ",," << p.x() << ',' << p.y() << ",\n";
        }
      }
      trace << std::setprecision(17) << "revision,cost_before,cost_after,model_gradient_norm,alpha,step_norm,predicted_decrease,linear_residual,cost_evaluations,factors_linearized\n";
      for (const auto& row : result.trace) {
        trace << row.revision << ',' << row.costBefore << ',' << row.costAfter << ',' << row.modelGradientNorm
              << ',' << row.alpha << ',' << row.stepNorm << ',' << row.predictedDecrease << ','
              << row.linearResidual << ',' << row.costEvaluations << ',' << row.factorsLinearized << '\n';
      }
      estimates.close();
      trace.close();
      summary["runs"].push_back(run);
      std::ofstream record(output/"summary.json");
      record.exceptions(std::ios::failbit|std::ios::badbit);
      record << summary.dump(2) << '\n';
      std::cout << run.dump() << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
