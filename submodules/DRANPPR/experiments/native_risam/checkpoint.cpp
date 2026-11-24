#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>
#include <risam/GraduatedKernel.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

using Json = nlohmann::json;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double number(const Json& row, const std::string& field) {
  require(row.at(field).is_number(), "nonnumeric " + field);
  double value = row.at(field).get<double>();
  require(std::isfinite(value), "nonfinite " + field);
  return value;
}

int identifier(const Json& row, const std::string& field) {
  require(row.at(field).is_number_integer(), "noninteger " + field);
  int value = row.at(field).get<int>();
  require(value >= 0, "negative " + field);
  return value;
}

struct ObservationRecord {
  int row, pose, subject;
  double time;
  boost::shared_ptr<Observation> plain, target;
};

int main(int argc, char** argv) {
  try {
    require(argc == 7, "usage: native-risam-checkpoint events.jsonl state.json output robot duration gaussian|sig_final");
    const int robot = std::stoi(argv[4]);
    const double duration = std::stod(argv[5]);
    const std::string mode(argv[6]);
    require(robot >= 0 && std::isfinite(duration) && duration > 0, "invalid robot or duration");
    require(mode == "gaussian" || mode == "sig_final", "unknown target objective");
    const std::filesystem::path output(argv[3]);
    require(!std::filesystem::exists(output), "output already exists");
    std::ifstream stream(argv[1]), snapshot(argv[2]);
    require(stream.good() && snapshot.good(), "cannot read inputs");
    Json state;
    snapshot >> state;
    require(state.is_array(), "state must be an array");
    const double shape = risam::SIGKernel::shapeParamFromInfThresh(.1, 2, .95);
    const auto estimator = gtsam::noiseModel::mEstimator::GemanMcClure::Create(shape);
    const auto polar = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05, .1));
    const gtsam::SharedNoiseModel robust = gtsam::noiseModel::Robust::Create(estimator, polar);
    const risam::SIGKernel kernel(shape, risam::SIGKernel::MuUpdateStrategy::STABLE,
                                 risam::SIGKernel::MuInitIncrementStrategy::EQUAL_5);
    gtsam::NonlinearFactorGraph graph, odometry, observations;
    std::vector<ObservationRecord> records;
    std::map<int, double> poseTimes;
    std::set<int> sourceRows;
    std::set<gtsam::Key> expectedKeys;
    int lastPose = -1;
    double first = -1, previousTime = -1;
    std::string line;
    while (std::getline(stream, line)) {
      const auto packet = Json::parse(line);
      const double t = number(packet, "timestamp_s");
      if (first < 0) first = t;
      require(t >= previousTime, "time decreased");
      previousTime = t;
      if (t-first > duration) break;
      if (identifier(packet, "robot") != robot) continue;
      require(number(packet, "available_time_s") == t &&
              number(packet, "support_max_time_s") <= t, "noncausal packet");
      const int index = identifier(packet, "pose_index");
      const gtsam::Key key = gtsam::Symbol('x', index);
      const std::string kind = packet.at("kind");
      if (kind == "pose") {
        require(index == lastPose+1, "nonconsecutive pose");
        if (index == 0) {
          require(packet.at("delta").is_null() && packet.at("from_pose_index").is_null(), "invalid root parent");
          graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
              key, gtsam::Pose2(), gtsam::noiseModel::Constrained::All(3));
        } else {
          require(identifier(packet, "from_pose_index") == lastPose, "invalid parent");
          const double dt = number(packet, "duration_s");
          require(dt > 0 && std::abs(t-poseTimes.at(lastPose)-dt) < 1e-6, "invalid odometry duration");
          const auto& delta = packet.at("delta");
          require(delta.is_array() && delta.size() == 3, "invalid odometry delta");
          const gtsam::Vector3 d(delta[0].get<double>(), delta[1].get<double>(), delta[2].get<double>());
          require(d.allFinite(), "nonfinite odometry delta");
          const auto noise = gtsam::noiseModel::Diagonal::Sigmas(
              (gtsam::Vector3(.1, .1, .05)*std::sqrt(dt)).eval());
          odometry.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
              gtsam::Symbol('x', lastPose), key, gtsam::Pose2(d[0], d[1], d[2]), noise);
          graph.push_back(odometry.back());
        }
        expectedKeys.insert(key);
        poseTimes[index] = t;
        lastPose = index;
      } else {
        require(kind == "observation", "unsupported packet kind");
        require(index == lastPose && poseTimes.at(index) == t, "observation timestamp or parent mismatch");
        const int row = identifier(packet, "source_row"), subject = identifier(packet, "landmark_subject");
        require(sourceRows.insert(row).second, "duplicate observation");
        const double range = number(packet, "range_m"), bearing = number(packet, "bearing_rad");
        require(range > 0, "nonpositive range");
        const gtsam::Key landmark = gtsam::Symbol('l', subject);
        auto plain = boost::make_shared<Observation>(key, landmark, gtsam::Rot2(bearing), range, polar);
        auto target = boost::make_shared<Observation>(key, landmark, gtsam::Rot2(bearing), range,
                                                     mode == "sig_final" ? robust : polar);
        records.push_back({row, index, subject, t, plain, target});
        observations.push_back(target);
        graph.push_back(target);
        expectedKeys.insert(landmark);
      }
    }
    require(lastPose >= 0, "empty robot prefix");
    gtsam::Values initial;
    std::set<gtsam::Key> receivedKeys;
    for (const auto& row : state) {
      if (identifier(row, "robot") != robot) continue;
      const int index = identifier(row, "index");
      const std::string kind = row.at("kind");
      require(kind == "pose" || kind == "landmark", "unsupported state kind");
      const gtsam::Key key = gtsam::Symbol(kind == "pose" ? 'x' : 'l', index);
      require(receivedKeys.insert(key).second, "duplicate state keys");
      const double x = number(row, "x"), y = number(row, "y");
      if (kind == "pose") {
        require(poseTimes.count(index) && number(row, "timestamp_s") == poseTimes.at(index),
                "state timestamp mismatch");
        initial.insert(key, gtsam::Pose2(x, y, number(row, "yaw")));
      } else initial.insert(key, gtsam::Point2(x, y));
    }
    require(receivedKeys == expectedKeys, "state and prefix keys disagree");
    const gtsam::Key root = gtsam::Symbol('x', 0);
    require(initial.at<gtsam::Pose2>(root).equals(gtsam::Pose2(), 1e-10), "nonidentity root gauge");
    gtsam::NonlinearFactorGraph physical = odometry;
    physical.push_back(observations);
    auto metrics = [&](const gtsam::Values& values) {
      const auto gradient = physical.linearize(values)->gradientAtZero();
      double norm2 = 0;
      for (const auto& item : gradient) if (item.first != root) norm2 += item.second.squaredNorm();
      const double cost = graph.error(values);
      require(std::isfinite(cost) && std::isfinite(norm2), "nonfinite objective or gradient");
      return Json{{"cost", cost}, {"odometry_cost", odometry.error(values)},
                  {"observation_cost", observations.error(values)}, {"gradient_norm", std::sqrt(norm2)}};
    };
    const auto before = metrics(initial);
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = 100;
    params.relativeErrorTol = 1e-10;
    params.absoluteErrorTol = 1e-10;
    const auto start = std::chrono::steady_clock::now();
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    const auto final = optimizer.optimize();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto after = metrics(final);
    require(after.at("cost").get<double>() <= before.at("cost").get<double>()+1e-8, "completion increased objective");
    require(final.at<gtsam::Pose2>(root).equals(gtsam::Pose2(), 1e-10), "completion changed root gauge");
    std::filesystem::create_directories(output);
    double maxCostDiff = 0, maxWeightDiff = 0;
    auto write = [&](const std::string& label, const gtsam::Values& values) {
      std::ofstream states(output/(label+"_estimates.csv")), weights(output/(label+"_weights.csv"));
      require(states.good() && weights.good(), "cannot create outputs");
      states << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
      weights << std::setprecision(17)
              << "source_row,pose_index,subject,timestamp_s,bearing_error,range_error,mahalanobis2,gm_weight\n";
      for (const auto& key : values.keys()) {
        const gtsam::Symbol symbol(key);
        if (symbol.chr() == 'x') {
          const auto& p = values.at<gtsam::Pose2>(key);
          require(gtsam::Vector3(p.x(), p.y(), p.theta()).allFinite(), "nonfinite final pose");
          states << robot << ",pose," << symbol.index() << ',' << poseTimes.at(symbol.index()) << ','
                 << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        } else {
          const auto& p = values.at<gtsam::Point2>(key);
          require(p.allFinite(), "nonfinite final landmark");
          states << robot << ",landmark," << symbol.index() << ",," << p.x() << ',' << p.y() << ",\n";
        }
      }
      for (const auto& record : records) {
        const auto residual = record.plain->unwhitenedError(values);
        const double r2 = 2*record.plain->error(values), weight = estimator->weight(std::sqrt(r2));
        if (mode == "sig_final") {
          // SIGKernel::error is already the scalar loss, despite its name.
          maxCostDiff = std::max(maxCostDiff, std::abs(record.target->error(values)
              -kernel.error(std::sqrt(r2), 1.)));
          maxWeightDiff = std::max(maxWeightDiff, std::abs(weight-kernel.weight(std::sqrt(r2), 1.)));
        }
        weights << record.row << ',' << record.pose << ',' << record.subject << ',' << record.time << ','
                << residual[0] << ',' << residual[1] << ',' << r2 << ',' << weight << '\n';
      }
    };
    write("before", initial);
    write("after", final);
    Json summary{{"robot", robot}, {"mode", mode}, {"shape", shape}, {"truth_used", false},
                 {"poses", poseTimes.size()}, {"observations", records.size()}, {"before", before}, {"after", after},
                 {"lm_iterations", optimizer.iterations()}, {"lm_seconds", seconds},
                 {"official_sig_cost_max_difference", maxCostDiff}, {"official_sig_weight_max_difference", maxWeightDiff},
                 {"scope", "Frozen-data batch local completion, not online or a global optimum certificate"}};
    std::ofstream summaryFile(output/"summary.json");
    require(summaryFile.good(), "cannot create summary");
    summaryFile << summary.dump(2) << '\n';
    std::cout << summary.dump(2) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
