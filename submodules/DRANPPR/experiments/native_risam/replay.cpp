#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>
#include <risam/GraduatedFactor.h>
#include <risam/RISAM.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

void require(bool valid, const std::string& message) {
  if (!valid) throw std::runtime_error(message);
}

double number(const Json& j, const std::string& key) {
  require(j.at(key).is_number(), "nonnumeric " + key);
  double value = j.at(key).get<double>();
  require(std::isfinite(value), "nonfinite " + key);
  return value;
}

int identifier(const Json& j, const std::string& key) {
  require(j.at(key).is_number_integer(), "noninteger " + key);
  int value = j.at(key).get<int>();
  require(value >= 0, "negative " + key);
  return value;
}

double millis(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct LocalRobot {
  std::unique_ptr<gtsam::ISAM2> gaussian;
  std::unique_ptr<risam::RISAM> robust;
  gtsam::NonlinearFactorGraph physical;
  gtsam::Values values;
  gtsam::Pose2 currentPose;
  int poseIndex{-1};
  double poseTime{0};
  std::map<int, double> poseTimes;
  std::set<int> landmarks, sources;

  explicit LocalRobot(const std::string& mode) {
    gtsam::ISAM2Params common;
    common.relinearizeThreshold = 1e-4;
    common.relinearizeSkip = 1;
    common.factorization = gtsam::ISAM2Params::QR;
    common.optimizationParams = gtsam::ISAM2GaussNewtonParams(1e-3);
    if (mode == "isam2") {
      gaussian = std::make_unique<gtsam::ISAM2>(common);
    } else {
      risam::RISAM::Parameters params;
      if (mode == "risam_gn") {
        params.isam2_params = common;
        params.optimization_params = std::nullopt;
      }
      robust = std::make_unique<risam::RISAM>(params);
    }
  }

  Json push(const std::vector<Json>& frame) {
    const auto start = Clock::now();
    const double t = number(frame.front(), "timestamp_s");
    const int robot = identifier(frame.front(), "robot");
    require(frame.front().at("kind") == "pose", "frame must start with pose");
    const int index = identifier(frame.front(), "pose_index");
    require(index == poseIndex + 1, "nonconsecutive pose or missing parent");
    for (const auto& packet : frame) {
      require(identifier(packet, "robot") == robot &&
              identifier(packet, "pose_index") == index &&
              number(packet, "timestamp_s") == t, "mixed sensor frame");
      require(number(packet, "available_time_s") == t &&
              number(packet, "support_max_time_s") <= t, "noncausal packet");
    }

    gtsam::Values initial;
    gtsam::NonlinearFactorGraph factors;
    const gtsam::Key key = gtsam::Symbol('x', index);
    gtsam::Pose2 predicted;
    if (poseIndex < 0) {
      require(frame.front().at("delta").is_null() &&
              frame.front().at("from_pose_index").is_null(), "invalid root parent");
      factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
          key, predicted, gtsam::noiseModel::Constrained::All(3));
    } else {
      require(identifier(frame.front(), "from_pose_index") == poseIndex, "invalid parent");
      const double dt = number(frame.front(), "duration_s");
      require(dt > 0 && std::abs(t-poseTime-dt) < 1e-6, "invalid odometry duration");
      const auto& delta = frame.front().at("delta");
      require(delta.is_array() && delta.size() == 3, "invalid odometry delta");
      const gtsam::Vector3 d(delta[0].get<double>(), delta[1].get<double>(), delta[2].get<double>());
      require(d.allFinite(), "nonfinite odometry delta");
      const gtsam::Pose2 relative(d[0], d[1], d[2]);
      predicted = currentPose.compose(relative);
      const auto noise = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector3(.1, .1, .05) * std::sqrt(dt)).eval());
      auto factor = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
          gtsam::Symbol('x', poseIndex), key, relative, noise);
      factors.push_back(factor);
      physical.push_back(factor);
    }
    initial.insert(key, predicted);
    const auto observationNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05, .1));
    for (std::size_t i = 1; i < frame.size(); ++i) {
      const auto& p = frame[i];
      require(p.at("kind") == "observation", "unsupported packet kind");
      const int row = identifier(p, "source_row");
      require(sources.insert(row).second, "duplicate observation");
      const int subject = identifier(p, "landmark_subject");
      const double distance = number(p, "range_m"), bearing = number(p, "bearing_rad");
      require(distance > 0, "nonpositive range");
      const gtsam::Key landmark = gtsam::Symbol('l', subject);
      if (landmarks.insert(subject).second) {
        initial.insert(landmark, predicted.transformFrom(
            gtsam::Point2(distance*std::cos(bearing), distance*std::sin(bearing))));
      }
      physical.emplace_shared<Observation>(key, landmark, gtsam::Rot2(bearing), distance, observationNoise);
      if (robust) {
        const auto kernel = boost::make_shared<risam::SIGKernel>(
            risam::SIGKernel::shapeParamFromInfThresh(.1, 2, .95),
            risam::SIGKernel::MuUpdateStrategy::STABLE,
            risam::SIGKernel::MuInitIncrementStrategy::EQUAL_5);
        factors.push_back(risam::make_shared_graduated<Observation>(
            kernel, key, landmark, gtsam::Rot2(bearing), distance, observationNoise));
      } else {
        factors.push_back(physical.back());
      }
    }

    const auto solveStart = Clock::now();
    gtsam::ISAM2Result result;
    std::size_t convexified = 0;
    if (gaussian) result = gaussian->update(factors, initial);
    else {
      const auto r = robust->update(factors, initial);
      result = r.isam2_result;
      convexified = r.convexified_factors.size();
    }
    const double solveMs = millis(solveStart);
    const auto estimateStart = Clock::now();
    values = gaussian ? gaussian->calculateEstimate() : robust->calculateEstimate();
    currentPose = values.at<gtsam::Pose2>(key);
    require(gtsam::Vector3(currentPose.x(), currentPose.y(), currentPose.theta()).allFinite(),
            "nonfinite pose estimate");
    const double estimateMs = millis(estimateStart);
    poseIndex = index;
    poseTime = t;
    poseTimes[index] = t;
    return Json{{"solve_ms",solveMs},{"estimate_ms",estimateMs},{"total_ms",millis(start)},
                {"new_factors",factors.size()},{"relinearized_first",result.variablesRelinearized},
                {"reeliminated_first",result.variablesReeliminated},{"convexified",convexified}};
  }
};

int main(int argc, char** argv) {
  try {
    require(argc == 5, "usage: native-risam-replay events.jsonl output mode duration_seconds");
    const std::string mode(argv[3]);
    require(mode == "isam2" || mode == "risam" || mode == "risam_gn", "unknown solver mode");
    const double duration = std::stod(argv[4]);
    require(std::isfinite(duration) && duration > 0, "invalid duration");
    const std::filesystem::path output(argv[2]);
    require(!std::filesystem::exists(output), "output already exists");
    std::filesystem::create_directories(output);
    std::ifstream input(argv[1]);
    require(input.good(), "cannot read input stream");
    std::ofstream causal(output/"causal_estimates.csv"), final(output/"final_estimates.csv"),
                  updates(output/"updates.csv");
    require(causal.good() && final.good() && updates.good(), "cannot create outputs");
    causal << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
    final << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
    updates << std::setprecision(17)
            << "robot,timestamp_s,elapsed_s,new_factors,solve_ms,estimate_ms,total_ms,relinearized_first,reeliminated_first,convexified,serial_backlog_s\n";
    std::map<int,std::unique_ptr<LocalRobot>> robots;
    std::map<int, std::vector<Json>> frames;
    double frameTime = -1;
    double first = -1, last = -1, finish = 0, maximumBacklog = 0, totalWork = 0;
    std::size_t frameCount = 0, packetCount = 0;
    auto processFrame = [&](const std::vector<Json>& frame) {
      const int robot = identifier(frame.front(), "robot");
      if (!robots.count(robot)) robots[robot] = std::make_unique<LocalRobot>(mode);
      auto& local = *robots.at(robot);
      const auto stats = local.push(frame);
      const double elapsed = local.poseTime - first;
      const double work = stats.at("total_ms").get<double>() / 1000.;
      finish = std::max(finish, elapsed) + work;
      totalWork += work;
      maximumBacklog = std::max(maximumBacklog, finish-elapsed);
      const auto& p = local.currentPose;
      causal << robot << ",pose," << local.poseIndex << ',' << local.poseTime << ','
             << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
      updates << robot << ',' << local.poseTime << ',' << elapsed;
      for (const auto& key : {"new_factors","solve_ms","estimate_ms","total_ms",
                             "relinearized_first","reeliminated_first","convexified"})
        updates << ',' << stats.at(key);
      updates << ',' << finish-elapsed << '\n';
      packetCount += frame.size();
      ++frameCount;
      if (frameCount % 1000 == 0)
        std::cerr << "frames=" << frameCount << " elapsed=" << elapsed << '\n';
    };
    auto flush = [&]() {
      for (const auto& item : frames) processFrame(item.second);
      frames.clear();
    };
    std::string line;
    while (std::getline(input, line)) {
      const Json packet = Json::parse(line);
      const double t = number(packet, "timestamp_s");
      if (first < 0) first = t;
      require(t >= last, "time decreased");
      last = t;
      if (t-first > duration) break;
      if (t != frameTime) flush();
      frameTime = t;
      frames[identifier(packet, "robot")].push_back(packet);
    }
    flush();
    require(frameCount > 0, "empty event stream");
    double physicalCost = 0;
    for (const auto& entry : robots) {
      const auto& robot = *entry.second;
      physicalCost += robot.physical.error(robot.values);
      for (const auto& key : robot.values.keys()) {
        const gtsam::Symbol symbol(key);
        if (symbol.chr() == 'x') {
          const auto& p = robot.values.at<gtsam::Pose2>(key);
          require(gtsam::Vector3(p.x(), p.y(), p.theta()).allFinite(), "nonfinite final pose");
          final << entry.first << ",pose," << symbol.index() << ',' << robot.poseTimes.at(symbol.index())
                << ',' << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        } else {
          const auto& p = robot.values.at<gtsam::Point2>(key);
          require(p.allFinite(), "nonfinite final landmark");
          final << entry.first << ",landmark," << symbol.index() << ",," << p.x() << ',' << p.y() << ",\n";
        }
      }
    }
    require(std::isfinite(physicalCost), "nonfinite physical cost");
    std::ofstream summary(output/"summary.json");
    summary << Json{{"mode",mode},{"frames",frameCount},{"packets",packetCount},
                    {"robots",robots.size()},{"total_work_s",totalWork},
                    {"max_serial_backlog_s",maximumBacklog},{"physical_gaussian_cost",physicalCost},
                    {"truth_used",false},{"communication",false},
                    {"scope","independent local native-factor bridge; not distributed riMESA/DRANPPR"}}
                    .dump(2) << '\n';
    require(causal.good() && final.good() && updates.good() && summary.good(), "output write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
