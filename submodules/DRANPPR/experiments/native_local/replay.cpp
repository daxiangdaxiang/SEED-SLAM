#include <DPGO/NativeBetweenPose2Factor.h>
#include <DPGO/NativeLocalOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>

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

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double number(const Json& packet, const std::string& key) {
  require(packet.at(key).is_number(), "nonnumeric " + key);
  const double value = packet.at(key).get<double>();
  require(std::isfinite(value), "nonfinite " + key);
  return value;
}

int identifier(const Json& packet, const std::string& key) {
  require(packet.at(key).is_number_integer(), "noninteger " + key);
  const auto value = packet.at(key).get<int64_t>();
  require(value >= 0 && value <= std::numeric_limits<int>::max(), "invalid " + key);
  return static_cast<int>(value);
}

double millis(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

const std::vector<std::string> columns{
  "new_factors", "physical_factors", "history_values", "prepare_ms", "insert_ms",
  "solve_ms", "estimate_ms", "total_ms", "insert_snapshot_ms", "insert_linearization_ms",
  "insert_contribution_ms", "graph_clone_ms", "export_ms", "response_ms", "interface_ms",
  "recovery_ms", "evaluation_ms", "publication_ms", "private_factorization_ms",
  "publication_snapshot_ms", "publication_linearization_ms", "publication_contribution_ms",
  "refresh_ms", "refresh_snapshot_ms", "refresh_linearization_ms", "refresh_contribution_ms",
  "model_refresh_keys", "model_refresh_factors", "model_refresh_reused",
  "factors_linearized", "variables_relinearized", "variables_reeliminated", "accepted_steps",
  "cost_before", "cost_after", "status"};

struct LocalRobot {
  bool cached{false};
  std::unique_ptr<DPGO::IncrementalLocalModel> native;
  std::unique_ptr<gtsam::ISAM2> isam;
  gtsam::NonlinearFactorGraph physical;
  gtsam::Values values;
  gtsam::Pose2 currentPose;
  std::vector<gtsam::Key> privateKeys;
  int poseIndex{-1};
  double poseTime{0.}, finish{0.}, maxBacklog{0.};
  std::map<int, double> poseTimes;
  std::set<int> landmarks, sources;

  explicit LocalRobot(const std::string& mode) {
    cached = mode == "native_cached";
    if (mode == "native" || cached) native = std::make_unique<DPGO::IncrementalLocalModel>();
    else {
      gtsam::ISAM2Params params;
      params.relinearizeThreshold = 1e-4;
      params.relinearizeSkip = 1;
      params.factorization = gtsam::ISAM2Params::QR;
      params.optimizationParams = gtsam::ISAM2GaussNewtonParams(1e-3);
      isam = std::make_unique<gtsam::ISAM2>(params);
    }
  }

  Json push(const std::vector<Json>& frame) {
    const auto start = Clock::now();
    const auto& first = frame.front();
    const int robot = identifier(first, "robot"), index = identifier(first, "pose_index");
    const double time = number(first, "timestamp_s");
    require(first.at("kind") == "pose", "frame must start with pose");
    require(index == poseIndex+1, "nonconsecutive pose or missing parent");
    for (const auto& packet : frame) {
      require(identifier(packet, "robot") == robot && identifier(packet, "pose_index") == index &&
              number(packet, "timestamp_s") == time, "mixed sensor frame");
      require(number(packet, "available_time_s") == time &&
              number(packet, "support_max_time_s") <= time, "noncausal packet");
    }
    DPGO::NativeModelUpdate insertion;
    gtsam::NonlinearFactorGraph factors;
    const auto key = gtsam::Symbol('x', index);
    gtsam::Pose2 predicted;
    auto add = [&](DPGO::NativeFactorId id, const gtsam::NonlinearFactor::shared_ptr& factor) {
      require(insertion.addFactors.emplace(id, factor).second, "duplicate factor identifier");
      factors.push_back(factor);
      physical.push_back(factor);
    };
    if (poseIndex < 0) {
      require(first.at("delta").is_null() && first.at("from_pose_index").is_null(), "invalid root parent");
      require(number(first, "duration_s") == 0, "invalid root duration");
      if (isam) factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
          key, predicted, gtsam::noiseModel::Constrained::All(3));
    } else {
      require(identifier(first, "from_pose_index") == poseIndex, "invalid parent");
      const double duration = number(first, "duration_s");
      require(duration > 0 && std::abs(time-poseTime-duration) < 1e-6, "invalid odometry duration");
      const auto& delta = first.at("delta");
      require(delta.is_array() && delta.size() == 3, "invalid odometry delta");
      const gtsam::Vector3 d(delta[0].get<double>(), delta[1].get<double>(), delta[2].get<double>());
      require(d.allFinite(), "nonfinite odometry delta");
      const gtsam::Pose2 relative(d[0], d[1], d[2]);
      predicted = currentPose.compose(relative);
      add(2*static_cast<DPGO::NativeFactorId>(index), boost::make_shared<DPGO::NativeBetweenPose2Factor>(
          gtsam::Symbol('x', poseIndex), key, relative, gtsam::noiseModel::Diagonal::Sigmas(
              (gtsam::Vector3(.1, .1, .05)*std::sqrt(duration)).eval())));
      privateKeys.push_back(key);
    }
    insertion.newValues.insert(key, predicted);
    const auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05, .1));
    for (std::size_t i = 1; i < frame.size(); ++i) {
      const auto& packet = frame[i];
      require(packet.at("kind") == "observation", "unsupported packet kind");
      const int row = identifier(packet, "source_row");
      require(sources.insert(row).second, "duplicate observation");
      const int subject = identifier(packet, "landmark_subject");
      const double distance = number(packet, "range_m"), bearing = number(packet, "bearing_rad");
      require(distance > 0, "nonpositive range");
      const auto landmark = gtsam::Symbol('l', subject);
      if (landmarks.insert(subject).second) insertion.newValues.insert(landmark, predicted.transformFrom(
          gtsam::Point2(distance*std::cos(bearing), distance*std::sin(bearing))));
      add(2*static_cast<DPGO::NativeFactorId>(row)+1, boost::make_shared<Observation>(
          key, landmark, gtsam::Rot2(bearing), distance, noise));
    }
    Json stats;
    for (const auto& column : columns) stats[column] = 0;
    stats["prepare_ms"] = millis(start);
    stats["new_factors"] = insertion.addFactors.size();
    if (native) {
      const auto insertStart = Clock::now();
      const auto inserted = native->update(insertion);
      if (cached) values.insert(insertion.newValues);
      stats["insert_ms"] = millis(insertStart);
      stats["insert_snapshot_ms"] = inserted.snapshotMs;
      stats["insert_linearization_ms"] = inserted.linearizationMs;
      stats["insert_contribution_ms"] = inserted.contributionMs;
      const auto solveStart = Clock::now();
      DPGO::NativeLocalOptions options;
      options.maxIterations = 1;
      const auto result = cached ? DPGO::optimizeNativeCachedLocal(
          *native, values, privateKeys, {gtsam::Symbol('x', 0)}, 1e-4, options) :
          DPGO::optimizeNativeLocal(*native, privateKeys, {gtsam::Symbol('x', 0)}, options);
      stats["solve_ms"] = millis(solveStart);
      stats["cost_before"] = result.initialCost;
      stats["cost_after"] = result.finalCost;
      stats["status"] = result.status;
      const auto& t = result.timing;
      stats["graph_clone_ms"] = t.graphCloneMs;
      stats["export_ms"] = t.exportMs;
      stats["response_ms"] = t.responseMs;
      stats["interface_ms"] = t.interfaceMs;
      stats["recovery_ms"] = t.recoveryMs;
      stats["evaluation_ms"] = t.evaluationMs;
      stats["publication_ms"] = t.publicationMs;
      stats["private_factorization_ms"] = t.privateFactorizationMs;
      stats["publication_snapshot_ms"] = t.publicationSnapshotMs;
      stats["publication_linearization_ms"] = t.publicationLinearizationMs;
      stats["publication_contribution_ms"] = t.publicationContributionMs;
      stats["refresh_ms"] = t.refreshMs;
      stats["refresh_snapshot_ms"] = t.refreshSnapshotMs;
      stats["refresh_linearization_ms"] = t.refreshLinearizationMs;
      stats["refresh_contribution_ms"] = t.refreshContributionMs;
      stats["model_refresh_keys"] = result.modelRefreshKeys;
      stats["model_refresh_factors"] = result.modelRefreshFactors;
      stats["model_refresh_reused"] = result.modelRefreshReused;
      std::size_t linearized = inserted.factorsLinearized+result.modelRefreshFactors, accepted = 0;
      for (const auto& row : result.trace) {
        linearized += row.factorsLinearized;
        accepted += row.alpha > 0;
      }
      stats["factors_linearized"] = linearized;
      stats["accepted_steps"] = accepted;
    } else {
      const auto solveStart = Clock::now();
      const auto result = isam->update(factors, insertion.newValues);
      stats["solve_ms"] = millis(solveStart);
      stats["variables_relinearized"] = result.variablesRelinearized;
      stats["variables_reeliminated"] = result.variablesReeliminated;
      stats["cost_before"] = stats["cost_after"] = nullptr;
      stats["status"] = "isam2_update";
    }
    const auto estimateStart = Clock::now();
    // Cached mode already publishes the complete estimate during optimization.
    if (!cached) values = native ? native->bases() : isam->calculateEstimate();
    currentPose = values.at<gtsam::Pose2>(key);
    require(gtsam::Vector3(currentPose.x(), currentPose.y(), currentPose.theta()).allFinite(), "nonfinite current pose");
    stats["estimate_ms"] = millis(estimateStart);
    poseIndex = index;
    poseTime = time;
    poseTimes[index] = time;
    stats["physical_factors"] = physical.size();
    stats["history_values"] = values.size();
    stats["total_ms"] = millis(start);
    return stats;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 5, "usage: native-causal-replay events.jsonl output native|native_cached|isam2_exact duration_seconds");
    const std::string mode(argv[3]);
    require(mode == "native" || mode == "native_cached" || mode == "isam2_exact", "unknown mode");
    const double duration = std::stod(argv[4]);
    require(std::isfinite(duration) && duration > 0, "invalid duration");
    std::ifstream input(argv[1]);
    require(input.good(), "cannot read input stream");
    const std::filesystem::path output(argv[2]);
    require(!std::filesystem::exists(output), "output already exists");
    std::filesystem::create_directories(output);
    std::ofstream causal(output/"causal_estimates.csv"), final(output/"final_estimates.csv"), updates(output/"updates.csv");
    require(causal.good() && final.good() && updates.good(), "cannot create outputs");
    causal << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
    final << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw\n";
    updates << std::setprecision(17) << "robot,timestamp_s,elapsed_s,initialization_ms";
    for (const auto& column : columns) updates << ',' << column;
    updates << ",serial_backlog_s,robot_backlog_s\n";
    std::map<int, std::unique_ptr<LocalRobot>> robots;
    std::map<int, std::vector<Json>> frames;
    double frameTime = -1., first = -1., last = -1., finish = 0., maxBacklog = 0., totalWork = 0.;
    std::size_t frameCount = 0, packetCount = 0;
    auto flush = [&]() {
      for (const auto& entry : frames) {
        const int robot = entry.first;
        const auto initializationStart = Clock::now();
        const bool initialize = !robots.count(robot);
        if (initialize) robots[robot] = std::make_unique<LocalRobot>(mode);
        const double initializationMs = initialize ? millis(initializationStart) : 0.;
        auto& local = *robots.at(robot);
        const auto stats = local.push(entry.second);
        const double elapsed = local.poseTime-first;
        const double work = (stats.at("total_ms").get<double>()+initializationMs)/1000.;
        totalWork += work;
        finish = std::max(finish, elapsed)+work;
        maxBacklog = std::max(maxBacklog, finish-elapsed);
        local.finish = std::max(local.finish, elapsed)+work;
        local.maxBacklog = std::max(local.maxBacklog, local.finish-elapsed);
        const auto& p = local.currentPose;
        causal << robot << ",pose," << local.poseIndex << ',' << local.poseTime << ',' << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        updates << robot << ',' << local.poseTime << ',' << elapsed << ',' << initializationMs;
        for (const auto& column : columns) {
          updates << ',';
          const auto& value = stats.at(column);
          if (value.is_string()) updates << value.get<std::string>();
          else if (!value.is_null()) updates << value;
        }
        updates << ',' << finish-elapsed << ',' << local.finish-elapsed << '\n';
        packetCount += entry.second.size();
        ++frameCount;
        if (frameCount%1000 == 0) std::cerr << "frames=" << frameCount << " elapsed=" << elapsed << '\n';
      }
      frames.clear();
    };
    std::string line;
    while (std::getline(input, line)) {
      const auto packet = Json::parse(line);
      const double time = number(packet, "timestamp_s");
      require(time >= 0, "negative timestamp");
      if (first < 0) first = time;
      require(time >= last, "time decreased");
      last = time;
      if (time-first > duration) break;
      if (time != frameTime) flush();
      frameTime = time;
      frames[identifier(packet, "robot")].push_back(packet);
    }
    flush();
    require(frameCount > 0, "empty event stream");
    double cost = 0.;
    Json robotBacklogs = Json::object();
    for (const auto& entry : robots) {
      const auto& local = *entry.second;
      cost += local.physical.error(local.values);
      robotBacklogs[std::to_string(entry.first)] = local.maxBacklog;
      for (const auto key : local.values.keys()) {
        const gtsam::Symbol symbol(key);
        if (symbol.chr() == 'x') {
          const auto& p = local.values.at<gtsam::Pose2>(key);
          require(gtsam::Vector3(p.x(), p.y(), p.theta()).allFinite(), "nonfinite final pose");
          final << entry.first << ",pose," << symbol.index() << ',' << local.poseTimes.at(symbol.index())
                << ',' << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        } else {
          const auto& p = local.values.at<gtsam::Point2>(key);
          require(p.allFinite(), "nonfinite final object");
          final << entry.first << ",landmark," << symbol.index() << ",," << p.x() << ',' << p.y() << ",\n";
        }
      }
    }
    require(std::isfinite(cost), "nonfinite physical cost");
    std::ofstream summary(output/"summary.json");
    summary << Json{{"mode",mode},{"frames",frameCount},{"packets",packetCount},{"robots",robots.size()},
      {"total_work_s",totalWork},{"max_serial_backlog_s",maxBacklog},{"max_robot_backlog_s",robotBacklogs},
      {"physical_gaussian_cost",cost},{"truth_used",false},{"communication",false},
      {"native_steps_per_frame",mode != "isam2_exact" ? Json(1) : Json(nullptr)},
      {"chart_refresh_tolerance",mode == "native_cached" ? Json(1e-4) : Json(nullptr)},
      {"scope","independent local exact-Jacobian Gaussian replay; not distributed or robust"},
      {"timing","frame service includes initialization; excludes input parsing, CSV and posthoc scoring"}}
      .dump(2) << '\n';
    require(causal.good() && final.good() && updates.good() && summary.good(), "output write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
