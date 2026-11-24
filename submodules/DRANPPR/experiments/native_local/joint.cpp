#include <DPGO/NativeBetweenPose2Factor.h>
#include "joint_package.h"
#include <DPGO/NativeInterface.h>
#include <DPGO/NativeLocalOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace DPGO;
using Json = nlohmann::json;
using Robot = std::uint32_t;
using Clock = std::chrono::steady_clock;
using Links = std::vector<std::pair<Robot, Robot>>;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

gtsam::Key globalKey(Robot robot, gtsam::Key key) {
  const gtsam::Symbol symbol(key);
  return symbol.chr() == 'x' ? X((std::uint64_t(robot)<<32)|symbol.index()) : key;
}

Json exportLinearization(const std::vector<Local>& locals) {
  Json rows = Json::array();
  for (const auto& local : locals) for (const auto& factor : local.graph) {
    const auto linear = factor->linearize(local.model->bases());
    const auto ab = linear->jacobian();
    Json row{{"robot", local.id}, {"kind", boost::dynamic_pointer_cast<Observation>(factor) ? "observation" : "odometry"},
             {"b", Json::array()}, {"terms", Json::array()}};
    for (Eigen::Index i=0; i<ab.second.size(); ++i) row["b"].push_back(ab.second[i]);
    Eigen::Index offset=0;
    for (auto key=linear->begin(); key!=linear->end(); ++key) {
      const auto size=linear->getDim(key);
      if (!local.fixed.count(*key)) {
        const gtsam::Symbol symbol(*key);
        Json term{{"kind", std::string(1,symbol.chr())}, {"index", symbol.index()}, {"a", Json::array()}};
        for (Eigen::Index i=0; i<ab.first.rows(); ++i) {
          Json values=Json::array();
          for (Eigen::Index j=0; j<size; ++j) values.push_back(ab.first(i,offset+j));
          term["a"].push_back(std::move(values));
        }
        row["terms"].push_back(std::move(term));
      }
      offset+=size;
    }
    rows.push_back(std::move(row));
  }
  return Json{{"schema","dranppr_physical_linearization_v1"}, {"truth_used",false},
              {"scope","Post-solve nominally whitened physical factors; one fixed root; no consensus penalty"},
              {"factors",std::move(rows)}};
}
Json centralized(std::vector<Local>& locals, Robot root, const std::string& mode, std::size_t iterations) {
  NativeModelUpdate input;
  std::uint64_t nextId = 0;
  for (const auto& local : locals) {
    std::map<gtsam::Key, gtsam::Key> keyMap;
    for (auto key : local.model->bases().keys()) {
      const auto global = globalKey(local.id, key);
      keyMap[key] = global;
      if (!input.newValues.exists(global)) input.newValues.insert(global, local.model->bases().at(key));
    }
    for (const auto& factor : local.graph) {
      // ExpressionFactor's generic rekey changes keys but not its expression tree.
      // Reconstruct polar factors so the centralized control uses the same model.
      if (const auto observation = boost::dynamic_pointer_cast<Observation>(factor)) {
        gtsam::Key pose=0, point=0;
        for (auto key : factor->keys()) {
          if (gtsam::Symbol(key).chr() == 'x') pose = keyMap.at(key);
          else point = keyMap.at(key);
        }
        input.addFactors[nextId++] = boost::make_shared<Observation>(
            pose, point, observation->measured(), observation->noiseModel());
      } else input.addFactors[nextId++] = factor->rekey(keyMap);
    }
  }
  const auto fixed = globalKey(root, X(0));
  gtsam::Values final;
  Json result{{"trace", Json::array()}, {"application_bytes", 0}, {"payload_bytes", 0}};
  if (mode == "central_gn") {
    IncrementalLocalModel model;
    model.update(input);
    NativeLocalOptions options;
    options.maxIterations = iterations;
    const auto solved = optimizeNativeLocal(model, model.linearSystem({}, {fixed}).keys, {fixed}, options);
    final = model.bases();
    result["status"] = solved.status;
    for (const auto& row : solved.trace) result["trace"].push_back({{"cost_before", row.costBefore},
        {"cost_after", row.costAfter}, {"alpha", row.alpha}, {"linear_residual", row.linearResidual}});
  } else {
    gtsam::NonlinearFactorGraph graph;
    for (const auto& factor : input.addFactors) graph.push_back(factor.second);
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(fixed, input.newValues.at<gtsam::Pose2>(fixed),
        gtsam::noiseModel::Constrained::All(3));
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = iterations;
    params.absoluteErrorTol = params.relativeErrorTol = 1e-12;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, input.newValues, params);
    final = optimizer.optimize();
    result["status"] = "batch_optimizer_returned";
    result["lm_iterations"] = optimizer.iterations();
  }
  for (auto& local : locals) {
    NativeModelUpdate update;
    for (auto key : local.model->bases().keys()) update.relinearizeValues.insert(key, final.at(globalKey(local.id, key)));
    local.model->update(update);
  }
  return result;
}

int main(int argc, char** argv) {
  try {
    require(argc == 5 || (argc == 6 && std::string(argv[5]) == "--linearization"),
        "usage: native-joint-audit package.json output routed|central_gn|central_lm iterations [--linearization]");
    const std::string mode(argv[3]);
    require(mode == "routed" || mode == "central_gn" || mode == "central_lm", "invalid solver mode");
    const int iterations = std::stoi(argv[4]);
    require(iterations > 0 && iterations <= 100, "iteration budget must be in 1..100");
    const std::filesystem::path output(argv[2]);
    require(!std::filesystem::exists(output), "output already exists");
    std::ifstream stream(argv[1]);
    require(stream.good(), "cannot read input");
    Json input; stream >> input;
    const Robot root = identifier(input, "root_robot");
    Links links;
    const auto prepareStart = Clock::now();
    auto locals = parse(input, root, links);
    const double setupSeconds = seconds(prepareStart);
    double initialCost=0.;
    for (const auto& local : locals) initialCost += local.graph.error(local.initial);
    require(std::isfinite(initialCost), "nonfinite initial objective");
    std::filesystem::create_directories(output);
    const auto start = Clock::now();
    Json summary = mode == "routed" ? routed(locals, links, iterations) : centralized(locals, root, mode, iterations);
    summary["solve_seconds"] = seconds(start);
    summary.update({{"mode", mode}, {"truth_used", false}, {"initial_cost", initialCost},
        {"setup_seconds", setupSeconds}, {"scope", "Frozen synchronous joint physical solve; not online"},
        {"ledger_exclusions", "Offline bootstrap/rosters, wire framing, retransmission, posthoc diagnostics; central zero bytes are not a communication baseline"}});
    // This central aggregation is posthoc only, never a routed decision input.
    double finalCost=0., consensus=0., rootMove=0., otherMove=0.;
    summary["physical_components"] = Json::array();
    std::map<gtsam::Key, gtsam::Vector> gradient;
    std::map<gtsam::Key, gtsam::Point2> points;
    std::ofstream estimates(output/"final_estimates.csv");
    estimates.exceptions(std::ios::failbit|std::ios::badbit);
    estimates << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw,cluster\n";
    for (const auto& local : locals) {
      const auto& values = local.model->bases();
      finalCost += local.graph.error(values);
      double beforeOdom=0., afterOdom=0., beforePolar=0., afterPolar=0.;
      std::size_t odomCount=0, polarCount=0;
      for (const auto& factor : local.graph) {
        if (boost::dynamic_pointer_cast<Observation>(factor)) {
          beforePolar += factor->error(local.initial);
          afterPolar += factor->error(values);
          ++polarCount;
        } else {
          beforeOdom += factor->error(local.initial);
          afterOdom += factor->error(values);
          ++odomCount;
        }
      }
      summary["physical_components"].push_back({{"robot", local.id}, {"odometry_count", odomCount},
          {"observation_count", polarCount}, {"initial_odometry_cost", beforeOdom},
          {"final_odometry_cost", afterOdom}, {"initial_observation_cost", beforePolar},
          {"final_observation_cost", afterPolar}});
      const double move = local.initial.at<gtsam::Pose2>(X(0)).localCoordinates(values.at<gtsam::Pose2>(X(0))).norm();
      if (local.id == root) rootMove = move; else otherMove = std::max(otherMove, move);
      for (const auto& entry : local.graph.linearize(values)->gradientAtZero()) {
        const auto global = globalKey(local.id, entry.first);
        if (global == globalKey(root, X(0))) continue;
        const auto inserted = gradient.emplace(global, entry.second);
        if (!inserted.second) inserted.first->second += entry.second;
      }
      for (const auto& p : local.times) {
        const auto& pose = values.at<gtsam::Pose2>(X(p.first));
        estimates << local.id << ",pose," << p.first << ',' << p.second << ',' << pose.x() << ',' << pose.y() << ',' << pose.theta() << ",\n";
      }
      for (const auto& item : local.labels) {
        const auto& p = values.at<gtsam::Point2>(L(item.first));
        const auto first = points.emplace(L(item.first), p);
        consensus = std::max(consensus, (first.first->second-p).norm());
        estimates << local.id << ",landmark," << item.second << ",," << p.x() << ',' << p.y() << ",," << item.first << '\n';
      }
    }
    estimates.close();
    double gradientSquared=0.;
    for (const auto& entry : gradient) gradientSquared += entry.second.squaredNorm();
    summary.update({{"final_cost", finalCost}, {"posthoc_gradient_norm", std::sqrt(gradientSquared)},
        {"consensus_max_m", consensus}, {"root_displacement", rootMove},
        {"nonroot_root_displacement_max", otherMove}});
    require(std::isfinite(finalCost) && finalCost <= initialCost+1e-8 && rootMove < 1e-9 && consensus == 0., "joint invariant failed");
    std::ofstream record(output/"summary.json");
    record.exceptions(std::ios::failbit|std::ios::badbit);
    record << summary.dump(2) << '\n';
    record.close();
    if (argc == 6) {
      std::ofstream linearization(output/"linearization.json");
      linearization.exceptions(std::ios::failbit|std::ios::badbit);
      linearization << exportLinearization(locals).dump() << '\n';
      linearization.close();
    }
    summary.erase("trace");
    std::cout << summary.dump() << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
