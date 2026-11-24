#pragma once

#include <DPGO/NativeInterface.h>
#include <DPGO/NativeLocalOptimizer.h>
#include <DPGO/IncrementalConditionalModel.h>
#include <gtsam/linear/VectorValues.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <functional>

using namespace DPGO;
using Json = nlohmann::json;
using Robot = std::uint32_t;
using Clock = std::chrono::steady_clock;
using Links = std::vector<std::pair<Robot, Robot>>;

inline void jointRequire(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
inline double jointSeconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}

struct Local {
  Robot id;
  std::map<gtsam::Key, gtsam::Key> interfaceKeys;
  std::unique_ptr<IncrementalLocalModel> model;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;
  std::set<gtsam::Key> fixed;
  std::vector<gtsam::Key> privateKeys;
  std::map<std::uint32_t, double> times;
  std::map<std::uint32_t, std::uint32_t> labels;
  bool reuseCharts{false};
  bool conditionalResponse{false};
  bool lazyRefresh{false}, refreshPending{false};
  gtsam::Values estimate;
  NativeLinearSystem cachedSystem;
  std::shared_ptr<const NativeLocalResponse> cachedResponse;
  std::unique_ptr<IncrementalConditionalModel> conditionalModel;
  IncrementalLocalModel::SquareRootFactors cachedFactors;
  std::set<NativeFactorId> conditionalFactors;
  std::set<gtsam::Key> cachedFixed;
  std::vector<gtsam::Key> cachedPrivate;
  std::function<void(const Local&,const NativeModelUpdate&,const std::string&)> auditUpdate;

  const gtsam::Values& current() const { return reuseCharts ? estimate : model->bases(); }
  NativeUpdateStatistics updateModel(const NativeModelUpdate& update,const std::string& stage) {
    if (auditUpdate && !update.relinearizeValues.empty()) auditUpdate(*this,update,stage);
    return model->update(update);
  }
};

inline NativeConditionalStatistics updateConditional(Local& local) {
  const auto factors = local.model->squareRootFactors();
  const auto layout = local.model->linearLayout({}, local.fixed);
  const bool fixedChanged = local.cachedFixed != local.fixed;
  if (!local.conditionalModel) local.conditionalModel = std::make_unique<IncrementalConditionalModel>();
  NativeGaussianUpdate update;
  auto included = local.conditionalFactors;
  for (auto id : local.conditionalFactors) if (!factors.count(id)) {
    update.removeFactors.insert(id); included.erase(id);
  }
  for (const auto& item : factors) {
    const auto old = local.cachedFactors.find(item.first);
    if (!fixedChanged && old != local.cachedFactors.end() && old->second == item.second) continue;
    std::vector<std::pair<gtsam::Key,gtsam::Matrix>> terms;
    for (auto it=item.second->begin(); it!=item.second->end(); ++it) {
      if (!local.fixed.count(*it)) terms.emplace_back(*it,item.second->getA(it));
    }
    if (terms.empty()) {
      if (included.erase(item.first)) update.removeFactors.insert(item.first);
      continue;
    }
    auto factor = boost::make_shared<gtsam::JacobianFactor>(terms,item.second->getb());
    if (included.count(item.first)) update.replaceFactors[item.first] = factor;
    else { update.addFactors[item.first] = factor; included.insert(item.first); }
  }
  const auto stats = local.conditionalModel->update(update,layout.keys,layout.dimensions,local.privateKeys);
  local.cachedResponse = local.conditionalModel->response();
  static_cast<NativeVariableLayout&>(local.cachedSystem) = layout;
  local.cachedFactors = factors;
  local.conditionalFactors = std::move(included);
  return stats;
}

inline Json routed(std::vector<Local>& locals, const Links& links, std::size_t maxIterations,
            bool verbose = true, std::uint64_t epoch = 0, std::size_t interfaceIterations = 500,
            bool rigidSubspace = false, bool fusedOneStep = false) {
  Json result{{"status", "iteration_limit"}, {"trace", Json::array()}};
  std::size_t bytes=0, payload=0, scalarBytes=0, messages=0, pcgIterations=0, applications=0;
  std::size_t responseBuilds=0, responseReuses=0, refreshedKeys=0, factorsLinearized=0;
  double maxResidual=0., modelSeconds=0., solveSeconds=0., updateSeconds=0., evaluationSeconds=0.;
  double snapshotSeconds=0., linearizationSeconds=0., contributionSeconds=0.;
  std::size_t conditionalsRebuilt=0, conditionalsReused=0, conditionalFullRebuilds=0;
  double conditionalSeconds=0., recoverySeconds=0., directionalSeconds=0.;
  std::size_t refreshRequests=0, refreshRounds=0;
  std::size_t subspaceSteps=0;
  auto countUpdate = [&](const NativeUpdateStatistics& stats) {
    snapshotSeconds += stats.snapshotMs/1000.;
    linearizationSeconds += stats.linearizationMs/1000.;
    contributionSeconds += stats.contributionMs/1000.;
  };
  for (std::size_t iteration=0; iteration<maxIterations; ++iteration) {
    const bool lazyRefresh = locals.front().lazyRefresh;
    const bool refreshNow = locals.front().refreshPending;
    for (const auto& local : locals) jointRequire(local.lazyRefresh == lazyRefresh &&
        local.refreshPending == refreshNow, "inconsistent network refresh decision");
    if (lazyRefresh && refreshNow) ++refreshRounds;
    const auto modelStart = Clock::now();
    std::vector<NativeInterfaceEndpoint> endpoints;
    std::map<Robot, const NativeLinearSystem*> systems;
    std::map<Robot, gtsam::Vector> displacements;
    for (auto& local : locals) {
      if (local.reuseCharts) {
        const auto delta = local.model->bases().localCoordinates(local.estimate);
        NativeModelUpdate refresh;
        for (auto key : local.estimate.keys()) {
          const bool stale = local.lazyRefresh ? local.refreshPending && delta.at(key).norm() > 0. :
              delta.at(key).lpNorm<Eigen::Infinity>() > 1e-4;
          if (!local.fixed.count(key) && stale) {
            refresh.relinearizeValues.insert(key, local.estimate.at(key));
          }
        }
        const auto stats = local.updateModel(refresh,"optimizer_refresh");
        countUpdate(stats);
        refreshedKeys += refresh.relinearizeValues.size();
        factorsLinearized += stats.factorsLinearized;
      }
      local.refreshPending = false;
      if (!local.reuseCharts || !local.cachedResponse ||
          local.cachedSystem.revision != local.model->revision() ||
          local.cachedFixed != local.fixed || local.cachedPrivate != local.privateKeys) {
        if (local.conditionalResponse) {
          const auto start = Clock::now();
          const auto stats = updateConditional(local);
          conditionalSeconds += jointSeconds(start);
          conditionalsRebuilt += stats.conditionalsRebuilt;
          conditionalsReused += stats.conditionalsReused;
          conditionalFullRebuilds += stats.fullRebuild;
        } else {
          local.cachedSystem = local.model->linearSystem({}, local.fixed);
          local.cachedResponse = std::make_shared<NativeSchurModel>(local.cachedSystem, local.privateKeys);
        }
        local.cachedFixed = local.fixed; local.cachedPrivate = local.privateKeys;
        ++responseBuilds;
      } else ++responseReuses;
      systems[local.id] = &local.cachedSystem;
      const auto& response = local.cachedResponse;
      NativeInterfaceEndpoint endpoint{local.id, response, {}, local.interfaceKeys};
      for (auto key : response->boundaryKeys()) endpoint.chartBases[key] = local.model->bases().at<gtsam::Point2>(key);
      auto& displacement = displacements[local.id];
      displacement = gtsam::Vector::Zero(local.cachedSystem.dimension);
      if (local.reuseCharts) {
        const auto delta = local.model->bases().localCoordinates(local.estimate);
        for (auto key : local.cachedSystem.keys) displacement.segment(local.cachedSystem.offsets.at(key),
            local.cachedSystem.dimensions.at(key)) = delta.at(key);
        for (auto key : response->boundaryKeys()) endpoint.chartDisplacements[key] = delta.at(key);
      }
      endpoints.push_back(std::move(endpoint));
    }
    modelSeconds += jointSeconds(modelStart);
    NativeInterfaceOptions options;
    options.epoch = epoch+iteration;
    options.maxIterations = interfaceIterations;
    options.rigidSubspace = rigidSubspace;
    options.fusedOneStep = fusedOneStep;
    const auto solveStart = Clock::now();
    const auto solved = solveNativeInterface(endpoints, links, options);
    solveSeconds += jointSeconds(solveStart);
    bytes += solved.applicationBytes; payload += solved.payloadBytes; messages += solved.messages.size();
    pcgIterations += solved.iterations;
    subspaceSteps += solved.subspaceSteps;
    for (const auto& count : solved.operatorApplications) applications += count.second;
    maxResidual = std::max(maxResidual, solved.relativeResidual);
    if (interfaceIterations == 500) {
      jointRequire(solved.relativeResidual <= 1e-8, "joint PCG failed verified residual tolerance");
    }
    std::map<Robot, gtsam::Vector> steps, scalars;
    const auto evaluationStart = Clock::now();
    for (std::size_t i=0; i<locals.size(); ++i) {
      const auto& local = locals[i];
      auto boundary = solved.localSteps.at(local.id);
      const auto& endpoint = endpoints[i];
      if (local.reuseCharts) for (std::size_t j=0; j<endpoint.response->boundaryKeys().size(); ++j) {
        boundary.segment<2>(2*j) += endpoint.chartDisplacements.at(endpoint.response->boundaryKeys()[j]);
      }
      const auto recoveryStart = Clock::now();
      gtsam::Vector step = endpoint.response->recover(boundary)-displacements.at(local.id);
      recoverySeconds += jointSeconds(recoveryStart);
      const auto& system = *systems.at(local.id);
      const auto directionalStart = Clock::now();
      gtsam::Vector2 terms;
      if (local.conditionalResponse) terms = local.model->directionalModel(system,displacements.at(local.id),step);
      else {
        gtsam::Vector gradient = system.gradient;
        if (local.reuseCharts) gradient += system.hessian*displacements.at(local.id);
        terms = gtsam::Vector2(gradient.dot(step),step.dot(system.hessian*step));
      }
      directionalSeconds += jointSeconds(directionalStart);
      scalars[local.id] = gtsam::Vector3(local.graph.error(local.current()),
          terms[0], terms[1]);
      steps[local.id] = std::move(step);
    }
    auto reduce = [&](const std::map<Robot, gtsam::Vector>& values) {
      const auto sum = sumNativeInterfaceScalars(endpoints, links, values, options);
      bytes += sum.applicationBytes; payload += sum.payloadBytes;
      scalarBytes += sum.applicationBytes; messages += sum.messages.size();
      return sum.sum;
    };
    const auto totals = reduce(scalars);
    bool accepted = false;
    double alpha = 1., finalCost = totals[0];
    const double fullPrediction = -totals[1]-.5*totals[2];
    Json fullDecrease = nullptr;
    bool requestRefresh = false;
    std::size_t trials = 0;
    for (; trials<30; ++trials, alpha *= .5) {
      const double prediction = -alpha*totals[1]-.5*alpha*alpha*totals[2];
      if (!std::isfinite(prediction) || prediction <= 0.) {
        if (trials == 0) requestRefresh = true;
        break;
      }
      std::vector<gtsam::Values> candidates;
      scalars.clear();
      for (const auto& local : locals) {
        const auto& system = *systems.at(local.id);
        gtsam::VectorValues delta;
        for (auto key : local.model->bases().keys()) {
          if (local.fixed.count(key)) delta.insert(key, gtsam::Vector::Zero(3));
          else delta.insert(key, (displacements.at(local.id).segment(system.offsets.at(key), system.dimensions.at(key))+
              alpha*steps.at(local.id).segment(system.offsets.at(key), system.dimensions.at(key))).eval());
        }
        candidates.push_back(local.model->bases().retract(delta));
        const double localCost = local.graph.error(candidates.back());
        scalars[local.id] = gtsam::Vector2(std::isfinite(localCost) ? localCost : 0.,
                                         std::isfinite(localCost) ? 0. : 1.);
      }
      const auto trialTotals = reduce(scalars);
      if (trials == 0) {
        requestRefresh = trialTotals[1] > 0. || !(trialTotals[0] < totals[0]);
        if (trialTotals[1] == 0.) fullDecrease = totals[0]-trialTotals[0];
      }
      if (trialTotals[1] > 0.) continue;
      const double cost = trialTotals[0];
      if (cost >= totals[0]) continue;
      evaluationSeconds += jointSeconds(evaluationStart);
      const auto updateStart = Clock::now();
      for (std::size_t i=0; i<locals.size(); ++i) {
        if (locals[i].reuseCharts) { locals[i].estimate = candidates[i]; continue; }
        NativeModelUpdate update;
        for (auto key : systems.at(locals[i].id)->keys) update.relinearizeValues.insert(key, candidates[i].at(key));
        const auto stats = locals[i].model->update(update);
        countUpdate(stats);
        refreshedKeys += update.relinearizeValues.size();
        factorsLinearized += stats.factorsLinearized;
      }
      updateSeconds += jointSeconds(updateStart);
      finalCost = cost;
      accepted = true;
      break;
    }
    if (!accepted) evaluationSeconds += jointSeconds(evaluationStart);
    if (lazyRefresh && requestRefresh) {
      ++refreshRequests;
      for (auto& local : locals) local.refreshPending = true;
    }
    result["trace"].push_back({{"iteration", iteration}, {"cost_before", totals[0]}, {"cost_after", finalCost},
        {"alpha", accepted ? alpha : 0.}, {"cost_trials", accepted ? trials+1 : trials},
        {"pcg_iterations", solved.iterations}, {"pcg_relative_residual", solved.relativeResidual},
        {"cumulative_application_bytes", bytes}, {"cumulative_scalar_bytes", scalarBytes},
        {"full_step_prediction", fullPrediction}, {"full_step_decrease", fullDecrease},
        {"refresh_requested",lazyRefresh && requestRefresh}, {"refresh_performed",lazyRefresh && refreshNow}});
    if (verbose) std::cout << result["trace"].back().dump() << std::endl;
    if (!accepted) {
      result["status"] = "no_decreasing_trial";
      if (lazyRefresh && requestRefresh) continue;
      break;
    }
    result["status"] = "iteration_limit";
  }
  result.update({{"application_bytes", bytes}, {"payload_bytes", payload}, {"messages", messages},
      {"scalar_collective_bytes", scalarBytes}, {"pcg_iterations", pcgIterations},
      {"pcg_max_relative_residual", maxResidual}, {"local_operator_applications", applications},
      {"model_seconds", modelSeconds}, {"interface_seconds", solveSeconds},
      {"publication_seconds", updateSeconds}, {"evaluation_and_collective_seconds", evaluationSeconds}});
  result.update({{"response_builds", responseBuilds}, {"response_reuses", responseReuses},
      {"refreshed_keys", refreshedKeys}, {"factors_linearized", factorsLinearized}});
  result.update({{"update_snapshot_seconds", snapshotSeconds},
      {"update_linearization_seconds", linearizationSeconds}, {"update_contribution_seconds", contributionSeconds}});
  result.update({{"conditionals_rebuilt",conditionalsRebuilt}, {"conditionals_reused",conditionalsReused},
      {"conditional_full_rebuilds",conditionalFullRebuilds}, {"conditional_seconds",conditionalSeconds},
      {"recovery_seconds",recoverySeconds}, {"directional_seconds",directionalSeconds}});
  result.update({{"refresh_requests",refreshRequests}, {"refresh_rounds",refreshRounds},
      {"subspace_steps",subspaceSteps}});
  return result;
}
