#include <DPGO/NativeLocalOptimizer.h>
#include <DPGO/NativeSchurModel.h>
#include <gtsam/linear/VectorValues.h>
#include <Eigen/Cholesky>

#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace DPGO {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
NativeLocalResult optimize(
    IncrementalLocalModel& model, const std::vector<gtsam::Key>& privateKeys,
    const std::set<gtsam::Key>& fixedKeys, const NativeLocalOptions& options,
    gtsam::Values* estimate, double relinearizeTolerance) {
  if (!std::isfinite(options.gradientTolerance) || options.gradientTolerance < 0. ||
      options.maxBacktracks == 0) throw std::invalid_argument("invalid native local solver options");
  NativeLocalResult result;
  result.status = "iteration_limit";
  const auto cloneStart = Clock::now();
  const auto graph = model.factorGraph();
  result.timing.graphCloneMs = milliseconds(cloneStart);
  for (std::size_t iteration = 0; iteration < options.maxIterations; ++iteration) {
    gtsam::VectorValues displacement;
    if (estimate) {
      const auto refreshStart = Clock::now();
      displacement = model.bases().localCoordinates(*estimate);
      NativeModelUpdate refresh;
      for (const auto key : estimate->keys()) {
        if (!fixedKeys.count(key) && displacement.at(key).lpNorm<Eigen::Infinity>() > relinearizeTolerance) {
          refresh.relinearizeValues.insert(key, estimate->at(key));
        }
      }
      const auto stats = model.update(refresh);
      result.modelRefreshKeys += refresh.relinearizeValues.size();
      result.modelRefreshFactors += stats.factorsLinearized;
      result.modelRefreshReused += stats.factorsReused;
      result.timing.refreshSnapshotMs += stats.snapshotMs;
      result.timing.refreshLinearizationMs += stats.linearizationMs;
      result.timing.refreshContributionMs += stats.contributionMs;
      for (const auto key : refresh.relinearizeValues.keys()) displacement.at(key).setZero();
      result.timing.refreshMs += milliseconds(refreshStart);
    }
    const auto exportStart = Clock::now();
    auto system = model.linearSystem({}, fixedKeys);
    if (estimate) {
      gtsam::Vector current = gtsam::Vector::Zero(system.gradient.size());
      for (const auto key : system.keys) {
        current.segment(system.offsets.at(key), system.dimensions.at(key)) = displacement.at(key);
      }
      // Solve for a correction to d0, not another absolute increment from theta.
      system.gradient += system.hessian*current;
    }
    result.timing.exportMs += milliseconds(exportStart);
    const auto responseStart = Clock::now();
    NativeSchurModel response(system, privateKeys);
    result.timing.responseMs += milliseconds(responseStart);
    result.timing.privateFactorizationMs += response.privateFactorizationMs();
    NativeLocalIteration row;
    row.revision = system.revision;
    const auto costStart = Clock::now();
    row.costBefore = row.costAfter = graph.error(estimate ? *estimate : model.bases());
    result.timing.evaluationMs += milliseconds(costStart);
    if (iteration == 0) result.initialCost = row.costBefore;
    result.finalCost = row.costBefore;
    row.modelGradientNorm = system.gradient.norm();
    if (!std::isfinite(row.costBefore)) throw std::runtime_error("nonfinite native physical objective");
    if (row.modelGradientNorm <= options.gradientTolerance) {
      result.status = estimate ? "cached_model_gradient_small" : "model_gradient_small";
      return result;
    }
    const auto nb = response.gradient().size();
    const auto interfaceStart = Clock::now();
    gtsam::Vector boundary(nb);
    if (nb > 0) {
      const gtsam::Matrix reduced = response.apply(gtsam::Matrix::Identity(nb, nb));
      Eigen::LDLT<gtsam::Matrix> solver(reduced);
      const double tolerance = std::numeric_limits<double>::epsilon()*nb*reduced.diagonal().cwiseAbs().maxCoeff();
      if (solver.info() != Eigen::Success || !solver.vectorD().allFinite() ||
          (solver.vectorD().array() <= tolerance).any()) {
        throw std::runtime_error("native interface is not numerically positive definite");
      }
      boundary = solver.solve(-response.gradient());
    }
    result.timing.interfaceMs += milliseconds(interfaceStart);
    const auto recoveryStart = Clock::now();
    const gtsam::Vector step = response.recover(boundary);
    row.stepNorm = step.norm();
    row.linearResidual = (system.hessian*step+system.gradient).norm()/std::max(1., system.gradient.norm());
    const double slope = system.gradient.dot(step), curvature = step.dot(system.hessian*step);
    result.timing.recoveryMs += milliseconds(recoveryStart);
    bool accepted = false;
    double alpha = 1.;
    for (std::size_t trial = 0; trial < options.maxBacktracks; ++trial, alpha *= .5) {
      const double predicted = -alpha*slope-.5*alpha*alpha*curvature;
      if (!std::isfinite(predicted) || predicted <= 0.) break;
      const auto evaluationStart = Clock::now();
      gtsam::VectorValues delta;
      for (auto key : model.bases().keys()) {
        if (fixedKeys.count(key)) delta.insert(key, gtsam::Vector::Zero(model.bases().at(key).dim()));
        else {
          gtsam::Vector value = alpha*step.segment(system.offsets.at(key), system.dimensions.at(key));
          if (estimate) value += displacement.at(key);
          delta.insert(key, value);
        }
      }
      const auto candidate = model.bases().retract(delta);
      const double cost = graph.error(candidate);
      result.timing.evaluationMs += milliseconds(evaluationStart);
      ++row.costEvaluations;
      if (!std::isfinite(cost) || cost >= row.costBefore) continue;
      const auto publicationStart = Clock::now();
      NativeUpdateStatistics stats;
      if (estimate) *estimate = candidate;
      else {
        NativeModelUpdate update;
        for (auto key : system.keys) update.relinearizeValues.insert(key, candidate.at(key));
        stats = model.update(update);
      }
      result.timing.publicationMs += milliseconds(publicationStart);
      result.timing.publicationSnapshotMs += stats.snapshotMs;
      result.timing.publicationLinearizationMs += stats.linearizationMs;
      result.timing.publicationContributionMs += stats.contributionMs;
      row.costAfter = cost;
      result.finalCost = cost;
      row.alpha = alpha;
      row.predictedDecrease = predicted;
      row.factorsLinearized = stats.factorsLinearized;
      accepted = true;
      break;
    }
    result.trace.push_back(row);
    if (!accepted) {
      result.status = "no_decreasing_trial";
      return result;
    }
  }
  return result;
}
}  // namespace

NativeLocalResult optimizeNativeLocal(
    IncrementalLocalModel& model, const std::vector<gtsam::Key>& privateKeys,
    const std::set<gtsam::Key>& fixedKeys, const NativeLocalOptions& options) {
  return optimize(model, privateKeys, fixedKeys, options, nullptr, 0.);
}

NativeLocalResult optimizeNativeCachedLocal(
    IncrementalLocalModel& model, gtsam::Values& estimate,
    const std::vector<gtsam::Key>& privateKeys, const std::set<gtsam::Key>& fixedKeys,
    double relinearizeTolerance, const NativeLocalOptions& options) {
  if (!std::isfinite(relinearizeTolerance) || relinearizeTolerance < 0.) {
    throw std::invalid_argument("invalid chart refresh tolerance");
  }
  if (model.bases().keys() != estimate.keys()) {
    throw std::invalid_argument("current estimates must match declared chart keys");
  }
  for (const auto key : estimate.keys()) {
    if (typeid(model.bases().at(key)) != typeid(estimate.at(key))) {
      throw std::invalid_argument("current estimate changed variable type");
    }
  }
  const auto displacement = model.bases().localCoordinates(estimate);
  for (const auto key : estimate.keys()) {
    if (!displacement.at(key).allFinite()) throw std::invalid_argument("nonfinite current estimate");
  }
  for (const auto key : fixedKeys) {
    if (!estimate.exists(key) || displacement.at(key).norm() > 1e-12) {
      throw std::invalid_argument("current estimate changed fixed root gauge");
    }
  }
  return optimize(model, privateKeys, fixedKeys, options, &estimate, relinearizeTolerance);
}

}  // namespace DPGO
