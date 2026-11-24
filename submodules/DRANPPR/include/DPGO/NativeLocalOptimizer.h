#pragma once

#include <DPGO/IncrementalLocalModel.h>
#include <string>
#include <limits>

namespace DPGO {

struct NativeLocalOptions {
  std::size_t maxIterations{100};
  std::size_t maxBacktracks{30};
  double gradientTolerance{1e-8};
};

struct NativeLocalIteration {
  std::uint64_t revision{0};
  double costBefore{0.}, costAfter{0.}, modelGradientNorm{0.};
  double alpha{0.}, stepNorm{0.}, predictedDecrease{0.}, linearResidual{0.};
  std::size_t costEvaluations{0}, factorsLinearized{0};
};

struct NativeLocalTiming {
  double graphCloneMs{0.}, exportMs{0.}, responseMs{0.}, interfaceMs{0.};
  double recoveryMs{0.}, evaluationMs{0.}, publicationMs{0.};
  double refreshMs{0.};
  // Nested counters, not additional disjoint time categories.
  double privateFactorizationMs{0.};
  double publicationSnapshotMs{0.}, publicationLinearizationMs{0.}, publicationContributionMs{0.};
  double refreshSnapshotMs{0.}, refreshLinearizationMs{0.}, refreshContributionMs{0.};
};

struct NativeLocalResult {
  std::string status;
  std::vector<NativeLocalIteration> trace;
  NativeLocalTiming timing;
  std::size_t modelRefreshKeys{0}, modelRefreshFactors{0}, modelRefreshReused{0};
  // Remain unavailable if maxIterations is zero.
  double initialCost{std::numeric_limits<double>::quiet_NaN()};
  double finalCost{std::numeric_limits<double>::quiet_NaN()};
};

// Small-graph correctness solver: fresh sparse private factorization plus a
// dense interface solve. Not incremental inference or a distributed optimizer.
// "model_gradient_small" is not certified true stationarity with approximate J.
NativeLocalResult optimizeNativeLocal(
    IncrementalLocalModel& model, const std::vector<gtsam::Key>& privateKeys,
    const std::set<gtsam::Key>& fixedKeys, const NativeLocalOptions& options = {});

// Same physical objective with separate current estimates and cached chart bases.
// Refresh a key when its chart displacement exceeds the supplied infinity-norm
// tolerance. Fresh factorization remains; cached model stationarity is NOT true
// nonlinear stationarity. The caller inserts new keys into both model and estimate.
NativeLocalResult optimizeNativeCachedLocal(
    IncrementalLocalModel& model, gtsam::Values& estimate,
    const std::vector<gtsam::Key>& privateKeys, const std::set<gtsam::Key>& fixedKeys,
    double relinearizeTolerance, const NativeLocalOptions& options = {});

}  // namespace DPGO
