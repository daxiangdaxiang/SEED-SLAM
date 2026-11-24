#pragma once

#include <DPGO/IncrementalLocalModel.h>
#include <DPGO/NativeLocalResponse.h>
#include <gtsam/linear/GaussianFactor.h>

namespace DPGO {

struct NativeGaussianUpdate {
  std::map<NativeFactorId, gtsam::GaussianFactor::shared_ptr> addFactors;
  std::map<NativeFactorId, gtsam::GaussianFactor::shared_ptr> replaceFactors;
  std::set<NativeFactorId> removeFactors;
};

struct NativeConditionalStatistics {
  std::uint64_t revision{0};
  std::size_t factorsCloned{0}, conditionalsRebuilt{0}, conditionalsReused{0};
  std::size_t originalFactorsEliminated{0}, childMessagesEliminated{0};
  bool fullRebuild{false};
  double snapshotMs{0.}, preparationMs{0.}, eliminationMs{0.}, boundaryMs{0.};
};

// Incremental elimination of a declared Gaussian model, not a nonlinear solver.
// Fixed gauge columns must already be removed, not replaced with soft priors.
// Uses GTSAM QR, retains original factors, and publishes immutable responses.
class IncrementalConditionalModel {
 public:
  IncrementalConditionalModel();
  ~IncrementalConditionalModel();
  IncrementalConditionalModel(const IncrementalConditionalModel&) = delete;
  IncrementalConditionalModel& operator=(const IncrementalConditionalModel&) = delete;

  NativeConditionalStatistics update(
      const NativeGaussianUpdate& update,
      const std::vector<gtsam::Key>& freeKeys,
      const std::map<gtsam::Key, Eigen::Index>& dimensions,
      const std::vector<gtsam::Key>& privateKeys);
  std::shared_ptr<const NativeLocalResponse> response() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace DPGO
