#pragma once

#include <DPGO/IncrementalLocalModel.h>
#include <DPGO/NativeLocalResponse.h>

namespace DPGO {

// Immutable, matrix-free private/interface response at one local revision.
// Factorizes H_PP once; no damping, global solve or network transport is added.
// Invertible H_PP does not imply that the remaining interface is observable.
class NativeSchurModel : public NativeLocalResponse {
 public:
  NativeSchurModel(const NativeLinearSystem& system,
                   const std::vector<gtsam::Key>& privateKeys);
  ~NativeSchurModel() override;
  NativeSchurModel(const NativeSchurModel&) = delete;
  NativeSchurModel& operator=(const NativeSchurModel&) = delete;

  std::uint64_t revision() const override;
  double privateFactorizationMs() const;
  const std::vector<gtsam::Key>& boundaryKeys() const override;
  const std::map<gtsam::Key, Eigen::Index>& boundaryDimensions() const override;
  const gtsam::Vector& gradient() const override;
  gtsam::Matrix apply(const gtsam::Matrix& boundaryVectors) const override;
  // Returns increments in the input system's full free-key order.
  gtsam::Vector recover(const gtsam::Vector& boundaryIncrement) const override;
  double privateRelativeResidual(const gtsam::Vector& fullIncrement) const override;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace DPGO
