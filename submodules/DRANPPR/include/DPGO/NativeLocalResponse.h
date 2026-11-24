#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <cstdint>
#include <map>
#include <vector>

namespace DPGO {

// Immutable response of one declared local quadratic; no transport or priors.
class NativeLocalResponse {
 public:
  virtual ~NativeLocalResponse() = default;
  virtual std::uint64_t revision() const = 0;
  virtual const std::vector<gtsam::Key>& boundaryKeys() const = 0;
  virtual const std::map<gtsam::Key, Eigen::Index>& boundaryDimensions() const = 0;
  virtual const gtsam::Vector& gradient() const = 0;
  virtual gtsam::Matrix apply(const gtsam::Matrix& boundaryVectors) const = 0;
  // Full increment in the declared free-key order, including private variables.
  virtual gtsam::Vector recover(const gtsam::Vector& boundaryIncrement) const = 0;
  virtual double privateRelativeResidual(const gtsam::Vector& fullIncrement) const = 0;
};

}  // namespace DPGO
