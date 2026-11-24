#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/BetweenFactor.h>

namespace DPGO {

// Same residual/noise as native BetweenFactor<Pose2>, with the complete chart
// chain rule. Opt-in: does not change upstream factors or compile-time settings.
class NativeBetweenPose2Factor : public gtsam::BetweenFactor<gtsam::Pose2> {
 public:
  using gtsam::BetweenFactor<gtsam::Pose2>::BetweenFactor;
  gtsam::Vector evaluateError(const gtsam::Pose2& first, const gtsam::Pose2& second,
      boost::optional<gtsam::Matrix&> h1 = boost::none,
      boost::optional<gtsam::Matrix&> h2 = boost::none) const override;
  gtsam::NonlinearFactor::shared_ptr clone() const override;
};

}  // namespace DPGO
