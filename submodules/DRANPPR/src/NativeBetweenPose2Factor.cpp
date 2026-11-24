#include <DPGO/NativeBetweenPose2Factor.h>

namespace DPGO {

gtsam::Vector NativeBetweenPose2Factor::evaluateError(
    const gtsam::Pose2& first, const gtsam::Pose2& second,
    boost::optional<gtsam::Matrix&> h1, boost::optional<gtsam::Matrix&> h2) const {
  const auto relative = gtsam::traits<gtsam::Pose2>::Between(first, second, h1, h2);
  gtsam::Matrix3 localJacobian;
  const auto residual = gtsam::traits<gtsam::Pose2>::Local(
      measured(), relative, boost::none, (h1 || h2) ? &localJacobian : nullptr);
  if (h1) *h1 = localJacobian*(*h1);
  if (h2) *h2 = localJacobian*(*h2);
  return residual;
}

gtsam::NonlinearFactor::shared_ptr NativeBetweenPose2Factor::clone() const {
  return boost::make_shared<NativeBetweenPose2Factor>(*this);
}

}  // namespace DPGO
