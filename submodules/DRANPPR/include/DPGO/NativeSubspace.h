#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace DPGO {

inline gtsam::Matrix6 nativeRigidPose3Basis(const gtsam::Pose3& pose) {
  return pose.inverse().AdjointMap();
}

inline gtsam::Matrix nativeRigidPointBasis(const gtsam::Vector& point) {
  const auto d = point.size();
  if ((d != 2 && d != 3) || !point.allFinite()) {
    throw std::invalid_argument("rigid point basis requires finite 2D or 3D position");
  }
  gtsam::Matrix basis = gtsam::Matrix::Zero(d,d+d*(d-1)/2);
  basis.leftCols(d).setIdentity();
  if (d == 2) basis.col(2) << -point[1],point[0];
  else basis.rightCols(3) << 0.,point[2],-point[1],
      -point[2],0.,point[0],point[1],-point[0],0.;
  return basis;
}

struct NativeSubspaceSolution {
  gtsam::Vector coefficients;
  std::size_t rank{0};
};

// Equilibrated PSD solve: discard numerical null directions, never add a prior.
inline NativeSubspaceSolution solveNativeSubspace(
    const gtsam::Matrix& hessian, const gtsam::Vector& gradient) {
  const auto n = gradient.size();
  if (!n || hessian.rows()!=n || hessian.cols()!=n ||
      !hessian.allFinite() || !gradient.allFinite()) {
    throw std::invalid_argument("invalid native subspace system");
  }
  gtsam::Vector scaling(n);
  for (Eigen::Index i=0; i<n; ++i) scaling[i] = hessian(i,i)>0. ? 1./std::sqrt(hessian(i,i)) : 1.;
  const gtsam::Matrix scaled = scaling.asDiagonal()*
      (.5*(hessian+hessian.transpose()))*scaling.asDiagonal();
  Eigen::SelfAdjointEigenSolver<gtsam::Matrix> eigen(scaled);
  if (eigen.info()!=Eigen::Success) throw std::runtime_error("native subspace eigensolve failed");
  const double tolerance = 64*n*std::numeric_limits<double>::epsilon()*
      std::max(1.,eigen.eigenvalues().cwiseAbs().maxCoeff());
  if (eigen.eigenvalues().minCoeff() < -tolerance) {
    throw std::runtime_error("native subspace curvature is indefinite");
  }
  NativeSubspaceSolution result{gtsam::Vector::Zero(n),0};
  const gtsam::Vector rhs = -scaling.cwiseProduct(gradient);
  for (Eigen::Index i=0; i<n; ++i) if (eigen.eigenvalues()[i]>tolerance) {
    const auto direction = eigen.eigenvectors().col(i);
    result.coefficients += direction*(direction.dot(rhs)/eigen.eigenvalues()[i]);
    ++result.rank;
  }
  result.coefficients = scaling.cwiseProduct(result.coefficients);
  return result;
}

}  // namespace DPGO
