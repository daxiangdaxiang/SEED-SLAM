#include <DPGO/NativeSubspace.h>
#include <DPGO/NativeSchurModel.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <Eigen/Cholesky>
#include <iomanip>
#include <sstream>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

TEST(NativeSubspace, SixDofBasisMatchesPhysicalLeftMotion) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(.3,-.4,.2),gtsam::Point3(2.,-1.,3.));
  const auto basis = nativeRigidPose3Basis(pose);
  for (int axis=0; axis<6; ++axis) {
    const gtsam::Vector6 delta = 1e-6*gtsam::Vector6::Unit(axis);
    const auto forward = gtsam::Pose3::Expmap(delta).compose(pose);
    const auto backward = gtsam::Pose3::Expmap(-delta).compose(pose);
    const gtsam::Vector6 numeric = (pose.localCoordinates(forward)-pose.localCoordinates(backward))/(2e-6);
    EXPECT_LE((numeric-basis.col(axis)).norm(),1e-8);
  }
  const auto points = nativeRigidPointBasis(pose.translation());
  for (int axis=0; axis<3; ++axis) {
    const gtsam::Vector3 delta = 1e-6*gtsam::Vector3::Unit(axis);
    const gtsam::Vector3 numeric = (gtsam::Rot3::Expmap(delta)*pose.translation()-
        gtsam::Rot3::Expmap(-delta)*pose.translation())/(2e-6);
    EXPECT_LE((numeric-points.col(3+axis)).norm(),1e-8);
  }
}

class ExactBetween3 : public gtsam::BetweenFactor<gtsam::Pose3> {
 public:
  using gtsam::BetweenFactor<gtsam::Pose3>::BetweenFactor;
  gtsam::Vector evaluateError(const gtsam::Pose3& a,const gtsam::Pose3& b,
      boost::optional<gtsam::Matrix&> h1=boost::none,
      boost::optional<gtsam::Matrix&> h2=boost::none) const override {
    const auto relative = a.between(b,h1,h2);
    gtsam::Matrix6 local;
    const auto residual = gtsam::traits<gtsam::Pose3>::Local(measured(),relative,
        boost::none,(h1 || h2) ? &local : nullptr);
    if (h1) *h1 = local*(*h1);
    if (h2) *h2 = local*(*h2);
    return residual;
  }
};

TEST(NativeSubspace, PhysicalPose3SchurModelKeepsRootAndContainsFineStep) {
  gtsam::Values values;
  for (int j=0; j<4; ++j) values.insert(X(j),gtsam::Pose3(
      gtsam::Rot3::RzRyRx(.1*j,-.07*j,.2*j),gtsam::Point3(.7*j,.2*j*j,.3*j)));
  for (int j=0; j<3; ++j) values.insert(L(j),gtsam::Pose3(
      gtsam::Rot3::RzRyRx(.3*j,.2*j,-.1*j),gtsam::Point3(2.+j,1.-j,2.+.5*j)));
  NativeLinearSystem system;
  for (auto key : values.keys()) if (key!=X(0)) {
    system.keys.push_back(key); system.offsets[key]=system.dimension;
    system.dimensions[key]=6; system.dimension+=6;
  }
  gtsam::Matrix h=gtsam::Matrix::Zero(system.dimension,system.dimension);
  system.gradient=gtsam::Vector::Zero(system.dimension);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6,.1);
  auto add = [&](gtsam::Key from,gtsam::Key to) {
    const auto measured = values.at<gtsam::Pose3>(from).between(values.at<gtsam::Pose3>(to)).retract(
        (gtsam::Vector6()<<.02,-.03,.01,.1,-.05,.08).finished());
    ExactBetween3 factor(from,to,measured,noise);
    const auto linear=factor.linearize(values);
    const auto ab=linear->jacobian();
    gtsam::Matrix a=gtsam::Matrix::Zero(6,system.dimension);
    int col=0;
    for (auto key : linear->keys()) {
      if (key!=X(0)) a.middleCols(system.offsets.at(key),6)=ab.first.middleCols(col,6);
      col+=6;
    }
    h+=a.transpose()*a; system.gradient-=a.transpose()*ab.second;
  };
  for (int j=1; j<4; ++j) add(X(j-1),X(j));
  for (int j=0; j<4; ++j) for (int k=0; k<3; ++k) add(X(j),L(k));
  system.hessian=h.sparseView();
  NativeSchurModel reduced(system,{X(1),X(2),X(3)});
  const auto s=reduced.apply(gtsam::Matrix::Identity(18,18));
  const auto& g=reduced.gradient();
  gtsam::Matrix v(18,7);
  for (int j=0; j<3; ++j) {
    v.block<6,6>(6*j,0)=nativeRigidPose3Basis(values.at<gtsam::Pose3>(L(j)));
    v.block<6,1>(6*j,6)=s.block<6,6>(6*j,6*j).ldlt().solve(-g.segment<6>(6*j));
  }
  const auto solution=solveNativeSubspace(v.transpose()*s*v,v.transpose()*g);
  const gtsam::Vector candidate=v*solution.coefficients;
  const gtsam::Vector direction=v.col(6);
  const gtsam::Vector old=(-g.dot(direction)/direction.dot(s*direction))*direction;
  const double oldModel=g.dot(old)+.5*old.dot(s*old);
  const double newModel=g.dot(candidate)+.5*candidate.dot(s*candidate);
  EXPECT_LT(newModel,oldModel);
  EXPECT_EQ(solution.rank,7u);
  EXPECT_EQ(system.offsets.count(X(0)),0u);
  const auto recovered=reduced.recover(candidate);
  EXPECT_LE(reduced.privateRelativeResidual(recovered),1e-10);
  EXPECT_LE((v.transpose()*(s*candidate+g)).norm(),1e-9);
  auto precise = [](double value) {
    std::ostringstream out; out << std::setprecision(17) << value; return out.str();
  };
  const double alpha=-g.dot(direction)/direction.dot(s*direction);
  const double fusedError=(reduced.apply(alpha*direction)-alpha*reduced.apply(direction)).norm()/
      std::max(1.,(alpha*reduced.apply(direction)).norm());
  EXPECT_LE(fusedError,1e-12);
  RecordProperty("fused_product_relative_error",precise(fusedError));
  RecordProperty("old_model_decrease",precise(-oldModel));
  RecordProperty("subspace_model_decrease",precise(-newModel));
  RecordProperty("private_relative_residual",precise(reduced.privateRelativeResidual(recovered)));
}
}  // namespace
