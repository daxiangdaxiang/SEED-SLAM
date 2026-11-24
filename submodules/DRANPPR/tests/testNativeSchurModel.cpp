#include <DPGO/NativeSchurModel.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <Eigen/Cholesky>

#include <iomanip>
#include <limits>
#include <sstream>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

NativeModelUpdate nativeFixture() {
  NativeModelUpdate input;
  input.newValues.insert(X(0), gtsam::Pose2());
  input.newValues.insert(X(1), gtsam::Pose2(1.1, .1, .1));
  input.newValues.insert(L(0), gtsam::Point2(2., 1.));
  input.newValues.insert(L(1), gtsam::Point2(3., -.5));
  input.addFactors[0] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      X(0), X(1), gtsam::Pose2(1., 0., 0.),
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1, .2, .05)));
  gtsam::Matrix2 covariance;
  covariance << .01, .015, .015, .25;
  auto noise = gtsam::noiseModel::Gaussian::Covariance(covariance);
  input.addFactors[1] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(0), gtsam::Rot2(.5), 2.2, noise);
  input.addFactors[2] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(1), L(0), gtsam::Rot2(.6), 1.4, noise);
  input.addFactors[3] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(1), L(1), gtsam::Rot2(-.3), 2.1, noise);
  return input;
}

std::string precise(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

TEST(NativeSchurModel, NativeFixedGaugeStepMatchesFullSolve) {
  IncrementalLocalModel model;
  model.update(nativeFixture());
  // Exercise a nontrivial key order, without an artificial anchor prior.
  auto system = model.linearSystem({L(1), X(1), L(0)}, {X(0)});
  NativeSchurModel schur(system, {X(1)});
  EXPECT_EQ(schur.revision(), model.revision());
  EXPECT_EQ(schur.boundaryKeys(), (std::vector<gtsam::Key>{L(1), L(0)}));
  const gtsam::Matrix h(system.hessian);
  const gtsam::Vector direct = h.ldlt().solve(-system.gradient);
  const gtsam::Matrix reduced = schur.apply(gtsam::Matrix::Identity(4, 4));
  const gtsam::Vector boundary = reduced.ldlt().solve(-schur.gradient());
  const gtsam::Vector recovered = schur.recover(boundary);
  const double stepError = (recovered-direct).norm()/std::max(1., direct.norm());
  const double residual = (h*recovered+system.gradient).norm()/std::max(1., system.gradient.norm());
  EXPECT_LE(stepError, 1e-9);
  EXPECT_LE(residual, 1e-10);
  EXPECT_LE(schur.privateRelativeResidual(recovered), 1e-10);
  const gtsam::Matrix fullCovariance = h.ldlt().solve(gtsam::Matrix::Identity(7, 7));
  const gtsam::Matrix objectCovariance = reduced.ldlt().solve(gtsam::Matrix::Identity(4, 4));
  const std::vector<int> boundaryColumns{0, 1, 5, 6};
  gtsam::Matrix marginal(4, 4);
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
    marginal(i,j) = fullCovariance(boundaryColumns[i], boundaryColumns[j]);
  }
  EXPECT_LE((objectCovariance-marginal).norm()/marginal.norm(), 1e-10);
  RecordProperty("relative_step_error", precise(stepError));
  RecordProperty("relative_full_residual", precise(residual));
  RecordProperty("relative_private_residual", precise(schur.privateRelativeResidual(recovered)));
}

TEST(NativeSchurModel, MatrixFreeProductMatchesExplicitDenseElimination) {
  NativeLinearSystem system;
  system.keys = {11, 42, 99};
  system.offsets = {{11, 0}, {42, 2}, {99, 5}};
  system.dimensions = {{11, 2}, {42, 3}, {99, 2}};
  gtsam::Matrix a(10, 7);
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) a(i, j) = std::sin(.31*(i+1)*(j+1));
  }
  const gtsam::Matrix h = a.transpose()*a+gtsam::Matrix::Identity(7, 7);
  system.hessian = h.sparseView();
  system.gradient = gtsam::Vector::LinSpaced(7, -.3, .7);
  NativeSchurModel schur(system, {11, 99});
  const std::vector<int> p{0, 1, 5, 6};
  gtsam::Matrix pp(4, 4), pb(4, 3);
  gtsam::Vector gp(4);
  for (int i = 0; i < 4; ++i) {
    gp(i) = system.gradient(p[i]);
    for (int j = 0; j < 4; ++j) pp(i, j) = h(p[i], p[j]);
    pb.row(i) = h.row(p[i]).segment(2, 3);
  }
  const gtsam::Matrix expected = h.block(2, 2, 3, 3)-pb.transpose()*pp.ldlt().solve(pb);
  const gtsam::Vector gradient = system.gradient.segment(2, 3)-pb.transpose()*pp.ldlt().solve(gp);
  gtsam::Matrix v(3, 2);
  v << .2, -.1, .6, .3, -.2, .9;
  const double productError = (schur.apply(v)-expected*v).norm()/std::max(1., (expected*v).norm());
  EXPECT_LE(productError, 1e-10);
  EXPECT_LE((schur.gradient()-gradient).norm(), 1e-10);
  RecordProperty("relative_product_error", precise(productError));
}

TEST(NativeSchurModel, EmptyAndAllPrivatePartitions) {
  IncrementalLocalModel model;
  model.update(nativeFixture());
  auto system = model.linearSystem({}, {X(0)});
  NativeSchurModel noPrivate(system, {});
  EXPECT_LE((noPrivate.apply(gtsam::Matrix::Identity(7, 7))-gtsam::Matrix(system.hessian)).norm(), 1e-12);
  EXPECT_TRUE(noPrivate.gradient().isApprox(system.gradient, 0.));
  const gtsam::Vector direct = gtsam::Matrix(system.hessian).ldlt().solve(-system.gradient);
  EXPECT_TRUE(noPrivate.recover(direct).isApprox(direct, 0.));
  NativeSchurModel allPrivate(system, system.keys);
  EXPECT_EQ(allPrivate.gradient().size(), 0);
  EXPECT_EQ(allPrivate.apply(gtsam::Matrix(0, 2)).rows(), 0);
  EXPECT_LE((allPrivate.recover(gtsam::Vector(0))-direct).norm(), 1e-9);
  IncrementalLocalModel empty;
  NativeSchurModel emptySchur(empty.linearSystem(), {});
  EXPECT_EQ(emptySchur.recover(gtsam::Vector(0)).size(), 0);
}

TEST(NativeSchurModel, SnapshotRetainsItsOwnRevisionAfterUpdate) {
  IncrementalLocalModel model;
  model.update(nativeFixture());
  const auto before = model.linearSystem({}, {X(0)});
  NativeSchurModel old(before, {X(1)});
  const auto oldProduct = old.apply(gtsam::Matrix::Identity(4, 4));
  NativeModelUpdate change;
  change.relinearizeValues.insert(X(1), gtsam::Pose2(.8, -.1, -.15));
  model.update(change);
  NativeSchurModel current(model.linearSystem({}, {X(0)}), {X(1)});
  EXPECT_LT(old.revision(), current.revision());
  EXPECT_TRUE(old.apply(gtsam::Matrix::Identity(4, 4)).isApprox(oldProduct, 0.));
  EXPECT_GT((current.gradient()-old.gradient()).norm(), 1e-3);
}

TEST(NativeSchurModel, InvalidAndSingularModelsAreExplicit) {
  IncrementalLocalModel model;
  model.update(nativeFixture());
  auto system = model.linearSystem({}, {X(0)});
  EXPECT_THROW(NativeSchurModel(system, {X(99)}), std::invalid_argument);
  EXPECT_THROW(NativeSchurModel(system, {X(1), X(1)}), std::invalid_argument);
  auto bad = system;
  bad.hessian.setZero();
  EXPECT_THROW(NativeSchurModel(bad, {X(1)}), std::runtime_error);
  bad = system;
  bad.hessian.coeffRef(0, 1) += 1.;
  EXPECT_THROW(NativeSchurModel(bad, {X(1)}), std::invalid_argument);
  bad = system;
  bad.gradient(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(NativeSchurModel(bad, {X(1)}), std::invalid_argument);
  bad = system;
  bad.offsets[L(0)] = 1;
  EXPECT_THROW(NativeSchurModel(bad, {X(1)}), std::invalid_argument);
  // Without fixing the component gauge, the full private system is singular.
  auto unfixed = model.linearSystem();
  EXPECT_THROW(NativeSchurModel(unfixed, unfixed.keys), std::runtime_error);
  NativeSchurModel valid(system, {X(1)});
  EXPECT_THROW(valid.apply(gtsam::Matrix::Zero(3, 2)), std::invalid_argument);
  EXPECT_THROW(valid.recover(gtsam::Vector::Zero(3)), std::invalid_argument);
  EXPECT_THROW(valid.apply(gtsam::Matrix::Constant(4, 1,
      std::numeric_limits<double>::infinity())), std::invalid_argument);
}
}  // namespace
