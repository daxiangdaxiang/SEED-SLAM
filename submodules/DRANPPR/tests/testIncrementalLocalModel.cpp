#include <DPGO/IncrementalLocalModel.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <limits>
#include <iomanip>
#include <sstream>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

NativeModelUpdate fixture() {
  NativeModelUpdate update;
  update.newValues.insert(X(0), gtsam::Pose2(.2, -.1, .05));
  update.newValues.insert(X(1), gtsam::Pose2(1.05, .12, .12));
  update.newValues.insert(L(0), gtsam::Point2(2.2, 1.));
  update.newValues.insert(L(1), gtsam::Point2(3., -.5));
  auto odom = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1, .2, .05));
  gtsam::Matrix2 covariance;
  covariance << .0025, .001, .001, .04;
  auto polar = gtsam::noiseModel::Gaussian::Covariance(covariance);
  update.addFactors[0] = boost::make_shared<gtsam::PriorFactor<gtsam::Pose2>>(
      X(0), gtsam::Pose2(), odom);
  update.addFactors[1] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      X(0), X(1), gtsam::Pose2(1., 0., 0.), odom);
  update.addFactors[2] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(0), gtsam::Rot2(.4), 2.1, polar);
  update.addFactors[3] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(1), L(0), gtsam::Rot2(.6), 1.5, polar);
  update.addFactors[4] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(1), L(1), gtsam::Rot2(-.3), 2.2, polar);
  return update;
}

gtsam::NonlinearFactorGraph graphFrom(const NativeModelUpdate& update) {
  gtsam::NonlinearFactorGraph graph;
  for (const auto& entry : update.addFactors) graph.push_back(entry.second);
  return graph;
}

void expectParity(const IncrementalLocalModel& model,
                  const gtsam::NonlinearFactorGraph& expected,
                  const std::set<gtsam::Key>& fixed = {}) {
  const auto system = model.linearSystem({}, fixed);
  const auto activeKeys = expected.keys();
  gtsam::Ordering fullOrder(std::vector<gtsam::Key>(activeKeys.begin(), activeKeys.end()));
  auto ab = expected.linearize(model.bases())->jacobian(fullOrder);
  std::map<gtsam::Key, Eigen::Index> offsets;
  Eigen::Index offset = 0;
  for (auto key : fullOrder) {
    offsets[key] = offset;
    offset += model.bases().at(key).dim();
  }
  gtsam::Matrix a = gtsam::Matrix::Zero(ab.first.rows(), system.gradient.size());
  for (auto key : system.keys) {
    if (!offsets.count(key)) continue;
    a.middleCols(system.offsets.at(key), system.dimensions.at(key)) =
        ab.first.middleCols(offsets.at(key), system.dimensions.at(key));
  }
  gtsam::Matrix h = a.transpose()*a;
  gtsam::Vector g = -a.transpose()*ab.second;
  const double hError = (gtsam::Matrix(system.hessian)-h).norm()/std::max(1., h.norm());
  const double gError = (system.gradient-g).norm()/std::max(1., g.norm());
  EXPECT_LE(hError, 1e-10);
  EXPECT_LE(gError, 1e-10);
  EXPECT_NEAR(system.constant, .5*ab.second.squaredNorm(), 1e-10);
  std::ostringstream hs, gs;
  hs << std::setprecision(17) << hError;
  gs << std::setprecision(17) << gError;
  ::testing::Test::RecordProperty("relative_hessian_error", hs.str());
  ::testing::Test::RecordProperty("relative_gradient_error", gs.str());
}

TEST(IncrementalLocalModel, NativePolarFreshParityAndGradientSign) {
  IncrementalLocalModel model;
  auto update = fixture();
  auto stats = model.update(update);
  EXPECT_EQ(stats.factorsLinearized, 5u);
  EXPECT_EQ(stats.factorsReused, 0u);
  EXPECT_GE(stats.snapshotMs, 0.);
  EXPECT_GE(stats.linearizationMs, 0.);
  EXPECT_GE(stats.contributionMs, 0.);
  EXPECT_EQ(model.factorCount(), 5u);
  expectParity(model, graphFrom(update));
  const auto system = model.linearSystem();
  gtsam::VectorValues delta;
  for (auto key : system.keys) delta.insert(key, gtsam::Vector::Ones(system.dimensions.at(key)));
  const double eps = 1e-6;
  auto graph = graphFrom(update);
  auto linear = graph.linearize(model.bases());
  double difference = (linear->error(eps*delta)-linear->error(-eps*delta))/(2*eps);
  EXPECT_NEAR(difference, system.gradient.sum(), 1e-5);
  // Pinned GTSAM uses approximate Between/Prior Jacobians away from zero
  // residual. Cache parity is not a claim of exact nonlinear differentiation.
  double physicalDifference = (graph.error(model.bases().retract(eps*delta))-
                               graph.error(model.bases().retract(-eps*delta)))/(2*eps);
  RecordProperty("native_physical_directional_gradient_gap",
                 std::to_string(std::abs(physicalDifference-system.gradient.sum())));
}

void expectSquareRootParity(const IncrementalLocalModel& square,
                           const IncrementalLocalModel& normal) {
  for (const auto& fixed : std::vector<std::set<gtsam::Key>>{{}, {X(0)}, {L(0)}}) {
    const auto layout = square.linearLayout({}, fixed);
    const auto reference = normal.linearSystem({}, fixed);
    ASSERT_EQ(layout.keys, reference.keys);
    gtsam::Matrix h = gtsam::Matrix::Zero(layout.dimension, layout.dimension);
    gtsam::Vector g = gtsam::Vector::Zero(layout.dimension);
    for (const auto& item : square.squareRootFactors()) {
      const auto& factor = *item.second;
      gtsam::Matrix a = gtsam::Matrix::Zero(factor.rows(), layout.dimension);
      for (auto it=factor.begin(); it!=factor.end(); ++it) if (!fixed.count(*it)) {
        a.middleCols(layout.offsets.at(*it), layout.dimensions.at(*it)) = factor.getA(it);
      }
      h += a.transpose()*a;
      g -= a.transpose()*factor.getb();
    }
    EXPECT_LE((h-gtsam::Matrix(reference.hessian)).norm()/std::max(1.,h.norm()), 1e-12);
    EXPECT_LE((g-reference.gradient).norm()/std::max(1.,g.norm()), 1e-12);
    const gtsam::Vector d = gtsam::Vector::LinSpaced(layout.dimension, -.03, .05);
    const gtsam::Vector s = gtsam::Vector::LinSpaced(layout.dimension, .02, -.04);
    const auto terms = square.directionalModel(layout, d, s);
    EXPECT_NEAR(terms[0], (reference.gradient+reference.hessian*d).dot(s), 1e-10);
    EXPECT_NEAR(terms[1], s.dot(reference.hessian*s), 1e-10);
  }
}

TEST(IncrementalLocalModel, SquareRootOnlyMatchesNormalWithoutNormalAssembly) {
  IncrementalLocalModel square(true), normal;
  const auto input = fixture();
  square.update(input); normal.update(input);
  expectSquareRootParity(square, normal);
  EXPECT_THROW(square.linearSystem(), std::logic_error);
  EXPECT_THROW(normal.squareRootFactors(), std::logic_error);
  EXPECT_THROW(square.linearLayout({}, {X(99)}), std::invalid_argument);
  EXPECT_THROW(square.linearLayout({L(0), L(0)}), std::invalid_argument);
}

TEST(IncrementalLocalModel, SquareRootIdentityTracksOnlyChangedFactors) {
  IncrementalLocalModel square(true), normal;
  const auto input = fixture();
  square.update(input); normal.update(input);
  const auto before = square.squareRootFactors();
  const auto oldLayout = square.linearLayout();
  NativeModelUpdate edit;
  edit.relinearizeValues.insert(X(1), gtsam::Pose2(1.2,.3,-.4));
  EXPECT_EQ(square.update(edit).factorsLinearized, 3u); normal.update(edit);
  const auto after = square.squareRootFactors();
  for (auto id : {0u, 2u}) EXPECT_EQ(before.at(id), after.at(id));
  for (auto id : {1u, 3u, 4u}) EXPECT_NE(before.at(id), after.at(id));
  expectSquareRootParity(square, normal);
  EXPECT_THROW(square.directionalModel(oldLayout, gtsam::Vector::Zero(oldLayout.dimension),
      gtsam::Vector::Zero(oldLayout.dimension)), std::invalid_argument);
  edit = NativeModelUpdate();
  edit.removeFactors.insert(2);
  edit.replaceFactors[3] = input.addFactors.at(4);
  edit.newValues.insert(X(2), gtsam::Pose2(2.,0.,0.));
  edit.addFactors[5] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      X(1), X(2), gtsam::Pose2(1.,0.,0.), gtsam::noiseModel::Isotropic::Sigma(3,.1));
  square.update(edit); normal.update(edit);
  expectSquareRootParity(square, normal);
  EXPECT_EQ(square.squareRootFactors().count(2), 0u);
}

TEST(IncrementalLocalModel, SquareRootFailedUpdatePreservesPublishedFactors) {
  IncrementalLocalModel square(true);
  square.update(fixture());
  const auto before = square.squareRootFactors();
  const auto revision = square.revision();
  NativeModelUpdate bad;
  bad.relinearizeValues.insert(X(1), gtsam::Pose2(1.,0.,0.));
  bad.addFactors[100] = nullptr;
  EXPECT_THROW(square.update(bad), std::invalid_argument);
  EXPECT_EQ(square.revision(), revision);
  EXPECT_EQ(square.squareRootFactors(), before);
  EXPECT_EQ(square.update(NativeModelUpdate()).factorsLinearized, 0u);
  EXPECT_EQ(square.squareRootFactors(), before);
}

TEST(IncrementalLocalModel, PolarGradientMatchesNonlinearFiniteDifference) {
  auto input = fixture();
  input.addFactors.erase(0);
  input.addFactors.erase(1);
  IncrementalLocalModel model;
  model.update(input);
  auto system = model.linearSystem();
  auto graph = graphFrom(input);
  gtsam::VectorValues delta;
  for (auto key : system.keys) delta.insert(key, gtsam::Vector::Ones(system.dimensions.at(key)));
  const double eps = 1e-6;
  const double difference = (graph.error(model.bases().retract(eps*delta))-
                            graph.error(model.bases().retract(-eps*delta)))/(2*eps);
  EXPECT_NEAR(difference, system.gradient.sum(), 1e-5);
}

TEST(IncrementalLocalModel, AppendingPoseAfterObjectsPreservesExistingOffsets) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  auto before = model.linearSystem();
  NativeModelUpdate append;
  append.newValues.insert(X(2), gtsam::Pose2(2., 0., 0.));
  append.addFactors[5] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      X(1), X(2), gtsam::Pose2(1., 0., 0.),
      gtsam::noiseModel::Isotropic::Sigma(3, .1));
  auto stats = model.update(append);
  EXPECT_EQ(stats.factorsLinearized, 1u);
  EXPECT_EQ(stats.factorsReused, 5u);
  auto after = model.linearSystem();
  for (auto key : before.keys) EXPECT_EQ(before.offsets.at(key), after.offsets.at(key));
  auto expected = graphFrom(input);
  expected.push_back(append.addFactors.at(5));
  expectParity(model, expected);
}

TEST(IncrementalLocalModel, ChangedChartInvalidatesAllIncidentOldFactors) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  auto before = model.linearSystem();
  NativeModelUpdate change;
  change.relinearizeValues.insert(X(1), gtsam::Pose2(1.2, .3, -.4));
  auto stats = model.update(change);
  EXPECT_EQ(stats.factorsLinearized, 3u);
  EXPECT_EQ(stats.factorsReused, 2u);
  EXPECT_DOUBLE_EQ(model.bases().at<gtsam::Pose2>(X(1)).x(), 1.2);
  EXPECT_GT((model.linearSystem().gradient-before.gradient).norm(), 1.);
  expectParity(model, graphFrom(input));
}

TEST(IncrementalLocalModel, RemoveAndReplaceSubtractExactlyOnce) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  NativeModelUpdate edit;
  edit.removeFactors.insert(2);
  edit.replaceFactors[3] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(1), gtsam::Rot2(-.1), 2.9,
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.1, .5)));
  auto stats = model.update(edit);
  EXPECT_EQ(stats.factorsLinearized, 1u);
  EXPECT_EQ(stats.factorsReused, 3u);
  EXPECT_EQ(stats.factorsRemoved, 1u);
  EXPECT_EQ(stats.factorsReplaced, 1u);
  EXPECT_EQ(model.factorCount(), 4u);
  input.addFactors.erase(2);
  input.addFactors[3] = edit.replaceFactors.at(3);
  expectParity(model, graphFrom(input));
  NativeModelUpdate chart;
  chart.relinearizeValues.insert(L(0), gtsam::Point2(2.3, .8));
  EXPECT_EQ(model.update(chart).factorsLinearized, 0u);
}

TEST(IncrementalLocalModel, ExactFixedColumnsAndInvalidOrdering) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  expectParity(model, graphFrom(input), {X(0)});
  auto system = model.linearSystem({}, {X(0)});
  EXPECT_EQ(system.gradient.size(), 7);
  EXPECT_THROW(model.linearSystem({L(0), L(0)}), std::invalid_argument);
  EXPECT_THROW(model.linearSystem({}, {X(99)}), std::invalid_argument);
}

TEST(IncrementalLocalModel, RejectedTransactionsLeavePublishedModelUnchanged) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  const auto before = model.linearSystem();
  const auto revision = model.revision();
  std::vector<NativeModelUpdate> bad(6);
  bad[0].newValues.insert(X(0), gtsam::Pose2());
  bad[1].addFactors[0] = input.addFactors.at(0);
  bad[2].removeFactors.insert(99);
  bad[3].relinearizeValues.insert(X(1), gtsam::Point2(1., 2.));
  bad[4].newValues.insert(X(2), gtsam::Pose2(std::numeric_limits<double>::quiet_NaN(), 0., 0.));
  bad[5].newValues.insert(X(2), gtsam::Pose2());
  bad[5].addFactors[5] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
      X(2), X(99), gtsam::Pose2(), gtsam::noiseModel::Isotropic::Sigma(3, .1));
  for (const auto& edit : bad) {
    EXPECT_THROW(model.update(edit), std::exception);
    EXPECT_EQ(model.revision(), revision);
    EXPECT_FALSE(model.bases().exists(X(2)));
    EXPECT_TRUE(gtsam::Matrix(model.linearSystem().hessian).isApprox(gtsam::Matrix(before.hessian), 0.));
    EXPECT_TRUE(model.linearSystem().gradient.isApprox(before.gradient, 0.));
    EXPECT_DOUBLE_EQ(model.linearSystem().constant, before.constant);
  }
}

TEST(IncrementalLocalModel, UnsupportedNoiseAndInvalidLinearizationsAreAtomic) {
  IncrementalLocalModel model;
  model.update(fixture());
  const auto before = model.linearSystem();
  const std::vector<gtsam::SharedNoiseModel> invalidNoise{
      gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(1.345),
                                      gtsam::noiseModel::Isotropic::Sigma(3, .1)),
      gtsam::noiseModel::Constrained::All(3),
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(-.1, .2, .3)),
      gtsam::SharedNoiseModel()};
  for (auto noise : invalidNoise) {
    NativeModelUpdate edit;
    edit.replaceFactors[1] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
        X(0), X(1), gtsam::Pose2(), noise);
    EXPECT_THROW(model.update(edit), std::invalid_argument);
    EXPECT_EQ(model.revision(), before.revision);
  }
  NativeModelUpdate edit;
  edit.newValues.insert(X(2), gtsam::Pose2());
  edit.replaceFactors[2] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(0), gtsam::Rot2(.3), std::numeric_limits<double>::quiet_NaN(),
      gtsam::noiseModel::Isotropic::Sigma(2, .1));
  EXPECT_THROW(model.update(edit), std::invalid_argument);
  EXPECT_FALSE(model.bases().exists(X(2)));
  edit.newValues.clear();
  edit.replaceFactors[2].reset();
  EXPECT_THROW(model.update(edit), std::invalid_argument);
  EXPECT_EQ(model.revision(), before.revision);
  EXPECT_TRUE(gtsam::Matrix(model.linearSystem().hessian).isApprox(gtsam::Matrix(before.hessian), 0.));
  EXPECT_TRUE(model.linearSystem().gradient.isApprox(before.gradient, 0.));
  EXPECT_DOUBLE_EQ(model.linearSystem().constant, before.constant);
}

TEST(IncrementalLocalModel, CovarianceReplacementAndChartChangeUseOneNewModel) {
  auto input = fixture();
  IncrementalLocalModel model;
  model.update(input);
  NativeModelUpdate edit;
  edit.replaceFactors[2] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(0), gtsam::Rot2(.4), 2.1,
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.02, .7)));
  edit.relinearizeValues.insert(X(1), gtsam::Pose2(1.2, .2, -.1));
  edit.removeFactors.insert(3);
  auto stats = model.update(edit);
  EXPECT_EQ(stats.factorsLinearized, 3u);
  EXPECT_EQ(stats.factorsReused, 1u);
  input.addFactors[2] = edit.replaceFactors.at(2);
  input.addFactors.erase(3);
  expectParity(model, graphFrom(input));
}

TEST(IncrementalLocalModel, ClonedFactorsIsolateCallerAndInspectionMutations) {
  auto input = fixture();
  IncrementalLocalModel model;
  model.update(input);
  const auto expected = model.factorGraph();
  input.addFactors.at(2)->keys()[0] = X(99);
  auto inspected = model.factorGraph();
  inspected.at(3)->keys()[0] = X(99);
  NativeModelUpdate edit;
  edit.relinearizeValues.insert(L(0), gtsam::Point2(2.1, .9));
  EXPECT_NO_THROW(model.update(edit));
  expectParity(model, expected);
}

TEST(IncrementalLocalModel, RemoveAllAndNoOpLeaveNoResidualInformation) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  const auto revision = model.revision();
  EXPECT_EQ(model.update(NativeModelUpdate()).revision, revision);
  NativeModelUpdate remove;
  for (const auto& entry : input.addFactors) remove.removeFactors.insert(entry.first);
  model.update(remove);
  auto system = model.linearSystem();
  EXPECT_EQ(system.hessian.nonZeros(), 0);
  EXPECT_DOUBLE_EQ(system.gradient.norm(), 0.);
  EXPECT_DOUBLE_EQ(system.constant, 0.);
  EXPECT_EQ(model.factorCount(), 0u);
  EXPECT_EQ(system.gradient.size(), 10);
  EXPECT_EQ(model.linearSystem({}, {X(0), X(1), L(0), L(1)}).gradient.size(), 0);
}

TEST(IncrementalLocalModel, RepeatedEditsMatchFreshGraphAtEveryRevision) {
  IncrementalLocalModel model;
  auto input = fixture();
  model.update(input);
  for (int step = 0; step < 40; ++step) {
    NativeModelUpdate edit;
    edit.relinearizeValues.insert(X(1), gtsam::Pose2(1.+.01*step, .1, .1-.005*step));
    edit.replaceFactors[2] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
        X(0), L(0), gtsam::Rot2(.4), 2.1,
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.03+.001*step, .3)));
    const auto stats = model.update(edit);
    EXPECT_EQ(stats.factorsLinearized, 4u);
    EXPECT_EQ(stats.factorsReused, 1u);
    input.addFactors[2] = edit.replaceFactors.at(2);
    expectParity(model, graphFrom(input));
  }
  NativeModelUpdate remove;
  for (const auto& entry : input.addFactors) remove.removeFactors.insert(entry.first);
  EXPECT_EQ(model.update(remove).factorsRemoved, 5u);
  EXPECT_EQ(model.linearSystem().hessian.nonZeros(), 0);
  EXPECT_DOUBLE_EQ(model.linearSystem().gradient.norm(), 0.);
  EXPECT_DOUBLE_EQ(model.linearSystem().constant, 0.);
}

TEST(IncrementalLocalModel, MixedReplacementAndStructuralEditsPreserveIncidence) {
  auto input = fixture();
  IncrementalLocalModel model;
  model.update(input);
  NativeModelUpdate edit;
  edit.removeFactors.insert(2);
  edit.replaceFactors[3] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(0), L(1), gtsam::Rot2(-.1), 2.9,
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.1, .5)));
  edit.replaceFactors[4] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
      X(1), L(1), gtsam::Rot2(-.2), 2.3,
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.2, .4)));
  const NativeFactorId largeId = std::numeric_limits<NativeFactorId>::max();
  edit.addFactors[largeId] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
      L(0), gtsam::Point2(2., 1.), gtsam::noiseModel::Isotropic::Sigma(2, .3));
  edit.relinearizeValues.insert(X(1), gtsam::Pose2(1.1, .2, .3));
  auto stats = model.update(edit);
  EXPECT_EQ(stats.factorsLinearized, 4u);
  EXPECT_EQ(stats.factorsReplaced, 2u);
  input.addFactors.erase(2);
  for (const auto& entry : edit.replaceFactors) input.addFactors[entry.first] = entry.second;
  input.addFactors[largeId] = edit.addFactors.at(largeId);
  expectParity(model, graphFrom(input));
  NativeModelUpdate refresh;
  refresh.relinearizeValues.insert(L(0), gtsam::Point2(2.2, .8));
  stats = model.update(refresh);
  EXPECT_EQ(stats.factorsLinearized, 1u);
  EXPECT_EQ(stats.factorsReused, 4u);
  expectParity(model, graphFrom(input));
  NativeModelUpdate remove;
  remove.removeFactors.insert(largeId);
  model.update(remove);
  EXPECT_EQ(model.update(refresh).factorsLinearized, 0u);
  input.addFactors.erase(largeId);
  expectParity(model, graphFrom(input));
}

TEST(IncrementalLocalModel, ReplacementAggregateOverflowRollsBack) {
  NativeModelUpdate input;
  input.newValues.insert(L(0), gtsam::Point2(0., 0.));
  for (NativeFactorId id : {8u, 17u}) {
    input.addFactors[id] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
        L(0), gtsam::Point2(0., 0.), gtsam::noiseModel::Isotropic::Sigma(2, 1.));
  }
  IncrementalLocalModel model;
  model.update(input);
  const auto before = model.linearSystem();
  NativeModelUpdate replace;
  for (const auto& entry : input.addFactors) {
    replace.replaceFactors[entry.first] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
        L(0), gtsam::Point2(0., 0.), gtsam::noiseModel::Isotropic::Sigma(2, 1e-154));
  }
  EXPECT_THROW(model.update(replace), std::invalid_argument);
  EXPECT_EQ(model.revision(), before.revision);
  EXPECT_TRUE(gtsam::Matrix(model.linearSystem().hessian).isApprox(gtsam::Matrix(before.hessian), 0.));
  EXPECT_TRUE(model.linearSystem().gradient.isApprox(before.gradient, 0.));
  EXPECT_DOUBLE_EQ(model.linearSystem().constant, before.constant);
  NativeModelUpdate refresh;
  refresh.relinearizeValues.insert(L(0), gtsam::Point2(.2, -.1));
  EXPECT_EQ(model.update(refresh).factorsLinearized, 2u);
  expectParity(model, graphFrom(input));
}
}  // namespace
