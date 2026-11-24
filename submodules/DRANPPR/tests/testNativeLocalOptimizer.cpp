#include <DPGO/NativeBetweenPose2Factor.h>
#include <DPGO/NativeLocalOptimizer.h>
#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <Eigen/Cholesky>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::L;

NativeModelUpdate smallProblem(bool exact) {
  NativeModelUpdate input;
  const std::vector<gtsam::Pose2> truth{{}, {1., .1, .2}, {2., .4, .4}};
  input.newValues.insert(X(0), truth[0]);
  input.newValues.insert(X(1), gtsam::Pose2(1.3, -.2, .4));
  input.newValues.insert(X(2), gtsam::Pose2(2.3, .7, .1));
  input.newValues.insert(L(0), gtsam::Point2(3.3, 1.3));
  input.newValues.insert(L(1), gtsam::Point2(.3, 2.3));
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1, .2, .07));
  for (int i = 1; i < 3; ++i) {
    const auto relative = truth[i-1].between(truth[i]);
    if (exact) input.addFactors[i] = boost::make_shared<NativeBetweenPose2Factor>(X(i-1), X(i), relative, noise);
    else input.addFactors[i] = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose2>>(X(i-1), X(i), relative, noise);
  }
  const std::vector<gtsam::Point2> landmarks{{3., 1.}, {0., 2.}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      input.addFactors[3+2*i+j] = boost::make_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
          X(i), L(j), truth[i].bearing(landmarks[j]), truth[i].range(landmarks[j]),
          gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.03, .2)));
    }
  }
  return input;
}

TEST(NativeLocalOptimizer, CompleteChainRuleKeepsNativeResidual) {
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1, .2, .05));
  NativeBetweenPose2Factor exact(X(0), X(1), gtsam::Pose2(1., -.1, .2), noise);
  gtsam::BetweenFactor<gtsam::Pose2> pinned(X(0), X(1), exact.measured(), noise);
  const gtsam::Pose2 a(.1, -.3, .4), b(1.6, .5, -.5);
  gtsam::Matrix ja, jb, pa, pb;
  EXPECT_TRUE(exact.evaluateError(a, b, ja, jb).isApprox(pinned.evaluateError(a, b, pa, pb), 0.));
  std::function<gtsam::Vector3(const gtsam::Pose2&, const gtsam::Pose2&)> error =
      [&](const gtsam::Pose2& x, const gtsam::Pose2& y) { return exact.evaluateError(x, y); };
  const auto na = gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose2, gtsam::Pose2>(error, a, b, 1e-6);
  const auto nb = gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose2, gtsam::Pose2>(error, a, b, 1e-6);
  EXPECT_LE((ja-na).norm(), 1e-8);
  EXPECT_LE((jb-nb).norm(), 1e-8);
  EXPECT_GT((pa-na).norm(), .1);
  EXPECT_TRUE(bool(boost::dynamic_pointer_cast<NativeBetweenPose2Factor>(exact.clone())));
}

TEST(NativeLocalOptimizer, TrueCostDescentAndFixedGauge) {
  for (bool exact : {false, true}) {
    IncrementalLocalModel model;
    model.update(smallProblem(exact));
    const double initial = model.factorGraph().error(model.bases());
    auto result = optimizeNativeLocal(model, {X(1), X(2)}, {X(0)});
    EXPECT_GE(result.timing.graphCloneMs, 0.);
    EXPECT_GE(result.timing.exportMs, 0.);
    EXPECT_GE(result.timing.responseMs, result.timing.privateFactorizationMs);
    EXPECT_GE(result.timing.interfaceMs, 0.);
    EXPECT_GE(result.timing.recoveryMs, 0.);
    EXPECT_GE(result.timing.evaluationMs, 0.);
    EXPECT_GE(result.timing.publicationMs, result.timing.publicationLinearizationMs);
    ASSERT_FALSE(result.trace.empty());
    double previous = initial;
    for (const auto& step : result.trace) {
      EXPECT_LE(step.costAfter, previous);
      EXPECT_LE(step.alpha, 1.);
      previous = step.costAfter;
    }
    EXPECT_LT(previous, 1e-14);
    EXPECT_TRUE(model.bases().at<gtsam::Pose2>(X(0)).equals(gtsam::Pose2(), 0.));
    EXPECT_TRUE(model.bases().at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(3., 1.), 1e-8));
  }
}

TEST(NativeLocalOptimizer, SchurFullAndBatchAgreeOnSmallGraph) {
  auto input = smallProblem(true);
  IncrementalLocalModel reduced, full;
  reduced.update(input);
  full.update(input);
  auto graph = reduced.factorGraph();
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
      X(0), gtsam::Pose2(), gtsam::noiseModel::Constrained::All(3));
  gtsam::LevenbergMarquardtParams params;
  params.maxIterations = 100;
  params.absoluteErrorTol = 1e-12;
  params.relativeErrorTol = 1e-12;
  const auto batch = gtsam::LevenbergMarquardtOptimizer(graph, input.newValues, params).optimize();
  optimizeNativeLocal(reduced, {X(1), X(2)}, {X(0)});
  optimizeNativeLocal(full, {X(1), X(2), L(0), L(1)}, {X(0)});
  EXPECT_LE(reduced.bases().localCoordinates(full.bases()).norm(), 1e-8);
  EXPECT_LE(reduced.bases().localCoordinates(batch).norm(), 1e-7);
}

TEST(NativeLocalOptimizer, InvalidOptionsAndZeroBudgetDoNotMutate) {
  IncrementalLocalModel model;
  model.update(smallProblem(true));
  const auto revision = model.revision();
  NativeLocalOptions options;
  options.gradientTolerance = -1.;
  EXPECT_THROW(optimizeNativeLocal(model, {X(1), X(2)}, {X(0)}, options), std::invalid_argument);
  options.gradientTolerance = 1e-8;
  options.maxIterations = 0;
  EXPECT_EQ(optimizeNativeLocal(model, {X(1), X(2)}, {X(0)}, options).status, "iteration_limit");
  EXPECT_EQ(model.revision(), revision);
}

class OppositeJacobian : public gtsam::NoiseModelFactor1<gtsam::Point2> {
 public:
  OppositeJacobian() : NoiseModelFactor1(gtsam::noiseModel::Unit::Create(2), L(0)) {}
  gtsam::Vector evaluateError(const gtsam::Point2& point,
      boost::optional<gtsam::Matrix&> h = boost::none) const override {
    if (h) *h = -gtsam::Matrix2::Identity();
    return point;
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<OppositeJacobian>(*this);
  }
};

TEST(NativeLocalOptimizer, NonDescendingDirectionLeavesStateUnchanged) {
  NativeModelUpdate input;
  input.newValues.insert(L(0), gtsam::Point2(1., 1.));
  input.addFactors[0] = boost::make_shared<OppositeJacobian>();
  IncrementalLocalModel model;
  model.update(input);
  const auto revision = model.revision();
  const auto result = optimizeNativeLocal(model, {L(0)}, {});
  EXPECT_EQ(result.status, "no_decreasing_trial");
  EXPECT_EQ(model.revision(), revision);
  EXPECT_TRUE(model.bases().at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(1., 1.), 0.));
  ASSERT_EQ(result.trace.size(), 1u);
  EXPECT_EQ(result.trace[0].costEvaluations, 30u);
  EXPECT_DOUBLE_EQ(result.trace[0].alpha, 0.);
}

TEST(NativeLocalOptimizer, CachedNonzeroDeltaUsesOneDeclaredChart) {
  NativeModelUpdate input;
  input.newValues.insert(L(0), gtsam::Point2(0., 0.));
  input.addFactors[0] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
      L(0), gtsam::Point2(2., 1.), gtsam::noiseModel::Unit::Create(2));
  IncrementalLocalModel model;
  model.update(input);
  auto estimate = input.newValues;
  estimate.update(L(0), gtsam::Point2(.5, .25));
  const auto revision = model.revision();
  const auto result = optimizeNativeCachedLocal(model, estimate, {L(0)}, {}, 10.);
  EXPECT_TRUE(estimate.at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(2., 1.), 1e-12));
  EXPECT_TRUE(model.bases().at<gtsam::Point2>(L(0)).isZero());
  EXPECT_EQ(model.revision(), revision);
  EXPECT_EQ(result.modelRefreshFactors, 0u);
  EXPECT_EQ(result.modelRefreshKeys, 0u);
  EXPECT_EQ(result.status, "cached_model_gradient_small");
  EXPECT_LT(result.finalCost, 1e-20);
  ASSERT_FALSE(result.trace.empty());
  EXPECT_NEAR(result.trace[0].predictedDecrease, result.initialCost, 1e-12);
}

TEST(NativeLocalOptimizer, ZeroThresholdMatchesFreshSteps) {
  IncrementalLocalModel fresh, cached;
  auto input = smallProblem(true);
  fresh.update(input);
  cached.update(input);
  auto estimate = input.newValues;
  NativeLocalOptions options;
  options.maxIterations = 1;
  for (int i = 0; i < 5; ++i) {
    optimizeNativeLocal(fresh, {X(1), X(2)}, {X(0)}, options);
    const auto result = optimizeNativeCachedLocal(cached, estimate, {X(1), X(2)}, {X(0)}, 0., options);
    EXPECT_LE(fresh.bases().localCoordinates(estimate).norm(), 1e-8);
    EXPECT_LE(result.finalCost, result.initialCost);
    EXPECT_TRUE(estimate.at<gtsam::Pose2>(X(0)).equals(gtsam::Pose2(), 0.));
  }
}

TEST(NativeLocalOptimizer, CachedFactorReplacementAndRemovalAreNotStale) {
  NativeModelUpdate input;
  input.newValues.insert(L(0), gtsam::Point2(0., 0.));
  input.addFactors[0] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
      L(0), gtsam::Point2(2., 0.), gtsam::noiseModel::Unit::Create(2));
  input.addFactors[1] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
      L(0), gtsam::Point2(0., 2.), gtsam::noiseModel::Unit::Create(2));
  IncrementalLocalModel model;
  model.update(input);
  auto estimate = input.newValues;
  optimizeNativeCachedLocal(model, estimate, {L(0)}, {}, 100.);
  EXPECT_TRUE(estimate.at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(1., 1.), 1e-12));
  NativeModelUpdate edit;
  edit.replaceFactors[1] = boost::make_shared<gtsam::PriorFactor<gtsam::Point2>>(
      L(0), gtsam::Point2(0., 2.), gtsam::noiseModel::Isotropic::Sigma(2, .5));
  model.update(edit);
  optimizeNativeCachedLocal(model, estimate, {L(0)}, {}, 100.);
  EXPECT_TRUE(estimate.at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(.4, 1.6), 1e-12));
  edit = NativeModelUpdate();
  edit.removeFactors.insert(1);
  model.update(edit);
  optimizeNativeCachedLocal(model, estimate, {L(0)}, {}, 100.);
  EXPECT_TRUE(estimate.at<gtsam::Point2>(L(0)).isApprox(gtsam::Point2(2., 0.), 1e-12));
}

TEST(NativeLocalOptimizer, PartialPoseChartRefreshMatchesIndependentFullCorrection) {
  auto input = smallProblem(true);
  IncrementalLocalModel cached, oracle;
  cached.update(input);
  auto estimate = input.newValues;
  estimate.update(X(1), gtsam::Pose2(1.2, -.1, .35));
  estimate.update(L(0), gtsam::Point2(3.30001, 1.3));
  const auto before = estimate;
  input.newValues.update(X(1), estimate.at<gtsam::Pose2>(X(1)));
  oracle.update(input);
  const auto system = oracle.linearSystem({}, {X(0)});
  const auto initialDelta = oracle.bases().localCoordinates(before);
  gtsam::Vector d0(system.gradient.size());
  for (auto key : system.keys) {
    d0.segment(system.offsets.at(key), system.dimensions.at(key)) = initialDelta.at(key);
  }
  const gtsam::Matrix hessian(system.hessian);
  const gtsam::Vector step = hessian.ldlt().solve(-(system.gradient+hessian*d0));
  NativeLocalOptions options;
  options.maxIterations = 1;
  const auto result = optimizeNativeCachedLocal(cached, estimate, {X(1), X(2)}, {X(0)}, 1e-4, options);
  ASSERT_EQ(result.trace.size(), 1u);
  ASSERT_GT(result.trace[0].alpha, 0.);
  EXPECT_EQ(result.modelRefreshKeys, 1u);
  EXPECT_EQ(result.modelRefreshFactors, 4u);
  EXPECT_EQ(result.modelRefreshReused, 4u);
  EXPECT_LE(cached.bases().localCoordinates(oracle.bases()).norm(), 1e-14);
  gtsam::VectorValues delta;
  delta.insert(X(0), gtsam::Vector3::Zero());
  for (auto key : system.keys) {
    delta.insert(key, initialDelta.at(key)+result.trace[0].alpha*
        step.segment(system.offsets.at(key), system.dimensions.at(key)));
  }
  EXPECT_LE(estimate.localCoordinates(oracle.bases().retract(delta)).norm(), 1e-9);
  EXPECT_LT(result.finalCost, result.initialCost);
}

TEST(NativeLocalOptimizer, CachedInvalidInputAndRejectedTrialDoNotPublish) {
  IncrementalLocalModel model;
  auto input = smallProblem(true);
  model.update(input);
  auto estimate = input.newValues;
  const auto revision = model.revision();
  EXPECT_THROW(optimizeNativeCachedLocal(model, estimate, {}, {X(0)}, -1.), std::invalid_argument);
  EXPECT_THROW(optimizeNativeCachedLocal(model, estimate, {}, {X(0)},
      std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  estimate.erase(L(1));
  EXPECT_THROW(optimizeNativeCachedLocal(model, estimate, {}, {X(0)}, 1e-4), std::invalid_argument);
  estimate = input.newValues;
  estimate.update(X(0), gtsam::Pose2(.1, 0., 0.));
  EXPECT_THROW(optimizeNativeCachedLocal(model, estimate, {}, {X(0)}, 1e-4), std::invalid_argument);
  EXPECT_EQ(model.revision(), revision);
  IncrementalLocalModel wrong;
  NativeModelUpdate opposite;
  opposite.newValues.insert(L(0), gtsam::Point2(1., 1.));
  opposite.addFactors[0] = boost::make_shared<OppositeJacobian>();
  wrong.update(opposite);
  auto current = opposite.newValues;
  const auto result = optimizeNativeCachedLocal(wrong, current, {L(0)}, {}, 1.);
  EXPECT_EQ(result.status, "no_decreasing_trial");
  EXPECT_LE(current.localCoordinates(opposite.newValues).norm(), 1e-15);
  EXPECT_EQ(result.modelRefreshFactors, 0u);
}
}  // namespace
