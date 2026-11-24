#include <DPGO/IncrementalConditionalModel.h>
#include <DPGO/NativeInterface.h>
#include <gtest/gtest.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <Eigen/Cholesky>

#include <limits>

namespace {
using namespace DPGO;
using Key = gtsam::Key;
using Factors = std::map<NativeFactorId, gtsam::GaussianFactor::shared_ptr>;
using Dimensions = std::map<Key, Eigen::Index>;
constexpr Key boundaryKey = 1000;

gtsam::GaussianFactor::shared_ptr unary(Key key, double scale, double rhs) {
  return boost::make_shared<gtsam::JacobianFactor>(key,
      (scale*gtsam::Matrix2::Identity()).eval(), gtsam::Vector2(rhs, -.3*rhs));
}
gtsam::GaussianFactor::shared_ptr binary(Key a, Key b, double scale = 1.) {
  return boost::make_shared<gtsam::JacobianFactor>(a, gtsam::Matrix2::Identity(),
      b, (-scale*gtsam::Matrix2::Identity()).eval(), gtsam::Vector2(.2, -.1));
}

struct Fixture {
  std::vector<Key> keys{boundaryKey}, privateKeys;
  Dimensions dimensions{{boundaryKey, 2}};
  Factors factors{{9000, unary(boundaryKey, .7, .3)}};
  void append(std::size_t count) {
    while (privateKeys.size() < count) {
      const Key key = privateKeys.size();
      keys.push_back(key); privateKeys.push_back(key); dimensions[key] = 2;
      factors[2*key] = key == 0 ? unary(key, 2., .1) : binary(key-1, key);
      factors[2*key+1] = binary(key, boundaryKey, .3);
    }
  }
};

NativeLinearSystem freshSystem(const Fixture& f) {
  NativeLinearSystem system;
  system.keys = f.keys; system.dimensions = f.dimensions;
  Eigen::Index size = 0;
  for (auto key : f.keys) { system.offsets[key] = size; size += f.dimensions.at(key); }
  gtsam::GaussianFactorGraph graph;
  for (const auto& factor : f.factors) graph.push_back(factor.second);
  if (graph.empty()) {
    system.hessian.resize(size, size); system.gradient = gtsam::Vector::Zero(size);
  } else {
    const auto hg = graph.hessian(gtsam::Ordering(f.keys));
    system.hessian = hg.first.sparseView(); system.gradient = -hg.second;
  }
  return system;
}

void parity(const Fixture& f, const std::shared_ptr<const NativeLocalResponse>& response) {
  const auto system = freshSystem(f);
  NativeSchurModel fresh(system, f.privateKeys);
  ASSERT_EQ(response->boundaryKeys(), fresh.boundaryKeys());
  ASSERT_EQ(response->boundaryDimensions(), fresh.boundaryDimensions());
  const auto n = fresh.gradient().size();
  const auto identity = gtsam::Matrix::Identity(n, n).eval();
  const auto expected = fresh.apply(identity);
  EXPECT_LE((response->apply(identity)-expected).norm()/std::max(1., expected.norm()), 1e-10);
  EXPECT_LE((response->gradient()-fresh.gradient()).norm()/std::max(1., fresh.gradient().norm()), 1e-10);
  const gtsam::Vector b = gtsam::Vector::LinSpaced(n, -.25, .15);
  const auto recovered = response->recover(b);
  EXPECT_LE((recovered-fresh.recover(b)).norm()/std::max(1., recovered.norm()), 1e-9);
  EXPECT_LE(fresh.privateRelativeResidual(recovered), 1e-10);
  EXPECT_LE(response->privateRelativeResidual(recovered), 1e-10);
}

TEST(IncrementalConditionalModel, FreshParityAndAppendReuse) {
  Fixture f; f.append(8);
  IncrementalConditionalModel model;
  NativeGaussianUpdate first; first.addFactors = f.factors;
  auto stats = model.update(first, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 8u);
  parity(f, model.response());
  const auto old = model.response();
  const auto oldProduct = old->apply(gtsam::Matrix2::Identity());
  f.append(9);
  NativeGaussianUpdate add;
  add.addFactors = {{16, f.factors.at(16)}, {17, f.factors.at(17)}};
  stats = model.update(add, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 2u);
  EXPECT_EQ(stats.conditionalsReused, 7u);
  EXPECT_FALSE(stats.fullRebuild);
  parity(f, model.response());
  EXPECT_LT(old->revision(), model.response()->revision());
  EXPECT_TRUE(old->apply(gtsam::Matrix2::Identity()).isApprox(oldProduct, 0.));
}

TEST(IncrementalConditionalModel, ReplacementRemovalAndWholeHistoryInvalidation) {
  Fixture f; f.append(8);
  IncrementalConditionalModel model;
  NativeGaussianUpdate first; first.addFactors = f.factors;
  model.update(first, f.keys, f.dimensions, f.privateKeys);
  NativeGaussianUpdate replace;
  f.factors[13] = binary(6, boundaryKey, 1.1);
  replace.replaceFactors[13] = f.factors.at(13);
  auto stats = model.update(replace, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 2u); EXPECT_EQ(stats.conditionalsReused, 6u);
  parity(f, model.response());
  NativeGaussianUpdate remove; remove.removeFactors.insert(13); f.factors.erase(13);
  model.update(remove, f.keys, f.dimensions, f.privateKeys);
  parity(f, model.response());
  replace.replaceFactors.clear();
  f.factors[0] = unary(0, 3., .4); replace.replaceFactors[0] = f.factors.at(0);
  stats = model.update(replace, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 8u); EXPECT_EQ(stats.conditionalsReused, 0u);
  parity(f, model.response());
}

TEST(IncrementalConditionalModel, IndependentBranchAndBoundaryUpdatesReuseConditionals) {
  Fixture f;
  f.keys = {boundaryKey, 1, 2, 3}; f.privateKeys = {1, 2, 3};
  f.dimensions = {{boundaryKey, 2}, {1, 2}, {2, 2}, {3, 2}};
  f.factors = {{0, unary(1, 2., .2)}, {1, binary(1,3)}, {2, unary(2, 2., .5)},
               {3, binary(2,3)}, {4, binary(3,boundaryKey)}, {5, unary(boundaryKey, 1., .1)}};
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors = f.factors;
  model.update(update, f.keys, f.dimensions, f.privateKeys);
  update = {}; f.factors[0] = unary(1, 4., .1); update.replaceFactors[0] = f.factors.at(0);
  auto stats = model.update(update, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 2u); EXPECT_EQ(stats.conditionalsReused, 1u);
  parity(f, model.response());
  update = {}; f.factors[5] = unary(boundaryKey, 3., .2); update.replaceFactors[5] = f.factors.at(5);
  stats = model.update(update, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 0u); EXPECT_EQ(stats.conditionalsReused, 3u);
  parity(f, model.response());
}

TEST(IncrementalConditionalModel, LayoutChangesRebuildAndNoOpPreservesRevision) {
  Fixture f; f.append(4);
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors = f.factors;
  model.update(update, f.keys, f.dimensions, f.privateKeys);
  const auto old = model.response();
  auto stats = model.update({}, f.keys, f.dimensions, f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt, 0u); EXPECT_EQ(model.response(), old);
  f.privateKeys = {3, 2, 1, 0};
  stats = model.update({}, f.keys, f.dimensions, f.privateKeys);
  EXPECT_TRUE(stats.fullRebuild); parity(f, model.response());
  f.privateKeys.pop_back();
  stats = model.update({}, f.keys, f.dimensions, f.privateKeys);
  EXPECT_TRUE(stats.fullRebuild); parity(f, model.response());
}

TEST(IncrementalConditionalModel, FailedEditsAreAtomicAndInputsAreIsolated) {
  Fixture f; f.append(3);
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors = f.factors;
  model.update(update, f.keys, f.dimensions, f.privateKeys);
  const auto old = model.response();
  update = {}; update.removeFactors.insert(12345);
  EXPECT_THROW(model.update(update, f.keys, f.dimensions, f.privateKeys), std::invalid_argument);
  EXPECT_EQ(old, model.response());
  update = {}; update.addFactors[0] = f.factors.at(0);
  EXPECT_THROW(model.update(update, f.keys, f.dimensions, f.privateKeys), std::invalid_argument);
  update = {}; update.replaceFactors[0] = unary(0, std::numeric_limits<double>::quiet_NaN(), 0.);
  EXPECT_THROW(model.update(update, f.keys, f.dimensions, f.privateKeys), std::invalid_argument);
  EXPECT_EQ(old, model.response());
  update = {}; update.removeFactors = {0,1,2};
  EXPECT_THROW(model.update(update, f.keys, f.dimensions, f.privateKeys), std::exception);
  EXPECT_EQ(old, model.response());
  auto invalid = f.dimensions; invalid[0] = 3;
  EXPECT_THROW(model.update({}, f.keys, invalid, f.privateKeys), std::invalid_argument);
  auto external = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(f.factors.at(0));
  external->getb().setConstant(500.);
  update = {}; update.replaceFactors[9000] = unary(boundaryKey, 2., .2);
  model.update(update, f.keys, f.dimensions, f.privateKeys);
  f.factors[0] = unary(0, 2., .1); f.factors[9000] = update.replaceFactors.at(9000);
  parity(f, model.response());
}

TEST(IncrementalConditionalModel, EmptyAllPrivateAndUnobservableBoundary) {
  IncrementalConditionalModel model;
  model.update({}, {}, {}, {});
  EXPECT_EQ(model.response()->recover(gtsam::Vector(0)).size(), 0);
  Fixture f; f.append(4);
  NativeGaussianUpdate update; update.addFactors = f.factors;
  f.privateKeys = f.keys;
  model.update(update, f.keys, f.dimensions, f.privateKeys);
  parity(f, model.response());
  f.privateKeys.clear();
  model.update({}, f.keys, f.dimensions, f.privateKeys); parity(f, model.response());
  IncrementalConditionalModel free;
  free.update({}, {boundaryKey}, {{boundaryKey,2}}, {});
  EXPECT_EQ(free.response()->apply(gtsam::Matrix2::Identity()).norm(), 0.);
  EXPECT_EQ(free.response()->gradient().norm(), 0.);
  EXPECT_THROW(free.update({}, {boundaryKey}, {{boundaryKey,2}}, {boundaryKey}), std::exception);
  EXPECT_THROW(free.response()->recover(gtsam::Vector(3)), std::invalid_argument);
  EXPECT_THROW(free.response()->apply(gtsam::Matrix::Constant(2, 1,
      std::numeric_limits<double>::infinity())), std::invalid_argument);
}

TEST(IncrementalConditionalModel, RoutedShiftedResponsesMatchFreshSchur) {
  std::vector<NativeInterfaceEndpoint> fresh, cached;
  for (std::uint32_t robot=0; robot<3; ++robot) {
    Fixture f; f.append(5);
    f.factors[0] = unary(0, 2.+robot, .1+robot);
    IncrementalConditionalModel model;
    NativeGaussianUpdate update; update.addFactors = f.factors;
    model.update(update, f.keys, f.dimensions, f.privateKeys);
    NativeInterfaceEndpoint endpoint;
    endpoint.robot = robot; endpoint.chartBases[boundaryKey] = gtsam::Point2(1.,2.);
    endpoint.chartDisplacements[boundaryKey] = gtsam::Point2(.1,-.2);
    auto system = freshSystem(f); system.revision = model.response()->revision();
    endpoint.response = std::make_shared<NativeSchurModel>(system, f.privateKeys);
    fresh.push_back(endpoint); endpoint.response = model.response(); cached.push_back(endpoint);
  }
  NativeInterfaceOptions options; options.relativeTolerance = 1e-11;
  const auto a = solveNativeInterface(fresh, {{0,1},{1,2}}, options);
  const auto b = solveNativeInterface(cached, {{0,1},{1,2}}, options);
  EXPECT_LE(b.relativeResidual, 1e-11);
  EXPECT_EQ(a.iterations, b.iterations); EXPECT_EQ(a.applicationBytes, b.applicationBytes);
  for (std::uint32_t r=0; r<3; ++r) EXPECT_LE((a.localSteps.at(r)-b.localSteps.at(r)).norm(), 1e-9);
  RecordProperty("pcg_iterations", int(b.iterations));
  RecordProperty("pcg_relative_residual", b.relativeResidual);
}

TEST(IncrementalConditionalModel, MixedDimensionsAndUnanchoredGauge) {
  Fixture f;
  f.keys={boundaryKey,1}; f.privateKeys={1}; f.dimensions={{boundaryKey,2},{1,3}};
  gtsam::Matrix a(5,5);
  for (int i=0; i<5; ++i) for (int j=0; j<5; ++j) a(i,j)=std::sin(.13*(i+1)*(j+1))+(i==j?2.:0.);
  f.factors={{0,boost::make_shared<gtsam::JacobianFactor>(1,a.leftCols(3).eval(),
      boundaryKey,a.rightCols(2).eval(),gtsam::Vector::LinSpaced(5,-.2,.4))}};
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors=f.factors;
  model.update(update,f.keys,f.dimensions,f.privateKeys); parity(f,model.response());
  IncrementalConditionalModel gauge;
  update={}; update.addFactors[0]=binary(1,2);
  EXPECT_THROW(gauge.update(update,{1,2},{{1,2},{2,2}},{1,2}),std::exception);
  EXPECT_THROW(gauge.response(),std::logic_error);
}

TEST(IncrementalConditionalModel, NewBoundaryAndDisconnectedAppendDoNotInvalidateOldBranch) {
  Fixture f; f.append(5);
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors=f.factors;
  model.update(update,f.keys,f.dimensions,f.privateKeys);
  f.keys.push_back(2000); f.dimensions[2000]=2;
  f.factors[10000]=unary(2000,2.,.4); update={}; update.addFactors[10000]=f.factors.at(10000);
  auto stats=model.update(update,f.keys,f.dimensions,f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt,0u); parity(f,model.response());
  f.keys.push_back(500); f.dimensions[500]=2; f.privateKeys.push_back(500);
  f.factors[10001]=binary(500,2000); update={}; update.addFactors[10001]=f.factors.at(10001);
  stats=model.update(update,f.keys,f.dimensions,f.privateKeys);
  EXPECT_EQ(stats.conditionalsRebuilt,1u); EXPECT_EQ(stats.conditionalsReused,5u);
  parity(f,model.response());
}

TEST(IncrementalConditionalModel, RejectsRoundoffGaugeAndOverflowWithoutPublishing) {
  IncrementalConditionalModel model;
  gtsam::Matrix a(3,1); a << 1.,std::sqrt(2.),std::acos(-1.);
  NativeGaussianUpdate update;
  update.addFactors[0]=boost::make_shared<gtsam::JacobianFactor>(1,a,2,a,gtsam::Vector3(.1,.2,.3));
  EXPECT_THROW(model.update(update,{1,2},{{1,1},{2,1}},{1,2}),std::exception);
  EXPECT_THROW(model.response(),std::logic_error);
  Fixture f; f.append(2);
  update={}; update.addFactors=f.factors;
  model.update(update,f.keys,f.dimensions,f.privateKeys);
  const auto old=model.response();
  update={};
  update.replaceFactors[0]=boost::make_shared<gtsam::JacobianFactor>(0,
      gtsam::Matrix::Constant(2,2,1e308),gtsam::Vector2(1e308,1e308));
  EXPECT_THROW(model.update(update,f.keys,f.dimensions,f.privateKeys),std::exception);
  EXPECT_EQ(model.response(),old);
}

TEST(IncrementalConditionalModel, ChangingFactorSupportAndRepeatedEditsMatchFresh) {
  Fixture f; f.append(10);
  IncrementalConditionalModel model;
  NativeGaussianUpdate update; update.addFactors=f.factors;
  model.update(update,f.keys,f.dimensions,f.privateKeys);
  for (std::size_t iteration=0; iteration<25; ++iteration) {
    const Key first=iteration%5, second=5+iteration%5;
    update={};
    f.factors[10000]=binary(first,second,.5+.03*iteration);
    if (iteration==0) update.addFactors[10000]=f.factors.at(10000);
    else update.replaceFactors[10000]=f.factors.at(10000);
    model.update(update,f.keys,f.dimensions,f.privateKeys);
    parity(f,model.response());
  }
  update={}; update.removeFactors.insert(10000); f.factors.erase(10000);
  model.update(update,f.keys,f.dimensions,f.privateKeys); parity(f,model.response());
}
}  // namespace
