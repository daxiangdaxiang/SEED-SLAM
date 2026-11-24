#include "../experiments/native_local/joint_step.h"
#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>

namespace {
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

class SquaredPointMeasurement : public gtsam::NoiseModelFactor1<gtsam::Point2> {
 public:
  explicit SquaredPointMeasurement(gtsam::Key key)
      : NoiseModelFactor1(gtsam::noiseModel::Isotropic::Sigma(2,1.),key) {}
  gtsam::Vector evaluateError(const gtsam::Point2& point,
      boost::optional<gtsam::Matrix&> H=boost::none) const override {
    if (H) { *H=gtsam::Matrix::Identity(2,2); (*H)(0,0)=2.*point.x(); }
    return gtsam::Vector2(point.x()*point.x()-1.,point.y());
  }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SquaredPointMeasurement>(*this);
  }
};

std::vector<Local> nonlinearPair(double x,bool conditional) {
  std::vector<Local> locals;
  for (Robot id:{1,2}) {
    Local local; local.id=id; local.reuseCharts=true; local.lazyRefresh=true;
    local.conditionalResponse=conditional;
    local.model=std::make_unique<IncrementalLocalModel>(conditional);
    NativeModelUpdate update;
    update.newValues.insert(L(0),gtsam::Point2(x,0.));
    update.newValues.insert(X(0),gtsam::Pose2());
    update.addFactors[0]=boost::make_shared<SquaredPointMeasurement>(L(0));
    local.model->update(update); local.estimate=local.model->bases();
    local.fixed.insert(X(0)); local.interfaceKeys[L(0)]=L(0);
    local.graph=local.model->factorGraph(); locals.push_back(std::move(local));
  }
  return locals;
}

TEST(NativeLazyRefresh, RejectedFullStepDefersRebaseWithoutExtraInterfaceIteration) {
  for (bool conditional:{false,true}) {
    auto locals=nonlinearPair(.1,conditional);
    const auto first=routed(locals,{{1,2}},1,false,0,1);
    EXPECT_EQ(first.at("refresh_requests"),1);
    EXPECT_EQ(first.at("refresh_rounds"),0);
    EXPECT_EQ(first.at("refreshed_keys"),0);
    EXPECT_EQ(first.at("pcg_iterations"),1);
    const auto& step=first.at("trace").at(0);
    EXPECT_LT(step.at("full_step_decrease").get<double>(),0.);
    EXPECT_GT(step.at("alpha").get<double>(),0.);
    EXPECT_LT(step.at("alpha").get<double>(),1.);
    EXPECT_LT(step.at("cost_after").get<double>(),step.at("cost_before").get<double>());
    const auto current=locals[0].current().at<gtsam::Point2>(L(0));
    for (const auto& local:locals) {
      EXPECT_TRUE(local.refreshPending);
      EXPECT_EQ(local.model->bases().at<gtsam::Point2>(L(0)).x(),.1);
    }
    const auto second=routed(locals,{{1,2}},1,false,1,1);
    EXPECT_EQ(second.at("refresh_rounds"),1);
    EXPECT_EQ(second.at("refresh_requests"),0);
    EXPECT_EQ(second.at("refreshed_keys"),2);
    EXPECT_EQ(second.at("pcg_iterations"),1);
    for (const auto& local:locals) {
      EXPECT_FALSE(local.refreshPending);
      EXPECT_LT((local.model->bases().at<gtsam::Point2>(L(0))-current).norm(),1e-12);
      EXPECT_TRUE(local.model->bases().at<gtsam::Pose2>(X(0)).equals(gtsam::Pose2()));
    }
  }
}

TEST(NativeLazyRefresh, SuccessfulFullStepRetainsChartOnFollowingIteration) {
  auto locals=nonlinearPair(1.1,true);
  const auto first=routed(locals,{{1,2}},1,false,0,1);
  EXPECT_EQ(first.at("refresh_requests"),0);
  EXPECT_GT(first.at("trace").at(0).at("full_step_decrease").get<double>(),0.);
  const auto second=routed(locals,{{1,2}},1,false,1,1);
  EXPECT_EQ(second.at("refresh_rounds"),0);
  EXPECT_EQ(second.at("response_reuses"),2);
  EXPECT_EQ(second.at("refreshed_keys"),0);
  for (const auto& local:locals) EXPECT_EQ(local.model->bases().at<gtsam::Point2>(L(0)).x(),1.1);
}

TEST(NativeLazyRefresh, RefreshDecisionMustBeSharedBeforeChangingCommonCharts) {
  auto locals=nonlinearPair(.1,true);
  locals[0].refreshPending=true;
  EXPECT_THROW(routed(locals,{{1,2}},1,false,0,1),std::runtime_error);
}
}  // namespace
