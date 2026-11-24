#include "../experiments/native_local/chart_audit.h"
#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::L;

NativeModelUpdate chartFixture() {
  NativeModelUpdate u;
  u.newValues.insert(X(0),gtsam::Pose2(.2,-.1,.2));
  u.newValues.insert(X(1),gtsam::Pose2(1.1,.3,.4));
  u.newValues.insert(X(2),gtsam::Pose2(2.2,1.,.5));
  u.newValues.insert(L(0),gtsam::Point2(3.,2.));
  u.addFactors[0]=boost::make_shared<NativeBetweenPose2Factor>(X(0),X(1),gtsam::Pose2(1.,0.,.1),
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1,.1,.05)));
  u.addFactors[1]=boost::make_shared<NativeBetweenPose2Factor>(X(1),X(2),gtsam::Pose2(1.,0.,.1),
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1,.1,.05)));
  for (int k=0;k<3;++k) u.addFactors[2+k]=boost::make_shared<ChartObservation>(X(k),L(0),
      gtsam::Rot2(.4),2.,gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05,.1)));
  return u;
}

NativeModelUpdate rigidUpdate(const gtsam::Values& values,const gtsam::Pose2& transform) {
  NativeModelUpdate update;
  for (auto key:values.keys()) {
    if (gtsam::Symbol(key).chr()=='x') update.relinearizeValues.insert(key,transform.compose(values.at<gtsam::Pose2>(key)));
    else update.relinearizeValues.insert(key,transform.transformFrom(values.at<gtsam::Point2>(key)));
  }
  return update;
}

TEST(NativeChartAudit, PureRigidMotionHasNoPhysicalChangeOrTransportedError) {
  IncrementalLocalModel model(true); model.update(chartFixture());
  const auto before=model.squareRootFactors();
  const auto revision=model.revision();
  const auto report=auditChartRefresh(model,rigidUpdate(model.bases(),gtsam::Pose2(2.,-3.,.8)),{});
  EXPECT_EQ(report.at("raw_trigger_pose_keys"),3);
  EXPECT_EQ(report.at("aligned_trigger_pose_keys"),0);
  EXPECT_EQ(report.at("aligned_trigger_point_keys"),0);
  EXPECT_LT(report.at("pose_translation_aligned_sq").get<double>(),1e-24);
  EXPECT_EQ(report.at("touched_factors"),5);
  for (const auto& name:{"odometry","observation"}) {
    const auto& r=report.at("families").at(name);
    EXPECT_LT(r.at("residual_change_sq").get<double>(),1e-24);
    EXPECT_LT(r.at("transported_error_sq").get<double>(),1e-24);
    EXPECT_LT(r.at("gauge_invariance_max").get<double>(),1e-12);
  }
  EXPECT_GT(report.at("families").at("observation").at("raw_error_sq").get<double>(),1.);
  EXPECT_EQ(model.revision(),revision);
  EXPECT_EQ(model.squareRootFactors(),before);
}

TEST(NativeChartAudit, DeformationRemainsAfterRigidAlignment) {
  IncrementalLocalModel model(true); model.update(chartFixture());
  NativeModelUpdate update;
  update.relinearizeValues.insert(X(1),gtsam::Pose2(1.3,.8,.8));
  const auto r=auditChartRefresh(model,update,{X(0)});
  EXPECT_EQ(r.at("requested_pose_keys"),1);
  EXPECT_GT(r.at("aligned_trigger_pose_keys").get<int>(),0);
  EXPECT_GT(r.at("pose_translation_aligned_sq").get<double>(),.01);
  EXPECT_EQ(r.at("pose_only_factors"),3);
  EXPECT_EQ(r.at("point_only_factors"),0);
  EXPECT_GT(r.at("fixed_aligned_max").get<double>(),0.);
  EXPECT_GT(r.at("families").at("odometry").at("residual_change_sq").get<double>(),1.);
}

TEST(NativeChartAudit, PointCoordinationDoesNotInventPoseMotion) {
  IncrementalLocalModel model(true); model.update(chartFixture());
  NativeModelUpdate update;
  update.relinearizeValues.insert(L(0),gtsam::Point2(3.1,2.2));
  const auto r=auditChartRefresh(model,update,{});
  EXPECT_EQ(r.at("requested_pose_keys"),0);
  EXPECT_EQ(r.at("requested_point_keys"),1);
  EXPECT_EQ(r.at("raw_trigger_pose_keys"),0);
  EXPECT_EQ(r.at("aligned_trigger_pose_keys"),0);
  EXPECT_EQ(r.at("point_only_factors"),3);
  EXPECT_EQ(r.at("touched_factors"),3);
  const auto& f=r.at("families").at("observation");
  EXPECT_GT(f.at("residual_change_sq").get<double>(),1.);
  EXPECT_NEAR(f.at("raw_error_sq").get<double>(),f.at("transported_error_sq").get<double>(),1e-12);
}

TEST(NativeChartAudit, PoseOriginCheckDoesNotMoveFixedRootToFitDeformation) {
  IncrementalLocalModel model(true); model.update(chartFixture());
  NativeModelUpdate update;
  update.relinearizeValues.insert(X(1),gtsam::Pose2(1.3,.8,.8));
  const auto r=auditChartRefresh(model,update,{X(0)},true);
  EXPECT_EQ(r.at("alignment_rotation_source"),"pose_origin");
  EXPECT_EQ(r.at("raw_trigger_pose_keys"),r.at("aligned_trigger_pose_keys"));
  EXPECT_LT(r.at("fixed_aligned_max").get<double>(),1e-14);
  const auto& f=r.at("families").at("observation");
  EXPECT_NEAR(f.at("raw_error_sq").get<double>(),f.at("transported_error_sq").get<double>(),1e-12);
  const auto rigid=auditChartRefresh(model,rigidUpdate(model.bases(),gtsam::Pose2(2.,3.,.9)),{},true);
  EXPECT_EQ(rigid.at("aligned_trigger_pose_keys"),0);
  EXPECT_EQ(rigid.at("aligned_trigger_point_keys"),0);
  EXPECT_LT(rigid.at("families").at("observation").at("transported_error_sq").get<double>(),1e-24);
}

TEST(NativeChartAudit, NoOpAndSinglePoseRigidMotionAreDefined) {
  IncrementalLocalModel model(true); model.update(chartFixture());
  const auto noop=auditChartRefresh(model,NativeModelUpdate(),{});
  EXPECT_EQ(noop.at("touched_factors"),0);
  EXPECT_EQ(noop.at("raw_trigger_pose_keys"),0);
  NativeModelUpdate u;
  u.newValues.insert(X(0),gtsam::Pose2()); u.newValues.insert(L(0),gtsam::Point2(2.,2.));
  u.addFactors[0]=boost::make_shared<ChartObservation>(X(0),L(0),gtsam::Rot2(.5),2.,
      gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05,.1)));
  IncrementalLocalModel single(true); single.update(u);
  const auto r=auditChartRefresh(single,rigidUpdate(single.bases(),gtsam::Pose2(2.,3.,.9)),{});
  EXPECT_EQ(r.at("alignment_rotation_source"),"pose_yaw");
  EXPECT_EQ(r.at("aligned_trigger_point_keys"),0);
  EXPECT_LT(r.at("families").at("observation").at("transported_error_sq").get<double>(),1e-24);
}

TEST(NativeChartAudit, RejectsNonrelativeFactorsAndUnsupportedInputs) {
  auto u=chartFixture();
  u.addFactors[99]=boost::make_shared<gtsam::PriorFactor<gtsam::Pose2>>(X(0),gtsam::Pose2(),
      gtsam::noiseModel::Isotropic::Sigma(3,.1));
  IncrementalLocalModel model(true); model.update(u);
  EXPECT_THROW(auditChartRefresh(model,rigidUpdate(model.bases(),gtsam::Pose2(1.,0.,.2)),{}),std::invalid_argument);
  IncrementalLocalModel normal; normal.update(chartFixture());
  EXPECT_THROW(auditChartRefresh(normal,NativeModelUpdate(),{}),std::logic_error);
}
}  // namespace
