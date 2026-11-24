#pragma once

#include <DPGO/IncrementalLocalModel.h>
#include <DPGO/NativeBetweenPose2Factor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

using ChartObservation = gtsam::BearingRangeFactor<gtsam::Pose2,gtsam::Point2>;

// Read-only diagnostic on historical factors, not a relinearization policy.
inline nlohmann::json auditChartRefresh(const DPGO::IncrementalLocalModel& model,
    const DPGO::NativeModelUpdate& update,const std::set<gtsam::Key>& fixed,bool poseOrigin=false) {
  using Json = nlohmann::json;
  const auto factors=model.squareRootFactors();
  const auto graph=model.factorGraph();
  const auto& bases=model.bases();
  gtsam::Values target=bases;
  for (auto key:update.relinearizeValues.keys()) target.update(key,update.relinearizeValues.at(key));
  std::vector<gtsam::Key> poses,points;
  gtsam::Point2 a=gtsam::Point2::Zero(),b=gtsam::Point2::Zero();
  double yawCos=0.,yawSin=0.;
  for (auto key:bases.keys()) {
    if (bases.at(key).dim()==3) {
      poses.push_back(key);
      const auto& x=bases.at<gtsam::Pose2>(key);
      const auto& y=target.at<gtsam::Pose2>(key);
      a+=x.translation(); b+=y.translation();
      yawCos+=std::cos(y.theta()-x.theta()); yawSin+=std::sin(y.theta()-x.theta());
    } else points.push_back(key);
  }
  if (poses.empty()) throw std::invalid_argument("chart audit requires Pose2 history");
  a/=poses.size(); b/=poses.size();
  double cross=0.,dot=0.,spreadA=0.,spreadB=0.;
  for (auto key:poses) {
    const gtsam::Point2 x=bases.at<gtsam::Pose2>(key).translation()-a;
    const gtsam::Point2 y=target.at<gtsam::Pose2>(key).translation()-b;
    cross+=x.x()*y.y()-x.y()*y.x(); dot+=x.dot(y);
    spreadA+=x.squaredNorm(); spreadB+=y.squaredNorm();
  }
  const bool spatial=std::hypot(cross,dot)>std::numeric_limits<double>::epsilon()*
      std::max(1.,std::sqrt(spreadA*spreadB));
  const double angle=spatial ? std::atan2(cross,dot) : std::atan2(yawSin,yawCos);
  const gtsam::Rot2 fittedRotation(angle);
  gtsam::Pose2 transform(fittedRotation,b-fittedRotation.rotate(a));
  if (poseOrigin) transform=target.at<gtsam::Pose2>(poses.front()).compose(
      bases.at<gtsam::Pose2>(poses.front()).inverse());
  const auto rotation=transform.rotation();
  const auto translation=transform.translation();
  gtsam::Values aligned;
  for (auto key:poses) aligned.insert(key,transform.compose(bases.at<gtsam::Pose2>(key)));
  for (auto key:points) aligned.insert(key,transform.transformFrom(bases.at<gtsam::Point2>(key)));
  const auto rawDelta=bases.localCoordinates(target),alignedDelta=aligned.localCoordinates(target);
  Json result{{"revision",model.revision()}, {"pose_count",poses.size()}, {"point_count",points.size()},
      {"alignment_rotation_source",poseOrigin ? "pose_origin" : (spatial ? "pose_positions" : "pose_yaw")},
      {"alignment",{translation.x(),translation.y(),transform.theta()}}, {"threshold",1e-4},
      {"requested_pose_keys",0}, {"requested_point_keys",0}, {"raw_trigger_pose_keys",0},
      {"aligned_trigger_pose_keys",0}, {"raw_trigger_point_keys",0}, {"aligned_trigger_point_keys",0},
      {"pose_translation_raw_sq",0.}, {"pose_translation_aligned_sq",0.},
      {"max_raw_delta",0.}, {"max_aligned_delta",0.}, {"fixed_aligned_max",0.},
      {"factors",factors.size()}, {"touched_factors",0}, {"pose_only_factors",0},
      {"point_only_factors",0}, {"both_factors",0}};
  auto increment=[](Json& r,const std::string& field) { r[field]=r.at(field).get<std::size_t>()+1; };
  auto add=[](Json& r,const std::string& field,double v) { r[field]=r.at(field).get<double>()+v; };
  auto maximum=[](Json& r,const std::string& field,double v) { r[field]=std::max(r.at(field).get<double>(),v); };
  for (auto key:bases.keys()) {
    const bool pose=bases.at(key).dim()==3;
    if (update.relinearizeValues.exists(key)) increment(result,pose ? "requested_pose_keys" : "requested_point_keys");
    const double raw=rawDelta.at(key).lpNorm<Eigen::Infinity>();
    const double remaining=alignedDelta.at(key).lpNorm<Eigen::Infinity>();
    if (!std::isfinite(raw) || !std::isfinite(remaining)) throw std::invalid_argument("nonfinite chart audit state");
    maximum(result,"max_raw_delta",raw); maximum(result,"max_aligned_delta",remaining);
    if (fixed.count(key)) maximum(result,"fixed_aligned_max",remaining);
    else {
      if (raw>1e-4) increment(result,pose ? "raw_trigger_pose_keys" : "raw_trigger_point_keys");
      if (remaining>1e-4) increment(result,pose ? "aligned_trigger_pose_keys" : "aligned_trigger_point_keys");
    }
    if (pose) {
      const auto y=target.at<gtsam::Pose2>(key).translation();
      add(result,"pose_translation_raw_sq",(y-bases.at<gtsam::Pose2>(key).translation()).squaredNorm());
      add(result,"pose_translation_aligned_sq",(y-aligned.at<gtsam::Pose2>(key).translation()).squaredNorm());
    }
  }
  const std::vector<double> edges{1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e-1,1.};
  result["norm_bin_upper_bounds"]=edges;
  const Json family{{"count",0}, {"residual_dimensions",0}, {"residual_change_sq",0.},
      {"raw_error_sq",0.}, {"transported_error_sq",0.}, {"target_residual_sq",0.},
      {"raw_error_max",0.}, {"transported_error_max",0.}, {"gauge_invariance_max",0.},
      {"cached_residual_max",0.}, {"raw_error_bins",std::vector<std::size_t>(edges.size()+1)},
      {"transported_error_bins",std::vector<std::size_t>(edges.size()+1)},
      {"residual_change_bins",std::vector<std::size_t>(edges.size()+1)}};
  result["families"]={{"odometry",family},{"observation",family}};
  std::size_t index=0;
  for (const auto& item:factors) {
    const auto& nonlinear=graph.at(index++);
    const bool odom=dynamic_cast<const DPGO::NativeBetweenPose2Factor*>(nonlinear.get());
    if (!odom && !dynamic_cast<const ChartObservation*>(nonlinear.get())) {
      throw std::invalid_argument("chart audit supports only relative odometry and polar observations");
    }
    bool changedPose=false,changedPoint=false;
    for (auto key:nonlinear->keys()) if (update.relinearizeValues.exists(key)) {
      if (bases.at(key).dim()==3) changedPose=true;
      else changedPoint=true;
    }
    if (!changedPose && !changedPoint) continue;
    increment(result,"touched_factors");
    increment(result,changedPose && changedPoint ? "both_factors" :
        (changedPose ? "pose_only_factors" : "point_only_factors"));
    auto& stats=result["families"][odom ? "odometry" : "observation"];
    increment(stats,"count");
    const auto noisy=boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(nonlinear);
    const auto baseResidual=noisy->whitenedError(bases),actual=noisy->whitenedError(target);
    gtsam::Vector raw=-item.second->getb(),transported=raw;
    for (auto it=item.second->begin();it!=item.second->end();++it) {
      raw.noalias()+=item.second->getA(it)*rawDelta.at(*it);
      if (bases.at(*it).dim()==3) transported.noalias()+=item.second->getA(it)*alignedDelta.at(*it);
      else transported.noalias()+=item.second->getA(it)*rotation.matrix().transpose()*alignedDelta.at(*it);
    }
    const double change=(actual-baseResidual).norm(),rawError=(actual-raw).norm();
    const double transportedError=(actual-transported).norm();
    if (!std::isfinite(change+rawError+transportedError)) throw std::invalid_argument("nonfinite chart audit residual");
    stats["residual_dimensions"]=stats.at("residual_dimensions").get<std::size_t>()+actual.size();
    add(stats,"residual_change_sq",change*change); add(stats,"target_residual_sq",actual.squaredNorm());
    add(stats,"raw_error_sq",rawError*rawError); add(stats,"transported_error_sq",transportedError*transportedError);
    maximum(stats,"raw_error_max",rawError); maximum(stats,"transported_error_max",transportedError);
    maximum(stats,"gauge_invariance_max",(noisy->whitenedError(aligned)-baseResidual).norm());
    maximum(stats,"cached_residual_max",(baseResidual+item.second->getb()).norm());
    for (const auto& entry:std::vector<std::pair<std::string,double>>{
        {"raw_error_bins",rawError},{"transported_error_bins",transportedError},{"residual_change_bins",change}}) {
      const auto bin=std::lower_bound(edges.begin(),edges.end(),entry.second)-edges.begin();
      stats[entry.first][bin]=stats[entry.first][bin].get<std::size_t>()+1;
    }
  }
  return result;
}
