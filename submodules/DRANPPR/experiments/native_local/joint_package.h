#pragma once
#include "joint_step.h"
#include <DPGO/NativeBetweenPose2Factor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/BearingRangeFactor.h>
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
double number(const Json& row, const std::string& key) {
  require(row.at(key).is_number(), "nonnumeric " + key);
  const double value = row.at(key).get<double>();
  require(std::isfinite(value), "nonfinite " + key);
  return value;
}
std::uint32_t identifier(const Json& row, const std::string& key) {
  require(row.at(key).is_number_integer(), "noninteger " + key);
  const auto value = row.at(key).get<std::int64_t>();
  require(value >= 0 && value < (1LL<<24), "identifier out of range " + key);
  return value;
}
gtsam::Vector vector(const Json& value, std::size_t size) {
  require(value.is_array() && value.size() == size, "invalid vector dimension");
  gtsam::Vector result(size);
  for (std::size_t k=0; k<size; ++k) {
    require(value[k].is_number(), "nonnumeric vector");
    result[k] = value[k].get<double>();
  }
  require(result.allFinite(), "nonfinite vector");
  return result;
}
gtsam::Key X(std::uint64_t index) { return gtsam::Symbol('x', index); }
gtsam::Key L(std::uint64_t index) { return gtsam::Symbol('l', index); }
gtsam::Vector factorSigmas(const Json& factor, bool explicitNoise,
                           const gtsam::Vector& nominal) {
  require(explicitNoise == factor.contains("sigmas"), "factor sigmas/schema mismatch");
  if (!explicitNoise) return nominal;
  const auto sigmas = vector(factor.at("sigmas"), nominal.size());
  require((sigmas.array() > 0.).all(), "nonpositive factor sigmas");
  return sigmas;
}
std::vector<Local> parse(const Json& input, Robot root, Links& links) {
  const bool explicitNoise = input.at("schema") == "dranppr_frozen_joint_v2";
  require(explicitNoise || input.at("schema") == "dranppr_frozen_joint_v1", "unknown package schema");
  std::vector<Local> locals;
  std::set<Robot> ids;
  std::map<gtsam::Key, gtsam::Point2> charts;
  std::map<std::uint32_t, std::set<Robot>> observers;
  std::map<std::uint32_t, std::uint32_t> labels;
  for (const auto& row : input.at("robots")) {
    Local local;
    local.id = identifier(row, "id");
    require(ids.insert(local.id).second, "duplicate robot");
    local.model = std::make_unique<IncrementalLocalModel>();
    NativeModelUpdate update;
    for (const auto& pose : row.at("poses")) {
      const auto index = identifier(pose, "index");
      require(index == local.times.size(), "nonconsecutive poses");
      const double t = number(pose, "timestamp_s");
      require(index == 0 || t > local.times.at(index-1), "nonincreasing pose time");
      local.times[index] = t;
      const auto p = vector(pose.at("value"), 3);
      update.newValues.insert(X(index), gtsam::Pose2(p[0], p[1], p[2]));
      if (local.id == root && index == 0) local.fixed.insert(X(0));
      else local.privateKeys.push_back(X(index));
    }
    require(!local.times.empty(), "robot without poses");
    for (const auto& point : row.at("objects")) {
      const auto cluster = identifier(point, "cluster"), label = identifier(point, "label");
      require(local.labels.emplace(cluster, label).second, "duplicate object");
      const auto inserted = labels.emplace(cluster, label);
      require(inserted.second || inserted.first->second == label, "inconsistent cluster label");
      const gtsam::Point2 p = vector(point.at("point"), 2);
      const auto chart = charts.emplace(L(cluster), p);
      require(chart.second || (chart.first->second-p).norm() == 0., "inconsistent shared charts");
      observers[cluster].insert(local.id);
      update.newValues.insert(L(cluster), p);
    }
    std::set<std::uint32_t> odometry, sources, observed;
    for (const auto& factor : row.at("odometry")) {
      const auto a = identifier(factor, "source"), b = identifier(factor, "target");
      require(b == a+1 && local.times.count(b) && odometry.insert(b).second, "invalid odometry endpoints");
      const double dt = number(factor, "dt");
      require(dt > 0. && std::abs(local.times.at(b)-local.times.at(a)-dt) < 1e-6, "invalid duration");
      const auto d = vector(factor.at("delta"), 3);
      const auto noise = gtsam::noiseModel::Diagonal::Sigmas(factorSigmas(
          factor, explicitNoise, (gtsam::Vector3(.1,.1,.05)*std::sqrt(dt)).eval()));
      update.addFactors[2*std::uint64_t(b)] = boost::make_shared<NativeBetweenPose2Factor>(
          X(a), X(b), gtsam::Pose2(d[0], d[1], d[2]), noise);
    }
    require(odometry.size()+1 == local.times.size(), "missing odometry");
    for (const auto& factor : row.at("observations")) {
      const auto p = identifier(factor, "pose"), c = identifier(factor, "cluster");
      const auto source = identifier(factor, "source");
      require(local.times.count(p) && local.labels.count(c) && sources.insert(source).second,
              "invalid or duplicate observation");
      const double range = number(factor, "range");
      require(range > 0., "nonpositive range");
      const auto polar = gtsam::noiseModel::Diagonal::Sigmas(
          factorSigmas(factor, explicitNoise, gtsam::Vector2(.05, .1)));
      update.addFactors[2*std::uint64_t(source)+1] = boost::make_shared<Observation>(
          X(p), L(c), gtsam::Rot2(number(factor, "bearing")), range, polar);
      observed.insert(c);
    }
    require(observed.size() == local.labels.size(), "unobserved object");
    local.model->update(update);
    local.initial = local.model->bases();
    local.graph = local.model->factorGraph();
    locals.push_back(std::move(local));
  }
  require(ids.size() >= 2 && ids.count(root), "invalid root or robot count");
  std::map<Robot, std::set<Robot>> adjacency;
  for (const auto& link : input.at("links")) {
    require(link.is_array() && link.size() == 2 && link[0].is_number_integer() && link[1].is_number_integer(), "invalid link");
    const Robot a = link[0].get<Robot>(), b = link[1].get<Robot>();
    require(a != b && ids.count(a) && ids.count(b) && adjacency[a].insert(b).second, "invalid or repeated link");
    adjacency[b].insert(a);
    links.emplace_back(a, b);
  }
  auto connected = [&](const std::set<Robot>& group) {
    std::set<Robot> visited{*group.begin()};
    std::vector<Robot> queue{*group.begin()};
    for (std::size_t i=0; i<queue.size(); ++i) for (auto next : adjacency[queue[i]]) {
      if (group.count(next) && visited.insert(next).second) queue.push_back(next);
    }
    return visited == group;
  };
  require(connected(ids), "disconnected communication topology");
  std::set<std::uint32_t> clusterIds;
  std::map<std::uint32_t, std::map<Robot, std::uint32_t>> labelClusters;
  for (const auto& cluster : input.at("clusters")) {
    const auto c = identifier(cluster, "cluster"), label = identifier(cluster, "label");
    require(clusterIds.insert(c).second && observers.count(c) && labels.at(c) == label, "invalid cluster roster");
    std::set<Robot> group;
    for (const auto& r : cluster.at("observers")) {
      require(r.is_number_integer() && group.insert(r.get<Robot>()).second, "invalid observer roster");
    }
    require(group == observers.at(c) && connected(group), "identity outside observer topology");
    require(identifier(cluster, "owner") == *group.begin(), "invalid cluster owner");
    for (auto r : group) require(labelClusters[label].emplace(r, c).second, "duplicate local identity");
  }
  require(clusterIds.size() == observers.size(), "incomplete cluster roster");
  for (const auto& label : labelClusters) for (const auto& edge : links) {
    const auto& map = label.second;
    require(!map.count(edge.first) || !map.count(edge.second) || map.at(edge.first) == map.at(edge.second),
            "neighbor identity incorrectly split");
  }
  std::sort(locals.begin(), locals.end(), [](const Local& a, const Local& b) { return a.id < b.id; });
  return locals;
}


