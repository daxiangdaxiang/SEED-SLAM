#include <DPGO/IncrementalLocalModel.h>

#include <gtsam/base/GenericValue.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/linear/GaussianFactor.h>
#include <Eigen/Cholesky>

#include <cmath>
#include <chrono>
#include <stdexcept>
#include <typeinfo>

namespace DPGO {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

struct CachedFactor {
  gtsam::NonlinearFactor::shared_ptr factor;
  std::vector<gtsam::Key> keys;
  std::vector<Eigen::Index> offsets, dimensions;
  gtsam::Matrix hessian;
  gtsam::Vector gradient;
  std::shared_ptr<const gtsam::JacobianFactor> squareRoot;
  double constant;
};

void validateValue(const gtsam::Value& value) {
  if (auto pose = dynamic_cast<const gtsam::GenericValue<gtsam::Pose2>*>(&value)) {
    if (pose->value().matrix().allFinite()) return;
  } else if (auto point = dynamic_cast<const gtsam::GenericValue<gtsam::Point2>*>(&value)) {
    if (point->value().allFinite()) return;
  }
  throw std::invalid_argument("native model requires finite Pose2 or Point2 values");
}

std::shared_ptr<const CachedFactor> linearize(
    const gtsam::NonlinearFactor::shared_ptr& input, const gtsam::Values& bases,
    bool squareRootOnly) {
  if (!input) throw std::invalid_argument("null native factor");
  auto factor = input->clone();
  auto noisy = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factor);
  auto gaussian = noisy ? boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(noisy->noiseModel()) : nullptr;
  if (!gaussian || gaussian->isConstrained()) {
    throw std::invalid_argument("native model requires unconstrained Gaussian physical noise");
  }
  const auto sigmas = gaussian->sigmas();
  const auto r = gaussian->R();
  const gtsam::Matrix information = r.transpose()*r;
  Eigen::LLT<gtsam::Matrix> noiseFactor(information);
  if (!sigmas.allFinite() || (sigmas.array() <= 0.).any() ||
      !r.allFinite() || !information.allFinite() ||
      noiseFactor.info() != Eigen::Success) {
    throw std::invalid_argument("native noise must be finite and positive definite");
  }
  std::set<gtsam::Key> unique;
  for (auto key : factor->keys()) {
    if (!bases.exists(key) || !unique.insert(key).second) {
      throw std::invalid_argument("factor has missing or repeated variable keys");
    }
  }
  if (unique.empty()) throw std::invalid_argument("factor has no variable keys");
  auto gaussianFactor = factor->linearize(bases);
  if (!gaussianFactor) throw std::invalid_argument("inactive factor has no tangent model");
  auto ab = gaussianFactor->jacobian();
  if (!ab.first.allFinite() || !ab.second.allFinite()) {
    throw std::invalid_argument("nonfinite native factor linearization");
  }
  auto cache = std::make_shared<CachedFactor>();
  cache->factor = factor;
  Eigen::Index offset = 0;
  for (auto it = gaussianFactor->begin(); it != gaussianFactor->end(); ++it) {
    const auto dim = gaussianFactor->getDim(it);
    if (dim != static_cast<Eigen::Index>(bases.at(*it).dim())) {
      throw std::invalid_argument("factor tangent dimension mismatch");
    }
    cache->keys.push_back(*it);
    cache->offsets.push_back(offset);
    cache->dimensions.push_back(dim);
    offset += dim;
  }
  if (offset != ab.first.cols() || ab.first.rows() != ab.second.size() ||
      std::set<gtsam::Key>(cache->keys.begin(), cache->keys.end()) != unique) {
    throw std::invalid_argument("invalid native Jacobian layout");
  }
  cache->constant = .5*ab.second.squaredNorm();
  if (squareRootOnly) {
    std::vector<std::pair<gtsam::Key,gtsam::Matrix>> terms;
    for (std::size_t i=0; i<cache->keys.size(); ++i) terms.emplace_back(cache->keys[i],
        ab.first.middleCols(cache->offsets[i], cache->dimensions[i]));
    cache->squareRoot = std::make_shared<const gtsam::JacobianFactor>(terms, ab.second);
  } else {
    cache->hessian = ab.first.transpose()*ab.first;
    cache->gradient = -ab.first.transpose()*ab.second;
  }
  if (!cache->hessian.allFinite() || !cache->gradient.allFinite() ||
      !std::isfinite(cache->constant)) {
    throw std::invalid_argument("native normal model overflow");
  }
  return cache;
}

}  // namespace

struct IncrementalLocalModel::State {
  bool squareRootOnly{false};
  gtsam::Values bases;
  std::vector<gtsam::Key> order;
  std::map<NativeFactorId, std::shared_ptr<const CachedFactor>> factors;
  std::map<gtsam::Key, std::set<NativeFactorId>> incidence;
  std::map<std::pair<gtsam::Key, gtsam::Key>, gtsam::Matrix> blocks;
  std::map<std::pair<gtsam::Key, gtsam::Key>, std::size_t> blockCounts;
  std::map<gtsam::Key, gtsam::Vector> gradients;
  double constant{0.};
  std::uint64_t revision{0};

  void accumulate(NativeFactorId id, const CachedFactor& cache, bool add) {
    const double sign = add ? 1. : -1.;
    constant += sign*cache.constant;
    if (squareRootOnly) {
      for (auto key : cache.keys) {
        if (add) incidence[key].insert(id);
        else incidence.at(key).erase(id);
      }
      return;
    }
    for (std::size_t i = 0; i < cache.keys.size(); ++i) {
      const auto key = cache.keys[i];
      gradients.at(key) += sign*cache.gradient.segment(cache.offsets[i], cache.dimensions[i]);
      if (add) incidence[key].insert(id);
      else incidence.at(key).erase(id);
      for (std::size_t j = 0; j < cache.keys.size(); ++j) {
        const auto pair = std::make_pair(key, cache.keys[j]);
        if (add && blockCounts[pair]++ == 0) {
          blocks[pair] = gtsam::Matrix::Zero(cache.dimensions[i], cache.dimensions[j]);
        }
        blocks.at(pair) += sign*cache.hessian.block(
            cache.offsets[i], cache.offsets[j], cache.dimensions[i], cache.dimensions[j]);
        if (!add && --blockCounts.at(pair) == 0) {
          blocks.erase(pair);
          blockCounts.erase(pair);
        }
      }
      if (!add && incidence.at(key).empty()) gradients.at(key).setZero();
    }
  }
};

IncrementalLocalModel::IncrementalLocalModel(bool squareRootOnly) : state_(new State) {
  state_->squareRootOnly = squareRootOnly;
}
IncrementalLocalModel::~IncrementalLocalModel() = default;

NativeUpdateStatistics IncrementalLocalModel::update(const NativeModelUpdate& input) {
  NativeUpdateStatistics stats;
  stats.revision = state_->revision;
  stats.factorsReused = state_->factors.size();
  if (input.newValues.empty() && input.relinearizeValues.empty() &&
      input.addFactors.empty() && input.replaceFactors.empty() && input.removeFactors.empty()) return stats;

  // Copy-on-publish provides rollback; only changed factors are relinearized.
  // This first implementation still copies O(graph) metadata and normal blocks.
  const auto snapshotStart = Clock::now();
  auto next = std::make_unique<State>(*state_);
  stats.snapshotMs = milliseconds(snapshotStart);
  for (auto key : input.newValues.keys()) {
    if (state_->bases.exists(key) || input.relinearizeValues.exists(key)) {
      throw std::invalid_argument("duplicate or conflicting new variable");
    }
    validateValue(input.newValues.at(key));
    next->bases.insert(key, input.newValues.at(key));
    next->order.push_back(key);
    if (!next->squareRootOnly) next->gradients[key] = gtsam::Vector::Zero(input.newValues.at(key).dim());
  }
  for (auto key : input.relinearizeValues.keys()) {
    if (!state_->bases.exists(key) || typeid(state_->bases.at(key)) != typeid(input.relinearizeValues.at(key))) {
      throw std::invalid_argument("unknown variable or changed variable type");
    }
    validateValue(input.relinearizeValues.at(key));
    next->bases.update(key, input.relinearizeValues.at(key));
  }
  for (const auto& entry : input.addFactors) {
    if (state_->factors.count(entry.first) || input.replaceFactors.count(entry.first) ||
        input.removeFactors.count(entry.first)) throw std::invalid_argument("duplicate or conflicting factor ID");
  }
  for (const auto& entry : input.replaceFactors) {
    if (!state_->factors.count(entry.first) || input.removeFactors.count(entry.first)) {
      throw std::invalid_argument("unknown or conflicting replacement ID");
    }
  }
  for (auto id : input.removeFactors) {
    if (!state_->factors.count(id)) throw std::invalid_argument("unknown removal ID");
  }

  auto pending = input.addFactors;
  pending.insert(input.replaceFactors.begin(), input.replaceFactors.end());
  for (auto key : input.relinearizeValues.keys()) {
    auto neighbors = state_->incidence.find(key);
    if (neighbors == state_->incidence.end()) continue;
    for (auto id : neighbors->second) {
      if (!input.removeFactors.count(id) && !pending.count(id)) {
        pending[id] = state_->factors.at(id)->factor;
      }
    }
  }
  std::map<NativeFactorId, std::shared_ptr<const CachedFactor>> prepared;
  const auto linearizationStart = Clock::now();
  for (const auto& entry : pending) prepared[entry.first] = linearize(entry.second, next->bases, next->squareRootOnly);
  stats.linearizationMs = milliseconds(linearizationStart);
  const auto contributionStart = Clock::now();
  std::set<NativeFactorId> subtract = input.removeFactors;
  for (const auto& entry : prepared) {
    if (state_->factors.count(entry.first)) subtract.insert(entry.first);
  }
  for (auto id : subtract) {
    next->accumulate(id, *next->factors.at(id), false);
    next->factors.erase(id);
  }
  for (const auto& entry : prepared) {
    next->accumulate(entry.first, *entry.second, true);
    next->factors[entry.first] = entry.second;
  }
  if (next->factors.empty()) next->constant = 0.;
  stats.contributionMs = milliseconds(contributionStart);
  if (!std::isfinite(next->constant)) throw std::invalid_argument("aggregate model overflow");
  for (const auto& block : next->blocks) {
    if (!block.second.allFinite()) throw std::invalid_argument("aggregate Hessian overflow");
  }
  for (const auto& gradient : next->gradients) {
    if (!gradient.second.allFinite()) throw std::invalid_argument("aggregate gradient overflow");
  }
  stats.revision = ++next->revision;
  stats.factorsLinearized = prepared.size();
  stats.factorsReused = next->factors.size()-prepared.size();
  stats.factorsAdded = input.addFactors.size();
  stats.factorsReplaced = input.replaceFactors.size();
  stats.factorsRemoved = input.removeFactors.size();
  state_.swap(next);
  return stats;
}

const gtsam::Values& IncrementalLocalModel::bases() const { return state_->bases; }
std::size_t IncrementalLocalModel::factorCount() const { return state_->factors.size(); }
std::uint64_t IncrementalLocalModel::revision() const { return state_->revision; }

gtsam::NonlinearFactorGraph IncrementalLocalModel::factorGraph() const {
  gtsam::NonlinearFactorGraph graph;
  for (const auto& entry : state_->factors) graph.push_back(entry.second->factor->clone());
  return graph;
}

NativeVariableLayout IncrementalLocalModel::linearLayout(
    const std::vector<gtsam::Key>& ordering, const std::set<gtsam::Key>& fixedKeys) const {
  for (auto key : fixedKeys) {
    if (!state_->bases.exists(key)) throw std::invalid_argument("unknown fixed key");
  }
  NativeVariableLayout system;
  system.revision = state_->revision;
  if (ordering.empty()) {
    for (auto key : state_->order) if (!fixedKeys.count(key)) system.keys.push_back(key);
  } else system.keys = ordering;
  if (system.keys.size() != state_->bases.size()-fixedKeys.size()) {
    throw std::invalid_argument("ordering must contain every free key once");
  }
  Eigen::Index total = 0;
  for (auto key : system.keys) {
    if (!state_->bases.exists(key) || fixedKeys.count(key) || system.offsets.count(key)) {
      throw std::invalid_argument("invalid free-key ordering");
    }
    system.offsets[key] = total;
    system.dimensions[key] = state_->bases.at(key).dim();
    total += system.dimensions.at(key);
  }
  system.dimension = total;
  return system;
}

NativeLinearSystem IncrementalLocalModel::linearSystem(
    const std::vector<gtsam::Key>& ordering, const std::set<gtsam::Key>& fixedKeys) const {
  if (state_->squareRootOnly) throw std::logic_error("square-root-only model has no normal system");
  NativeLinearSystem system;
  static_cast<NativeVariableLayout&>(system) = linearLayout(ordering, fixedKeys);
  const auto total = system.dimension;
  system.constant = state_->constant;
  system.gradient = gtsam::Vector::Zero(total);
  for (auto key : system.keys) {
    system.gradient.segment(system.offsets.at(key), system.dimensions.at(key)) = state_->gradients.at(key);
  }
  std::vector<Eigen::Triplet<double>> entries;
  for (const auto& entry : state_->blocks) {
    auto row = system.offsets.find(entry.first.first);
    auto col = system.offsets.find(entry.first.second);
    if (row == system.offsets.end() || col == system.offsets.end()) continue;
    for (Eigen::Index i = 0; i < entry.second.rows(); ++i) {
      for (Eigen::Index j = 0; j < entry.second.cols(); ++j) {
        if (entry.second(i, j) != 0.) entries.emplace_back(row->second+i, col->second+j, entry.second(i, j));
      }
    }
  }
  system.hessian.resize(total, total);
  system.hessian.setFromTriplets(entries.begin(), entries.end());
  return system;
}

IncrementalLocalModel::SquareRootFactors IncrementalLocalModel::squareRootFactors() const {
  if (!state_->squareRootOnly) throw std::logic_error("normal model has no square-root cache");
  SquareRootFactors factors;
  for (const auto& entry : state_->factors) factors.emplace(entry.first, entry.second->squareRoot);
  return factors;
}

gtsam::Vector2 IncrementalLocalModel::directionalModel(const NativeVariableLayout& layout,
    const gtsam::Vector& displacement, const gtsam::Vector& step) const {
  if (!state_->squareRootOnly) throw std::logic_error("directional square-root model unavailable");
  if (layout.revision != revision() || displacement.size() != layout.dimension ||
      step.size() != layout.dimension || !displacement.allFinite() || !step.allFinite()) {
    throw std::invalid_argument("invalid or stale directional model input");
  }
  gtsam::Vector2 result = gtsam::Vector2::Zero();
  for (const auto& entry : state_->factors) {
    const auto& factor = *entry.second->squareRoot;
    gtsam::Vector residual = -factor.getb(), change = gtsam::Vector::Zero(factor.rows());
    for (auto it=factor.begin(); it!=factor.end(); ++it) {
      const auto offset = layout.offsets.find(*it);
      if (offset == layout.offsets.end()) continue;
      const auto dimension = layout.dimensions.at(*it);
      residual.noalias() += factor.getA(it)*displacement.segment(offset->second, dimension);
      change.noalias() += factor.getA(it)*step.segment(offset->second, dimension);
    }
    result[0] += residual.dot(change);
    result[1] += change.squaredNorm();
  }
  return result;
}

}  // namespace DPGO
