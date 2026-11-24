#include <DPGO/IncrementalConditionalModel.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/VectorValues.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>

namespace DPGO {
namespace {
using Key = gtsam::Key;
using Bucket = std::optional<Key>;
using Dimensions = std::map<Key, Eigen::Index>;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

struct Conditional {
  gtsam::GaussianConditional::shared_ptr factor;
  gtsam::JacobianFactor::shared_ptr remaining;
  Bucket parent;
};

class Response final : public NativeLocalResponse {
 public:
  std::uint64_t version{0};
  std::vector<Key> keys, boundary, privateKeys;
  Dimensions dimensions, boundaryDims;
  gtsam::Matrix hessian;
  gtsam::Vector reducedGradient;
  std::vector<std::shared_ptr<const Conditional>> conditionals;

  std::uint64_t revision() const override { return version; }
  const std::vector<Key>& boundaryKeys() const override { return boundary; }
  const Dimensions& boundaryDimensions() const override { return boundaryDims; }
  const gtsam::Vector& gradient() const override { return reducedGradient; }
  gtsam::Matrix apply(const gtsam::Matrix& v) const override {
    if (v.rows() != hessian.cols() || !v.allFinite()) {
      throw std::invalid_argument("invalid conditional response product input");
    }
    gtsam::Matrix result = hessian*v;
    if (!result.allFinite()) throw std::runtime_error("nonfinite conditional response product");
    return result;
  }
  gtsam::Vector recover(const gtsam::Vector& b) const override {
    if (b.size() != hessian.cols() || !b.allFinite()) {
      throw std::invalid_argument("invalid conditional response boundary increment");
    }
    gtsam::VectorValues values;
    Eigen::Index offset = 0;
    for (auto key : boundary) {
      values.insert(key, b.segment(offset, dimensions.at(key)));
      offset += dimensions.at(key);
    }
    for (auto it=conditionals.rbegin(); it!=conditionals.rend(); ++it) {
      values.insert((*it)->factor->solve(values));
    }
    Eigen::Index total = 0;
    for (auto key : keys) total += dimensions.at(key);
    gtsam::Vector result(total); offset = 0;
    for (auto key : keys) {
      result.segment(offset, dimensions.at(key)) = values.at(key);
      offset += dimensions.at(key);
    }
    if (!result.allFinite()) throw std::runtime_error("nonfinite conditional recovery");
    return result;
  }
  double privateRelativeResidual(const gtsam::Vector& full) const override {
    Eigen::Index total = 0;
    for (auto key : keys) total += dimensions.at(key);
    if (full.size() != total || !full.allFinite()) throw std::invalid_argument("invalid conditional full increment");
    if (conditionals.empty()) return 0.;
    gtsam::VectorValues values;
    Eigen::Index offset = 0;
    for (auto key : keys) { values.insert(key, full.segment(offset, dimensions.at(key))); offset += dimensions.at(key); }
    auto zeroPrivate = values;
    for (auto key : privateKeys) zeroPrivate.at(key).setZero();
    gtsam::GaussianFactorGraph graph;
    for (const auto& conditional : conditionals) graph.push_back(conditional->factor);
    const auto residual = graph.gradient(values), rhs = graph.gradient(zeroPrivate);
    double residualSquared = 0., rhsSquared = 0.;
    for (auto key : privateKeys) {
      residualSquared += residual.at(key).squaredNorm(); rhsSquared += rhs.at(key).squaredNorm();
    }
    return std::sqrt(residualSquared)/std::max(1., std::sqrt(rhsSquared));
  }
};

void validateLayout(const std::vector<Key>& keys, const Dimensions& dimensions,
                    const std::vector<Key>& privateKeys) {
  std::set<Key> seen;
  Eigen::Index total = 0;
  if (keys.size() != dimensions.size()) throw std::invalid_argument("conditional layout size mismatch");
  for (auto key : keys) {
    auto dim = dimensions.find(key);
    if (!seen.insert(key).second || dim == dimensions.end() || dim->second <= 0 ||
        dim->second > std::numeric_limits<Eigen::Index>::max()-total) {
      throw std::invalid_argument("invalid conditional free-key layout");
    }
    total += dim->second;
  }
  seen.clear();
  for (auto key : privateKeys) {
    if (!dimensions.count(key) || !seen.insert(key).second) {
      throw std::invalid_argument("unknown or repeated conditional private key");
    }
  }
}

void validateKeys(const gtsam::GaussianFactor& factor, const Dimensions& dimensions) {
  std::set<Key> seen;
  for (auto it=factor.begin(); it!=factor.end(); ++it) {
    auto dim = dimensions.find(*it);
    if (dim == dimensions.end() || dim->second != factor.getDim(it) || !seen.insert(*it).second) {
      throw std::invalid_argument("invalid conditional factor key or dimension");
    }
  }
}

gtsam::GaussianFactor::shared_ptr prepare(
    const gtsam::GaussianFactor::shared_ptr& input, const Dimensions& dimensions,
    double& coefficientScale) {
  if (!input) throw std::invalid_argument("null conditional factor");
  const auto factor = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(input->clone());
  if (!factor || factor->empty()) throw std::invalid_argument("expected a nonempty Jacobian factor");
  validateKeys(*factor, dimensions);
  const auto noise = factor->get_model();
  if (noise && (noise->isConstrained() || !noise->sigmas().allFinite() ||
                (noise->sigmas().array() <= 0.).any())) {
    throw std::invalid_argument("conditional factor noise must be finite and unconstrained");
  }
  const auto ab = factor->jacobian();
  if (!ab.first.allFinite() || !ab.second.allFinite()) {
    throw std::invalid_argument("nonfinite conditional Jacobian");
  }
  coefficientScale = ab.first.size() ? ab.first.cwiseAbs().maxCoeff() : 0.;
  return factor;
}
}  // namespace

struct IncrementalConditionalModel::State {
  std::vector<Key> keys, privateKeys;
  Dimensions dimensions;
  std::map<Key, std::size_t> ranks;
  std::map<NativeFactorId, gtsam::GaussianFactor::shared_ptr> factors;
  std::map<NativeFactorId, double> factorScales;
  std::map<Bucket, std::set<NativeFactorId>> buckets;
  std::map<Bucket, std::set<Key>> children;
  std::map<Key, std::shared_ptr<const Conditional>> nodes;
  std::shared_ptr<const Response> response;

  Bucket owner(const gtsam::GaussianFactor& factor) const {
    Bucket result;
    std::size_t best = privateKeys.size();
    for (auto key : factor.keys()) {
      const auto rank = ranks.find(key);
      if (rank != ranks.end() && rank->second < best) { result=key; best=rank->second; }
    }
    return result;
  }
};

IncrementalConditionalModel::IncrementalConditionalModel() : state_(new State) {}
IncrementalConditionalModel::~IncrementalConditionalModel() = default;

std::shared_ptr<const NativeLocalResponse> IncrementalConditionalModel::response() const {
  if (!state_->response) throw std::logic_error("conditional model has no published revision");
  return state_->response;
}

NativeConditionalStatistics IncrementalConditionalModel::update(
    const NativeGaussianUpdate& input, const std::vector<Key>& keys,
    const Dimensions& dimensions, const std::vector<Key>& privateKeys) {
  validateLayout(keys, dimensions, privateKeys);
  NativeConditionalStatistics stats;
  const auto& old = *state_;
  if (old.response && input.addFactors.empty() && input.replaceFactors.empty() &&
      input.removeFactors.empty() && old.keys == keys && old.dimensions == dimensions &&
      old.privateKeys == privateKeys) {
    stats.revision = old.response->version; stats.conditionalsReused = privateKeys.size();
    return stats;
  }
  for (const auto& dim : old.dimensions) {
    if (dimensions.count(dim.first) && dimensions.at(dim.first) != dim.second) {
      throw std::invalid_argument("conditional variable dimension changed");
    }
  }
  const auto snapshotStart = Clock::now();
  // Metadata is copied for rollback; numeric factors/conditionals are immutable.
  auto next = std::make_unique<State>(old);
  stats.snapshotMs = ms(snapshotStart);
  const auto preparationStart = Clock::now();
  auto& s = *next;
  const bool appendOrder = privateKeys.size() >= old.privateKeys.size() &&
      std::equal(old.privateKeys.begin(), old.privateKeys.end(), privateKeys.begin());
  bool reusable = old.response && appendOrder;
  if (appendOrder) for (std::size_t i=old.privateKeys.size(); i<privateKeys.size(); ++i) {
    if (old.dimensions.count(privateKeys[i])) reusable = false;
  }
  s.keys=keys; s.dimensions=dimensions; s.privateKeys=privateKeys; s.ranks.clear();
  for (std::size_t i=0; i<privateKeys.size(); ++i) s.ranks[privateKeys[i]]=i;
  stats.fullRebuild = !reusable;
  std::map<NativeFactorId, gtsam::GaussianFactor::shared_ptr> prepared;
  std::map<NativeFactorId, double> preparedScales;
  for (const auto& entry : input.addFactors) {
    if (old.factors.count(entry.first) || input.replaceFactors.count(entry.first) || input.removeFactors.count(entry.first)) {
      throw std::invalid_argument("duplicate or conflicting conditional factor ID");
    }
    prepared[entry.first] = prepare(entry.second, dimensions, preparedScales[entry.first]);
  }
  for (const auto& entry : input.replaceFactors) {
    if (!old.factors.count(entry.first) || input.removeFactors.count(entry.first)) {
      throw std::invalid_argument("unknown or conflicting conditional replacement ID");
    }
    prepared[entry.first] = prepare(entry.second, dimensions, preparedScales[entry.first]);
  }
  std::set<NativeFactorId> subtract = input.removeFactors;
  for (auto id : subtract) if (!old.factors.count(id)) throw std::invalid_argument("unknown conditional removal ID");
  for (const auto& entry : input.replaceFactors) subtract.insert(entry.first);
  std::set<std::size_t> dirty;
  auto mark = [&](Bucket key) { if (key) dirty.insert(s.ranks.at(*key)); };
  for (auto id : subtract) {
    if (reusable) {
      const auto bucket = old.owner(*old.factors.at(id));
      s.buckets[bucket].erase(id); mark(bucket);
    }
    s.factors.erase(id);
    s.factorScales.erase(id);
  }
  for (const auto& entry : prepared) {
    s.factors[entry.first] = entry.second;
    s.factorScales[entry.first] = preparedScales.at(entry.first);
    if (reusable) {
      const auto bucket = s.owner(*entry.second);
      s.buckets[bucket].insert(entry.first); mark(bucket);
    }
  }
  for (const auto& factor : s.factors) validateKeys(*factor.second, dimensions);
  if (!reusable) {
    s.nodes.clear(); s.children.clear(); s.buckets.clear();
    for (const auto& entry : s.factors) s.buckets[s.owner(*entry.second)].insert(entry.first);
    for (std::size_t i=0; i<privateKeys.size(); ++i) dirty.insert(i);
  } else {
    for (std::size_t i=old.privateKeys.size(); i<privateKeys.size(); ++i) dirty.insert(i);
  }
  stats.factorsCloned = prepared.size();
  stats.preparationMs = ms(preparationStart);
  const auto eliminationStart = Clock::now();
  double coefficientScale = 0.;
  for (const auto& scale : s.factorScales) coefficientScale = std::max(coefficientScale, scale.second);
  Eigen::Index privateDimension = 0;
  for (auto key : privateKeys) privateDimension += dimensions.at(key);
  // Scale rank checks by the original model, not a nearly zero trailing pivot.
  const double rankTolerance = std::numeric_limits<double>::epsilon()*
      std::max<Eigen::Index>(1, privateDimension)*coefficientScale;
  while (!dirty.empty()) {
    const auto rank = *dirty.begin(); dirty.erase(dirty.begin());
    const auto key = privateKeys[rank];
    const auto previous = s.nodes.find(key);
    if (previous != s.nodes.end()) {
      s.children[previous->second->parent].erase(key); mark(previous->second->parent);
    }
    gtsam::GaussianFactorGraph graph;
    for (auto id : s.buckets[key]) { graph.push_back(s.factors.at(id)); ++stats.originalFactorsEliminated; }
    for (auto child : s.children[key]) { graph.push_back(s.nodes.at(child)->remaining); ++stats.childMessagesEliminated; }
    if (graph.empty()) throw std::runtime_error("unconstrained private conditional variable");
    const auto eliminated = gtsam::EliminateQR(graph, gtsam::Ordering{key});
    const gtsam::Matrix r = eliminated.first->R();
    const double scale = r.size() ? r.cwiseAbs().maxCoeff() : 0.;
    if (r.rows() != dimensions.at(key) || r.cols() != dimensions.at(key) || !r.allFinite() || scale <= 0. ||
        !eliminated.first->d().allFinite() || !eliminated.first->S().allFinite() ||
        (r.diagonal().cwiseAbs().array() <= rankTolerance).any()) {
      throw std::runtime_error("private conditional is numerically rank deficient");
    }
    auto node = std::make_shared<Conditional>();
    node->factor = eliminated.first; node->remaining = eliminated.second;
    node->parent = s.owner(*node->remaining);
    if (node->parent && s.ranks.at(*node->parent) <= rank) throw std::logic_error("invalid elimination dependency");
    s.nodes[key] = node; s.children[node->parent].insert(key); mark(node->parent);
    ++stats.conditionalsRebuilt;
  }
  stats.conditionalsReused = privateKeys.size()-stats.conditionalsRebuilt;
  stats.eliminationMs = ms(eliminationStart);
  const auto boundaryStart = Clock::now();
  auto response = std::make_shared<Response>();
  response->version = old.response ? old.response->version+1 : 1;
  response->keys=keys; response->dimensions=dimensions; response->privateKeys=privateKeys;
  Dimensions offsets;
  Eigen::Index n = 0;
  for (auto key : keys) if (!s.ranks.count(key)) {
    response->boundary.push_back(key); response->boundaryDims[key]=dimensions.at(key);
    offsets[key]=n; n+=dimensions.at(key);
  }
  for (auto key : privateKeys) response->conditionals.push_back(s.nodes.at(key));
  response->hessian = gtsam::Matrix::Zero(n,n); response->reducedGradient = gtsam::Vector::Zero(n);
  gtsam::GaussianFactorGraph reduced;
  for (auto id : s.buckets[std::nullopt]) reduced.push_back(s.factors.at(id));
  for (auto key : s.children[std::nullopt]) reduced.push_back(s.nodes.at(key)->remaining);
  for (const auto& factor : reduced) {
    const auto ab = factor->jacobian();
    if (!ab.first.allFinite() || !ab.second.allFinite()) throw std::runtime_error("nonfinite reduced conditional model");
    gtsam::Matrix a = gtsam::Matrix::Zero(ab.first.rows(), n);
    Eigen::Index col = 0;
    for (auto key : factor->keys()) {
      a.middleCols(offsets.at(key), dimensions.at(key)) = ab.first.middleCols(col, dimensions.at(key));
      col += dimensions.at(key);
    }
    response->hessian.noalias() += a.transpose()*a;
    response->reducedGradient.noalias() -= a.transpose()*ab.second;
  }
  if (!response->hessian.allFinite() || !response->reducedGradient.allFinite()) {
    throw std::runtime_error("conditional boundary model overflow");
  }
  s.response=response; stats.revision=response->version; stats.boundaryMs=ms(boundaryStart);
  state_.swap(next);
  return stats;
}
}  // namespace DPGO
