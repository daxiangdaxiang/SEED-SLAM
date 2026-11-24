#include <DPGO/NativeSchurModel.h>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace DPGO {
namespace {
using Sparse = Eigen::SparseMatrix<double>;

void validate(const NativeLinearSystem& system) {
  const auto n = system.gradient.size();
  if (system.hessian.rows() != n || system.hessian.cols() != n ||
      !system.gradient.allFinite() || !std::isfinite(system.constant) ||
      system.offsets.size() != system.keys.size() ||
      system.dimensions.size() != system.keys.size()) {
    throw std::invalid_argument("invalid native tangent system");
  }
  std::set<gtsam::Key> seen;
  Eigen::Index offset = 0;
  for (auto key : system.keys) {
    auto position = system.offsets.find(key);
    auto dimension = system.dimensions.find(key);
    if (!seen.insert(key).second || position == system.offsets.end() ||
        dimension == system.dimensions.end() || position->second != offset ||
        dimension->second <= 0 || dimension->second > n-offset) {
      throw std::invalid_argument("invalid native key layout");
    }
    offset += dimension->second;
  }
  if (offset != n) throw std::invalid_argument("native key dimensions do not cover system");
  for (int col = 0; col < system.hessian.outerSize(); ++col) {
    for (Sparse::InnerIterator it(system.hessian, col); it; ++it) {
      if (!std::isfinite(it.value())) throw std::invalid_argument("nonfinite native Hessian");
    }
  }
  const Sparse difference = system.hessian-Sparse(system.hessian.transpose());
  if (difference.norm() > 1e-12*std::max(1., system.hessian.norm())) {
    throw std::invalid_argument("native Hessian is not symmetric");
  }
}
}  // namespace

struct NativeSchurModel::State {
  std::uint64_t revision;
  double privateFactorizationMs{0.};
  std::vector<gtsam::Key> boundaryKeys;
  std::map<gtsam::Key, Eigen::Index> boundaryDimensions;
  std::vector<Eigen::Index> privateColumns, boundaryColumns;
  Sparse pp, pb, bb;
  Eigen::SimplicialLDLT<Sparse> solver;
  gtsam::Vector gp, gb, reducedGradient;

  gtsam::Matrix solvePrivate(const gtsam::Matrix& rhs) const {
    if (!rhs.allFinite()) throw std::runtime_error("nonfinite private right-hand side");
    if (rhs.rows() == 0 || rhs.cols() == 0) return rhs;
    gtsam::Matrix result = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !result.allFinite()) {
      throw std::runtime_error("native private solve failed");
    }
    return result;
  }
};

NativeSchurModel::NativeSchurModel(
    const NativeLinearSystem& system, const std::vector<gtsam::Key>& privateKeys)
    : state_(new State) {
  validate(system);
  std::set<gtsam::Key> privateSet;
  for (auto key : privateKeys) {
    if (!system.offsets.count(key) || !privateSet.insert(key).second) {
      throw std::invalid_argument("unknown or repeated native private key");
    }
  }
  auto& s = *state_;
  s.revision = system.revision;
  const auto n = system.gradient.size();
  std::vector<Eigen::Index> privateIndex(n, -1), boundaryIndex(n, -1);
  for (auto key : system.keys) {
    const bool isPrivate = privateSet.count(key);
    auto& columns = isPrivate ? s.privateColumns : s.boundaryColumns;
    auto& index = isPrivate ? privateIndex : boundaryIndex;
    if (!isPrivate) {
      s.boundaryKeys.push_back(key);
      s.boundaryDimensions[key] = system.dimensions.at(key);
    }
    for (Eigen::Index i = 0; i < system.dimensions.at(key); ++i) {
      const auto column = system.offsets.at(key)+i;
      index[column] = columns.size();
      columns.push_back(column);
    }
  }
  const Eigen::Index np = s.privateColumns.size(), nb = s.boundaryColumns.size();
  s.gp.resize(np);
  s.gb.resize(nb);
  for (Eigen::Index i = 0; i < np; ++i) s.gp(i) = system.gradient(s.privateColumns[i]);
  for (Eigen::Index i = 0; i < nb; ++i) s.gb(i) = system.gradient(s.boundaryColumns[i]);
  std::vector<Eigen::Triplet<double>> pp, pb, bb;
  // Only average roundoff after the explicit symmetry check, never repair a
  // nonsymmetric model or add a prior to make an unobservable system invertible.
  const Sparse h = .5*(system.hessian+Sparse(system.hessian.transpose()));
  for (int col = 0; col < h.outerSize(); ++col) {
    for (Sparse::InnerIterator it(h, col); it; ++it) {
      const auto pRow = privateIndex[it.row()], pCol = privateIndex[it.col()];
      if (pRow >= 0 && pCol >= 0) pp.emplace_back(pRow, pCol, it.value());
      else if (pRow >= 0) pb.emplace_back(pRow, boundaryIndex[it.col()], it.value());
      else if (pCol < 0) bb.emplace_back(boundaryIndex[it.row()], boundaryIndex[it.col()], it.value());
    }
  }
  s.pp.resize(np, np);
  s.pb.resize(np, nb);
  s.bb.resize(nb, nb);
  s.pp.setFromTriplets(pp.begin(), pp.end());
  s.pb.setFromTriplets(pb.begin(), pb.end());
  s.bb.setFromTriplets(bb.begin(), bb.end());
  if (np > 0) {
    const auto factorStart = std::chrono::steady_clock::now();
    s.solver.compute(s.pp);
    s.privateFactorizationMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-factorStart).count();
    if (s.solver.info() != Eigen::Success) {
      throw std::runtime_error("native private factorization failed");
    }
    const auto pivots = s.solver.vectorD();
    const double scale = s.pp.diagonal().cwiseAbs().maxCoeff();
    const double tolerance = std::numeric_limits<double>::epsilon()*np*scale;
    if (!pivots.allFinite() || scale <= 0. || (pivots.array() <= tolerance).any()) {
      throw std::runtime_error("native private system is not numerically positive definite");
    }
  }
  s.reducedGradient = s.gb-s.pb.transpose()*s.solvePrivate(s.gp);
  if (!s.reducedGradient.allFinite()) throw std::runtime_error("nonfinite native reduced gradient");
}

NativeSchurModel::~NativeSchurModel() = default;
std::uint64_t NativeSchurModel::revision() const { return state_->revision; }
double NativeSchurModel::privateFactorizationMs() const { return state_->privateFactorizationMs; }
const std::vector<gtsam::Key>& NativeSchurModel::boundaryKeys() const { return state_->boundaryKeys; }
const std::map<gtsam::Key, Eigen::Index>& NativeSchurModel::boundaryDimensions() const {
  return state_->boundaryDimensions;
}
const gtsam::Vector& NativeSchurModel::gradient() const { return state_->reducedGradient; }

gtsam::Matrix NativeSchurModel::apply(const gtsam::Matrix& v) const {
  const auto& s = *state_;
  if (v.rows() != s.gb.size() || !v.allFinite()) {
    throw std::invalid_argument("invalid native interface product input");
  }
  const gtsam::Matrix result = s.bb*v-s.pb.transpose()*s.solvePrivate(s.pb*v);
  if (!result.allFinite()) throw std::runtime_error("nonfinite native interface product");
  return result;
}

gtsam::Vector NativeSchurModel::recover(const gtsam::Vector& boundary) const {
  const auto& s = *state_;
  if (boundary.size() != s.gb.size() || !boundary.allFinite()) {
    throw std::invalid_argument("invalid native interface increment");
  }
  const gtsam::Vector privateStep = -s.solvePrivate(s.gp+s.pb*boundary);
  gtsam::Vector full(s.gp.size()+s.gb.size());
  for (Eigen::Index i = 0; i < s.gp.size(); ++i) full(s.privateColumns[i]) = privateStep(i);
  for (Eigen::Index i = 0; i < s.gb.size(); ++i) full(s.boundaryColumns[i]) = boundary(i);
  return full;
}

double NativeSchurModel::privateRelativeResidual(const gtsam::Vector& full) const {
  const auto& s = *state_;
  if (full.size() != s.gp.size()+s.gb.size() || !full.allFinite()) {
    throw std::invalid_argument("invalid native full increment");
  }
  gtsam::Vector privateStep(s.gp.size()), boundary(s.gb.size());
  for (Eigen::Index i = 0; i < s.gp.size(); ++i) privateStep(i) = full(s.privateColumns[i]);
  for (Eigen::Index i = 0; i < s.gb.size(); ++i) boundary(i) = full(s.boundaryColumns[i]);
  const gtsam::Vector rhs = -s.gp-s.pb*boundary;
  return (s.pp*privateStep-rhs).norm()/std::max(1., rhs.norm());
}

}  // namespace DPGO
