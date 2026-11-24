#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/JacobianFactor.h>
#include <Eigen/SparseCore>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace DPGO {

using NativeFactorId = std::uint64_t;

struct NativeModelUpdate {
  gtsam::Values newValues;
  gtsam::Values relinearizeValues;
  std::map<NativeFactorId, gtsam::NonlinearFactor::shared_ptr> addFactors;
  std::map<NativeFactorId, gtsam::NonlinearFactor::shared_ptr> replaceFactors;
  std::set<NativeFactorId> removeFactors;
};

struct NativeUpdateStatistics {
  std::uint64_t revision{0};
  std::size_t factorsLinearized{0};
  std::size_t factorsReused{0};
  std::size_t factorsAdded{0};
  std::size_t factorsReplaced{0};
  std::size_t factorsRemoved{0};
  double snapshotMs{0.}, linearizationMs{0.}, contributionMs{0.};
};

struct NativeVariableLayout {
  std::uint64_t revision{0};
  std::vector<gtsam::Key> keys;
  std::map<gtsam::Key, Eigen::Index> offsets;
  std::map<gtsam::Key, Eigen::Index> dimensions;
  Eigen::Index dimension{0};
};

struct NativeLinearSystem : NativeVariableLayout {
  Eigen::SparseMatrix<double> hessian;
  // Derivative of the cached quadratic, using the native factor Jacobians.
  // Some GTSAM configurations approximate the nonlinear pose-factor Jacobian.
  gtsam::Vector gradient;
  double constant{0.};
};

// Native Pose2/Point2, fixed-Gaussian local model; not an incremental optimizer.
// A transaction publishes only after all changed factors linearize successfully.
class IncrementalLocalModel {
 public:
  explicit IncrementalLocalModel(bool squareRootOnly = false);
  ~IncrementalLocalModel();
  IncrementalLocalModel(const IncrementalLocalModel&) = delete;
  IncrementalLocalModel& operator=(const IncrementalLocalModel&) = delete;

  NativeUpdateStatistics update(const NativeModelUpdate& update);
  const gtsam::Values& bases() const;
  std::size_t factorCount() const;
  std::uint64_t revision() const;
  gtsam::NonlinearFactorGraph factorGraph() const;

  using SquareRootFactors = std::map<NativeFactorId, std::shared_ptr<const gtsam::JacobianFactor>>;
  // Immutable whitened factors. Unchanged factors retain pointer identity.
  SquareRootFactors squareRootFactors() const;
  NativeVariableLayout linearLayout(
      const std::vector<gtsam::Key>& ordering = {},
      const std::set<gtsam::Key>& fixedKeys = {}) const;
  // Directional slope and curvature at a displacement from the cached bases.
  gtsam::Vector2 directionalModel(const NativeVariableLayout& layout,
      const gtsam::Vector& displacement, const gtsam::Vector& step) const;

  // Empty ordering uses stable insertion order. Fixed keys have zero increment.
  NativeLinearSystem linearSystem(
      const std::vector<gtsam::Key>& ordering = {},
      const std::set<gtsam::Key>& fixedKeys = {}) const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace DPGO
