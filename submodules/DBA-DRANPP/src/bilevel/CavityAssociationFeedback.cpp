#include <DPGO/CavityAssociationFeedback.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace DPGO {
namespace {

double rankThreshold(const Vector &eigenvalues, std::size_t dimension) {
  const double scale =
      eigenvalues.size() == 0
          ? 1.0
          : std::max(1.0, eigenvalues.cwiseAbs().maxCoeff());
  return std::numeric_limits<double>::epsilon() *
         static_cast<double>(std::max<std::size_t>(1u, dimension)) * scale;
}

Vector positiveEigenvalues(const Matrix &matrix, const char *name) {
  if (matrix.rows() != matrix.cols() || !matrix.allFinite()) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite square matrix");
  }
  const Matrix symmetric = 0.5 * (matrix + matrix.transpose());
  const double symmetryScale = std::max(1.0, matrix.cwiseAbs().maxCoeff());
  const double symmetryTolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, matrix.rows())) *
      symmetryScale;
  if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() >
      symmetryTolerance) {
    throw std::invalid_argument(std::string(name) + " must be symmetric");
  }
  Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed to factor ") + name);
  }
  const Vector values = solver.eigenvalues();
  const double threshold =
      rankThreshold(values, static_cast<std::size_t>(matrix.rows()));
  if (values.size() > 0 && values.minCoeff() <= threshold) {
    throw std::invalid_argument(std::string(name) +
                                " must be positive definite");
  }
  return values;
}

CavityPairBlockEvidence invalidPairEvidence(const std::string &status,
                                            std::size_t numMatches,
                                            std::size_t gaugeRank = 0u) {
  CavityPairBlockEvidence evidence;
  evidence.status = status;
  evidence.numMatches = numMatches;
  evidence.gaugeRank = gaugeRank;
  return evidence;
}

}  // namespace

CavityPairBlockEvidence profileCavityPairBlockEvidence(
    const Matrix &firstPoints, const Matrix &secondPoints,
    const Matrix &firstCovariance, const Matrix &secondCovariance) {
  if (firstPoints.cols() != 2 || secondPoints.rows() != firstPoints.rows() ||
      secondPoints.cols() != firstPoints.cols() || !firstPoints.allFinite() ||
      !secondPoints.allFinite()) {
    throw std::invalid_argument("point blocks must be finite matching Nx2 matrices");
  }
  const std::size_t numMatches =
      static_cast<std::size_t>(firstPoints.rows());
  if (firstCovariance.rows() != firstPoints.rows() ||
      firstCovariance.cols() != firstPoints.rows() ||
      secondCovariance.rows() != firstPoints.rows() ||
      secondCovariance.cols() != firstPoints.rows()) {
    throw std::invalid_argument("pair covariances have incompatible dimensions");
  }
  if (numMatches < 2u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches);
  }

  const Matrix objectCovariance = firstCovariance + secondCovariance;
  const Vector covarianceEigenvalues =
      positiveEigenvalues(objectCovariance, "pair object covariance");
  const Matrix objectInformation = objectCovariance.ldlt().solve(
      Matrix::Identity(objectCovariance.rows(), objectCovariance.cols()));
  const Vector ones = Vector::Ones(firstPoints.rows());
  const Vector centroidWeights = objectInformation * ones;
  const double centroidDenominator = ones.dot(centroidWeights);
  if (!std::isfinite(centroidDenominator) || centroidDenominator <= 0.0) {
    throw std::runtime_error("invalid pair centroid information");
  }
  const Eigen::RowVector2d firstCentroid =
      centroidWeights.transpose() * firstPoints / centroidDenominator;
  const Eigen::RowVector2d secondCentroid =
      centroidWeights.transpose() * secondPoints / centroidDenominator;
  const Matrix firstCentered = firstPoints.rowwise() - firstCentroid;
  const Matrix secondCentered = secondPoints.rowwise() - secondCentroid;
  const Matrix crossCovariance =
      firstCentered.transpose() * objectInformation * secondCentered;
  Eigen::JacobiSVD<Matrix> svd(crossCovariance,
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix orientation = Matrix::Identity(2, 2);
  orientation(1, 1) =
      (svd.matrixU() * svd.matrixV().transpose()).determinant();
  const Matrix rotation =
      svd.matrixU() * orientation * svd.matrixV().transpose();
  const Eigen::RowVector2d translation =
      firstCentroid - secondCentroid * rotation.transpose();
  const Matrix predicted =
      (rotation * secondPoints.transpose()).transpose().rowwise() +
      translation;

  Matrix gaugeJacobian = Matrix::Zero(2 * firstPoints.rows(), 3);
  for (Eigen::Index match = 0; match < predicted.rows(); ++match) {
    const double x = predicted(match, 0) - firstCentroid(0);
    const double y = predicted(match, 1) - firstCentroid(1);
    gaugeJacobian(2 * match, 0) = y;
    gaugeJacobian(2 * match + 1, 0) = -x;
    gaugeJacobian.block<2, 2>(2 * match, 1) = -Matrix::Identity(2, 2);
  }
  Matrix information = Matrix::Zero(2 * firstPoints.rows(),
                                    2 * firstPoints.rows());
  for (Eigen::Index row = 0; row < objectInformation.rows(); ++row) {
    for (Eigen::Index col = 0; col < objectInformation.cols(); ++col) {
      information.block<2, 2>(2 * row, 2 * col) =
          objectInformation(row, col) * Matrix::Identity(2, 2);
    }
  }
  Matrix gaugeInformation =
      gaugeJacobian.transpose() * information * gaugeJacobian;
  gaugeInformation = 0.5 * (gaugeInformation + gaugeInformation.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix> gaugeSolver(gaugeInformation);
  if (gaugeSolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor pair gauge information");
  }
  const Vector gaugeEigenvalues = gaugeSolver.eigenvalues();
  const double gaugeThreshold = rankThreshold(gaugeEigenvalues, 3u);
  const std::size_t gaugeRank = static_cast<std::size_t>(
      (gaugeEigenvalues.array() > gaugeThreshold).count());
  if (gaugeRank != 3u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches,
                               gaugeRank);
  }

  const Matrix reversePredictedCentered =
      (rotation.transpose() * firstCentered.transpose()).transpose();
  Matrix reverseGaugeJacobian = Matrix::Zero(2 * firstPoints.rows(), 3);
  for (Eigen::Index match = 0; match < reversePredictedCentered.rows();
       ++match) {
    const double x = reversePredictedCentered(match, 0);
    const double y = reversePredictedCentered(match, 1);
    reverseGaugeJacobian(2 * match, 0) = y;
    reverseGaugeJacobian(2 * match + 1, 0) = -x;
    reverseGaugeJacobian.block<2, 2>(2 * match, 1) =
        -Matrix::Identity(2, 2);
  }
  Matrix reverseGaugeInformation = reverseGaugeJacobian.transpose() *
                                   information * reverseGaugeJacobian;
  reverseGaugeInformation =
      0.5 * (reverseGaugeInformation + reverseGaugeInformation.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix> reverseGaugeSolver(
      reverseGaugeInformation);
  if (reverseGaugeSolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor reverse pair gauge information");
  }
  const Vector reverseGaugeEigenvalues = reverseGaugeSolver.eigenvalues();
  const double reverseGaugeThreshold = rankThreshold(reverseGaugeEigenvalues, 3u);
  const std::size_t reverseGaugeRank = static_cast<std::size_t>(
      (reverseGaugeEigenvalues.array() > reverseGaugeThreshold).count());
  if (reverseGaugeRank != 3u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches,
                               std::min(gaugeRank, reverseGaugeRank));
  }

  Vector residual(2 * firstPoints.rows());
  for (Eigen::Index match = 0; match < firstPoints.rows(); ++match) {
    residual.segment<2>(2 * match) =
        (firstPoints.row(match) - predicted.row(match)).transpose();
  }
  const Vector rhs = gaugeJacobian.transpose() * information * residual;
  const Vector profiledResidual =
      residual - gaugeJacobian * gaugeInformation.ldlt().solve(rhs);
  const double profileChiSquared =
      profiledResidual.dot(information * profiledResidual);
  const double covarianceLogDet =
      2.0 * covarianceEigenvalues.array().log().sum();
  const double gaugeInformationLogDet =
      0.5 * (gaugeEigenvalues.array().log().sum() +
             reverseGaugeEigenvalues.array().log().sum());
  const double residualDimension = static_cast<double>(2u * numMatches);

  CavityPairBlockEvidence evidence;
  evidence.valid = true;
  evidence.status = "ok";
  evidence.profileChiSquared = profileChiSquared;
  evidence.covarianceLogDet = covarianceLogDet;
  evidence.gaugeInformationLogDet = gaugeInformationLogDet;
  evidence.score =
      0.5 * (profileChiSquared + covarianceLogDet +
             gaugeInformationLogDet +
             (residualDimension - 3.0) *
                 std::log(2.0 * std::acos(-1.0)));
  evidence.gaugeRank = gaugeRank;
  evidence.numMatches = numMatches;
  return evidence;
}

CavityPairBlockEvidence profileDiagonalCavityPairBlockEvidence(
    const Matrix &firstPoints, const Matrix &secondPoints,
    const Vector &firstVariance, const Vector &secondVariance) {
  if (firstPoints.cols() != 2 || secondPoints.rows() != firstPoints.rows() ||
      secondPoints.cols() != firstPoints.cols() || !firstPoints.allFinite() ||
      !secondPoints.allFinite()) {
    throw std::invalid_argument(
        "point blocks must be finite matching Nx2 matrices");
  }
  const std::size_t numMatches =
      static_cast<std::size_t>(firstPoints.rows());
  if (firstVariance.size() != firstPoints.rows() ||
      secondVariance.size() != firstPoints.rows() ||
      !firstVariance.allFinite() || !secondVariance.allFinite()) {
    throw std::invalid_argument(
        "pair variance vectors have incompatible dimensions");
  }
  if (numMatches < 2u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches);
  }
  const Vector covarianceDiagonal = firstVariance + secondVariance;
  if ((covarianceDiagonal.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "pair object covariance must be positive definite");
  }
  const Vector weights = covarianceDiagonal.cwiseInverse();
  const double weightSum = weights.sum();
  if (!std::isfinite(weightSum) || weightSum <= 0.0) {
    throw std::runtime_error("invalid pair centroid information");
  }
  const Eigen::RowVector2d firstCentroid =
      weights.transpose() * firstPoints / weightSum;
  const Eigen::RowVector2d secondCentroid =
      weights.transpose() * secondPoints / weightSum;
  const Matrix firstCentered = firstPoints.rowwise() - firstCentroid;
  const Matrix secondCentered = secondPoints.rowwise() - secondCentroid;
  Matrix crossCovariance = Matrix::Zero(2, 2);
  for (Eigen::Index match = 0; match < firstPoints.rows(); ++match) {
    crossCovariance.noalias() +=
        weights(match) * firstCentered.row(match).transpose() *
        secondCentered.row(match);
  }
  Eigen::JacobiSVD<Matrix> svd(crossCovariance,
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix orientation = Matrix::Identity(2, 2);
  orientation(1, 1) =
      (svd.matrixU() * svd.matrixV().transpose()).determinant();
  const Matrix rotation =
      svd.matrixU() * orientation * svd.matrixV().transpose();
  const Eigen::RowVector2d translation =
      firstCentroid - secondCentroid * rotation.transpose();
  const Matrix predicted =
      (rotation * secondPoints.transpose()).transpose().rowwise() +
      translation;

  Matrix gaugeInformation = Matrix::Zero(3, 3);
  Matrix reverseGaugeInformation = Matrix::Zero(3, 3);
  Vector rhs = Vector::Zero(3);
  Matrix residuals(firstPoints.rows(), 2);
  for (Eigen::Index match = 0; match < firstPoints.rows(); ++match) {
    Eigen::Matrix<double, 2, 3> jacobian =
        Eigen::Matrix<double, 2, 3>::Zero();
    jacobian(0, 0) = predicted(match, 1) - firstCentroid(1);
    jacobian(1, 0) = -(predicted(match, 0) - firstCentroid(0));
    jacobian.block<2, 2>(0, 1) = -Matrix::Identity(2, 2);
    const Vector residual =
        (firstPoints.row(match) - predicted.row(match)).transpose();
    residuals.row(match) = residual.transpose();
    const double weight = weights(match);
    gaugeInformation.noalias() +=
        weight * jacobian.transpose() * jacobian;
    rhs.noalias() += weight * jacobian.transpose() * residual;

    const Eigen::RowVector2d reversePoint =
        firstCentered.row(match) * rotation;
    Eigen::Matrix<double, 2, 3> reverseJacobian =
        Eigen::Matrix<double, 2, 3>::Zero();
    reverseJacobian(0, 0) = reversePoint(1);
    reverseJacobian(1, 0) = -reversePoint(0);
    reverseJacobian.block<2, 2>(0, 1) = -Matrix::Identity(2, 2);
    reverseGaugeInformation.noalias() +=
        weight * reverseJacobian.transpose() * reverseJacobian;
  }
  gaugeInformation = 0.5 * (gaugeInformation + gaugeInformation.transpose());
  reverseGaugeInformation =
      0.5 * (reverseGaugeInformation + reverseGaugeInformation.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix> gaugeSolver(gaugeInformation);
  if (gaugeSolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor pair gauge information");
  }
  const Vector gaugeEigenvalues = gaugeSolver.eigenvalues();
  const double gaugeThreshold = rankThreshold(gaugeEigenvalues, 3u);
  const std::size_t gaugeRank = static_cast<std::size_t>(
      (gaugeEigenvalues.array() > gaugeThreshold).count());
  if (gaugeRank != 3u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches,
                               gaugeRank);
  }
  Eigen::SelfAdjointEigenSolver<Matrix> reverseGaugeSolver(
      reverseGaugeInformation);
  if (reverseGaugeSolver.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor reverse pair gauge information");
  }
  const Vector reverseGaugeEigenvalues = reverseGaugeSolver.eigenvalues();
  const double reverseGaugeThreshold = rankThreshold(reverseGaugeEigenvalues, 3u);
  const std::size_t reverseGaugeRank = static_cast<std::size_t>(
      (reverseGaugeEigenvalues.array() > reverseGaugeThreshold).count());
  if (reverseGaugeRank != 3u) {
    return invalidPairEvidence("unobservable_pair_gauge", numMatches,
                               std::min(gaugeRank, reverseGaugeRank));
  }
  const Vector gaugeStep = gaugeInformation.ldlt().solve(rhs);
  double profileChiSquared = 0.0;
  for (Eigen::Index match = 0; match < firstPoints.rows(); ++match) {
    Eigen::Matrix<double, 2, 3> jacobian =
        Eigen::Matrix<double, 2, 3>::Zero();
    jacobian(0, 0) = predicted(match, 1) - firstCentroid(1);
    jacobian(1, 0) = -(predicted(match, 0) - firstCentroid(0));
    jacobian.block<2, 2>(0, 1) = -Matrix::Identity(2, 2);
    const Vector profiledResidual =
        residuals.row(match).transpose() - jacobian * gaugeStep;
    profileChiSquared +=
        weights(match) * profiledResidual.squaredNorm();
  }
  const double covarianceLogDet =
      2.0 * covarianceDiagonal.array().log().sum();
  const double gaugeInformationLogDet =
      0.5 * (gaugeEigenvalues.array().log().sum() +
             reverseGaugeEigenvalues.array().log().sum());
  const double residualDimension = static_cast<double>(2u * numMatches);

  CavityPairBlockEvidence evidence;
  evidence.valid = true;
  evidence.status = "ok";
  evidence.profileChiSquared = profileChiSquared;
  evidence.covarianceLogDet = covarianceLogDet;
  evidence.gaugeInformationLogDet = gaugeInformationLogDet;
  evidence.score =
      0.5 * (profileChiSquared + covarianceLogDet +
             gaugeInformationLogDet +
             (residualDimension - 3.0) *
                 std::log(2.0 * std::acos(-1.0)));
  evidence.gaugeRank = gaugeRank;
  evidence.numMatches = numMatches;
  return evidence;
}

CavityComponentAssignment solveCavityComponentAssignment(
    const std::vector<std::size_t> &candidateFirstObjects,
    const std::vector<std::size_t> &activeSecondObjects,
    const Matrix &costs) {
  const std::size_t candidateCount = candidateFirstObjects.size();
  const std::size_t activeCount = activeSecondObjects.size();
  if (std::set<std::size_t>(candidateFirstObjects.begin(),
                            candidateFirstObjects.end()).size() !=
          candidateCount ||
      std::set<std::size_t>(activeSecondObjects.begin(),
                            activeSecondObjects.end()).size() != activeCount) {
    throw std::invalid_argument("object identifiers must be unique");
  }
  if (candidateCount < activeCount) {
    throw std::invalid_argument("fewer candidates than active objects");
  }
  if (costs.rows() != static_cast<Eigen::Index>(candidateCount) ||
      costs.cols() != static_cast<Eigen::Index>(activeCount) ||
      !costs.allFinite()) {
    throw std::invalid_argument("assignment cost matrix is invalid");
  }
  if (activeCount == 0u) {
    return {};
  }

  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> rowPotential(activeCount + 1u, 0.0);
  std::vector<double> columnPotential(candidateCount + 1u, 0.0);
  std::vector<std::size_t> matching(candidateCount + 1u, 0u);
  std::vector<std::size_t> predecessor(candidateCount + 1u, 0u);
  for (std::size_t active = 1u; active <= activeCount; ++active) {
    matching[0] = active;
    std::size_t candidate0 = 0u;
    std::vector<double> minimum(candidateCount + 1u, infinity);
    std::vector<bool> used(candidateCount + 1u, false);
    do {
      used[candidate0] = true;
      const std::size_t active0 = matching[candidate0];
      double delta = infinity;
      std::size_t candidate1 = 0u;
      for (std::size_t candidate = 1u; candidate <= candidateCount;
           ++candidate) {
        if (used[candidate]) {
          continue;
        }
        const double reduced =
            costs(static_cast<Eigen::Index>(candidate - 1u),
                  static_cast<Eigen::Index>(active0 - 1u)) -
            rowPotential[active0] - columnPotential[candidate];
        if (reduced < minimum[candidate]) {
          minimum[candidate] = reduced;
          predecessor[candidate] = candidate0;
        }
        if (minimum[candidate] < delta) {
          delta = minimum[candidate];
          candidate1 = candidate;
        }
      }
      for (std::size_t candidate = 0u; candidate <= candidateCount;
           ++candidate) {
        if (used[candidate]) {
          rowPotential[matching[candidate]] += delta;
          columnPotential[candidate] -= delta;
        } else {
          minimum[candidate] -= delta;
        }
      }
      candidate0 = candidate1;
    } while (matching[candidate0] != 0u);
    do {
      const std::size_t candidate1 = predecessor[candidate0];
      matching[candidate0] = matching[candidate1];
      candidate0 = candidate1;
    } while (candidate0 != 0u);
  }

  CavityComponentAssignment result;
  for (std::size_t candidate = 1u; candidate <= candidateCount; ++candidate) {
    if (matching[candidate] == 0u) {
      continue;
    }
    const std::size_t active = matching[candidate] - 1u;
    result.assignments.push_back(
        {candidateFirstObjects[candidate - 1u], activeSecondObjects[active]});
    result.totalCost +=
        costs(static_cast<Eigen::Index>(candidate - 1u),
              static_cast<Eigen::Index>(active));
  }
  std::sort(result.assignments.begin(), result.assignments.end());
  return result;
}

namespace {

struct CavityStateIndex {
  std::map<std::size_t, std::size_t> byObject;
};

CavityStateIndex validateCavityRobotState(
    const CavityRobotAssociationState &state, const char *name) {
  const std::size_t count = state.objectIds.size();
  if (state.categories.size() != count ||
      state.points.rows() != static_cast<Eigen::Index>(count) ||
      state.points.cols() != 2 ||
      state.covarianceDiagonal.size() != static_cast<Eigen::Index>(count) ||
      !state.points.allFinite() || !state.covarianceDiagonal.allFinite() ||
      (state.covarianceDiagonal.array() <= 0.0).any()) {
    throw std::invalid_argument(std::string(name) +
                                " cavity state is inconsistent");
  }
  CavityStateIndex index;
  for (std::size_t position = 0; position < count; ++position) {
    if (!index.byObject.emplace(state.objectIds[position], position).second) {
      throw std::invalid_argument(std::string(name) +
                                  " object identifiers must be unique");
    }
  }
  return index;
}

double scoreCavityAssignments(
    const CavityRobotAssociationState &first,
    const CavityRobotAssociationState &second,
    const CavityStateIndex &firstIndex, const CavityStateIndex &secondIndex,
    std::vector<std::pair<std::size_t, std::size_t>> assignments) {
  std::sort(assignments.begin(), assignments.end());
  Matrix firstPoints(assignments.size(), 2);
  Matrix secondPoints(assignments.size(), 2);
  Vector firstVariance(assignments.size());
  Vector secondVariance(assignments.size());
  for (std::size_t match = 0; match < assignments.size(); ++match) {
    const auto firstEntry = firstIndex.byObject.find(assignments[match].first);
    const auto secondEntry =
        secondIndex.byObject.find(assignments[match].second);
    if (firstEntry == firstIndex.byObject.end() ||
        secondEntry == secondIndex.byObject.end()) {
      throw std::invalid_argument("assignment references unavailable cavity state");
    }
    firstPoints.row(static_cast<Eigen::Index>(match)) =
        first.points.row(static_cast<Eigen::Index>(firstEntry->second));
    secondPoints.row(static_cast<Eigen::Index>(match)) =
        second.points.row(static_cast<Eigen::Index>(secondEntry->second));
    firstVariance(static_cast<Eigen::Index>(match)) =
        first.covarianceDiagonal(static_cast<Eigen::Index>(firstEntry->second));
    secondVariance(static_cast<Eigen::Index>(match)) =
        second.covarianceDiagonal(
            static_cast<Eigen::Index>(secondEntry->second));
  }
  const CavityPairBlockEvidence evidence =
      profileDiagonalCavityPairBlockEvidence(
          firstPoints, secondPoints, firstVariance, secondVariance);
  if (!evidence.valid) {
    throw std::runtime_error("invalid cavity pair evidence: " +
                             evidence.status);
  }
  return evidence.score;
}

}  // namespace

CavityPairReassignment reassignCavityPairIdentities(
    const CavityRobotAssociationState &first,
    const CavityRobotAssociationState &second,
    const std::vector<std::pair<std::size_t, std::size_t>>
        &activeAssignments) {
  const CavityStateIndex firstIndex =
      validateCavityRobotState(first, "first");
  const CavityStateIndex secondIndex =
      validateCavityRobotState(second, "second");
  std::vector<std::pair<std::size_t, std::size_t>> original =
      activeAssignments;
  std::sort(original.begin(), original.end());
  if (std::unique(original.begin(), original.end()) != original.end()) {
    throw std::invalid_argument("active assignments must be unique");
  }
  std::set<std::size_t> usedFirst;
  std::set<std::size_t> usedSecond;
  std::map<std::size_t,
           std::vector<std::pair<std::size_t, std::size_t>>>
      assignmentsByCategory;
  for (const auto &assignment : original) {
    const auto firstEntry = firstIndex.byObject.find(assignment.first);
    const auto secondEntry = secondIndex.byObject.find(assignment.second);
    if (firstEntry == firstIndex.byObject.end() ||
        secondEntry == secondIndex.byObject.end()) {
      throw std::invalid_argument("active assignment lacks cavity state");
    }
    if (!usedFirst.insert(assignment.first).second ||
        !usedSecond.insert(assignment.second).second) {
      throw std::invalid_argument("active assignments must be one-to-one");
    }
    const std::size_t firstCategory = first.categories[firstEntry->second];
    const std::size_t secondCategory = second.categories[secondEntry->second];
    if (firstCategory != secondCategory) {
      throw std::invalid_argument("active assignment violates semantic category");
    }
    assignmentsByCategory[firstCategory].push_back(assignment);
  }

  CavityPairReassignment result;
  result.assignmentsBefore = original;
  result.assignmentsAfter = original;
  result.payloadBytes =
      (first.objectIds.size() + second.objectIds.size()) * 3u * sizeof(double);
  result.componentCount = assignmentsByCategory.size();
  result.scoreBefore = scoreCavityAssignments(
      first, second, firstIndex, secondIndex, original);

  std::vector<std::pair<std::size_t, std::size_t>> candidateAssignments;
  for (const auto &component : assignmentsByCategory) {
    const std::size_t category = component.first;
    std::vector<std::size_t> activeSecondObjects;
    for (const auto &assignment : component.second) {
      activeSecondObjects.push_back(assignment.second);
    }
    std::sort(activeSecondObjects.begin(), activeSecondObjects.end());
    std::vector<std::size_t> candidateFirstObjects;
    for (std::size_t index = 0; index < first.objectIds.size(); ++index) {
      if (first.categories[index] == category) {
        candidateFirstObjects.push_back(first.objectIds[index]);
      }
    }
    std::sort(candidateFirstObjects.begin(), candidateFirstObjects.end());

    std::vector<std::pair<std::size_t, std::size_t>> componentExcluded;
    for (const auto &assignment : original) {
      const std::size_t firstPosition =
          firstIndex.byObject.at(assignment.first);
      if (first.categories[firstPosition] != category) {
        componentExcluded.push_back(assignment);
      }
    }
    Matrix costs(candidateFirstObjects.size(), activeSecondObjects.size());
    for (std::size_t row = 0; row < candidateFirstObjects.size(); ++row) {
      for (std::size_t col = 0; col < activeSecondObjects.size(); ++col) {
        std::vector<std::pair<std::size_t, std::size_t>> singleton =
            componentExcluded;
        singleton.push_back(
            {candidateFirstObjects[row], activeSecondObjects[col]});
        costs(static_cast<Eigen::Index>(row),
              static_cast<Eigen::Index>(col)) =
            scoreCavityAssignments(first, second, firstIndex, secondIndex,
                                   std::move(singleton));
        ++result.candidateEvaluationCount;
      }
    }
    const CavityComponentAssignment assignment =
        solveCavityComponentAssignment(candidateFirstObjects,
                                       activeSecondObjects, costs);
    candidateAssignments.insert(candidateAssignments.end(),
                                assignment.assignments.begin(),
                                assignment.assignments.end());
  }
  std::sort(candidateAssignments.begin(), candidateAssignments.end());
  result.scoreAfterCandidate = scoreCavityAssignments(
      first, second, firstIndex, secondIndex, candidateAssignments);
  result.accepted = result.scoreAfterCandidate < result.scoreBefore;
  if (result.accepted) {
    result.assignmentsAfter = candidateAssignments;
  } else {
    result.scoreAfterCandidate = result.scoreBefore;
  }
  std::set<std::pair<std::size_t, std::size_t>> after(
      result.assignmentsAfter.begin(), result.assignmentsAfter.end());
  result.changedEdgeCount = static_cast<std::size_t>(std::count_if(
      original.begin(), original.end(),
      [&](const auto &assignment) { return after.count(assignment) == 0u; }));
  return result;
}

std::vector<std::pair<std::size_t, std::size_t>>
extractCavityActiveAssignments(const Matrix &weights) {
  if (!weights.allFinite()) {
    throw std::invalid_argument("association weights must be finite");
  }
  std::vector<Eigen::Index> rowChoice(
      static_cast<std::size_t>(weights.rows()), -1);
  std::vector<Eigen::Index> columnChoice(
      static_cast<std::size_t>(weights.cols()), -1);
  for (Eigen::Index row = 0; row < weights.rows(); ++row) {
    double best = 0.0;
    for (Eigen::Index col = 0; col < weights.cols(); ++col) {
      if (weights(row, col) > best) {
        best = weights(row, col);
        rowChoice[static_cast<std::size_t>(row)] = col;
      }
    }
  }
  for (Eigen::Index col = 0; col < weights.cols(); ++col) {
    double best = 0.0;
    for (Eigen::Index row = 0; row < weights.rows(); ++row) {
      if (weights(row, col) > best) {
        best = weights(row, col);
        columnChoice[static_cast<std::size_t>(col)] = row;
      }
    }
  }
  std::vector<std::pair<std::size_t, std::size_t>> active;
  for (Eigen::Index row = 0; row < weights.rows(); ++row) {
    const Eigen::Index col = rowChoice[static_cast<std::size_t>(row)];
    if (col >= 0 &&
        columnChoice[static_cast<std::size_t>(col)] == row) {
      active.push_back({static_cast<std::size_t>(row),
                        static_cast<std::size_t>(col)});
    }
  }
  return active;
}

}  // namespace DPGO
