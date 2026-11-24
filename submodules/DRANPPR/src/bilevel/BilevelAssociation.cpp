/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/BilevelAssociation.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/QuadraticProblem.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>

#include <boost/math/distributions/students_t.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DPGO {
namespace {

double finiteNonnegative(double value) {
  if (!std::isfinite(value) || value < 0.0) {
    return 0.0;
  }
  return value;
}

void validateSinkhornOptions(const PartialSinkhornOptions &options) {
  if (!std::isfinite(options.stepSize) || options.stepSize < 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.stepSize must be "
                                "finite and nonnegative");
  }
  if (!std::isfinite(options.temperature) || options.temperature <= 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.temperature must be "
                                "finite and positive");
  }
  if (!std::isfinite(options.rowCapacity) || options.rowCapacity < 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.rowCapacity must be "
                                "finite and nonnegative");
  }
  if (!std::isfinite(options.columnCapacity) ||
      options.columnCapacity < 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.columnCapacity must be "
                                "finite and nonnegative");
  }
  if (!std::isfinite(options.temporalBlend) || options.temporalBlend < 0.0 ||
      options.temporalBlend > 1.0) {
    throw std::invalid_argument("PartialSinkhornOptions.temporalBlend must be "
                                "finite and in [0, 1]");
  }
  if (!std::isfinite(options.temporalKlWeight) ||
      options.temporalKlWeight < 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.temporalKlWeight "
                                "must be finite and nonnegative");
  }
  if (!std::isfinite(options.temporalKlEpsilon) ||
      options.temporalKlEpsilon <= 0.0) {
    throw std::invalid_argument("PartialSinkhornOptions.temporalKlEpsilon "
                                "must be finite and positive");
  }
  if (!std::isfinite(options.entropyRegularizationWeight) ||
      options.entropyRegularizationWeight < 0.0) {
    throw std::invalid_argument(
        "PartialSinkhornOptions.entropyRegularizationWeight must be finite "
        "and nonnegative");
  }
  if (!std::isinf(options.unmatchedCost) &&
      !std::isfinite(options.unmatchedCost)) {
    throw std::invalid_argument(
        "PartialSinkhornOptions.unmatchedCost must be finite or infinity");
  }
  if (!std::isfinite(options.previousSupportThreshold) ||
      options.previousSupportThreshold < 0.0) {
    throw std::invalid_argument(
        "PartialSinkhornOptions.previousSupportThreshold must be finite "
        "and nonnegative");
  }
}

struct Entry {
  int row{0};
  int col{0};
  double value{0.0};
};

struct TransformInlierEntry {
  std::size_t candidateIndex{0};
  double residual{0.0};
  double weight{0.0};
};

bool validTransformCandidate(const TransformAssociationCandidate &candidate,
                             std::size_t numLocalObjects,
                             std::size_t numTargetObjects,
                             const TransformAssociationVerificationOptions
                                 &options) {
  if (candidate.localObjectId >= numLocalObjects ||
      candidate.targetObjectId >= numTargetObjects) {
    return false;
  }
  if (!std::isfinite(candidate.score) || candidate.score < options.minScore ||
      candidate.score <= 0.0) {
    return false;
  }
  if (candidate.localPoint.rows() != static_cast<int>(options.dimension) ||
      candidate.targetPoint.rows() != static_cast<int>(options.dimension) ||
      candidate.localPoint.cols() != 1 || candidate.targetPoint.cols() != 1) {
    return false;
  }
  return candidate.localPoint.allFinite() && candidate.targetPoint.allFinite();
}

bool rigidSeedPairwiseDistancesConsistent(
    const std::vector<TransformAssociationCandidate> &candidates,
    const std::vector<std::size_t> &indices,
    const TransformAssociationVerificationOptions &options) {
  const double tolerance =
      2.0 * std::max(options.maxResidual,
                     std::numeric_limits<double>::epsilon()) +
      1e-12;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const TransformAssociationCandidate &a = candidates[indices[i]];
    for (std::size_t j = i + 1; j < indices.size(); ++j) {
      const TransformAssociationCandidate &b = candidates[indices[j]];
      const double localDistance = (a.localPoint - b.localPoint).norm();
      const double targetDistance = (a.targetPoint - b.targetPoint).norm();
      if (!std::isfinite(localDistance) || !std::isfinite(targetDistance)) {
        return false;
      }
      if (std::abs(localDistance - targetDistance) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool estimateRigidTransform(
    const std::vector<TransformAssociationCandidate> &candidates,
    const std::vector<std::size_t> &indices, unsigned dimension,
    Matrix &transform) {
  if (indices.size() < 2 || dimension < 2) {
    return false;
  }
  if (dimension == 2 && indices.size() == 2) {
    const TransformAssociationCandidate &first = candidates[indices[0]];
    const TransformAssociationCandidate &second = candidates[indices[1]];
    const Vector sourceDelta = second.targetPoint - first.targetPoint;
    const Vector targetDelta = second.localPoint - first.localPoint;
    const double sourceNorm = sourceDelta.norm();
    const double targetNorm = targetDelta.norm();
    if (sourceNorm < 1e-18 || targetNorm < 1e-18) {
      return false;
    }
    const double invNormProduct = 1.0 / (sourceNorm * targetNorm);
    const double cosine = std::clamp(
        sourceDelta.dot(targetDelta) * invNormProduct, -1.0, 1.0);
    const double sine =
        (sourceDelta(0) * targetDelta(1) -
         sourceDelta(1) * targetDelta(0)) *
        invNormProduct;
    Matrix rotation(2, 2);
    rotation << cosine, -sine, sine, cosine;
    const Vector translation =
        first.localPoint - rotation * first.targetPoint;
    transform = Matrix::Zero(2, 3);
    transform.leftCols(2) = rotation;
    transform.col(2) = translation;
    return transform.allFinite();
  }

  Vector sourceMean = Vector::Zero(dimension);
  Vector targetMean = Vector::Zero(dimension);
  for (std::size_t index : indices) {
    sourceMean += candidates[index].targetPoint;
    targetMean += candidates[index].localPoint;
  }
  sourceMean /= static_cast<double>(indices.size());
  targetMean /= static_cast<double>(indices.size());

  Matrix covariance = Matrix::Zero(dimension, dimension);
  double sourceScatter = 0.0;
  double targetScatter = 0.0;
  for (std::size_t index : indices) {
    const Vector sourceCentered =
        candidates[index].targetPoint - sourceMean;
    const Vector targetCentered =
        candidates[index].localPoint - targetMean;
    covariance.noalias() += sourceCentered * targetCentered.transpose();
    sourceScatter += sourceCentered.squaredNorm();
    targetScatter += targetCentered.squaredNorm();
  }
  if (sourceScatter < 1e-18 || targetScatter < 1e-18) {
    return false;
  }

  Eigen::JacobiSVD<Matrix> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix u = svd.matrixU();
  Matrix v = svd.matrixV();
  Matrix rotation = v * u.transpose();
  if (rotation.determinant() < 0.0) {
    v.col(static_cast<int>(dimension - 1)) *= -1.0;
    rotation = v * u.transpose();
  }
  const Vector translation = targetMean - rotation * sourceMean;

  transform = Matrix::Zero(dimension, dimension + 1);
  transform.leftCols(static_cast<int>(dimension)) = rotation;
  transform.col(static_cast<int>(dimension)) = translation;
  return transform.allFinite();
}

std::vector<TransformInlierEntry> transformInliers(
    const std::vector<TransformAssociationCandidate> &candidates,
    const std::vector<std::size_t> &validIndices, const Matrix &transform,
    const TransformAssociationVerificationOptions &options) {
  std::vector<TransformInlierEntry> entries;
  entries.reserve(validIndices.size());
  if (options.dimension == 2 && transform.rows() == 2 &&
      transform.cols() >= 3) {
    const double r00 = transform(0, 0);
    const double r01 = transform(0, 1);
    const double r10 = transform(1, 0);
    const double r11 = transform(1, 1);
    const double tx = transform(0, 2);
    const double ty = transform(1, 2);
    const double sigma =
        std::max(options.maxResidual, std::numeric_limits<double>::epsilon());
    const double invSigmaSquared = 1.0 / (sigma * sigma);
    const double maxResidualSquared =
        options.maxResidual * options.maxResidual;
    for (std::size_t index : validIndices) {
      const TransformAssociationCandidate &candidate = candidates[index];
      const double targetX = candidate.targetPoint(0);
      const double targetY = candidate.targetPoint(1);
      const double dx =
          candidate.localPoint(0) - (r00 * targetX + r01 * targetY + tx);
      const double dy =
          candidate.localPoint(1) - (r10 * targetX + r11 * targetY + ty);
      const double residualSquared = dx * dx + dy * dy;
      if (!std::isfinite(residualSquared) ||
          residualSquared > maxResidualSquared) {
        continue;
      }
      const double robust = std::exp(-0.5 * residualSquared * invSigmaSquared);
      const double weight = std::max(
          0.0, std::min(1.0, options.weightScale * candidate.score * robust));
      if (weight <= 0.0) {
        continue;
      }
      entries.push_back(
          TransformInlierEntry{index, std::sqrt(residualSquared), weight});
    }
    std::sort(entries.begin(), entries.end(),
              [&](const TransformInlierEntry &a,
                  const TransformInlierEntry &b) {
                const TransformAssociationCandidate &ca =
                    candidates[a.candidateIndex];
                const TransformAssociationCandidate &cb =
                    candidates[b.candidateIndex];
                if (a.residual != b.residual) {
                  return a.residual < b.residual;
                }
                if (ca.score != cb.score) {
                  return ca.score > cb.score;
                }
                if (ca.localObjectId != cb.localObjectId) {
                  return ca.localObjectId < cb.localObjectId;
                }
                return ca.targetObjectId < cb.targetObjectId;
              });
    return entries;
  }
  const Matrix rotation =
      transform.leftCols(static_cast<int>(options.dimension));
  const Vector translation =
      transform.col(static_cast<int>(options.dimension));
  const double sigma =
      std::max(options.maxResidual, std::numeric_limits<double>::epsilon());
  const double invSigmaSquared = 1.0 / (sigma * sigma);
  const double maxResidualSquared =
      options.maxResidual * options.maxResidual;
  for (std::size_t index : validIndices) {
    const TransformAssociationCandidate &candidate = candidates[index];
    const Vector predicted = rotation * candidate.targetPoint + translation;
    const double residualSquared =
        (candidate.localPoint - predicted).squaredNorm();
    if (!std::isfinite(residualSquared) ||
        residualSquared > maxResidualSquared) {
      continue;
    }
    const double robust = std::exp(-0.5 * residualSquared * invSigmaSquared);
    const double weight = std::max(
        0.0, std::min(1.0, options.weightScale * candidate.score * robust));
    if (weight <= 0.0) {
      continue;
    }
    entries.push_back(
        TransformInlierEntry{index, std::sqrt(residualSquared), weight});
  }
  std::sort(entries.begin(), entries.end(),
            [&](const TransformInlierEntry &a,
                const TransformInlierEntry &b) {
              const TransformAssociationCandidate &ca =
                  candidates[a.candidateIndex];
              const TransformAssociationCandidate &cb =
                  candidates[b.candidateIndex];
              if (a.residual != b.residual) {
                return a.residual < b.residual;
              }
              if (ca.score != cb.score) {
                return ca.score > cb.score;
              }
              if (ca.localObjectId != cb.localObjectId) {
                return ca.localObjectId < cb.localObjectId;
              }
              return ca.targetObjectId < cb.targetObjectId;
            });
  return entries;
}

std::vector<TransformInlierEntry> selectOneToOneInliers(
    const std::vector<TransformAssociationCandidate> &candidates,
    const std::vector<TransformInlierEntry> &entries,
    std::size_t numLocalObjects, std::size_t numTargetObjects,
    std::vector<int> &usedLocal, std::vector<int> &usedTarget, int &stamp) {
  if (usedLocal.size() != numLocalObjects) {
    usedLocal.assign(numLocalObjects, 0);
  }
  if (usedTarget.size() != numTargetObjects) {
    usedTarget.assign(numTargetObjects, 0);
  }
  if (stamp == std::numeric_limits<int>::max()) {
    std::fill(usedLocal.begin(), usedLocal.end(), 0);
    std::fill(usedTarget.begin(), usedTarget.end(), 0);
    stamp = 0;
  }
  ++stamp;
  std::vector<TransformInlierEntry> selected;
  selected.reserve(entries.size());
  for (const TransformInlierEntry &entry : entries) {
    const TransformAssociationCandidate &candidate =
        candidates[entry.candidateIndex];
    if (usedLocal[candidate.localObjectId] == stamp ||
        usedTarget[candidate.targetObjectId] == stamp) {
      continue;
    }
    usedLocal[candidate.localObjectId] = stamp;
    usedTarget[candidate.targetObjectId] = stamp;
    selected.push_back(entry);
  }
  return selected;
}

void keepLargestNonzeros(Matrix &weights, std::size_t maxNonzeros) {
  if (maxNonzeros == 0 ||
      maxNonzeros >= static_cast<std::size_t>(weights.size())) {
    return;
  }

  std::vector<Entry> entries;
  entries.reserve(static_cast<std::size_t>(weights.size()));
  for (int row = 0; row < weights.rows(); ++row) {
    for (int col = 0; col < weights.cols(); ++col) {
      const double value = weights(row, col);
      if (std::isfinite(value) && value > 0.0) {
        entries.push_back(Entry{row, col, value});
      }
    }
  }
  if (entries.size() <= maxNonzeros) {
    return;
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) {
              if (a.value != b.value) {
                return a.value > b.value;
              }
              if (a.row != b.row) {
                return a.row < b.row;
              }
              return a.col < b.col;
            });

  Matrix pruned = Matrix::Zero(weights.rows(), weights.cols());
  for (std::size_t idx = 0; idx < maxNonzeros; ++idx) {
    const Entry &entry = entries[idx];
    pruned(entry.row, entry.col) = entry.value;
  }
  weights = std::move(pruned);
}

void scaleRowsToCapacity(Matrix &weights, double capacity) {
  if (capacity < 0.0) {
    return;
  }
  for (int row = 0; row < weights.rows(); ++row) {
    double sum = weights.row(row).sum();
    if (sum > capacity && sum > 0.0) {
      weights.row(row) *= capacity / sum;
    }
  }
}

void scaleColumnsToCapacity(Matrix &weights, double capacity) {
  if (capacity < 0.0) {
    return;
  }
  for (int col = 0; col < weights.cols(); ++col) {
    double sum = weights.col(col).sum();
    if (sum > capacity && sum > 0.0) {
      weights.col(col) *= capacity / sum;
    }
  }
}

void scalePositiveRowsToCapacity(Matrix &weights, double capacity) {
  if (capacity < 0.0) {
    return;
  }
  for (int row = 0; row < weights.rows(); ++row) {
    double sum = weights.row(row).sum();
    if (sum > 0.0) {
      weights.row(row) *= capacity / sum;
    }
  }
}

void scalePositiveColumnsToCapacity(Matrix &weights, double capacity) {
  if (capacity < 0.0) {
    return;
  }
  for (int col = 0; col < weights.cols(); ++col) {
    double sum = weights.col(col).sum();
    if (sum > 0.0) {
      weights.col(col) *= capacity / sum;
    }
  }
}

void validatePoseBlock(const Matrix &pose, unsigned dimension) {
  if (pose.rows() != static_cast<int>(dimension) ||
      pose.cols() != static_cast<int>(dimension + 1)) {
    throw std::invalid_argument(
        "ObjectCopyConsistencyTerm.targetPose has wrong dimensions");
  }
  if (!pose.allFinite()) {
    throw std::invalid_argument(
        "ObjectCopyConsistencyTerm.targetPose must be finite");
  }
}

void validateLiftedPoseBlock(const Matrix &pose, int rank,
                             unsigned dimension) {
  if (rank < static_cast<int>(dimension) ||
      pose.rows() != rank ||
      pose.cols() != static_cast<int>(dimension + 1)) {
    throw std::invalid_argument(
        "ObjectCopyConsistencyTerm.targetPose has wrong lifted dimensions");
  }
  if (!pose.allFinite()) {
    throw std::invalid_argument(
        "ObjectCopyConsistencyTerm.targetPose must be finite");
  }
}

Matrix yawSymmetryRotation(unsigned dimension, std::size_t branch,
                           std::size_t order) {
  Matrix rotation = Matrix::Identity(dimension, dimension);
  if (order <= 1 || dimension < 2) {
    return rotation;
  }
  const double theta =
      2.0 * M_PI * static_cast<double>(branch % order) /
      static_cast<double>(order);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  rotation(0, 0) = c;
  rotation(0, 1) = -s;
  rotation(1, 0) = s;
  rotation(1, 1) = c;
  return rotation;
}

Matrix applyYawSymmetryBranch(const Matrix &pose, unsigned dimension,
                              std::size_t branch, std::size_t order) {
  Matrix candidate = pose;
  candidate.leftCols(static_cast<int>(dimension)).noalias() =
      pose.leftCols(static_cast<int>(dimension)) *
      yawSymmetryRotation(dimension, branch, order);
  return candidate;
}

Matrix closestYawSymmetryPose(const Matrix &reference, const Matrix &target,
                              unsigned dimension, std::size_t order) {
  if (order <= 1) {
    return target;
  }
  Matrix best = target;
  double bestCost = std::numeric_limits<double>::infinity();
  for (std::size_t branch = 0; branch < order; ++branch) {
    const Matrix candidate =
        applyYawSymmetryBranch(target, dimension, branch, order);
    const double cost = (reference - candidate).squaredNorm();
    if (cost < bestCost) {
      bestCost = cost;
      best = candidate;
    }
  }
  return best;
}

}  // namespace

TrackEmpiricalInformationEstimate estimateTrackEmpiricalInformation(
    const std::vector<Vector> &centers,
    const std::vector<double> &translationInformation) {
  TrackEmpiricalInformationEstimate result;
  result.sampleCount = centers.size();
  result.effectiveSampleSize = static_cast<double>(centers.size());
  if (centers.size() < 2 || centers.size() != translationInformation.size()) {
    return result;
  }

  const Eigen::Index dimension = centers.front().size();
  if (dimension <= 0) {
    return result;
  }
  Vector weightedMean = Vector::Zero(dimension);
  double informationSum = 0.0;
  double uncenteredEnergy = 0.0;
  for (std::size_t index = 0; index < centers.size(); ++index) {
    const double information = translationInformation[index];
    if (centers[index].size() != dimension || !centers[index].allFinite() ||
        !std::isfinite(information) || information <= 0.0) {
      return result;
    }
    weightedMean.noalias() += information * centers[index];
    informationSum += information;
    uncenteredEnergy += information * centers[index].squaredNorm();
  }
  if (!std::isfinite(informationSum) || informationSum <= 0.0 ||
      !std::isfinite(uncenteredEnergy)) {
    return result;
  }
  weightedMean /= informationSum;

  std::vector<Vector> whitenedResiduals;
  whitenedResiduals.reserve(centers.size());
  double residualEnergy = 0.0;
  for (std::size_t index = 0; index < centers.size(); ++index) {
    Vector residual = std::sqrt(translationInformation[index]) *
                      (centers[index] - weightedMean);
    residualEnergy += residual.squaredNorm();
    whitenedResiduals.push_back(std::move(residual));
  }
  const double zeroEnergyTolerance =
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, uncenteredEnergy);
  if (!std::isfinite(residualEnergy) ||
      residualEnergy <= zeroEnergyTolerance) {
    return result;
  }

  const double sampleCount = static_cast<double>(centers.size());
  const double dispersionDenominator =
      static_cast<double>(dimension) * (sampleCount - 1.0);
  result.overdispersion =
      std::max(1.0, residualEnergy / dispersionDenominator);

  const auto lagCorrelation = [&](std::size_t lag) {
    double numerator = 0.0;
    for (std::size_t index = 0; index + lag < centers.size(); ++index) {
      numerator += whitenedResiduals[index].dot(
          whitenedResiduals[index + lag]);
    }
    return numerator / residualEnergy;
  };

  double positiveCorrelationSum = 0.0;
  for (std::size_t firstLag = 1; firstLag + 1 < centers.size();
       firstLag += 2) {
    const double pairSum =
        lagCorrelation(firstLag) + lagCorrelation(firstLag + 1);
    if (!std::isfinite(pairSum) || pairSum <= 0.0) {
      break;
    }
    positiveCorrelationSum += pairSum;
  }
  const double autocorrelationInflation =
      1.0 + 2.0 * positiveCorrelationSum;
  result.effectiveSampleSize = std::clamp(
      sampleCount / autocorrelationInflation, 1.0, sampleCount);
  result.effectiveSampleRatio = result.effectiveSampleSize / sampleCount;
  result.informationScale =
      result.effectiveSampleRatio / result.overdispersion;
  result.empirical = std::isfinite(result.informationScale) &&
                     result.informationScale > 0.0;
  if (!result.empirical) {
    result.informationScale = 1.0;
    result.overdispersion = 1.0;
    result.effectiveSampleSize = sampleCount;
    result.effectiveSampleRatio = 1.0;
  }
  return result;
}

double AssociationCostTerms::weightedCost(
    const AssociationCostWeights &weights) const {
  return weights.semantic * semantic + weights.shape * shape +
         weights.descriptor * descriptor + weights.geometry * geometry +
         weights.temporal * temporal;
}

TransformAssociationVerificationResult verifyTransformAwareAssociation(
    std::size_t numLocalObjects, std::size_t numTargetObjects,
    const std::vector<TransformAssociationCandidate> &candidates,
    const TransformAssociationVerificationOptions &options) {
  if (options.dimension < 2) {
    throw std::invalid_argument(
        "TransformAssociationVerificationOptions.dimension must be >= 2");
  }
  if (!std::isfinite(options.maxResidual) || options.maxResidual <= 0.0) {
    throw std::invalid_argument(
        "TransformAssociationVerificationOptions.maxResidual must be finite "
        "and positive");
  }
  if (!std::isfinite(options.minScore)) {
    throw std::invalid_argument(
        "TransformAssociationVerificationOptions.minScore must be finite");
  }
  if (!std::isfinite(options.weightScale) || options.weightScale < 0.0) {
    throw std::invalid_argument(
        "TransformAssociationVerificationOptions.weightScale must be finite "
        "and nonnegative");
  }

  TransformAssociationVerificationResult result;
  result.rmse = std::numeric_limits<double>::infinity();
  result.weights = Matrix::Zero(numLocalObjects, numTargetObjects);
  result.transform = Matrix::Zero(options.dimension, options.dimension + 1);
  result.transform.leftCols(static_cast<int>(options.dimension)).setIdentity();

  std::vector<std::size_t> validIndices;
  validIndices.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (validTransformCandidate(candidates[index], numLocalObjects,
                                numTargetObjects, options)) {
      validIndices.push_back(index);
    }
  }
  const std::size_t seedSize =
      std::max<std::size_t>(2, static_cast<std::size_t>(options.dimension));
  const std::size_t minInliers =
      std::max<std::size_t>(seedSize, options.minInliers);
  if (validIndices.size() < minInliers) {
    return result;
  }

  std::vector<std::size_t> seedIndices = validIndices;
  std::sort(seedIndices.begin(), seedIndices.end(),
            [&](std::size_t a, std::size_t b) {
              const TransformAssociationCandidate &ca = candidates[a];
              const TransformAssociationCandidate &cb = candidates[b];
              if (ca.score != cb.score) {
                return ca.score > cb.score;
              }
              if (ca.localObjectId != cb.localObjectId) {
                return ca.localObjectId < cb.localObjectId;
              }
              return ca.targetObjectId < cb.targetObjectId;
            });
  if (options.maxSeedCandidates > 0 &&
      seedIndices.size() > options.maxSeedCandidates) {
    seedIndices.resize(options.maxSeedCandidates);
  }
  if (seedSize >= 3 && seedIndices.size() > 30) {
    std::vector<std::size_t> compatibility(seedIndices.size(), 0u);
    std::vector<std::size_t> pair(2, 0u);
    for (std::size_t first = 0; first + 1 < seedIndices.size(); ++first) {
      const TransformAssociationCandidate &firstCandidate =
          candidates[seedIndices[first]];
      for (std::size_t second = first + 1; second < seedIndices.size();
           ++second) {
        const TransformAssociationCandidate &secondCandidate =
            candidates[seedIndices[second]];
        if (firstCandidate.localObjectId == secondCandidate.localObjectId ||
            firstCandidate.targetObjectId == secondCandidate.targetObjectId) {
          continue;
        }
        pair[0] = seedIndices[first];
        pair[1] = seedIndices[second];
        if (rigidSeedPairwiseDistancesConsistent(candidates, pair, options)) {
          ++compatibility[first];
          ++compatibility[second];
        }
      }
    }
    std::vector<std::size_t> order(seedIndices.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                               std::size_t rhs) {
      if (compatibility[lhs] != compatibility[rhs]) {
        return compatibility[lhs] > compatibility[rhs];
      }
      const TransformAssociationCandidate &a = candidates[seedIndices[lhs]];
      const TransformAssociationCandidate &b = candidates[seedIndices[rhs]];
      if (a.score != b.score) {
        return a.score > b.score;
      }
      if (a.localObjectId != b.localObjectId) {
        return a.localObjectId < b.localObjectId;
      }
      return a.targetObjectId < b.targetObjectId;
    });
    std::vector<std::size_t> ranked;
    ranked.reserve(seedIndices.size());
    for (std::size_t index : order) {
      ranked.push_back(seedIndices[index]);
    }
    seedIndices = std::move(ranked);
    seedIndices.resize(30);
  }

  std::vector<TransformInlierEntry> bestInliers;
  Matrix bestTransform = result.transform;
  double bestWeight = -1.0;
  double bestSquaredResidual = std::numeric_limits<double>::infinity();
  std::vector<int> usedLocalInlierStamp(numLocalObjects, 0);
  std::vector<int> usedTargetInlierStamp(numTargetObjects, 0);
  int inlierSelectionStamp = 0;
  auto evaluateSeedHypothesis =
      [&](const std::vector<std::size_t> &selectedSeedIndices) {
        if (!rigidSeedPairwiseDistancesConsistent(candidates,
                                                  selectedSeedIndices,
                                                  options)) {
          ++result.seedHypothesesRejectedByDistance;
          return;
        }
        ++result.seedHypothesesEvaluated;
        Matrix transform;
        if (!estimateRigidTransform(candidates, selectedSeedIndices,
                                    options.dimension, transform)) {
          return;
        }
        const std::vector<TransformInlierEntry> selected =
            selectOneToOneInliers(
                candidates,
                transformInliers(candidates, validIndices, transform,
                                 options),
                numLocalObjects, numTargetObjects, usedLocalInlierStamp,
                usedTargetInlierStamp, inlierSelectionStamp);
        if (selected.size() < minInliers) {
          return;
        }
        double weight = 0.0;
        double squaredResidual = 0.0;
        for (const TransformInlierEntry &entry : selected) {
          weight += entry.weight;
          squaredResidual += entry.residual * entry.residual;
        }
        const bool better =
            selected.size() > bestInliers.size() ||
            (selected.size() == bestInliers.size() && weight > bestWeight) ||
            (selected.size() == bestInliers.size() &&
             weight == bestWeight &&
             squaredResidual < bestSquaredResidual);
        if (better) {
          bestInliers = selected;
          bestTransform = transform;
          bestWeight = weight;
          bestSquaredResidual = squaredResidual;
        }
      };

  if (seedSize == 2) {
    std::vector<std::size_t> selectedSeedIndices(2, 0);
    for (std::size_t first = 0; first + 1 < seedIndices.size(); ++first) {
      const TransformAssociationCandidate &firstCandidate =
          candidates[seedIndices[first]];
      for (std::size_t second = first + 1; second < seedIndices.size();
           ++second) {
        const TransformAssociationCandidate &secondCandidate =
            candidates[seedIndices[second]];
        if (firstCandidate.localObjectId == secondCandidate.localObjectId ||
            firstCandidate.targetObjectId == secondCandidate.targetObjectId) {
          continue;
        }
        selectedSeedIndices[0] = seedIndices[first];
        selectedSeedIndices[1] = seedIndices[second];
        evaluateSeedHypothesis(selectedSeedIndices);
      }
    }
  } else {
    std::vector<std::size_t> combination(seedSize, 0);
    for (std::size_t i = 0; i < seedSize; ++i) {
      combination[i] = i;
    }
    while (seedIndices.size() >= seedSize) {
      std::vector<bool> usedLocal(numLocalObjects, false);
      std::vector<bool> usedTarget(numTargetObjects, false);
      bool repeated = false;
      std::vector<std::size_t> selectedSeedIndices;
      selectedSeedIndices.reserve(seedSize);
      for (std::size_t position : combination) {
        const std::size_t seedIndex = seedIndices[position];
        const TransformAssociationCandidate &candidate = candidates[seedIndex];
        if (usedLocal[candidate.localObjectId] ||
            usedTarget[candidate.targetObjectId]) {
          repeated = true;
          break;
        }
        usedLocal[candidate.localObjectId] = true;
        usedTarget[candidate.targetObjectId] = true;
        selectedSeedIndices.push_back(seedIndex);
      }
      if (!repeated) {
        evaluateSeedHypothesis(selectedSeedIndices);
      }

      bool advanced = false;
      for (std::size_t offset = 0; offset < seedSize; ++offset) {
        const std::size_t pos = seedSize - 1 - offset;
        const std::size_t maxValue = seedIndices.size() - seedSize + pos;
        if (combination[pos] < maxValue) {
          ++combination[pos];
          for (std::size_t j = pos + 1; j < seedSize; ++j) {
            combination[j] = combination[j - 1] + 1;
          }
          advanced = true;
          break;
        }
      }
      if (!advanced) {
        break;
      }
    }
  }
  if (bestInliers.size() < minInliers) {
    return result;
  }

  std::vector<std::size_t> refinedIndices;
  refinedIndices.reserve(bestInliers.size());
  for (const TransformInlierEntry &entry : bestInliers) {
    refinedIndices.push_back(entry.candidateIndex);
  }
  Matrix refinedTransform;
  if (estimateRigidTransform(candidates, refinedIndices, options.dimension,
                             refinedTransform)) {
    bestTransform = refinedTransform;
    bestInliers = selectOneToOneInliers(
        candidates,
        transformInliers(candidates, validIndices, bestTransform, options),
        numLocalObjects, numTargetObjects, usedLocalInlierStamp,
        usedTargetInlierStamp, inlierSelectionStamp);
  }
  if (bestInliers.size() < minInliers) {
    return result;
  }

  double squaredResidual = 0.0;
  result.accepted = true;
  result.inlierCount = bestInliers.size();
  result.transform = bestTransform;
  result.inliers.reserve(bestInliers.size());
  for (const TransformInlierEntry &entry : bestInliers) {
    const TransformAssociationCandidate &candidate =
        candidates[entry.candidateIndex];
    result.weights(static_cast<int>(candidate.localObjectId),
                   static_cast<int>(candidate.targetObjectId)) =
        entry.weight;
    result.inliers.push_back(candidate);
    squaredResidual += entry.residual * entry.residual;
  }
  result.rmse =
      std::sqrt(squaredResidual / static_cast<double>(bestInliers.size()));
  return result;
}

AssociationCycleConsistencyStats pruneAssociationCycleInconsistencies(
    std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleConsistencyOptions &options) {
  if (!std::isfinite(options.minDirectWeight) ||
      options.minDirectWeight < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleConsistencyOptions.minDirectWeight must be finite "
        "and nonnegative");
  }
  if (!std::isfinite(options.minComposedWeight) ||
      options.minComposedWeight < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleConsistencyOptions.minComposedWeight must be finite "
        "and nonnegative");
  }

  AssociationCycleConsistencyStats stats;
  if (options.minTwoHopSupport == 0) {
    return stats;
  }

  const std::size_t numRobots = associationWeights.size();
  for (const std::vector<Matrix> &row : associationWeights) {
    if (row.size() != numRobots) {
      throw std::invalid_argument(
          "associationWeights must be a square robot-by-robot matrix");
    }
  }

  const std::vector<std::vector<Matrix>> original = associationWeights;
  auto activeWeight = [&](double weight, double threshold) {
    return std::isfinite(weight) && weight >= threshold && weight > 0.0;
  };

  for (std::size_t sourceRobot = 0; sourceRobot < numRobots; ++sourceRobot) {
    for (std::size_t targetRobot = 0; targetRobot < numRobots;
         ++targetRobot) {
      if (sourceRobot == targetRobot) {
        continue;
      }
      Matrix &direct = associationWeights[sourceRobot][targetRobot];
      const Matrix &directOriginal = original[sourceRobot][targetRobot];
      if (direct.rows() != directOriginal.rows() ||
          direct.cols() != directOriginal.cols()) {
        throw std::invalid_argument(
            "associationWeights changed shape during cycle pruning");
      }
      for (int sourceObject = 0; sourceObject < directOriginal.rows();
           ++sourceObject) {
        for (int targetObject = 0; targetObject < directOriginal.cols();
             ++targetObject) {
          const double directWeight = directOriginal(sourceObject, targetObject);
          if (!activeWeight(directWeight, options.minDirectWeight)) {
            continue;
          }
          ++stats.checkedEntries;

          bool hasTwoHopEvidence = false;
          std::size_t support = 0;
          for (std::size_t middleRobot = 0; middleRobot < numRobots;
               ++middleRobot) {
            if (middleRobot == sourceRobot || middleRobot == targetRobot) {
              continue;
            }
            const Matrix &firstHop = original[sourceRobot][middleRobot];
            const Matrix &secondHop = original[middleRobot][targetRobot];
            if (sourceObject >= firstHop.rows() ||
                targetObject >= secondHop.cols()) {
              continue;
            }

            bool middleSupportsDirect = false;
            const int sharedObjects =
                std::min(firstHop.cols(), secondHop.rows());
            for (int middleObject = 0; middleObject < sharedObjects;
                 ++middleObject) {
              const double firstWeight =
                  firstHop(sourceObject, middleObject);
              if (!activeWeight(firstWeight, options.minDirectWeight)) {
                continue;
              }
              for (int composedTarget = 0;
                   composedTarget < secondHop.cols(); ++composedTarget) {
                const double secondWeight =
                    secondHop(middleObject, composedTarget);
                if (!activeWeight(secondWeight, options.minDirectWeight)) {
                  continue;
                }
                const double composedWeight = firstWeight * secondWeight;
                if (!activeWeight(composedWeight,
                                  options.minComposedWeight)) {
                  continue;
                }
                hasTwoHopEvidence = true;
                if (composedTarget == targetObject) {
                  middleSupportsDirect = true;
                }
              }
            }
            if (middleSupportsDirect) {
              ++support;
            }
          }

          const bool keep =
              support >= options.minTwoHopSupport ||
              (!hasTwoHopEvidence && options.keepPairsWithoutTwoHopEvidence);
          if (keep) {
            ++stats.keptEntries;
          } else {
            direct(sourceObject, targetObject) = 0.0;
            ++stats.prunedEntries;
          }
        }
      }
    }
  }
  return stats;
}

AssociationCycleObjectiveResult evaluateAssociationCycleObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.weight must be finite and "
        "nonnegative");
  }
  if (!std::isfinite(options.minEdgeMass) || options.minEdgeMass < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.minEdgeMass must be finite and "
        "nonnegative");
  }

  AssociationCycleObjectiveResult result;
  const std::size_t numRobots = associationWeights.size();
  result.gradient.resize(numRobots);
  result.blockLipschitz.assign(
      numRobots, std::vector<double>(numRobots, 0.0));
  for (std::size_t robot = 0; robot < numRobots; ++robot) {
    if (associationWeights[robot].size() != numRobots) {
      throw std::invalid_argument(
          "associationWeights must be a square robot-by-robot matrix");
    }
    result.gradient[robot].resize(numRobots);
    for (std::size_t neighbor = 0; neighbor < numRobots; ++neighbor) {
      result.gradient[robot][neighbor] =
          Matrix::Zero(associationWeights[robot][neighbor].rows(),
                       associationWeights[robot][neighbor].cols());
    }
  }
  if (options.weight == 0.0) {
    return result;
  }

  auto activeEdge = [&](const Matrix &weights) {
    return weights.size() > 0 && weights.cwiseAbs().sum() >= options.minEdgeMass;
  };
  auto squaredSpectralNormUpperBound = [](const Matrix &matrix) {
    if (matrix.size() == 0) {
      return 0.0;
    }
    const Matrix absolute = matrix.cwiseAbs();
    const double maxRowSum = absolute.rowwise().sum().maxCoeff();
    const double maxColumnSum = absolute.colwise().sum().maxCoeff();
    return maxRowSum * maxColumnSum;
  };

  for (std::size_t sourceRobot = 0; sourceRobot < numRobots; ++sourceRobot) {
    for (std::size_t targetRobot = 0; targetRobot < numRobots;
         ++targetRobot) {
      if (sourceRobot == targetRobot ||
          !activeEdge(associationWeights[sourceRobot][targetRobot])) {
        continue;
      }
      for (std::size_t middleRobot = 0; middleRobot < numRobots;
           ++middleRobot) {
        if (middleRobot == sourceRobot || middleRobot == targetRobot) {
          continue;
        }
        const Matrix &direct =
            associationWeights[sourceRobot][targetRobot];
        const Matrix &firstHop =
            associationWeights[sourceRobot][middleRobot];
        const Matrix &secondHop =
            associationWeights[middleRobot][targetRobot];
        if (!activeEdge(firstHop) || !activeEdge(secondHop)) {
          continue;
        }
        if (direct.rows() != firstHop.rows() ||
            direct.cols() != secondHop.cols() ||
            firstHop.cols() != secondHop.rows()) {
          throw std::invalid_argument(
              "associationWeights cycle matrices have incompatible shapes");
        }

        const Matrix residual = direct - firstHop * secondHop;
        result.cost +=
            0.5 * options.weight * residual.squaredNorm();
        ++result.activeTriplets;
        result.gradient[sourceRobot][targetRobot] +=
            options.weight * residual;
        result.gradient[sourceRobot][middleRobot] +=
            -options.weight * residual * secondHop.transpose();
        result.gradient[middleRobot][targetRobot] +=
            -options.weight * firstHop.transpose() * residual;
        result.blockLipschitz[sourceRobot][targetRobot] += options.weight;
        result.blockLipschitz[sourceRobot][middleRobot] +=
            options.weight *
            squaredSpectralNormUpperBound(secondHop);
        result.blockLipschitz[middleRobot][targetRobot] +=
            options.weight * squaredSpectralNormUpperBound(firstHop);
      }
    }
  }
  return result;
}

AssociationCycleObjectiveResult evaluateAssociationPartialCycleAffinityObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.weight must be finite and "
        "nonnegative");
  }
  if (!std::isfinite(options.minEdgeMass) || options.minEdgeMass < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.minEdgeMass must be finite and "
        "nonnegative");
  }

  AssociationCycleObjectiveResult result;
  const std::size_t numRobots = associationWeights.size();
  result.gradient.resize(numRobots);
  result.blockLipschitz.assign(
      numRobots, std::vector<double>(numRobots, 0.0));
  for (std::size_t robot = 0; robot < numRobots; ++robot) {
    if (associationWeights[robot].size() != numRobots) {
      throw std::invalid_argument(
          "associationWeights must be a square robot-by-robot matrix");
    }
    result.gradient[robot].resize(numRobots);
    for (std::size_t neighbor = 0; neighbor < numRobots; ++neighbor) {
      result.gradient[robot][neighbor] =
          Matrix::Zero(associationWeights[robot][neighbor].rows(),
                       associationWeights[robot][neighbor].cols());
    }
  }
  if (options.weight == 0.0) {
    return result;
  }

  auto activeEdge = [&](const Matrix &weights) {
    return weights.size() > 0 &&
           weights.cwiseAbs().sum() >= options.minEdgeMass;
  };
  auto addSharedGradient = [&](std::size_t first, std::size_t second,
                               const Matrix &gradient) {
    if (result.gradient[first][second].rows() == gradient.rows() &&
        result.gradient[first][second].cols() == gradient.cols()) {
      result.gradient[first][second] += gradient;
    }
    if (result.gradient[second][first].rows() == gradient.cols() &&
        result.gradient[second][first].cols() == gradient.rows()) {
      result.gradient[second][first] += gradient.transpose();
    }
  };

  for (std::size_t first = 0; first < numRobots; ++first) {
    for (std::size_t middle = first + 1; middle < numRobots; ++middle) {
      for (std::size_t last = middle + 1; last < numRobots; ++last) {
        const Matrix &firstToMiddle = associationWeights[first][middle];
        const Matrix &middleToLast = associationWeights[middle][last];
        const Matrix &firstToLast = associationWeights[first][last];
        if (!activeEdge(firstToMiddle) || !activeEdge(middleToLast) ||
            !activeEdge(firstToLast)) {
          continue;
        }
        if (firstToMiddle.cols() != middleToLast.rows() ||
            firstToMiddle.rows() != firstToLast.rows() ||
            middleToLast.cols() != firstToLast.cols()) {
          throw std::invalid_argument(
              "associationWeights partial-cycle matrices have incompatible "
              "shapes");
        }

        const Matrix composition = firstToMiddle * middleToLast;
        result.cost -= options.weight *
                       (firstToLast.array() * composition.array()).sum();
        ++result.activeTriplets;
        addSharedGradient(first, last, -options.weight * composition);
        addSharedGradient(
            first, middle,
            -options.weight * firstToLast * middleToLast.transpose());
        addSharedGradient(
            middle, last,
            -options.weight * firstToMiddle.transpose() * firstToLast);
      }
    }
  }
  return result;
}

AssociationCycleObjectiveResult evaluateAssociationPartialCycleClosureObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.weight must be finite and "
        "nonnegative");
  }
  if (!std::isfinite(options.minEdgeMass) || options.minEdgeMass < 0.0) {
    throw std::invalid_argument(
        "AssociationCycleObjectiveOptions.minEdgeMass must be finite and "
        "nonnegative");
  }

  AssociationCycleObjectiveResult result;
  const std::size_t numRobots = associationWeights.size();
  result.gradient.resize(numRobots);
  result.blockLipschitz.assign(
      numRobots, std::vector<double>(numRobots, 0.0));
  for (std::size_t robot = 0; robot < numRobots; ++robot) {
    if (associationWeights[robot].size() != numRobots) {
      throw std::invalid_argument(
          "associationWeights must be a square robot-by-robot matrix");
    }
    result.gradient[robot].resize(numRobots);
    for (std::size_t neighbor = 0; neighbor < numRobots; ++neighbor) {
      result.gradient[robot][neighbor] =
          Matrix::Zero(associationWeights[robot][neighbor].rows(),
                       associationWeights[robot][neighbor].cols());
    }
  }
  if (options.weight == 0.0) {
    return result;
  }

  auto activeEdge = [&](const Matrix &weights) {
    return weights.size() > 0 &&
           weights.cwiseAbs().sum() >= options.minEdgeMass;
  };
  auto addSharedGradient = [&](std::size_t first, std::size_t second,
                               const Matrix &gradient) {
    if (result.gradient[first][second].rows() == gradient.rows() &&
        result.gradient[first][second].cols() == gradient.cols()) {
      result.gradient[first][second] += gradient;
    }
    if (result.gradient[second][first].rows() == gradient.cols() &&
        result.gradient[second][first].cols() == gradient.rows()) {
      result.gradient[second][first] += gradient.transpose();
    }
  };

  for (std::size_t first = 0; first < numRobots; ++first) {
    for (std::size_t middle = first + 1; middle < numRobots; ++middle) {
      for (std::size_t last = middle + 1; last < numRobots; ++last) {
        const Matrix &firstToMiddle = associationWeights[first][middle];
        const Matrix &middleToLast = associationWeights[middle][last];
        const Matrix &firstToLast = associationWeights[first][last];
        if (!activeEdge(firstToMiddle) || !activeEdge(middleToLast) ||
            !activeEdge(firstToLast)) {
          continue;
        }
        if (firstToMiddle.cols() != middleToLast.rows() ||
            firstToMiddle.rows() != firstToLast.rows() ||
            middleToLast.cols() != firstToLast.cols()) {
          throw std::invalid_argument(
              "associationWeights partial-cycle matrices have incompatible "
              "shapes");
        }

        const Matrix closeFirstToLast =
            (firstToMiddle * middleToLast - firstToLast).cwiseMax(0.0);
        const Matrix closeFirstToMiddle =
            (firstToLast * middleToLast.transpose() - firstToMiddle)
                .cwiseMax(0.0);
        const Matrix closeMiddleToLast =
            (firstToMiddle.transpose() * firstToLast - middleToLast)
                .cwiseMax(0.0);
        result.cost += 0.5 * options.weight *
                       (closeFirstToLast.squaredNorm() +
                        closeFirstToMiddle.squaredNorm() +
                        closeMiddleToLast.squaredNorm());
        ++result.activeTriplets;

        Matrix firstToMiddleGradient =
            closeFirstToLast * middleToLast.transpose() -
            closeFirstToMiddle +
            firstToLast * closeMiddleToLast.transpose();
        Matrix middleToLastGradient =
            firstToMiddle.transpose() * closeFirstToLast +
            closeFirstToMiddle.transpose() * firstToLast -
            closeMiddleToLast;
        Matrix firstToLastGradient =
            -closeFirstToLast + closeFirstToMiddle * middleToLast +
            firstToMiddle * closeMiddleToLast;
        addSharedGradient(first, middle,
                          options.weight * firstToMiddleGradient);
        addSharedGradient(middle, last,
                          options.weight * middleToLastGradient);
        addSharedGradient(first, last,
                          options.weight * firstToLastGradient);
      }
    }
  }
  return result;
}

AssociationRelationalGeometryResult evaluateAssociationRelationalGeometryObjective(
    const Matrix &sourceDistances, const Matrix &targetDistances,
    const Matrix &assignment,
    const AssociationRelationalGeometryOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationRelationalGeometryOptions.weight must be finite and "
        "nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationRelationalGeometryOptions.distanceScale must be finite "
        "and nonnegative");
  }
  if (sourceDistances.rows() != sourceDistances.cols() ||
      targetDistances.rows() != targetDistances.cols()) {
    throw std::invalid_argument(
        "relational geometry distance matrices must be square");
  }
  if (assignment.rows() != sourceDistances.rows() ||
      assignment.cols() != targetDistances.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match relational geometry distances");
  }

  AssociationRelationalGeometryResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!sourceDistances.allFinite() || !targetDistances.allFinite() ||
      !assignment.allFinite()) {
    throw std::invalid_argument(
        "relational geometry objective inputs must be finite");
  }

  const Vector rowMass = assignment.rowwise().sum();
  const Vector colMass = assignment.colwise().sum().transpose();

  double scale = options.distanceScale;
  if (scale == 0.0) {
    const Vector positiveRowMass = rowMass.cwiseMax(0.0);
    const Vector positiveColMass = colMass.cwiseMax(0.0);
    const Matrix sourcePairMass =
        positiveRowMass * positiveRowMass.transpose();
    const Matrix targetPairMass =
        positiveColMass * positiveColMass.transpose();
    const double pairMass = sourcePairMass.sum() + targetPairMass.sum();
    if (pairMass > 1e-12) {
      const double weightedSquaredNorm =
          sourcePairMass.cwiseProduct(sourceDistances.array().square().matrix())
              .sum() +
          targetPairMass.cwiseProduct(targetDistances.array().square().matrix())
              .sum();
      scale = std::sqrt(std::max(0.0, weightedSquaredNorm / pairMass));
    } else {
      const double squaredNorm = sourceDistances.squaredNorm() +
                                 targetDistances.squaredNorm();
      const double count = static_cast<double>(sourceDistances.size() +
                                               targetDistances.size());
      scale = count > 0.0 ? std::sqrt(squaredNorm / count) : 1.0;
    }
    if (!std::isfinite(scale) || scale <= 1e-12) {
      scale = 1.0;
    }
  }

  const Matrix source = sourceDistances / scale;
  const Matrix target = targetDistances / scale;

  const Matrix sourceSquared = source.array().square().matrix();
  const Matrix targetSquared = target.array().square().matrix();
  const double sourceTerm = rowMass.dot(sourceSquared * rowMass);
  const double targetTerm = colMass.dot(targetSquared * colMass);
  const Matrix crossMatrix = source * assignment * target.transpose();
  const double crossTerm = assignment.cwiseProduct(crossMatrix).sum();
  const double rawCost =
      0.5 * options.weight * (sourceTerm + targetTerm - 2.0 * crossTerm);

  const Vector sourceGradient = sourceSquared * rowMass;
  const Vector targetGradient = targetSquared * colMass;
  const Matrix rawGradient =
      sourceGradient.replicate(1, assignment.cols()) +
      targetGradient.transpose().replicate(assignment.rows(), 1) -
      2.0 * crossMatrix;
  const double assignmentMass = assignment.sum();
  if (!std::isfinite(assignmentMass) || assignmentMass <= 1e-12) {
    return result;
  }
  if (options.normalizeBySquaredAssignmentMass) {
    const double squaredMass = assignmentMass * assignmentMass;
    result.cost = rawCost / squaredMass;
    result.gradient =
        (options.weight / squaredMass) * rawGradient -
        (2.0 * rawCost / (squaredMass * assignmentMass)) *
            Matrix::Ones(assignment.rows(), assignment.cols());
  } else {
    result.cost = rawCost / assignmentMass;
    result.gradient =
        (options.weight / assignmentMass) * rawGradient -
        (rawCost / (assignmentMass * assignmentMass)) *
            Matrix::Ones(assignment.rows(), assignment.cols());
  }
  return result;
}

AssociationRelationalNullEvidenceResult
evaluateAssociationRelationalNullEvidence(
    const Matrix &sourceDistances, const Matrix &targetDistances,
    const Matrix &assignment, const Matrix &supportCosts,
    const std::vector<std::size_t> &sourceCategories,
    const std::vector<std::size_t> &targetCategories,
    const AssociationRelationalNullEvidenceOptions &options) {
  if (sourceDistances.rows() != sourceDistances.cols() ||
      targetDistances.rows() != targetDistances.cols() ||
      assignment.rows() != sourceDistances.rows() ||
      assignment.cols() != targetDistances.rows() ||
      supportCosts.rows() != assignment.rows() ||
      supportCosts.cols() != assignment.cols() ||
      sourceCategories.size() !=
          static_cast<std::size_t>(assignment.rows()) ||
      targetCategories.size() !=
          static_cast<std::size_t>(assignment.cols())) {
    throw std::invalid_argument(
        "relational null evidence inputs have incompatible shapes");
  }
  if (!sourceDistances.allFinite() || !targetDistances.allFinite() ||
      !assignment.allFinite()) {
    throw std::invalid_argument(
        "relational null evidence distances and assignment must be finite");
  }
  if (!std::isfinite(options.fixedAnchorThreshold) ||
      options.fixedAnchorThreshold < 0.0 ||
      options.fixedMinimumAnchors == 0u ||
      !std::isfinite(options.fixedStudentTDegreesOfFreedom) ||
      options.fixedStudentTDegreesOfFreedom <= 0.0 ||
      options.geometricDimension == 0u) {
    throw std::invalid_argument(
        "relational null evidence options must be finite and positive");
  }

  AssociationRelationalNullEvidenceResult result;
  result.costOffset = Matrix::Zero(assignment.rows(), assignment.cols());
  struct Anchor {
    int row{0};
    int col{0};
    double weight{0.0};
  };
  std::vector<Anchor> anchors;
  if (options.calibration ==
      AssociationRelationalNullCalibration::FixedStudentT) {
    std::vector<Anchor> candidates;
    for (int row = 0; row < assignment.rows(); ++row) {
      for (int col = 0; col < assignment.cols(); ++col) {
        const double weight = assignment(row, col);
        if (weight > options.fixedAnchorThreshold &&
            std::isfinite(supportCosts(row, col)) &&
            sourceCategories[static_cast<std::size_t>(row)] ==
                targetCategories[static_cast<std::size_t>(col)]) {
          candidates.push_back(Anchor{row, col, weight});
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Anchor &first, const Anchor &second) {
                if (first.weight != second.weight) {
                  return first.weight > second.weight;
                }
                if (first.row != second.row) {
                  return first.row < second.row;
                }
                return first.col < second.col;
              });
    std::vector<bool> usedRows(static_cast<std::size_t>(assignment.rows()),
                               false);
    std::vector<bool> usedCols(static_cast<std::size_t>(assignment.cols()),
                               false);
    for (const Anchor &candidate : candidates) {
      if (usedRows[static_cast<std::size_t>(candidate.row)] ||
          usedCols[static_cast<std::size_t>(candidate.col)]) {
        continue;
      }
      usedRows[static_cast<std::size_t>(candidate.row)] = true;
      usedCols[static_cast<std::size_t>(candidate.col)] = true;
      anchors.push_back(candidate);
    }
  } else {
    const Vector rowMass = assignment.cwiseMax(0.0).rowwise().sum();
    const Vector colMass = assignment.cwiseMax(0.0).colwise().sum().transpose();
    std::vector<int> rowBest(static_cast<std::size_t>(assignment.rows()), -1);
    std::vector<int> colBest(static_cast<std::size_t>(assignment.cols()), -1);
    for (int row = 0; row < assignment.rows(); ++row) {
      for (int col = 0; col < assignment.cols(); ++col) {
        if (!std::isfinite(supportCosts(row, col)) ||
            sourceCategories[static_cast<std::size_t>(row)] !=
                targetCategories[static_cast<std::size_t>(col)]) {
          continue;
        }
        const int previousCol = rowBest[static_cast<std::size_t>(row)];
        if (previousCol < 0 ||
            assignment(row, col) > assignment(row, previousCol)) {
          rowBest[static_cast<std::size_t>(row)] = col;
        }
        const int previousRow = colBest[static_cast<std::size_t>(col)];
        if (previousRow < 0 ||
            assignment(row, col) > assignment(previousRow, col)) {
          colBest[static_cast<std::size_t>(col)] = row;
        }
      }
    }
    for (int row = 0; row < assignment.rows(); ++row) {
      const int col = rowBest[static_cast<std::size_t>(row)];
      if (col < 0 || colBest[static_cast<std::size_t>(col)] != row) {
        continue;
      }
      const double weight = assignment(row, col);
      const double rowUnmatched = std::max(0.0, 1.0 - rowMass(row));
      const double colUnmatched = std::max(0.0, 1.0 - colMass(col));
      if (weight > rowUnmatched && weight > colUnmatched) {
        anchors.push_back(Anchor{row, col, weight});
      }
    }
  }
  result.anchorCount = anchors.size();
  const std::size_t minimumAnchors =
      options.calibration ==
              AssociationRelationalNullCalibration::FixedStudentT
          ? options.fixedMinimumAnchors
          : options.geometricDimension + 1u;
  if (anchors.size() < minimumAnchors) {
    return result;
  }

  auto median = [](std::vector<double> values) {
    if (values.empty()) {
      return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2u;
    if (values.size() % 2u == 0u) {
      return 0.5 * (values[middle - 1u] + values[middle]);
    }
    return values[middle];
  };

  std::vector<double> anchorResiduals;
  anchorResiduals.reserve(anchors.size() * (anchors.size() - 1u) / 2u);
  for (std::size_t first = 0; first < anchors.size(); ++first) {
    for (std::size_t second = first + 1u; second < anchors.size(); ++second) {
      anchorResiduals.push_back(std::abs(
          sourceDistances(anchors[first].row, anchors[second].row) -
          targetDistances(anchors[first].col, anchors[second].col)));
    }
  }
  result.distanceScale = 1.4826 * median(anchorResiduals);
  if (!std::isfinite(result.distanceScale) ||
      result.distanceScale <= 1e-12) {
    return result;
  }

  auto relationalScore = [&](int row, int col) {
    double score = 0.0;
    std::size_t count = 0u;
    for (const Anchor &anchor : anchors) {
      if (anchor.row == row || anchor.col == col) {
        continue;
      }
      const double residual = std::abs(
          sourceDistances(row, anchor.row) -
          targetDistances(col, anchor.col));
      const double normalized = residual / result.distanceScale;
      score += std::log1p(normalized * normalized);
      ++count;
    }
    return count > 0u ? score / static_cast<double>(count)
                      : std::numeric_limits<double>::infinity();
  };

  std::vector<double> logNullScores;
  constexpr double kScoreEpsilon = 1e-12;
  for (std::size_t first = 0; first < anchors.size(); ++first) {
    const Anchor &source = anchors[first];
    for (std::size_t second = 0; second < anchors.size(); ++second) {
      const Anchor &target = anchors[second];
      if (first == second ||
          sourceCategories[static_cast<std::size_t>(source.row)] !=
              targetCategories[static_cast<std::size_t>(target.col)] ||
          !std::isfinite(supportCosts(source.row, target.col))) {
        continue;
      }
      const double score = relationalScore(source.row, target.col);
      if (std::isfinite(score) && score >= 0.0) {
        logNullScores.push_back(std::log(score + kScoreEpsilon));
      }
    }
  }
  result.nullSampleCount = logNullScores.size();
  if (logNullScores.empty() ||
      (options.calibration ==
           AssociationRelationalNullCalibration::FixedStudentT &&
       logNullScores.size() < options.fixedMinimumAnchors)) {
    return result;
  }
  if (options.calibration ==
      AssociationRelationalNullCalibration::AdaptiveEmpiricalRank) {
    std::vector<double> nullScores;
    nullScores.reserve(logNullScores.size());
    for (const double logScore : logNullScores) {
      nullScores.push_back(std::exp(logScore));
    }
    std::sort(nullScores.begin(), nullScores.end());
    const double denominator =
        static_cast<double>(nullScores.size() + 1u);
    for (std::size_t rank = 1u; rank <= nullScores.size() + 1u; ++rank) {
      result.centeredNullLogMean +=
          std::log(static_cast<double>(rank) / denominator);
    }
    result.centeredNullLogMean /= denominator;
    result.empiricalRankResolution = 1.0 / denominator;
    for (int row = 0; row < assignment.rows(); ++row) {
      for (int col = 0; col < assignment.cols(); ++col) {
        if (!std::isfinite(supportCosts(row, col)) ||
            sourceCategories[static_cast<std::size_t>(row)] !=
                targetCategories[static_cast<std::size_t>(col)]) {
          continue;
        }
        const double score = relationalScore(row, col);
        if (!std::isfinite(score) || score < 0.0) {
          continue;
        }
        const std::size_t lowerTailCount = static_cast<std::size_t>(
            std::upper_bound(nullScores.begin(), nullScores.end(), score) -
            nullScores.begin());
        const double probability =
            static_cast<double>(lowerTailCount + 1u) / denominator;
        result.costOffset(row, col) =
            std::log(probability) - result.centeredNullLogMean;
      }
    }
    result.valid = true;
    return result;
  }
  result.logNullMedian = median(logNullScores);
  std::vector<double> deviations;
  deviations.reserve(logNullScores.size());
  for (const double value : logNullScores) {
    deviations.push_back(std::abs(value - result.logNullMedian));
  }
  result.logNullScale = 1.4826 * median(deviations);
  if (!std::isfinite(result.logNullScale) ||
      result.logNullScale <= 1e-12) {
    return result;
  }

  const boost::math::students_t_distribution<double> distribution(
      options.fixedStudentTDegreesOfFreedom);
  constexpr double kMinimumProbability = 1e-12;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      if (!std::isfinite(supportCosts(row, col)) ||
          sourceCategories[static_cast<std::size_t>(row)] !=
              targetCategories[static_cast<std::size_t>(col)]) {
        continue;
      }
      const double score = relationalScore(row, col);
      if (!std::isfinite(score) || score < 0.0) {
        continue;
      }
      const double standardized =
          (std::log(score + kScoreEpsilon) - result.logNullMedian) /
          result.logNullScale;
      const double probability = std::clamp(
          boost::math::cdf(distribution, standardized), kMinimumProbability,
          1.0);
      result.costOffset(row, col) = std::log(probability);
    }
  }
  result.valid = true;
  return result;
}

double partialAssignmentSupportCoupling(
    double rowMass, double columnMass, double entryMass,
    double rowCapacity, double columnCapacity) {
  if (!std::isfinite(rowMass) || !std::isfinite(columnMass) ||
      !std::isfinite(entryMass) ||
      !std::isfinite(rowCapacity) || rowCapacity <= 0.0 ||
      !std::isfinite(columnCapacity) || columnCapacity <= 0.0) {
    return 0.0;
  }
  const double support = std::max(0.0, entryMass);
  const double rowAvailability = std::clamp(
      (rowCapacity - std::max(0.0, rowMass) + support) / rowCapacity,
      0.0, 1.0);
  const double columnAvailability = std::clamp(
      (columnCapacity - std::max(0.0, columnMass) + support) /
          columnCapacity,
      0.0, 1.0);
  return std::sqrt(rowAvailability * columnAvailability);
}

AssociationRotoTranslationAlignmentResult
evaluateAssociationRotoTranslationAlignmentObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const AssociationRotoTranslationAlignmentOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationAlignmentOptions.weight must be finite "
        "and nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationAlignmentOptions.distanceScale must be "
        "finite and nonnegative");
  }
  if (sourcePoints.cols() != targetPoints.cols()) {
    throw std::invalid_argument(
        "roto-translation alignment point dimensions must match");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match alignment point counts");
  }

  const int dimension = sourcePoints.cols();
  AssociationRotoTranslationAlignmentResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  result.rotation = Matrix::Identity(dimension, dimension);
  result.translation = Vector::Zero(dimension);
  if (options.weight == 0.0 || assignment.size() == 0 ||
      sourcePoints.rows() == 0 || targetPoints.rows() == 0 ||
      dimension == 0) {
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !assignment.allFinite()) {
    throw std::invalid_argument(
        "roto-translation alignment objective inputs must be finite");
  }

  const double mass = assignment.sum();
  if (!std::isfinite(mass) || std::abs(mass) <= 1e-12) {
    return result;
  }
  result.mass = mass;

  Vector sourceMean = Vector::Zero(dimension);
  Vector targetMean = Vector::Zero(dimension);
  for (int sourceIndex = 0; sourceIndex < assignment.rows(); ++sourceIndex) {
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      const double weight = assignment(sourceIndex, targetIndex);
      sourceMean.noalias() +=
          weight * sourcePoints.row(sourceIndex).transpose();
      targetMean.noalias() +=
          weight * targetPoints.row(targetIndex).transpose();
    }
  }
  sourceMean /= mass;
  targetMean /= mass;

  Matrix covariance = Matrix::Zero(dimension, dimension);
  double weightedScatter = 0.0;
  for (int sourceIndex = 0; sourceIndex < assignment.rows(); ++sourceIndex) {
    const Vector sourceCentered =
        sourcePoints.row(sourceIndex).transpose() - sourceMean;
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      const double weight = assignment(sourceIndex, targetIndex);
      const Vector targetCentered =
          targetPoints.row(targetIndex).transpose() - targetMean;
      covariance.noalias() +=
          weight * targetCentered * sourceCentered.transpose();
      weightedScatter +=
          weight * (sourceCentered.squaredNorm() +
                    targetCentered.squaredNorm());
    }
  }

  Eigen::JacobiSVD<Matrix> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix u = svd.matrixU();
  Matrix v = svd.matrixV();
  Matrix rotation = v * u.transpose();
  if (dimension > 1 && rotation.determinant() < 0.0) {
    v.col(dimension - 1) *= -1.0;
    rotation = v * u.transpose();
  }
  result.rotation = rotation;
  result.translation = sourceMean - rotation * targetMean;

  double scale = options.distanceScale;
  if (scale == 0.0) {
    scale = std::sqrt(
        std::max(0.0, weightedScatter / std::abs(mass)));
    if (!std::isfinite(scale) || scale <= 1e-12) {
      scale = 1.0;
    }
  }
  const double scaleSquared = scale * scale;
  for (int sourceIndex = 0; sourceIndex < assignment.rows(); ++sourceIndex) {
    const Vector sourcePoint = sourcePoints.row(sourceIndex).transpose();
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      const Vector alignedTarget =
          rotation * targetPoints.row(targetIndex).transpose() +
          result.translation;
      const double entryCost =
          0.5 * options.weight *
          (sourcePoint - alignedTarget).squaredNorm() / scaleSquared;
      result.gradient(sourceIndex, targetIndex) = entryCost;
      result.cost += assignment(sourceIndex, targetIndex) * entryCost;
    }
  }
  return result;
}

AssociationRotoTranslationEstimate
estimateSparseAssociationRotoTranslationAlignment(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const std::vector<SparseAssignmentWeightEntry> &entries) {
  if (sourcePoints.cols() != targetPoints.cols()) {
    throw std::invalid_argument(
        "sparse roto-translation point dimensions must match");
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite()) {
    throw std::invalid_argument(
        "sparse roto-translation points must be finite");
  }
  const int dimension = sourcePoints.cols();
  AssociationRotoTranslationEstimate result;
  result.rotation = Matrix::Identity(dimension, dimension);
  result.translation = Vector::Zero(dimension);
  if (dimension == 0) {
    return result;
  }

  Vector sourceMean = Vector::Zero(dimension);
  Vector targetMean = Vector::Zero(dimension);
  for (const SparseAssignmentWeightEntry &entry : entries) {
    if (entry.row < 0 || entry.col < 0 ||
        entry.row >= sourcePoints.rows() || entry.col >= targetPoints.rows()) {
      throw std::invalid_argument(
          "sparse roto-translation entry is outside point matrices");
    }
    if (!std::isfinite(entry.value) || entry.value <= 0.0) {
      continue;
    }
    result.mass += entry.value;
    sourceMean.noalias() +=
        entry.value * sourcePoints.row(entry.row).transpose();
    targetMean.noalias() +=
        entry.value * targetPoints.row(entry.col).transpose();
  }
  if (!std::isfinite(result.mass) || result.mass <= 1e-12) {
    result.mass = 0.0;
    return result;
  }
  sourceMean /= result.mass;
  targetMean /= result.mass;

  Matrix covariance = Matrix::Zero(dimension, dimension);
  for (const SparseAssignmentWeightEntry &entry : entries) {
    if (!std::isfinite(entry.value) || entry.value <= 0.0) {
      continue;
    }
    const Vector sourceCentered =
        sourcePoints.row(entry.row).transpose() - sourceMean;
    const Vector targetCentered =
        targetPoints.row(entry.col).transpose() - targetMean;
    covariance.noalias() +=
        entry.value * targetCentered * sourceCentered.transpose();
  }

  Eigen::JacobiSVD<Matrix> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix u = svd.matrixU();
  Matrix v = svd.matrixV();
  Matrix rotation = v * u.transpose();
  if (dimension > 1 && rotation.determinant() < 0.0) {
    v.col(dimension - 1) *= -1.0;
    rotation = v * u.transpose();
  }
  result.rotation = rotation;
  result.translation = sourceMean - rotation * targetMean;
  return result;
}

AssociationRotoTranslationEstimate estimateSparseAssociationPoseAlignment(
    const std::vector<Matrix> &sourcePoses,
    const std::vector<Matrix> &targetPoses,
    const std::vector<SparsePoseAlignmentWeightEntry> &entries) {
  if (sourcePoses.empty() || targetPoses.empty()) {
    return {};
  }
  const int dimension = sourcePoses.front().rows();
  if (dimension <= 0) {
    throw std::invalid_argument(
        "sparse pose alignment dimension must be positive");
  }
  for (const Matrix &pose : sourcePoses) {
    validatePoseBlock(pose, static_cast<unsigned>(dimension));
  }
  for (const Matrix &pose : targetPoses) {
    validatePoseBlock(pose, static_cast<unsigned>(dimension));
  }

  AssociationRotoTranslationEstimate result;
  result.rotation = Matrix::Identity(dimension, dimension);
  result.translation = Vector::Zero(dimension);
  double translationMass = 0.0;
  Vector sourceMean = Vector::Zero(dimension);
  Vector targetMean = Vector::Zero(dimension);
  for (const SparsePoseAlignmentWeightEntry &entry : entries) {
    if (entry.row < 0 || entry.col < 0 ||
        entry.row >= static_cast<int>(sourcePoses.size()) ||
        entry.col >= static_cast<int>(targetPoses.size())) {
      throw std::invalid_argument(
          "sparse pose alignment entry is outside pose arrays");
    }
    if (!std::isfinite(entry.associationWeight) ||
        !std::isfinite(entry.rotationInformation) ||
        !std::isfinite(entry.translationInformation) ||
        entry.associationWeight < 0.0 || entry.rotationInformation < 0.0 ||
        entry.translationInformation < 0.0) {
      throw std::invalid_argument(
          "sparse pose alignment weights must be finite and nonnegative");
    }
    if (entry.associationWeight <= 0.0) {
      continue;
    }
    result.mass += entry.associationWeight;
    const double weight =
        entry.associationWeight * entry.translationInformation;
    translationMass += weight;
    sourceMean.noalias() +=
        weight * sourcePoses[static_cast<std::size_t>(entry.row)].col(
                     dimension);
    targetMean.noalias() +=
        weight * targetPoses[static_cast<std::size_t>(entry.col)].col(
                     dimension);
  }
  if (!std::isfinite(result.mass) || result.mass <= 1e-12 ||
      !std::isfinite(translationMass) || translationMass <= 1e-12) {
    result.mass = 0.0;
    return result;
  }
  sourceMean /= translationMass;
  targetMean /= translationMass;

  Matrix covariance = Matrix::Zero(dimension, dimension);
  for (const SparsePoseAlignmentWeightEntry &entry : entries) {
    if (entry.associationWeight <= 0.0) {
      continue;
    }
    const Matrix &sourcePose =
        sourcePoses[static_cast<std::size_t>(entry.row)];
    const Matrix &targetPose =
        targetPoses[static_cast<std::size_t>(entry.col)];
    const double translationWeight =
        entry.associationWeight * entry.translationInformation;
    if (translationWeight > 0.0) {
      const Vector sourceCentered = sourcePose.col(dimension) - sourceMean;
      const Vector targetCentered = targetPose.col(dimension) - targetMean;
      covariance.noalias() +=
          translationWeight * targetCentered * sourceCentered.transpose();
    }
    const double rotationWeight =
        entry.associationWeight * entry.rotationInformation;
    if (rotationWeight > 0.0) {
      covariance.noalias() +=
          rotationWeight * targetPose.leftCols(dimension) *
          sourcePose.leftCols(dimension).transpose();
    }
  }

  Eigen::JacobiSVD<Matrix> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix u = svd.matrixU();
  Matrix v = svd.matrixV();
  Matrix rotation = v * u.transpose();
  if (dimension > 1 && rotation.determinant() < 0.0) {
    v.col(dimension - 1) *= -1.0;
    rotation = v * u.transpose();
  }
  result.rotation = rotation;
  result.translation = sourceMean - rotation * targetMean;
  return result;
}

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options) {
  return evaluateAssociationRotoTranslationHypothesisMixtureObjective(
      sourcePoints, targetPoints, assignment, Matrix(), rotations,
      translations, options);
}

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &gradientSupport,
    const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions.weight must be "
        "finite and nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions.distanceScale "
        "must be finite and nonnegative");
  }
  if (!std::isfinite(options.softminTemperature) ||
      options.softminTemperature < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions."
        "softminTemperature must be finite and nonnegative");
  }
  if (rotations.size() != translations.size()) {
    throw std::invalid_argument(
        "roto-translation hypothesis rotation/translation counts must match");
  }
  if (sourcePoints.cols() != targetPoints.cols()) {
    throw std::invalid_argument(
        "roto-translation hypothesis point dimensions must match");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match hypothesis point counts");
  }
  const bool useGradientSupport =
      gradientSupport.rows() > 0 || gradientSupport.cols() > 0;
  if (useGradientSupport &&
      (gradientSupport.rows() != assignment.rows() ||
       gradientSupport.cols() != assignment.cols())) {
    throw std::invalid_argument(
        "roto-translation hypothesis gradient support shape must match "
        "assignment");
  }

  const int dimension = sourcePoints.cols();
  AssociationRotoTranslationHypothesisMixtureResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  result.hypothesisWeights =
      Vector::Zero(static_cast<int>(rotations.size()));
  if (options.weight == 0.0 || assignment.size() == 0 ||
      rotations.empty() || sourcePoints.rows() == 0 ||
      targetPoints.rows() == 0 || dimension == 0) {
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !assignment.allFinite()) {
    throw std::invalid_argument(
        "roto-translation hypothesis objective inputs must be finite");
  }

  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    if (rotations[hypothesis].rows() != dimension ||
        rotations[hypothesis].cols() != dimension ||
        translations[hypothesis].size() != dimension) {
      throw std::invalid_argument(
          "roto-translation hypothesis dimensions must match point "
          "dimension");
    }
    if (!rotations[hypothesis].allFinite() ||
        !translations[hypothesis].allFinite()) {
      throw std::invalid_argument(
          "roto-translation hypothesis transforms must be finite");
    }
  }

  double scale = options.distanceScale;
  if (scale == 0.0) {
    Vector sourceMean = Vector::Zero(dimension);
    Vector targetMean = Vector::Zero(dimension);
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      sourceMean.noalias() += sourcePoints.row(row).transpose();
    }
    for (int row = 0; row < targetPoints.rows(); ++row) {
      targetMean.noalias() += targetPoints.row(row).transpose();
    }
    sourceMean /= static_cast<double>(sourcePoints.rows());
    targetMean /= static_cast<double>(targetPoints.rows());
    double scatter = 0.0;
    int count = 0;
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      scatter +=
          (sourcePoints.row(row).transpose() - sourceMean).squaredNorm();
      ++count;
    }
    for (int row = 0; row < targetPoints.rows(); ++row) {
      scatter +=
          (targetPoints.row(row).transpose() - targetMean).squaredNorm();
      ++count;
    }
    scale = count > 0 ? std::sqrt(scatter / static_cast<double>(count))
                      : 1.0;
    if (!std::isfinite(scale) || scale <= 1e-12) {
      scale = 1.0;
    }
  }

  const double scaleSquared = scale * scale;

  struct ActivePair {
    int source{0};
    int target{0};
    double weight{0.0};
  };
  std::vector<ActivePair> activePairs;
  activePairs.reserve(static_cast<std::size_t>(assignment.size()));
  for (int sourceIndex = 0; sourceIndex < assignment.rows();
       ++sourceIndex) {
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      const double weight = assignment(sourceIndex, targetIndex);
      if (std::isfinite(weight) && std::abs(weight) > 1e-12) {
        activePairs.push_back(
            ActivePair{sourceIndex, targetIndex, weight});
      }
    }
  }

  struct SupportPair {
    int source{0};
    int target{0};
  };
  std::vector<SupportPair> supportPairs;
  supportPairs.reserve(useGradientSupport
                           ? static_cast<std::size_t>(gradientSupport.size())
                           : static_cast<std::size_t>(assignment.size()));
  for (int sourceIndex = 0; sourceIndex < assignment.rows();
       ++sourceIndex) {
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      if (useGradientSupport &&
          !std::isfinite(gradientSupport(sourceIndex, targetIndex))) {
        continue;
      }
      supportPairs.push_back(SupportPair{sourceIndex, targetIndex});
    }
  }

  std::vector<Matrix> hypothesisGradients;
  hypothesisGradients.reserve(rotations.size());
  Vector hypothesisCosts =
      Vector::Zero(static_cast<int>(rotations.size()));
  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    Matrix gradient = Matrix::Zero(assignment.rows(), assignment.cols());
    for (const SupportPair &support : supportPairs) {
      const Vector sourcePoint =
          sourcePoints.row(support.source).transpose();
      const Vector alignedTarget =
          rotations[hypothesis] *
              targetPoints.row(support.target).transpose() +
          translations[hypothesis];
      gradient(support.source, support.target) =
          0.5 * options.weight *
          (sourcePoint - alignedTarget).squaredNorm() / scaleSquared;
    }
    for (const ActivePair &active : activePairs) {
      const Vector sourcePoint =
          sourcePoints.row(active.source).transpose();
      const Vector alignedTarget =
          rotations[hypothesis] *
              targetPoints.row(active.target).transpose() +
          translations[hypothesis];
      const double entryCost =
          0.5 * options.weight *
          (sourcePoint - alignedTarget).squaredNorm() / scaleSquared;
      hypothesisCosts(static_cast<int>(hypothesis)) +=
          active.weight * entryCost;
    }
    hypothesisGradients.push_back(std::move(gradient));
  }

  if (options.softminTemperature == 0.0) {
    Eigen::Index minIndex = 0;
    result.cost = hypothesisCosts.minCoeff(&minIndex);
    result.hypothesisWeights(minIndex) = 1.0;
    result.gradient = hypothesisGradients[static_cast<std::size_t>(minIndex)];
    return result;
  }

  const double minCost = hypothesisCosts.minCoeff();
  double expSum = 0.0;
  for (int hypothesis = 0; hypothesis < hypothesisCosts.size();
       ++hypothesis) {
    const double value = std::exp(
        -(hypothesisCosts(hypothesis) - minCost) /
        options.softminTemperature);
    result.hypothesisWeights(hypothesis) = value;
    expSum += value;
  }
  if (!std::isfinite(expSum) || expSum <= 0.0) {
    throw std::runtime_error(
        "roto-translation hypothesis softmin normalization failed");
  }
  result.hypothesisWeights /= expSum;
  result.cost =
      minCost - options.softminTemperature * std::log(expSum) +
      options.softminTemperature *
          std::log(static_cast<double>(rotations.size()));
  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    result.gradient.noalias() +=
        result.hypothesisWeights(static_cast<int>(hypothesis)) *
        hypothesisGradients[hypothesis];
  }
  return result;
}

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const std::vector<SparseAssignmentCostEntry> &supportEntries,
    const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions.weight must be "
        "finite and nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions.distanceScale "
        "must be finite and nonnegative");
  }
  if (!std::isfinite(options.softminTemperature) ||
      options.softminTemperature < 0.0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationHypothesisMixtureOptions."
        "softminTemperature must be finite and nonnegative");
  }
  if (rotations.size() != translations.size()) {
    throw std::invalid_argument(
        "roto-translation hypothesis rotation/translation counts must match");
  }
  if (sourcePoints.cols() != targetPoints.cols()) {
    throw std::invalid_argument(
        "roto-translation hypothesis point dimensions must match");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match hypothesis point counts");
  }

  const int dimension = sourcePoints.cols();
  AssociationRotoTranslationHypothesisMixtureResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  result.hypothesisWeights =
      Vector::Zero(static_cast<int>(rotations.size()));
  if (options.weight == 0.0 || assignment.size() == 0 ||
      rotations.empty() || sourcePoints.rows() == 0 ||
      targetPoints.rows() == 0 || dimension == 0 ||
      supportEntries.empty()) {
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite()) {
    throw std::invalid_argument(
        "sparse roto-translation hypothesis points must be finite");
  }

  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    if (rotations[hypothesis].rows() != dimension ||
        rotations[hypothesis].cols() != dimension ||
        translations[hypothesis].size() != dimension) {
      throw std::invalid_argument(
          "roto-translation hypothesis dimensions must match point "
          "dimension");
    }
    if (!rotations[hypothesis].allFinite() ||
        !translations[hypothesis].allFinite()) {
      throw std::invalid_argument(
          "roto-translation hypothesis transforms must be finite");
    }
  }

  double scale = options.distanceScale;
  if (scale == 0.0) {
    Vector sourceMean = Vector::Zero(dimension);
    Vector targetMean = Vector::Zero(dimension);
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      sourceMean.noalias() += sourcePoints.row(row).transpose();
    }
    for (int row = 0; row < targetPoints.rows(); ++row) {
      targetMean.noalias() += targetPoints.row(row).transpose();
    }
    sourceMean /= static_cast<double>(sourcePoints.rows());
    targetMean /= static_cast<double>(targetPoints.rows());
    double scatter = 0.0;
    int count = 0;
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      scatter +=
          (sourcePoints.row(row).transpose() - sourceMean).squaredNorm();
      ++count;
    }
    for (int row = 0; row < targetPoints.rows(); ++row) {
      scatter +=
          (targetPoints.row(row).transpose() - targetMean).squaredNorm();
      ++count;
    }
    scale = count > 0 ? std::sqrt(scatter / static_cast<double>(count))
                      : 1.0;
    if (!std::isfinite(scale) || scale <= 1e-12) {
      scale = 1.0;
    }
  }

  const double scaleSquared = scale * scale;
  struct Pair {
    int source{0};
    int target{0};
    double weight{0.0};
  };
  std::vector<Pair> supportPairs;
  std::vector<Pair> activePairs;
  supportPairs.reserve(supportEntries.size());
  activePairs.reserve(supportEntries.size());
  for (const SparseAssignmentCostEntry &entry : supportEntries) {
    if (entry.row < 0 || entry.col < 0 ||
        entry.row >= assignment.rows() || entry.col >= assignment.cols()) {
      throw std::invalid_argument(
          "sparse roto-translation support entry is outside matrix shape");
    }
    const double weight = assignment(entry.row, entry.col);
    if (!std::isfinite(weight)) {
      throw std::invalid_argument(
          "sparse roto-translation assignment support values must be finite");
    }
    supportPairs.push_back(Pair{entry.row, entry.col, weight});
    if (std::abs(weight) > 1e-12) {
      activePairs.push_back(Pair{entry.row, entry.col, weight});
    }
  }
  if (activePairs.empty()) {
    return result;
  }

  Vector hypothesisCosts =
      Vector::Zero(static_cast<int>(rotations.size()));
  auto entryCost = [&](std::size_t hypothesis, const Pair &pair) {
    if (dimension == 2) {
      const double r00 = rotations[hypothesis](0, 0);
      const double r01 = rotations[hypothesis](0, 1);
      const double r10 = rotations[hypothesis](1, 0);
      const double r11 = rotations[hypothesis](1, 1);
      const double tx = translations[hypothesis](0);
      const double ty = translations[hypothesis](1);
      const double targetX = targetPoints(pair.target, 0);
      const double targetY = targetPoints(pair.target, 1);
      const double dx =
          sourcePoints(pair.source, 0) -
          (r00 * targetX + r01 * targetY + tx);
      const double dy =
          sourcePoints(pair.source, 1) -
          (r10 * targetX + r11 * targetY + ty);
      return 0.5 * options.weight * (dx * dx + dy * dy) / scaleSquared;
    }
    const Vector sourcePoint =
        sourcePoints.row(pair.source).transpose();
    const Vector alignedTarget =
        rotations[hypothesis] *
            targetPoints.row(pair.target).transpose() +
        translations[hypothesis];
    return 0.5 * options.weight *
           (sourcePoint - alignedTarget).squaredNorm() / scaleSquared;
  };
  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    for (const Pair &active : activePairs) {
      hypothesisCosts(static_cast<int>(hypothesis)) +=
          active.weight * entryCost(hypothesis, active);
    }
  }

  if (options.softminTemperature == 0.0) {
    Eigen::Index minIndex = 0;
    result.cost = hypothesisCosts.minCoeff(&minIndex);
    result.hypothesisWeights(minIndex) = 1.0;
    for (const Pair &support : supportPairs) {
      result.gradient(support.source, support.target) =
          entryCost(static_cast<std::size_t>(minIndex), support);
    }
    return result;
  }

  const double minCost = hypothesisCosts.minCoeff();
  double expSum = 0.0;
  for (int hypothesis = 0; hypothesis < hypothesisCosts.size();
       ++hypothesis) {
    const double value = std::exp(
        -(hypothesisCosts(hypothesis) - minCost) /
        options.softminTemperature);
    result.hypothesisWeights(hypothesis) = value;
    expSum += value;
  }
  if (!std::isfinite(expSum) || expSum <= 0.0) {
    throw std::runtime_error(
        "sparse roto-translation hypothesis softmin normalization failed");
  }
  result.hypothesisWeights /= expSum;
  result.cost =
      minCost - options.softminTemperature * std::log(expSum) +
      options.softminTemperature *
          std::log(static_cast<double>(rotations.size()));
  for (std::size_t hypothesis = 0; hypothesis < rotations.size();
       ++hypothesis) {
    const double weight = result.hypothesisWeights(
        static_cast<int>(hypothesis));
    for (const Pair &support : supportPairs) {
      result.gradient(support.source, support.target) +=
          weight * entryCost(hypothesis, support);
    }
  }
  return result;
}

AssociationPairwiseTransformValidationResult
evaluateAssociationPairwiseTransformValidationObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &rotation,
    const AssociationPairwiseTransformValidationOptions &options) {
  Matrix gradientSupport =
      Matrix::Zero(assignment.rows(), assignment.cols());
  return evaluateAssociationPairwiseTransformValidationObjective(
      sourcePoints, targetPoints, assignment, rotation, gradientSupport,
      options);
}

AssociationPairwiseTransformValidationResult
evaluateAssociationPairwiseTransformValidationObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &rotation,
    const Matrix &gradientSupport,
    const AssociationPairwiseTransformValidationOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationPairwiseTransformValidationOptions.weight must be finite "
        "and nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationPairwiseTransformValidationOptions.distanceScale must be "
        "finite and nonnegative");
  }
  if (sourcePoints.cols() != targetPoints.cols()) {
    throw std::invalid_argument(
        "pairwise transform validation point dimensions must match");
  }
  if (rotation.rows() != sourcePoints.cols() ||
      rotation.cols() != sourcePoints.cols()) {
    throw std::invalid_argument(
        "pairwise transform validation rotation dimension must match points");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match pairwise validation point counts");
  }
  if (gradientSupport.rows() != assignment.rows() ||
      gradientSupport.cols() != assignment.cols()) {
    throw std::invalid_argument(
        "pairwise transform validation gradient support must match "
        "assignment shape");
  }

  AssociationPairwiseTransformValidationResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0 ||
      sourcePoints.rows() == 0 || targetPoints.rows() == 0 ||
      sourcePoints.cols() == 0) {
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !assignment.allFinite() || !rotation.allFinite()) {
    throw std::invalid_argument(
        "pairwise transform validation objective inputs must be finite");
  }

  double scale = options.distanceScale;
  if (scale == 0.0) {
    double squaredNorm = 0.0;
    std::size_t count = 0;
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      for (int other = row + 1; other < sourcePoints.rows(); ++other) {
        squaredNorm +=
            (sourcePoints.row(row) - sourcePoints.row(other)).squaredNorm();
        ++count;
      }
    }
    for (int row = 0; row < targetPoints.rows(); ++row) {
      for (int other = row + 1; other < targetPoints.rows(); ++other) {
        squaredNorm +=
            (rotation *
             (targetPoints.row(row) - targetPoints.row(other)).transpose())
                .squaredNorm();
        ++count;
      }
    }
    scale = count > 0 ? std::sqrt(squaredNorm / static_cast<double>(count))
                      : 1.0;
    if (!std::isfinite(scale) || scale <= 1e-12) {
      scale = 1.0;
    }
  }
  const double scaleSquared = scale * scale;
  const Matrix rotatedTargetPoints = targetPoints * rotation.transpose();

  struct ActivePair {
    int source{0};
    int target{0};
    double weight{0.0};
  };
  std::vector<ActivePair> activePairs;
  activePairs.reserve(static_cast<std::size_t>(assignment.size()));
  for (int source = 0; source < assignment.rows(); ++source) {
    for (int target = 0; target < assignment.cols(); ++target) {
      const double weight = assignment(source, target);
      if (std::isfinite(weight) && std::abs(weight) > 1e-12) {
        activePairs.push_back(ActivePair{source, target, weight});
      }
    }
  }
  if (activePairs.empty()) {
    return result;
  }

  struct SupportPair {
    int source{0};
    int target{0};
  };
  std::vector<SupportPair> supportPairs;
  supportPairs.reserve(static_cast<std::size_t>(gradientSupport.size()));
  for (int source = 0; source < gradientSupport.rows(); ++source) {
    for (int target = 0; target < gradientSupport.cols(); ++target) {
      if (!std::isfinite(gradientSupport(source, target))) {
        continue;
      }
      supportPairs.push_back(SupportPair{source, target});
    }
  }

  for (const SupportPair &support : supportPairs) {
    double gradientEntry = 0.0;
    for (const ActivePair &active : activePairs) {
      double residualSquared = 0.0;
      for (int dim = 0; dim < sourcePoints.cols(); ++dim) {
        const double sourceDelta =
            sourcePoints(support.source, dim) -
            sourcePoints(active.source, dim);
        const double targetDelta =
            rotatedTargetPoints(support.target, dim) -
            rotatedTargetPoints(active.target, dim);
        const double residual = sourceDelta - targetDelta;
        residualSquared += residual * residual;
      }
      gradientEntry += active.weight * residualSquared / scaleSquared;
    }
    result.gradient(support.source, support.target) =
        options.weight * gradientEntry;
  }

  result.cost =
      0.5 * assignment.cwiseProduct(result.gradient).sum();
  return result;
}

AssociationRotoTranslationRingHypothesisResult
generateAssociationRotoTranslationRingHypotheses(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const AssociationRotoTranslationRingHypothesisOptions &options) {
  if (options.angleBins == 0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationRingHypothesisOptions.angleBins must be "
        "positive");
  }
  if (sourcePoints.cols() != 2 || targetPoints.cols() != 2) {
    throw std::invalid_argument(
        "ring-style roto-translation hypotheses require 2D points");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match ring hypothesis point counts");
  }
  AssociationRotoTranslationRingHypothesisResult result;
  if (options.maxHypotheses == 0 || assignment.size() == 0 ||
      sourcePoints.rows() == 0 || targetPoints.rows() == 0) {
    result.scores = Vector::Zero(0);
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !assignment.allFinite()) {
    throw std::invalid_argument(
        "ring-style roto-translation hypothesis inputs must be finite");
  }

  const double twoPi = 2.0 * M_PI;
  const double angleStep =
      twoPi / static_cast<double>(options.angleBins);
  Vector sourceMass = assignment.rowwise().sum();
  Vector targetMass = assignment.colwise().sum().transpose();
  const double totalMass = assignment.sum();
  if (!std::isfinite(totalMass)) {
    throw std::invalid_argument(
        "ring-style roto-translation assignment mass must be finite");
  }
  if (std::abs(totalMass) <= 1e-12) {
    sourceMass.setOnes();
    targetMass.setOnes();
  }

  Vector sourceMean = Vector::Zero(2);
  double sourceWeightSum = 0.0;
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    const double weight = std::max(0.0, sourceMass(row));
    sourceMean(0) += weight * sourcePoints(row, 0);
    sourceMean(1) += weight * sourcePoints(row, 1);
    sourceWeightSum += weight;
  }
  if (sourceWeightSum <= 1e-12) {
    sourceWeightSum = static_cast<double>(sourcePoints.rows());
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      sourceMean(0) += sourcePoints(row, 0);
      sourceMean(1) += sourcePoints(row, 1);
    }
  }
  sourceMean /= sourceWeightSum;

  Vector targetMean = Vector::Zero(2);
  double targetWeightSum = 0.0;
  for (int row = 0; row < targetPoints.rows(); ++row) {
    const double weight = std::max(0.0, targetMass(row));
    targetMean(0) += weight * targetPoints(row, 0);
    targetMean(1) += weight * targetPoints(row, 1);
    targetWeightSum += weight;
  }
  if (targetWeightSum <= 1e-12) {
    targetWeightSum = static_cast<double>(targetPoints.rows());
    for (int row = 0; row < targetPoints.rows(); ++row) {
      targetMean(0) += targetPoints(row, 0);
      targetMean(1) += targetPoints(row, 1);
    }
  }
  targetMean /= targetWeightSum;

  std::vector<double> binScores(options.angleBins, 0.0);
  for (int sourceIndex = 0; sourceIndex < assignment.rows();
       ++sourceIndex) {
    const Vector sourceCentered =
        sourcePoints.row(sourceIndex).transpose() - sourceMean;
    if (sourceCentered.squaredNorm() <= 1e-12) {
      continue;
    }
    const double sourceAngle =
        std::atan2(sourceCentered(1), sourceCentered(0));
    for (int targetIndex = 0; targetIndex < assignment.cols();
         ++targetIndex) {
      double vote = assignment(sourceIndex, targetIndex);
      if (std::abs(totalMass) <= 1e-12) {
        vote = 1.0;
      }
      if (!std::isfinite(vote) || vote <= 0.0) {
        continue;
      }
      const Vector targetCentered =
          targetPoints.row(targetIndex).transpose() - targetMean;
      if (targetCentered.squaredNorm() <= 1e-12) {
        continue;
      }
      const double targetAngle =
          std::atan2(targetCentered(1), targetCentered(0));
      double yaw = std::fmod(sourceAngle - targetAngle, twoPi);
      if (yaw < 0.0) {
        yaw += twoPi;
      }
      std::size_t bin = static_cast<std::size_t>(
          std::floor(yaw / angleStep + 0.5));
      bin %= options.angleBins;
      binScores[bin] += vote;
    }
  }

  std::vector<std::pair<double, std::size_t>> rankedBins;
  rankedBins.reserve(options.angleBins);
  for (std::size_t bin = 0; bin < options.angleBins; ++bin) {
    rankedBins.emplace_back(binScores[bin], bin);
  }
  std::sort(rankedBins.begin(), rankedBins.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });

  const std::size_t count =
      std::min(options.maxHypotheses, rankedBins.size());
  result.rotations.reserve(count);
  result.translations.reserve(count);
  result.scores = Vector::Zero(static_cast<int>(count));
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t bin = rankedBins[index].second;
    const double yaw = static_cast<double>(bin) * angleStep;
    Matrix rotation(2, 2);
    rotation << std::cos(yaw), -std::sin(yaw),
                std::sin(yaw), std::cos(yaw);
    Vector translation = sourceMean - rotation * targetMean;
    result.rotations.push_back(rotation);
    result.translations.push_back(translation);
    result.scores(static_cast<int>(index)) = rankedBins[index].first;
  }
  return result;
}

AssociationRotoTranslationRingHypothesisResult
generateAssociationRotoTranslationRingHypotheses(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const std::vector<SparseAssignmentCostEntry> &supportEntries,
    const AssociationRotoTranslationRingHypothesisOptions &options) {
  if (options.angleBins == 0) {
    throw std::invalid_argument(
        "AssociationRotoTranslationRingHypothesisOptions.angleBins must be "
        "positive");
  }
  if (sourcePoints.cols() != 2 || targetPoints.cols() != 2) {
    throw std::invalid_argument(
        "sparse ring-style roto-translation hypotheses require 2D points");
  }
  if (assignment.rows() != sourcePoints.rows() ||
      assignment.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "assignment matrix shape must match sparse ring hypothesis point "
        "counts");
  }
  AssociationRotoTranslationRingHypothesisResult result;
  if (options.maxHypotheses == 0 || assignment.size() == 0 ||
      sourcePoints.rows() == 0 || targetPoints.rows() == 0 ||
      supportEntries.empty()) {
    result.scores = Vector::Zero(0);
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite()) {
    throw std::invalid_argument(
        "sparse ring-style roto-translation hypothesis points must be finite");
  }

  const double twoPi = 2.0 * M_PI;
  const double angleStep =
      twoPi / static_cast<double>(options.angleBins);
  Vector sourceMass = Vector::Zero(assignment.rows());
  Vector targetMass = Vector::Zero(assignment.cols());
  double totalMass = 0.0;
  for (const SparseAssignmentCostEntry &entry : supportEntries) {
    if (entry.row < 0 || entry.col < 0 ||
        entry.row >= assignment.rows() || entry.col >= assignment.cols()) {
      throw std::invalid_argument(
          "sparse ring support entry is outside assignment shape");
    }
    const double weight = assignment(entry.row, entry.col);
    if (!std::isfinite(weight)) {
      throw std::invalid_argument(
          "sparse ring assignment support values must be finite");
    }
    sourceMass(entry.row) += weight;
    targetMass(entry.col) += weight;
    totalMass += weight;
  }
  if (std::abs(totalMass) <= 1e-12) {
    return generateAssociationRotoTranslationRingHypotheses(
        sourcePoints, targetPoints, assignment, options);
  }

  Vector sourceMean = Vector::Zero(2);
  double sourceWeightSum = 0.0;
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    const double weight = std::max(0.0, sourceMass(row));
    sourceMean.noalias() += weight * sourcePoints.row(row).transpose();
    sourceWeightSum += weight;
  }
  if (sourceWeightSum <= 1e-12) {
    sourceWeightSum = static_cast<double>(sourcePoints.rows());
    for (int row = 0; row < sourcePoints.rows(); ++row) {
      sourceMean.noalias() += sourcePoints.row(row).transpose();
    }
  }
  sourceMean /= sourceWeightSum;

  Vector targetMean = Vector::Zero(2);
  double targetWeightSum = 0.0;
  for (int row = 0; row < targetPoints.rows(); ++row) {
    const double weight = std::max(0.0, targetMass(row));
    targetMean.noalias() += weight * targetPoints.row(row).transpose();
    targetWeightSum += weight;
  }
  if (targetWeightSum <= 1e-12) {
    targetWeightSum = static_cast<double>(targetPoints.rows());
    for (int row = 0; row < targetPoints.rows(); ++row) {
      targetMean.noalias() += targetPoints.row(row).transpose();
    }
  }
  targetMean /= targetWeightSum;

  std::vector<double> binScores(options.angleBins, 0.0);
  for (const SparseAssignmentCostEntry &entry : supportEntries) {
    const double vote = assignment(entry.row, entry.col);
    if (!std::isfinite(vote) || vote <= 0.0) {
      continue;
    }
    const double sourceDx = sourcePoints(entry.row, 0) - sourceMean(0);
    const double sourceDy = sourcePoints(entry.row, 1) - sourceMean(1);
    if (sourceDx * sourceDx + sourceDy * sourceDy <= 1e-12) {
      continue;
    }
    const double targetDx = targetPoints(entry.col, 0) - targetMean(0);
    const double targetDy = targetPoints(entry.col, 1) - targetMean(1);
    if (targetDx * targetDx + targetDy * targetDy <= 1e-12) {
      continue;
    }
    const double sourceAngle = std::atan2(sourceDy, sourceDx);
    const double targetAngle = std::atan2(targetDy, targetDx);
    double yaw = std::fmod(sourceAngle - targetAngle, twoPi);
    if (yaw < 0.0) {
      yaw += twoPi;
    }
    std::size_t bin = static_cast<std::size_t>(
        std::floor(yaw / angleStep + 0.5));
    bin %= options.angleBins;
    binScores[bin] += vote;
  }

  std::vector<std::pair<double, std::size_t>> rankedBins;
  rankedBins.reserve(options.angleBins);
  for (std::size_t bin = 0; bin < options.angleBins; ++bin) {
    rankedBins.emplace_back(binScores[bin], bin);
  }
  std::sort(rankedBins.begin(), rankedBins.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first > rhs.first;
              }
              return lhs.second < rhs.second;
            });

  const std::size_t count =
      std::min(options.maxHypotheses, rankedBins.size());
  result.rotations.reserve(count);
  result.translations.reserve(count);
  result.scores = Vector::Zero(static_cast<int>(count));
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t bin = rankedBins[index].second;
    const double yaw = static_cast<double>(bin) * angleStep;
    Matrix rotation(2, 2);
    rotation << std::cos(yaw), -std::sin(yaw),
                std::sin(yaw), std::cos(yaw);
    Vector translation = sourceMean - rotation * targetMean;
    result.rotations.push_back(rotation);
    result.translations.push_back(translation);
    result.scores(static_cast<int>(index)) = rankedBins[index].first;
  }
  return result;
}

Matrix centerAssignmentGradientRowsOnFiniteSupport(const Matrix &gradient,
                                                   const Matrix &supportCosts) {
  if (gradient.rows() != supportCosts.rows() ||
      gradient.cols() != supportCosts.cols()) {
    throw std::invalid_argument(
        "gradient and support cost matrix shapes must match");
  }
  Matrix centered = Matrix::Zero(gradient.rows(), gradient.cols());
  for (int row = 0; row < gradient.rows(); ++row) {
    double sum = 0.0;
    int count = 0;
    for (int col = 0; col < gradient.cols(); ++col) {
      if (!std::isfinite(supportCosts(row, col))) {
        continue;
      }
      const double value = std::isfinite(gradient(row, col))
                               ? gradient(row, col)
                               : 0.0;
      sum += value;
      ++count;
    }
    if (count == 0) {
      continue;
    }
    const double mean = sum / static_cast<double>(count);
    for (int col = 0; col < gradient.cols(); ++col) {
      if (!std::isfinite(supportCosts(row, col))) {
        continue;
      }
      const double value = std::isfinite(gradient(row, col))
                               ? gradient(row, col)
                               : 0.0;
      centered(row, col) = value - mean;
    }
  }
  return centered;
}

Matrix centerAndNormalizeAssignmentGradientOnFiniteSupport(
    const Matrix &gradient, const Matrix &supportCosts, bool centerRows,
    bool centerColumns, double targetRms) {
  return centerAndNormalizeAssignmentGradientOnFiniteSupport(
      gradient, supportCosts, supportCosts, centerRows, centerColumns,
      targetRms);
}

Matrix centerAndNormalizeAssignmentGradientOnFiniteSupport(
    const Matrix &gradient, const Matrix &centeringSupportCosts,
    const Matrix &normalizationSupportCosts, bool centerRows,
    bool centerColumns, double targetRms) {
  if (gradient.rows() != centeringSupportCosts.rows() ||
      gradient.cols() != centeringSupportCosts.cols() ||
      gradient.rows() != normalizationSupportCosts.rows() ||
      gradient.cols() != normalizationSupportCosts.cols()) {
    throw std::invalid_argument(
        "gradient and support cost matrix shapes must match");
  }
  Matrix result = Matrix::Zero(gradient.rows(), gradient.cols());
  std::size_t finiteCount = 0;
  for (int row = 0; row < gradient.rows(); ++row) {
    for (int col = 0; col < gradient.cols(); ++col) {
      if (!std::isfinite(centeringSupportCosts(row, col))) {
        continue;
      }
      result(row, col) =
          std::isfinite(gradient(row, col)) ? gradient(row, col) : 0.0;
      ++finiteCount;
    }
  }
  if (finiteCount == 0) {
    return result;
  }

  const int centeringPasses = centerRows && centerColumns ? 8 : 1;
  for (int pass = 0; pass < centeringPasses; ++pass) {
    if (centerRows) {
      for (int row = 0; row < result.rows(); ++row) {
        double sum = 0.0;
        int count = 0;
        for (int col = 0; col < result.cols(); ++col) {
          if (std::isfinite(centeringSupportCosts(row, col))) {
            sum += result(row, col);
            ++count;
          }
        }
        if (count == 0) {
          continue;
        }
        const double mean = sum / static_cast<double>(count);
        for (int col = 0; col < result.cols(); ++col) {
          if (std::isfinite(centeringSupportCosts(row, col))) {
            result(row, col) -= mean;
          }
        }
      }
    }
    if (centerColumns) {
      for (int col = 0; col < result.cols(); ++col) {
        double sum = 0.0;
        int count = 0;
        for (int row = 0; row < result.rows(); ++row) {
          if (std::isfinite(centeringSupportCosts(row, col))) {
            sum += result(row, col);
            ++count;
          }
        }
        if (count == 0) {
          continue;
        }
        const double mean = sum / static_cast<double>(count);
        for (int row = 0; row < result.rows(); ++row) {
          if (std::isfinite(centeringSupportCosts(row, col))) {
            result(row, col) -= mean;
          }
        }
      }
    }
  }

  if (!std::isfinite(targetRms) || targetRms <= 0.0) {
    return result;
  }
  const double requestedRms = targetRms;
  double squaredNorm = 0.0;
  std::size_t normalizationCount = 0;
  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      if (std::isfinite(centeringSupportCosts(row, col)) &&
          std::isfinite(normalizationSupportCosts(row, col))) {
        squaredNorm += result(row, col) * result(row, col);
        ++normalizationCount;
      }
    }
  }
  if (normalizationCount == 0) {
    return result;
  }
  const double rms =
      std::sqrt(squaredNorm / static_cast<double>(normalizationCount));
  if (std::isfinite(rms) && rms > 1e-12) {
    result *= requestedRms / rms;
  }
  return result;
}

double associationMeritGateFromDecrease(double meritDecrease,
                                        double trustScale) {
  if (!std::isfinite(trustScale) || trustScale <= 0.0) {
    return 1.0;
  }
  if (!std::isfinite(meritDecrease)) {
    return 0.0;
  }
  if (meritDecrease >= 0.0) {
    return 1.0;
  }
  const double ratio = -meritDecrease / trustScale;
  if (!std::isfinite(ratio)) {
    return 0.0;
  }
  const double ratioSquared = ratio * ratio;
  if (!std::isfinite(ratioSquared)) {
    return 0.0;
  }
  return 1.0 / (1.0 + ratioSquared);
}

double associationFreshSupportStateValidationMeritGate(double baseMerit,
                                                       double candidateMerit,
                                                       double trustScale) {
  if (!std::isfinite(trustScale) || trustScale <= 0.0) {
    return 1.0;
  }
  if (!std::isfinite(baseMerit) || !std::isfinite(candidateMerit)) {
    return 0.0;
  }
  return associationMeritGateFromDecrease(baseMerit - candidateMerit,
                                          trustScale);
}

double stateValidationTransformConfidencePenalty(double weightedConfidence) {
  const double confidence =
      std::isfinite(weightedConfidence)
          ? std::clamp(weightedConfidence, 0.0, 1.0)
          : 0.0;
  const double residual = 1.0 - confidence;
  return residual * residual;
}

double objectObservationUncertaintyReliability(std::size_t observationCount,
                                               double meanDistance,
                                               double countPrior,
                                               double distanceScale,
                                               double minReliability) {
  if ((!std::isfinite(countPrior) || countPrior <= 0.0) &&
      (!std::isfinite(distanceScale) || distanceScale <= 0.0)) {
    return 1.0;
  }
  const double minScale =
      std::clamp(std::isfinite(minReliability) ? minReliability : 0.0, 0.0,
                 1.0);
  double scale = 1.0;
  if (std::isfinite(countPrior) && countPrior > 0.0) {
    const double count = static_cast<double>(observationCount);
    const double countFactor =
        count > 0.0 ? std::sqrt(count / (count + countPrior)) : 0.0;
    scale *= countFactor;
  }
  if (std::isfinite(distanceScale) && distanceScale > 0.0) {
    const double distance =
        std::isfinite(meanDistance) ? std::max(0.0, meanDistance)
                                    : std::numeric_limits<double>::infinity();
    const double distanceFactor =
        std::isfinite(distance) ? distanceScale / (distanceScale + distance)
                                : 0.0;
    scale *= distanceFactor;
  }
  if (!std::isfinite(scale)) {
    return minScale;
  }
  return std::clamp(std::max(minScale, scale), 0.0, 1.0);
}

AssociationCoveragePriorResult evaluateAssociationCoveragePrior(
    const Matrix &assignment, const Matrix &reference,
    const AssociationCoveragePriorOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationCoveragePriorOptions.weight must be finite and "
        "nonnegative");
  }
  if (assignment.rows() != reference.rows() ||
      assignment.cols() != reference.cols()) {
    throw std::invalid_argument(
        "assignment and reference coverage prior matrices must match");
  }
  AssociationCoveragePriorResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!assignment.allFinite() || !reference.allFinite()) {
    throw std::invalid_argument(
        "coverage prior objective inputs must be finite");
  }
  double effectiveWeight = options.weight;
  if (options.normalizeByReferenceMass) {
    const double referenceMass = reference.sum();
    const double assignmentMass = assignment.sum();
    double normalizer = 1.0;
    if (std::isfinite(referenceMass) && referenceMass > normalizer) {
      normalizer = referenceMass;
    }
    if (std::isfinite(assignmentMass) && assignmentMass > normalizer) {
      normalizer = assignmentMass;
    }
    if (normalizer > 1e-12) {
      effectiveWeight /= normalizer;
    }
  }

  if (options.matchRowColumnMarginals) {
    const Eigen::VectorXd rowDiff =
        assignment.rowwise().sum() - reference.rowwise().sum();
    const Eigen::RowVectorXd colDiff =
        assignment.colwise().sum() - reference.colwise().sum();
    result.cost =
        0.5 * effectiveWeight *
        (rowDiff.squaredNorm() + colDiff.squaredNorm());
    result.gradient =
        effectiveWeight *
        (rowDiff.replicate(1, assignment.cols()) +
         colDiff.replicate(assignment.rows(), 1));
    return result;
  }

  const double massDiff = assignment.sum() - reference.sum();
  result.cost = 0.5 * effectiveWeight * massDiff * massDiff;
  result.gradient =
      Matrix::Constant(assignment.rows(), assignment.cols(),
                       effectiveWeight * massDiff);
  return result;
}

Matrix associationConfidenceGatedSignedRecoveryGate(
    const Matrix &signedGradient, const Matrix &confidence,
    double sigmoidScale, double maxBlend) {
  if (signedGradient.rows() != confidence.rows() ||
      signedGradient.cols() != confidence.cols()) {
    throw std::invalid_argument(
        "signed recovery gate gradient and confidence shapes must match");
  }
  if (!std::isfinite(sigmoidScale) || sigmoidScale <= 0.0) {
    throw std::invalid_argument(
        "signed recovery gate sigmoid scale must be finite and positive");
  }
  if (!std::isfinite(maxBlend) || maxBlend < 0.0) {
    throw std::invalid_argument(
        "signed recovery gate max blend must be finite and nonnegative");
  }

  Matrix gate = Matrix::Zero(signedGradient.rows(), signedGradient.cols());
  if (maxBlend == 0.0 || signedGradient.size() == 0) {
    return gate;
  }
  for (int row = 0; row < gate.rows(); ++row) {
    for (int col = 0; col < gate.cols(); ++col) {
      if (!std::isfinite(signedGradient(row, col)) ||
          !std::isfinite(confidence(row, col))) {
        continue;
      }
      const double clippedConfidence =
          std::clamp(confidence(row, col), 0.0, 1.0);
      const double argument = -signedGradient(row, col) / sigmoidScale;
      double logistic = 0.5;
      if (argument > 40.0) {
        logistic = 1.0;
      } else if (argument < -40.0) {
        logistic = std::exp(argument);
      } else {
        logistic = 1.0 / (1.0 + std::exp(-argument));
      }
      gate(row, col) = maxBlend * clippedConfidence * logistic;
    }
  }
  return gate;
}

Matrix associationReciprocalFreshSupportGate(
    const Matrix &localWeights, const Matrix &reverseTransposedWeights,
    const Matrix &confidence, double maxBlend) {
  if (localWeights.rows() != reverseTransposedWeights.rows() ||
      localWeights.cols() != reverseTransposedWeights.cols() ||
      localWeights.rows() != confidence.rows() ||
      localWeights.cols() != confidence.cols()) {
    throw std::invalid_argument(
        "reciprocal fresh support gate matrices must have matching shapes");
  }
  if (!std::isfinite(maxBlend) || maxBlend < 0.0) {
    throw std::invalid_argument(
        "reciprocal fresh support gate max blend must be finite and "
        "nonnegative");
  }

  Matrix gate = Matrix::Zero(localWeights.rows(), localWeights.cols());
  if (maxBlend == 0.0 || localWeights.size() == 0) {
    return gate;
  }
  for (int row = 0; row < gate.rows(); ++row) {
    for (int col = 0; col < gate.cols(); ++col) {
      if (!std::isfinite(localWeights(row, col)) ||
          !std::isfinite(reverseTransposedWeights(row, col)) ||
          !std::isfinite(confidence(row, col))) {
        continue;
      }
      const double localMass =
          std::clamp(localWeights(row, col), 0.0, 1.0);
      const double reverseMass =
          std::clamp(reverseTransposedWeights(row, col), 0.0, 1.0);
      const double positiveReciprocalResidual =
          std::max(0.0, reverseMass - localMass);
      const double clippedConfidence =
          std::clamp(confidence(row, col), 0.0, 1.0);
      gate(row, col) =
          std::min(maxBlend,
                   maxBlend * clippedConfidence *
                       positiveReciprocalResidual);
    }
  }
  return gate;
}

Matrix associationReciprocalUniquenessConfidence(
    const Matrix &localWeights, const Matrix &reverseTransposedWeights,
    double competitionScale) {
  if (localWeights.rows() != reverseTransposedWeights.rows() ||
      localWeights.cols() != reverseTransposedWeights.cols()) {
    throw std::invalid_argument(
        "reciprocal uniqueness confidence matrices must have matching shapes");
  }
  Matrix confidence = Matrix::Ones(localWeights.rows(), localWeights.cols());
  if (!std::isfinite(competitionScale) || competitionScale <= 0.0 ||
      localWeights.size() == 0) {
    return confidence;
  }

  Matrix support = Matrix::Zero(localWeights.rows(), localWeights.cols());
  for (int row = 0; row < support.rows(); ++row) {
    for (int col = 0; col < support.cols(); ++col) {
      const double local =
          std::isfinite(localWeights(row, col))
              ? std::clamp(localWeights(row, col), 0.0, 1.0)
              : 0.0;
      const double reverse =
          std::isfinite(reverseTransposedWeights(row, col))
              ? std::clamp(reverseTransposedWeights(row, col), 0.0, 1.0)
              : 0.0;
      support(row, col) = std::max(local, reverse);
    }
  }

  Vector rowSums = support.rowwise().sum();
  Vector colSums = support.colwise().sum().transpose();
  constexpr double eps = 1e-12;
  for (int row = 0; row < support.rows(); ++row) {
    for (int col = 0; col < support.cols(); ++col) {
      const double mass = support(row, col);
      const double rowCompetition = std::max(0.0, rowSums(row) - mass);
      const double colCompetition = std::max(0.0, colSums(col) - mass);
      const double competition = rowCompetition + colCompetition;
      confidence(row, col) =
          mass / (mass + competitionScale * competition + eps);
    }
  }
  return confidence;
}

AssociationPairSupportGateResult applyAssociationPairSupportGate(
    const Matrix &assignment,
    const AssociationPairSupportGateOptions &options) {
  if (!std::isfinite(options.minEffectiveSupport) ||
      options.minEffectiveSupport < 0.0) {
    throw std::invalid_argument(
        "AssociationPairSupportGateOptions.minEffectiveSupport must be "
        "finite and nonnegative");
  }
  if (!std::isfinite(options.power) || options.power < 0.0) {
    throw std::invalid_argument(
        "AssociationPairSupportGateOptions.power must be finite and "
        "nonnegative");
  }

  AssociationPairSupportGateResult result;
  result.weights = assignment;
  double mass = 0.0;
  double squaredMass = 0.0;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double value = assignment(row, col);
      if (!std::isfinite(value) || value <= 0.0) {
        if (!std::isfinite(value)) {
          result.weights(row, col) = 0.0;
        }
        continue;
      }
      mass += value;
      squaredMass += value * value;
    }
  }
  result.effectiveSupport =
      squaredMass > 1e-24 ? (mass * mass) / squaredMass : 0.0;

  if (options.power <= 0.0 || options.minEffectiveSupport <= 0.0) {
    result.gate = 1.0;
    return result;
  }
  const double ratio = std::clamp(
      result.effectiveSupport / options.minEffectiveSupport, 0.0, 1.0);
  result.gate = std::pow(ratio, options.power);
  if (!std::isfinite(result.gate)) {
    result.gate = 0.0;
  }
  result.weights *= result.gate;
  return result;
}

double associationPairSupportConfidence(
    const Matrix &assignment, const AssociationPairSupportGateOptions &options) {
  if (!std::isfinite(options.minEffectiveSupport) ||
      options.minEffectiveSupport < 0.0) {
    throw std::invalid_argument(
        "AssociationPairSupportGateOptions.minEffectiveSupport must be "
        "finite and nonnegative");
  }
  if (!std::isfinite(options.power) || options.power < 0.0) {
    throw std::invalid_argument(
        "AssociationPairSupportGateOptions.power must be finite and "
        "nonnegative");
  }

  double mass = 0.0;
  double squaredMass = 0.0;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double value = assignment(row, col);
      if (!std::isfinite(value) || value <= 0.0) {
        continue;
      }
      mass += value;
      squaredMass += value * value;
    }
  }
  if (!std::isfinite(mass) || mass <= 0.0 || squaredMass <= 1e-24) {
    return 0.0;
  }
  const double effectiveSupport = (mass * mass) / squaredMass;
  if (!std::isfinite(effectiveSupport) || effectiveSupport <= 0.0) {
    return 0.0;
  }

  if (options.minEffectiveSupport <= 0.0) {
    return std::clamp(mass, 0.0, 1.0);
  }
  const double supportRatio =
      std::clamp(effectiveSupport / options.minEffectiveSupport, 0.0, 1.0);
  const double massRatio =
      std::clamp(mass / options.minEffectiveSupport, 0.0, 1.0);
  const double supportPower = options.power > 0.0 ? options.power : 1.0;
  const double poweredSupportRatio = std::pow(supportRatio, supportPower);
  const double confidence = poweredSupportRatio * massRatio;
  return std::isfinite(confidence) ? std::clamp(confidence, 0.0, 1.0) : 0.0;
}

AssociationPairSupportLatentContinuationResult
applyAssociationPairSupportLatentContinuation(
    const Matrix &candidate, const Matrix &reference, double latentConfidence,
    const AssociationPairSupportGateOptions &options) {
  if (candidate.rows() != reference.rows() ||
      candidate.cols() != reference.cols()) {
    throw std::invalid_argument(
        "pair support latent continuation matrices must match");
  }
  if (!candidate.allFinite() || !reference.allFinite()) {
    throw std::invalid_argument(
        "pair support latent continuation inputs must be finite");
  }
  if (!std::isfinite(latentConfidence) || latentConfidence < 0.0) {
    throw std::invalid_argument(
        "pair support latent confidence must be finite and nonnegative");
  }

  AssociationPairSupportLatentContinuationResult result;
  result.weights = candidate;
  result.referenceConfidence =
      associationPairSupportConfidence(reference, options);
  result.candidateConfidence =
      associationPairSupportConfidence(candidate, options);
  auto positiveMass = [](const Matrix &assignment) {
    double mass = 0.0;
    for (int row = 0; row < assignment.rows(); ++row) {
      for (int col = 0; col < assignment.cols(); ++col) {
        const double value = assignment(row, col);
        if (std::isfinite(value) && value > 0.0) {
          mass += value;
        }
      }
    }
    return std::isfinite(mass) ? mass : 0.0;
  };
  const double referenceMass = positiveMass(reference);
  if (referenceMass > 1e-12) {
    const double coverage =
        std::clamp(positiveMass(candidate) / referenceMass, 0.0, 1.0);
    result.candidateConfidence =
        std::min(result.candidateConfidence, coverage);
  }
  result.latentConfidence =
      std::clamp(std::max(latentConfidence, result.referenceConfidence), 0.0,
                 1.0);
  if (result.latentConfidence <= result.candidateConfidence) {
    return result;
  }

  result.blend =
      std::clamp(result.latentConfidence - result.candidateConfidence, 0.0,
                 1.0);
  result.weights = candidate + result.blend * (reference - candidate);
  return result;
}

bool associationPairSupportReferenceShouldUpdate(
    const Matrix &candidate, const Matrix &reference,
    const AssociationPairSupportGateOptions &options) {
  if (candidate.rows() != reference.rows() ||
      candidate.cols() != reference.cols()) {
    throw std::invalid_argument(
        "pair support reference update matrices must match");
  }
  if (!candidate.allFinite() || !reference.allFinite()) {
    throw std::invalid_argument(
        "pair support reference update inputs must be finite");
  }
  auto positiveMass = [](const Matrix &assignment) {
    double mass = 0.0;
    for (int row = 0; row < assignment.rows(); ++row) {
      for (int col = 0; col < assignment.cols(); ++col) {
        const double value = assignment(row, col);
        if (std::isfinite(value) && value > 0.0) {
          mass += value;
        }
      }
    }
    return std::isfinite(mass) ? mass : 0.0;
  };

  const double candidateConfidence =
      associationPairSupportConfidence(candidate, options);
  const double referenceConfidence =
      associationPairSupportConfidence(reference, options);
  constexpr double kTolerance = 1e-12;
  if (candidateConfidence > referenceConfidence + kTolerance) {
    return true;
  }
  if (candidateConfidence + kTolerance < referenceConfidence) {
    return false;
  }
  return positiveMass(candidate) + kTolerance >= positiveMass(reference);
}

double objectCopyResidualReliability(double residualNorm,
                                     double residualScale,
                                     double minReliability) {
  if (!std::isfinite(residualNorm) || residualNorm <= 0.0) {
    return 1.0;
  }
  if (!std::isfinite(residualScale) || residualScale <= 0.0) {
    return 1.0;
  }
  const double floor = std::clamp(
      std::isfinite(minReliability) ? minReliability : 0.0, 0.0, 1.0);
  const double ratio = residualNorm / residualScale;
  return floor + (1.0 - floor) / (1.0 + ratio * ratio);
}

double objectCopyResidualReliabilityFromActiveRms(
    double residualNorm, double activeResidualRms, double minReliability) {
  return objectCopyResidualReliability(residualNorm, activeResidualRms,
                                       minReliability);
}

Matrix objectCopyResidualReliabilityMatrixFromActiveRms(
    const Matrix &residualNorms, const Matrix &assignment,
    double minReliability) {
  if (residualNorms.rows() != assignment.rows() ||
      residualNorms.cols() != assignment.cols()) {
    throw std::invalid_argument(
        "residualNorms and assignment dimensions must match");
  }
  Matrix reliability = Matrix::Ones(residualNorms.rows(), residualNorms.cols());
  if (residualNorms.size() == 0) {
    return reliability;
  }

  double mass = 0.0;
  double squaredMass = 0.0;
  double weightedSquaredResidual = 0.0;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double weight = assignment(row, col);
      const double residual = residualNorms(row, col);
      if (!std::isfinite(weight) || weight <= 1e-12 ||
          !std::isfinite(residual) || residual < 0.0) {
        continue;
      }
      mass += weight;
      squaredMass += weight * weight;
      weightedSquaredResidual += weight * residual * residual;
    }
  }

  const double effectiveSupport =
      squaredMass > 1e-24 ? (mass * mass) / squaredMass : 0.0;
  if (mass <= 1e-12 || effectiveSupport <= 1.0 + 1e-12) {
    return reliability;
  }

  const double activeResidualRms =
      std::sqrt(std::max(0.0, weightedSquaredResidual / mass));
  if (!std::isfinite(activeResidualRms) || activeResidualRms <= 0.0) {
    return reliability;
  }
  for (int row = 0; row < residualNorms.rows(); ++row) {
    for (int col = 0; col < residualNorms.cols(); ++col) {
      const double weight = assignment(row, col);
      const double residual = residualNorms(row, col);
      if (!std::isfinite(weight) || weight <= 1e-12 ||
          !std::isfinite(residual) || residual < 0.0) {
        continue;
      }
      reliability(row, col) = objectCopyResidualReliabilityFromActiveRms(
          residual, activeResidualRms, minReliability);
    }
  }
  return reliability;
}

std::vector<double> objectResidualReliabilityVectorFromActiveRms(
    const std::vector<double> &residualNorms, double minReliability) {
  std::vector<double> reliability(residualNorms.size(), 1.0);
  double squaredResidualSum = 0.0;
  std::size_t count = 0;
  for (const double residual : residualNorms) {
    if (!std::isfinite(residual) || residual < 0.0) {
      continue;
    }
    squaredResidualSum += residual * residual;
    ++count;
  }
  if (count <= 1 || squaredResidualSum <= 0.0) {
    return reliability;
  }
  const double activeResidualRms =
      std::sqrt(squaredResidualSum / static_cast<double>(count));
  if (!std::isfinite(activeResidualRms) || activeResidualRms <= 0.0) {
    return reliability;
  }
  for (std::size_t i = 0; i < residualNorms.size(); ++i) {
    const double residual = residualNorms[i];
    if (!std::isfinite(residual) || residual < 0.0) {
      continue;
    }
    reliability[i] = objectCopyResidualReliabilityFromActiveRms(
        residual, activeResidualRms, minReliability);
  }
  return reliability;
}

double objectCopyReliabilityUpperCost(double reliability, double weight) {
  if (!std::isfinite(weight) || weight <= 0.0) {
    return 0.0;
  }
  const double clipped =
      std::clamp(std::isfinite(reliability) ? reliability : 1.0, 1e-12, 1.0);
  return weight * -std::log(clipped);
}

AssociationSupportSharpnessResult evaluateAssociationSupportSharpnessObjective(
    const Matrix &assignment,
    const AssociationSupportSharpnessOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationSupportSharpnessOptions.weight must be finite and "
        "nonnegative");
  }
  AssociationSupportSharpnessResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!assignment.allFinite()) {
    throw std::invalid_argument(
        "support sharpness objective input must be finite");
  }
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double value = assignment(row, col);
      result.cost += 0.5 * options.weight * value * (1.0 - value);
      result.gradient(row, col) = options.weight * (0.5 - value);
    }
  }
  return result;
}

double adaptiveAssociationSupportSharpnessWeight(
    double unitSharpnessGradientNorm, double localGradientNorm,
    const AssociationSupportSharpnessAdaptiveOptions &options) {
  if (!std::isfinite(options.maxWeight) || options.maxWeight < 0.0) {
    throw std::invalid_argument(
        "AssociationSupportSharpnessAdaptiveOptions.maxWeight must be finite "
        "and nonnegative");
  }
  if (!std::isfinite(options.targetGradientRatio) ||
      options.targetGradientRatio < 0.0) {
    throw std::invalid_argument(
        "AssociationSupportSharpnessAdaptiveOptions.targetGradientRatio must "
        "be finite and nonnegative");
  }
  if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
    throw std::invalid_argument(
        "AssociationSupportSharpnessAdaptiveOptions.epsilon must be finite "
        "and positive");
  }
  if (!std::isfinite(unitSharpnessGradientNorm) ||
      unitSharpnessGradientNorm < 0.0 ||
      !std::isfinite(localGradientNorm) || localGradientNorm < 0.0) {
    throw std::invalid_argument(
        "support sharpness gradient norms must be finite and nonnegative");
  }
  if (options.maxWeight == 0.0 ||
      unitSharpnessGradientNorm <= options.epsilon) {
    return 0.0;
  }
  if (options.targetGradientRatio == 0.0) {
    return options.maxWeight;
  }
  const double targetGradientNorm =
      options.targetGradientRatio * localGradientNorm;
  if (targetGradientNorm <= options.epsilon) {
    return options.maxWeight;
  }
  const double rawGradientNorm =
      options.maxWeight * unitSharpnessGradientNorm;
  const double gate =
      rawGradientNorm / (rawGradientNorm + targetGradientNorm);
  return options.maxWeight * std::max(0.0, std::min(1.0, gate));
}

AssociationStructuralExpansionResult
evaluateAssociationStructuralExpansionObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &reference,
    const Matrix &rotation, const Vector &translation,
    const AssociationStructuralExpansionOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationStructuralExpansionOptions.weight must be finite and "
        "nonnegative");
  }
  if (!std::isfinite(options.distanceScale) ||
      options.distanceScale < 0.0) {
    throw std::invalid_argument(
        "AssociationStructuralExpansionOptions.distanceScale must be finite "
        "and nonnegative");
  }
  if (sourcePoints.rows() != assignment.rows() ||
      targetPoints.rows() != assignment.cols() ||
      assignment.rows() != reference.rows() ||
      assignment.cols() != reference.cols()) {
    throw std::invalid_argument(
        "structural expansion assignment/reference shapes must match points");
  }
  const int dimension = sourcePoints.cols();
  if (dimension <= 0 || targetPoints.cols() != dimension ||
      rotation.rows() != dimension || rotation.cols() != dimension ||
      translation.rows() != dimension || translation.cols() != 1) {
    throw std::invalid_argument(
        "structural expansion transform dimensions are inconsistent");
  }
  AssociationStructuralExpansionResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !assignment.allFinite() || !reference.allFinite() ||
      !rotation.allFinite() || !translation.allFinite()) {
    throw std::invalid_argument(
        "structural expansion objective inputs must be finite");
  }

  double squaredResidualSum = 0.0;
  std::size_t residualCount = 0;
  Matrix squaredResiduals(assignment.rows(), assignment.cols());
  for (int row = 0; row < assignment.rows(); ++row) {
    const Vector source = sourcePoints.row(row).transpose();
    for (int col = 0; col < assignment.cols(); ++col) {
      const Vector target = targetPoints.row(col).transpose();
      const double squaredResidual =
          (source - (rotation * target + translation)).squaredNorm();
      squaredResiduals(row, col) = squaredResidual;
      squaredResidualSum += squaredResidual;
      ++residualCount;
    }
  }
  double scale = options.distanceScale;
  if (scale <= 0.0) {
    scale = residualCount > 0
                ? std::sqrt(squaredResidualSum /
                            static_cast<double>(residualCount))
                : 1.0;
  }
  scale = std::max(scale, std::numeric_limits<double>::epsilon());
  const double invTwoScaleSquared = 0.5 / (scale * scale);
  Matrix confidence(assignment.rows(), assignment.cols());
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      confidence(row, col) =
          std::exp(-squaredResiduals(row, col) * invTwoScaleSquared);
    }
  }
  const Vector rowMean =
      confidence.rowwise().mean();
  const Vector colMean =
      confidence.colwise().mean().transpose();
  const double globalMean = confidence.mean();
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double referenceWeight =
          std::max(0.0, std::min(1.0, reference(row, col)));
      const double novelty = 1.0 - referenceWeight;
      if (novelty <= 0.0) {
        continue;
      }
      double competitionScore = 0.0;
      if (assignment.rows() > 1 && assignment.cols() > 1) {
        competitionScore =
            2.0 * (confidence(row, col) - rowMean(row) - colMean(col) +
                   globalMean);
      } else if (assignment.cols() > 1) {
        competitionScore = 2.0 * (confidence(row, col) - rowMean(row));
      } else if (assignment.rows() > 1) {
        competitionScore = 2.0 * (confidence(row, col) - colMean(col));
      }
      const double reward = options.weight * novelty * competitionScore;
      result.gradient(row, col) = -reward;
      result.cost += assignment(row, col) * result.gradient(row, col);
    }
  }
  return result;
}

Matrix associationProposalSupportGate(const Matrix &proposal) {
  if (!proposal.allFinite()) {
    throw std::invalid_argument("association proposal gate input must be finite");
  }
  Matrix clipped = Matrix::Zero(proposal.rows(), proposal.cols());
  Vector rowMax = Vector::Zero(proposal.rows());
  Vector colMax = Vector::Zero(proposal.cols());
  for (int row = 0; row < proposal.rows(); ++row) {
    for (int col = 0; col < proposal.cols(); ++col) {
      const double mass = std::max(0.0, std::min(1.0, proposal(row, col)));
      clipped(row, col) = mass;
      rowMax(row) = std::max(rowMax(row), mass);
      colMax(col) = std::max(colMax(col), mass);
    }
  }
  Matrix gate = Matrix::Zero(proposal.rows(), proposal.cols());
  for (int row = 0; row < proposal.rows(); ++row) {
    for (int col = 0; col < proposal.cols(); ++col) {
      const double mass = clipped(row, col);
      if (mass <= 0.0 || rowMax(row) <= 0.0 || colMax(col) <= 0.0) {
        continue;
      }
      const double normalizer = std::sqrt(rowMax(row) * colMax(col));
      gate(row, col) =
          std::sqrt(std::max(0.0, std::min(1.0, mass / normalizer)));
    }
  }
  return gate;
}

Matrix associationUnionSupportGate(const Matrix &first, const Matrix &second) {
  if (first.rows() != second.rows() || first.cols() != second.cols()) {
    throw std::invalid_argument("association union support gate inputs must "
                                "have matching shape");
  }
  const Matrix firstGate = associationProposalSupportGate(first);
  const Matrix secondGate = associationProposalSupportGate(second);
  Matrix gate = firstGate;
  for (int row = 0; row < gate.rows(); ++row) {
    for (int col = 0; col < gate.cols(); ++col) {
      gate(row, col) = std::max(firstGate(row, col), secondGate(row, col));
    }
  }
  return gate;
}

Matrix associationResidualConfidenceGate(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation,
    double distanceScale) {
  if (!std::isfinite(distanceScale) || distanceScale < 0.0) {
    throw std::invalid_argument(
        "association residual confidence distance scale must be finite and "
        "nonnegative");
  }
  const int dimension = sourcePoints.cols();
  if (dimension <= 0 || targetPoints.cols() != dimension ||
      rotation.rows() != dimension || rotation.cols() != dimension ||
      translation.rows() != dimension || translation.cols() != 1) {
    throw std::invalid_argument(
        "association residual confidence dimensions are inconsistent");
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !rotation.allFinite() || !translation.allFinite()) {
    throw std::invalid_argument(
        "association residual confidence inputs must be finite");
  }
  Matrix squaredResiduals(sourcePoints.rows(), targetPoints.rows());
  double squaredResidualSum = 0.0;
  std::size_t residualCount = 0;
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    const Vector source = sourcePoints.row(row).transpose();
    for (int col = 0; col < targetPoints.rows(); ++col) {
      const Vector target = targetPoints.row(col).transpose();
      const double squaredResidual =
          (source - (rotation * target + translation)).squaredNorm();
      squaredResiduals(row, col) = squaredResidual;
      squaredResidualSum += squaredResidual;
      ++residualCount;
    }
  }
  double scale = distanceScale;
  if (scale <= 0.0) {
    scale = residualCount > 0
                ? std::sqrt(squaredResidualSum /
                            static_cast<double>(residualCount))
                : 1.0;
  }
  scale = std::max(scale, std::numeric_limits<double>::epsilon());
  const double invTwoScaleSquared = 0.5 / (scale * scale);
  Matrix gate(sourcePoints.rows(), targetPoints.rows());
  for (int row = 0; row < gate.rows(); ++row) {
    for (int col = 0; col < gate.cols(); ++col) {
      gate(row, col) =
          std::exp(-squaredResiduals(row, col) * invTwoScaleSquared);
    }
  }
  return gate;
}

Matrix associationRelativeResidualConfidenceGate(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation,
    double distanceScale) {
  return associationProposalSupportGate(
      associationResidualConfidenceGate(sourcePoints, targetPoints, rotation,
                                        translation, distanceScale));
}

Matrix associationRelativeResidualConfidenceGateOnSupport(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation, double distanceScale,
    const Matrix &support) {
  if (!std::isfinite(distanceScale) || distanceScale < 0.0) {
    throw std::invalid_argument(
        "association residual confidence distance scale must be finite and "
        "nonnegative");
  }
  const int dimension = sourcePoints.cols();
  if (dimension <= 0 || targetPoints.cols() != dimension ||
      rotation.rows() != dimension || rotation.cols() != dimension ||
      translation.rows() != dimension || translation.cols() != 1 ||
      support.rows() != sourcePoints.rows() ||
      support.cols() != targetPoints.rows()) {
    throw std::invalid_argument(
        "association residual confidence support dimensions are inconsistent");
  }
  if (!sourcePoints.allFinite() || !targetPoints.allFinite() ||
      !rotation.allFinite() || !translation.allFinite() ||
      !support.allFinite()) {
    throw std::invalid_argument(
        "association residual confidence support inputs must be finite");
  }

  Matrix rawGate = Matrix::Zero(sourcePoints.rows(), targetPoints.rows());
  std::vector<Entry> activeEntries;
  activeEntries.reserve(std::min<std::size_t>(
      1024u, static_cast<std::size_t>(support.size())));
  double squaredResidualSum = 0.0;
  for (int row = 0; row < support.rows(); ++row) {
    const Vector source = sourcePoints.row(row).transpose();
    for (int col = 0; col < support.cols(); ++col) {
      if (support(row, col) <= 0.0) {
        continue;
      }
      const Vector target = targetPoints.row(col).transpose();
      const double squaredResidual =
          (source - (rotation * target + translation)).squaredNorm();
      activeEntries.push_back(Entry{row, col, squaredResidual});
      squaredResidualSum += squaredResidual;
    }
  }
  if (activeEntries.empty()) {
    return rawGate;
  }
  double scale = distanceScale;
  if (scale <= 0.0) {
    scale = std::sqrt(squaredResidualSum /
                      static_cast<double>(activeEntries.size()));
  }
  scale = std::max(scale, std::numeric_limits<double>::epsilon());
  const double invTwoScaleSquared = 0.5 / (scale * scale);
  for (const Entry &entry : activeEntries) {
    rawGate(entry.row, entry.col) =
        std::exp(-entry.value * invTwoScaleSquared);
  }
  return associationProposalSupportGate(rawGate);
}

AssociationReferenceTrustResult evaluateAssociationReferenceTrust(
    const Matrix &assignment, const Matrix &reference,
    const AssociationReferenceTrustOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationReferenceTrustOptions.weight must be finite and "
        "nonnegative");
  }
  if (assignment.rows() != reference.rows() ||
      assignment.cols() != reference.cols()) {
    throw std::invalid_argument(
        "assignment and reference trust matrices must match");
  }
  AssociationReferenceTrustResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!assignment.allFinite() || !reference.allFinite()) {
    throw std::invalid_argument(
        "reference trust objective inputs must be finite");
  }
  const Matrix diff = assignment - reference;
  result.cost = 0.5 * options.weight * diff.squaredNorm();
  result.gradient = options.weight * diff;
  return result;
}

AssociationReferenceSupportSharpeningResult
evaluateAssociationReferenceSupportSharpening(
    const Matrix &assignment, const Matrix &reference,
    const AssociationReferenceSupportSharpeningOptions &options) {
  if (!std::isfinite(options.weight) || options.weight < 0.0) {
    throw std::invalid_argument(
        "AssociationReferenceSupportSharpeningOptions.weight must be finite "
        "and nonnegative");
  }
  if (!std::isfinite(options.target) || options.target < 0.0) {
    throw std::invalid_argument(
        "AssociationReferenceSupportSharpeningOptions.target must be finite "
        "and nonnegative");
  }
  if (assignment.rows() != reference.rows() ||
      assignment.cols() != reference.cols()) {
    throw std::invalid_argument(
        "assignment and reference support sharpening matrices must match");
  }
  AssociationReferenceSupportSharpeningResult result;
  result.gradient = Matrix::Zero(assignment.rows(), assignment.cols());
  if (options.weight == 0.0 || assignment.size() == 0) {
    return result;
  }
  if (!assignment.allFinite() || !reference.allFinite()) {
    throw std::invalid_argument(
        "reference support sharpening objective inputs must be finite");
  }
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double support =
          std::max(0.0, std::min(1.0, reference(row, col)));
      if (support <= 0.0) {
        continue;
      }
      const double diff = assignment(row, col) - options.target;
      result.cost += 0.5 * options.weight * support * diff * diff;
      result.gradient(row, col) = options.weight * support * diff;
    }
  }
  return result;
}

double scaleAssociationReferenceTrustWeightByCostRms(
    double baseWeight, const Matrix &costs, double minimumCostRmsScale) {
  if (!std::isfinite(baseWeight) || baseWeight < 0.0) {
    throw std::invalid_argument(
        "reference trust base weight must be finite and nonnegative");
  }
  if (!std::isfinite(minimumCostRmsScale) || minimumCostRmsScale < 0.0) {
    throw std::invalid_argument(
        "reference trust minimum cost scale must be finite and nonnegative");
  }
  if (baseWeight == 0.0) {
    return 0.0;
  }
  double squaredNorm = 0.0;
  std::size_t count = 0;
  for (int row = 0; row < costs.rows(); ++row) {
    for (int col = 0; col < costs.cols(); ++col) {
      const double value = costs(row, col);
      if (!std::isfinite(value)) {
        continue;
      }
      squaredNorm += value * value;
      ++count;
    }
  }
  double costRms = minimumCostRmsScale;
  if (count > 0) {
    costRms = std::max(costRms,
                       std::sqrt(squaredNorm / static_cast<double>(count)));
  }
  return baseWeight * costRms;
}

double associationConsensusWeightPower(double associationWeight,
                                       double power) {
  if (!std::isfinite(power) || power <= 0.0) {
    throw std::invalid_argument(
        "association consensus weight power must be finite and positive");
  }
  if (!std::isfinite(associationWeight) || associationWeight <= 0.0) {
    return 0.0;
  }
  if (power == 1.0) {
    return associationWeight;
  }
  return std::pow(associationWeight, power);
}

double associationHighConfidenceAnchorScale(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition,
    double transformConfidence, double gain) {
  if (!std::isfinite(gain) || gain < 0.0) {
    throw std::invalid_argument(
        "high-confidence anchor gain must be finite and nonnegative");
  }
  if (gain == 0.0) {
    return 1.0;
  }
  const double forward = std::clamp(finiteNonnegative(associationWeight),
                                    0.0, 1.0);
  const double reciprocal =
      std::clamp(finiteNonnegative(reciprocalAssociationWeight), 0.0, 1.0);
  const double rowOther = finiteNonnegative(rowCompetition);
  const double colOther = finiteNonnegative(columnCompetition);
  const double transform =
      std::clamp(finiteNonnegative(transformConfidence), 0.0, 1.0);
  const double support = std::sqrt(forward * reciprocal);
  const double competition = rowOther + colOther;
  constexpr double kEps = 1e-12;
  const double uniqueness =
      competition <= 0.0
          ? 1.0
          : support / (support + competition + kEps);
  return 1.0 + gain * support * uniqueness * transform;
}

double associationContinuousSupportGate(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition,
    double transformConfidence, double reliability, double power) {
  if (!std::isfinite(power) || power < 0.0) {
    throw std::invalid_argument(
        "continuous support gate power must be finite and nonnegative");
  }
  if (power == 0.0) {
    return 1.0;
  }
  const double forward = std::clamp(finiteNonnegative(associationWeight),
                                    0.0, 1.0);
  const double reciprocal =
      std::clamp(finiteNonnegative(reciprocalAssociationWeight), 0.0, 1.0);
  const double rowOther = finiteNonnegative(rowCompetition);
  const double colOther = finiteNonnegative(columnCompetition);
  const double transform =
      std::clamp(finiteNonnegative(transformConfidence), 0.0, 1.0);
  const double reliable =
      std::clamp(finiteNonnegative(reliability), 0.0, 1.0);
  const double support = std::sqrt(forward * reciprocal);
  const double competition = rowOther + colOther;
  constexpr double kEps = 1e-12;
  const double uniqueness =
      competition <= 0.0
          ? 1.0
          : support / (support + competition + kEps);
  const double confidence =
      std::clamp(support * uniqueness * transform * reliable, 0.0, 1.0);
  return std::pow(confidence, power);
}

double associationBoundedResidualReliabilityBlend(
    double sharedReliability, double currentResidualReliability) {
  const double shared = std::isfinite(sharedReliability)
                            ? std::clamp(sharedReliability, 0.0, 1.0)
                            : 0.0;
  if (shared <= 0.0) {
    return 0.0;
  }
  const double current =
      std::isfinite(currentResidualReliability)
          ? std::clamp(currentResidualReliability, 0.0, 1.0)
          : shared;
  const double excess = std::max(0.0, shared - current);
  return shared / (1.0 + excess);
}

double associationSupportConfidenceReliability(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition) {
  const double forward = std::clamp(finiteNonnegative(associationWeight),
                                    0.0, 1.0);
  const double reciprocal =
      std::clamp(finiteNonnegative(reciprocalAssociationWeight), 0.0, 1.0);
  const double rowOther = finiteNonnegative(rowCompetition);
  const double colOther = finiteNonnegative(columnCompetition);
  const double support = std::sqrt(forward * reciprocal);
  const double competition = rowOther + colOther;
  constexpr double kEps = 1e-12;
  const double uniqueness =
      competition <= 0.0
          ? 1.0
          : support / (support + competition + kEps);
  return std::clamp(support * uniqueness, 0.0, 1.0);
}

double associationTrajectoryFeedbackTrustScale(double poseTrust) {
  const double trust = std::isfinite(poseTrust)
                           ? std::clamp(poseTrust, 0.0, 1.0)
                           : 0.0;
  return trust * trust;
}

double associationTrajectoryFeedbackObjectMassNormalizer(double objectMass) {
  if (!std::isfinite(objectMass)) {
    return 0.0;
  }
  const double mass = std::max(0.0, objectMass);
  if (mass <= 1.0) {
    return 1.0;
  }
  return 1.0 / mass;
}

Matrix projectAssociationForLowerGraph(const Matrix &assignment,
                                       double power) {
  if (!std::isfinite(power) || power < 1.0) {
    throw std::invalid_argument(
        "lower association projection power must be finite and >= 1");
  }
  Matrix finite = assignment.unaryExpr([](double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
  });
  if (power == 1.0 || finite.size() == 0) {
    return finite;
  }

  Matrix projected = finite.unaryExpr([power](double value) {
    return value > 0.0 ? std::pow(value, power) : 0.0;
  });
  if (projected.sum() <= 0.0) {
    return projected;
  }

  Vector rowTargets(finite.rows());
  for (int row = 0; row < finite.rows(); ++row) {
    rowTargets(row) = finite.row(row).sum();
  }
  Vector colTargets(finite.cols());
  for (int col = 0; col < finite.cols(); ++col) {
    colTargets(col) = finite.col(col).sum();
  }

  constexpr int kIterations = 40;
  constexpr double kEps = 1e-15;
  for (int iter = 0; iter < kIterations; ++iter) {
    for (int row = 0; row < projected.rows(); ++row) {
      const double sum = projected.row(row).sum();
      if (rowTargets(row) <= 0.0) {
        projected.row(row).setZero();
      } else if (sum > kEps) {
        projected.row(row) *= rowTargets(row) / sum;
      }
    }
    for (int col = 0; col < projected.cols(); ++col) {
      const double sum = projected.col(col).sum();
      if (colTargets(col) <= 0.0) {
        projected.col(col).setZero();
      } else if (sum > kEps) {
        projected.col(col) *= colTargets(col) / sum;
      }
    }
  }
  return projected;
}

Matrix projectAssociationForLowerGraphHard(const Matrix &assignment) {
  struct Entry {
    int row{0};
    int col{0};
    double weight{0.0};
  };
  std::vector<Entry> entries;
  entries.reserve(static_cast<std::size_t>(assignment.size()));
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      const double weight = assignment(row, col);
      if (std::isfinite(weight) && weight > 0.0) {
        entries.push_back({row, col, weight});
      }
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) {
              if (a.weight != b.weight) {
                return a.weight > b.weight;
              }
              if (a.row != b.row) {
                return a.row < b.row;
              }
              return a.col < b.col;
            });
  Matrix projected = Matrix::Zero(assignment.rows(), assignment.cols());
  std::vector<bool> usedRows(static_cast<std::size_t>(assignment.rows()),
                             false);
  std::vector<bool> usedCols(static_cast<std::size_t>(assignment.cols()),
                             false);
  for (const Entry &entry : entries) {
    if (usedRows[static_cast<std::size_t>(entry.row)] ||
        usedCols[static_cast<std::size_t>(entry.col)]) {
      continue;
    }
    projected(entry.row, entry.col) = entry.weight;
    usedRows[static_cast<std::size_t>(entry.row)] = true;
    usedCols[static_cast<std::size_t>(entry.col)] = true;
  }
  return projected;
}

double backendFeedbackGeometryCost(
    double geometryCost, double localGradientNorm, double targetGradientNorm,
    const BackendAssociationFeedbackOptions &options) {
  if (!std::isfinite(geometryCost) || geometryCost <= 0.0) {
    return finiteNonnegative(geometryCost);
  }
  if (!std::isfinite(options.gradientScale) ||
      options.gradientScale <= 0.0) {
    return geometryCost;
  }
  const double local =
      std::isfinite(localGradientNorm) && localGradientNorm > 0.0
          ? localGradientNorm
          : 0.0;
  const double target =
      std::isfinite(targetGradientNorm) && targetGradientNorm > 0.0
          ? targetGradientNorm
          : 0.0;
  const double averageGradient = 0.5 * (local + target);
  const double minTrust =
      std::isfinite(options.minTrust)
          ? std::max(0.0, std::min(1.0, options.minTrust))
          : 0.0;
  const double trust =
      std::max(minTrust, 1.0 / (1.0 + options.gradientScale *
                                          averageGradient));
  return geometryCost * trust;
}

double backendResponseGeometryCost(
    double geometryCost, double localGradientDotResidual,
    double targetGradientDotResidual, double localStiffness,
    double targetStiffness,
    const BackendAssociationResponseOptions &options) {
  if (!std::isfinite(geometryCost) || geometryCost <= 0.0) {
    return finiteNonnegative(geometryCost);
  }
  if (!std::isfinite(options.responseGain) ||
      options.responseGain <= 0.0) {
    return geometryCost;
  }

  const double localDot =
      std::isfinite(localGradientDotResidual) &&
              localGradientDotResidual > 0.0
          ? localGradientDotResidual
          : 0.0;
  const double targetDot =
      std::isfinite(targetGradientDotResidual) &&
              targetGradientDotResidual > 0.0
          ? targetGradientDotResidual
          : 0.0;
  const double helpfulAlignment = localDot + targetDot;
  if (helpfulAlignment <= 0.0) {
    return geometryCost;
  }

  const double localCurvature =
      std::isfinite(localStiffness) && localStiffness > 0.0
          ? localStiffness
          : 0.0;
  const double targetCurvature =
      std::isfinite(targetStiffness) && targetStiffness > 0.0
          ? targetStiffness
          : 0.0;
  const double damping =
      std::isfinite(options.curvatureDamping) &&
              options.curvatureDamping > 0.0
          ? options.curvatureDamping
          : 1.0;
  const double rawReduction =
      options.responseGain * helpfulAlignment /
      (damping + localCurvature + targetCurvature);
  if (!std::isfinite(rawReduction) || rawReduction <= 0.0) {
    return geometryCost;
  }

  const double maxFraction =
      std::isfinite(options.maxCostReductionFraction)
          ? std::max(0.0, std::min(1.0,
                                   options.maxCostReductionFraction))
          : 0.5;
  const double maxReduction = maxFraction * geometryCost;
  const double reduction = std::min(rawReduction, maxReduction);
  return finiteNonnegative(geometryCost - reduction);
}

double backendAdjointGeometryCost(
    double geometryCost, double localGradientDotResidual,
    double targetGradientDotResidual, double localStiffness,
    double targetStiffness,
    const BackendAssociationAdjointOptions &options) {
  if (!std::isfinite(geometryCost) || geometryCost <= 0.0) {
    return finiteNonnegative(geometryCost);
  }
  if (!std::isfinite(options.adjointGain) || options.adjointGain <= 0.0) {
    return geometryCost;
  }

  const double damping =
      std::isfinite(options.curvatureDamping) &&
              options.curvatureDamping > 0.0
          ? options.curvatureDamping
          : 1.0;
  const double localCurvature =
      std::isfinite(localStiffness) && localStiffness > 0.0
          ? localStiffness
          : 0.0;
  const double targetCurvature =
      std::isfinite(targetStiffness) && targetStiffness > 0.0
          ? targetStiffness
          : 0.0;
  const double localDot =
      std::isfinite(localGradientDotResidual) ? localGradientDotResidual : 0.0;
  const double targetDot =
      std::isfinite(targetGradientDotResidual) ? targetGradientDotResidual
                                               : 0.0;
  const double alignment =
      localDot / (damping + localCurvature) +
      targetDot / (damping + targetCurvature);
  const double signedDelta = -options.adjointGain * alignment;
  if (!std::isfinite(signedDelta) || signedDelta == 0.0) {
    return geometryCost;
  }

  const double maxReductionFraction =
      std::isfinite(options.maxCostReductionFraction)
          ? std::max(0.0, std::min(1.0,
                                   options.maxCostReductionFraction))
          : 0.5;
  const double maxIncreaseFraction =
      std::isfinite(options.maxCostIncreaseFraction)
          ? std::max(0.0, options.maxCostIncreaseFraction)
          : 0.5;
  if (signedDelta < 0.0) {
    if (!options.allowCostReduction) {
      return geometryCost;
    }
    const double maxReduction = maxReductionFraction * geometryCost;
    return finiteNonnegative(geometryCost -
                             std::min(-signedDelta, maxReduction));
  }
  if (!options.allowCostIncrease) {
    return geometryCost;
  }
  const double maxIncrease = maxIncreaseFraction * geometryCost;
  return geometryCost + std::min(signedDelta, maxIncrease);
}

Matrix solveBackendCoupledAdjoint(const SparseMatrix &hessian,
                                  const Matrix &gradient,
                                  double damping) {
  if (hessian.rows() != hessian.cols()) {
    throw std::invalid_argument("backend coupled adjoint Hessian must be "
                                "square");
  }
  if (gradient.cols() != hessian.cols()) {
    throw std::invalid_argument("backend coupled adjoint gradient has wrong "
                                "column count");
  }
  if (!std::isfinite(damping) || damping < 0.0) {
    throw std::invalid_argument("backend coupled adjoint damping must be "
                                "finite and nonnegative");
  }
  Matrix dense = Matrix(hessian);
  if (!dense.allFinite() || !gradient.allFinite()) {
    throw std::invalid_argument("backend coupled adjoint inputs must be "
                                "finite");
  }
  dense.diagonal().array() += damping;
  Eigen::LDLT<Matrix> ldlt(dense);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("backend coupled adjoint solve failed");
  }
  Matrix adjoint(gradient.rows(), gradient.cols());
  for (int row = 0; row < gradient.rows(); ++row) {
    adjoint.row(row) = ldlt.solve(gradient.row(row).transpose()).transpose();
  }
  if (!adjoint.allFinite()) {
    throw std::runtime_error("backend coupled adjoint solve produced a "
                             "non-finite result");
  }
  return adjoint;
}

Matrix centerDenseMatrixRowsColumns(const Matrix &input, bool centerRows,
                                    bool centerColumns) {
  Matrix centered = input;
  if (centered.size() == 0) {
    return centered;
  }
  const int passes = centerRows && centerColumns ? 4 : 1;
  for (int pass = 0; pass < passes; ++pass) {
    if (centerRows) {
      for (int row = 0; row < centered.rows(); ++row) {
        double sum = 0.0;
        int count = 0;
        for (int col = 0; col < centered.cols(); ++col) {
          const double value = centered(row, col);
          if (std::isfinite(value)) {
            sum += value;
            ++count;
          }
        }
        if (count > 0) {
          const double mean = sum / static_cast<double>(count);
          for (int col = 0; col < centered.cols(); ++col) {
            if (std::isfinite(centered(row, col))) {
              centered(row, col) -= mean;
            }
          }
        }
      }
    }
    if (centerColumns) {
      for (int col = 0; col < centered.cols(); ++col) {
        double sum = 0.0;
        int count = 0;
        for (int row = 0; row < centered.rows(); ++row) {
          const double value = centered(row, col);
          if (std::isfinite(value)) {
            sum += value;
            ++count;
          }
        }
        if (count > 0) {
          const double mean = sum / static_cast<double>(count);
          for (int row = 0; row < centered.rows(); ++row) {
            if (std::isfinite(centered(row, col))) {
              centered(row, col) -= mean;
            }
          }
        }
      }
    }
  }
  return centered;
}

DistributedUpperTrackingResult updateDistributedUpperTracker(
    DistributedUpperTrackingState &state, const Matrix &localGradient,
    const std::vector<Matrix> &neighborTrackers,
    const DistributedUpperTrackingOptions &options) {
  if (!std::isfinite(options.consensusWeight) ||
      options.consensusWeight < 0.0 || options.consensusWeight > 1.0) {
    throw std::invalid_argument(
        "DistributedUpperTrackingOptions.consensusWeight must be in [0, 1]");
  }
  if (!std::isfinite(options.correctionWeight) ||
      options.correctionWeight < 0.0) {
    throw std::invalid_argument(
        "DistributedUpperTrackingOptions.correctionWeight must be finite and "
        "nonnegative");
  }

  DistributedUpperTrackingResult result;
  if (localGradient.size() == 0 || localGradient.rows() == 0 ||
      localGradient.cols() == 0) {
    return result;
  }

  Matrix activeLocalGradient = localGradient;
  std::vector<unsigned char> activeMask(
      static_cast<std::size_t>(localGradient.size()), 0);
  bool hasActiveEntry = false;
  for (int col = 0; col < activeLocalGradient.cols(); ++col) {
    for (int row = 0; row < activeLocalGradient.rows(); ++row) {
      const std::size_t index =
          static_cast<std::size_t>(row + col * activeLocalGradient.rows());
      if (std::isfinite(activeLocalGradient(row, col))) {
        activeMask[index] = 1;
        hasActiveEntry = true;
      } else {
        activeLocalGradient(row, col) = 0.0;
      }
    }
  }
  if (!hasActiveEntry) {
    return result;
  }

  if (!state.initialized ||
      state.tracker.rows() != activeLocalGradient.rows() ||
      state.tracker.cols() != activeLocalGradient.cols() ||
      state.previousLocalGradient.rows() != activeLocalGradient.rows() ||
      state.previousLocalGradient.cols() != activeLocalGradient.cols()) {
    state.tracker = activeLocalGradient;
    state.previousLocalGradient = activeLocalGradient;
    state.initialized = true;
    result.trackedGradient = state.tracker;
    result.correction = Matrix::Zero(activeLocalGradient.rows(),
                                     activeLocalGradient.cols());
    result.trackerNorm = state.tracker.norm();
    return result;
  }

  Matrix neighborMean = Matrix::Zero(activeLocalGradient.rows(),
                                     activeLocalGradient.cols());
  std::size_t validNeighbors = 0;
  for (const Matrix &neighbor : neighborTrackers) {
    if (neighbor.rows() != activeLocalGradient.rows() ||
        neighbor.cols() != activeLocalGradient.cols() ||
        !neighbor.allFinite()) {
      continue;
    }
    neighborMean += neighbor;
    ++validNeighbors;
  }
  Matrix consensusTracker = state.tracker;
  if (validNeighbors > 0) {
    neighborMean /= static_cast<double>(validNeighbors);
    result.consensusResidualNorm = (state.tracker - neighborMean).norm();
    consensusTracker =
        (1.0 - options.consensusWeight) * state.tracker +
        options.consensusWeight * neighborMean;
  }

  const Matrix localDelta = activeLocalGradient - state.previousLocalGradient;
  result.localGradientDeltaNorm = localDelta.norm();
  state.tracker = consensusTracker + localDelta;
  state.previousLocalGradient = activeLocalGradient;
  result.trackedGradient = state.tracker;
  result.trackerNorm = state.tracker.norm();
  result.correction =
      options.correctionWeight * (state.tracker - activeLocalGradient);
  if (options.centerRows || options.centerColumns) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int col = 0; col < result.correction.cols(); ++col) {
      for (int row = 0; row < result.correction.rows(); ++row) {
        const std::size_t index =
            static_cast<std::size_t>(row + col * result.correction.rows());
        if (index >= activeMask.size() || activeMask[index] == 0) {
          result.correction(row, col) = nan;
        }
      }
    }
  }
  result.correction = centerDenseMatrixRowsColumns(
      result.correction, options.centerRows, options.centerColumns);
  for (int col = 0; col < result.correction.cols(); ++col) {
    for (int row = 0; row < result.correction.rows(); ++row) {
      const std::size_t index =
          static_cast<std::size_t>(row + col * result.correction.rows());
      if (index >= activeMask.size() || activeMask[index] == 0 ||
          !std::isfinite(result.correction(row, col))) {
        result.correction(row, col) = 0.0;
      }
    }
  }
  result.correctionNorm = result.correction.norm();
  return result;
}

LowerGaugeStationarityDiagnostics evaluateLowerGaugeStationarity(
    const Matrix &gradient, unsigned dimension) {
  if (dimension == 0 || gradient.rows() < static_cast<int>(dimension) ||
      gradient.cols() == 0 || !gradient.allFinite()) {
    throw std::invalid_argument(
        "lower gauge stationarity gradient has incompatible dimensions");
  }
  LowerGaugeStationarityDiagnostics diagnostics;
  diagnostics.rawGradientNorm = gradient.norm();
  const unsigned blockWidth = dimension + 1;
  const std::size_t numBlocks =
      static_cast<std::size_t>(gradient.cols()) / blockWidth;
  if (numBlocks == 0) {
    diagnostics.quotientGradientNorm = diagnostics.rawGradientNorm;
    return diagnostics;
  }

  Vector meanTranslation = Vector::Zero(gradient.rows());
  std::size_t validBlocks = 0;
  for (std::size_t block = 0; block < numBlocks; ++block) {
    const int col = static_cast<int>(block * blockWidth + dimension);
    if (col >= gradient.cols()) {
      continue;
    }
    meanTranslation += gradient.col(col);
    ++validBlocks;
  }
  if (validBlocks == 0) {
    diagnostics.quotientGradientNorm = diagnostics.rawGradientNorm;
    return diagnostics;
  }
  meanTranslation /= static_cast<double>(validBlocks);

  Matrix gaugeComponent = Matrix::Zero(gradient.rows(), gradient.cols());
  for (std::size_t block = 0; block < numBlocks; ++block) {
    const int col = static_cast<int>(block * blockWidth + dimension);
    if (col < gradient.cols()) {
      gaugeComponent.col(col) = meanTranslation;
    }
  }
  const Matrix quotient = gradient - gaugeComponent;
  diagnostics.gaugeComponentNorm = gaugeComponent.norm();
  diagnostics.quotientGradientNorm = quotient.norm();
  diagnostics.gaugeComponentRatio =
      diagnostics.rawGradientNorm > 1e-12
          ? diagnostics.gaugeComponentNorm / diagnostics.rawGradientNorm
          : 0.0;
  if (diagnostics.gaugeComponentRatio > 1.0 &&
      diagnostics.gaugeComponentRatio < 1.0 + 1e-12) {
    diagnostics.gaugeComponentRatio = 1.0;
  }
  return diagnostics;
}

LowerGaugeStationarityDiagnostics evaluateLowerGaugeStationarity(
    const Matrix &state, const Matrix &gradient, unsigned dimension) {
  if (dimension == 0 || state.rows() < static_cast<int>(dimension) ||
      state.rows() != gradient.rows() || state.cols() != gradient.cols() ||
      state.cols() == 0 ||
      state.cols() % static_cast<int>(dimension + 1u) != 0 ||
      !state.allFinite() || !gradient.allFinite()) {
    throw std::invalid_argument(
        "lower SE gauge stationarity state/gradient dimensions mismatch");
  }

  const Matrix quotient =
      projectLowerGaugeTangent(state, gradient, dimension);
  const Matrix gaugeComponent = gradient - quotient;
  LowerGaugeStationarityDiagnostics diagnostics;
  diagnostics.rawGradientNorm = gradient.norm();
  diagnostics.quotientGradientNorm = quotient.norm();
  diagnostics.gaugeComponentNorm = gaugeComponent.norm();
  diagnostics.gaugeComponentRatio =
      diagnostics.rawGradientNorm > 1e-12
          ? diagnostics.gaugeComponentNorm / diagnostics.rawGradientNorm
          : 0.0;
  if (diagnostics.gaugeComponentRatio > 1.0 &&
      diagnostics.gaugeComponentRatio < 1.0 + 1e-12) {
    diagnostics.gaugeComponentRatio = 1.0;
  }
  return diagnostics;
}

Matrix projectLowerGaugeTangent(const Matrix &state, const Matrix &tangent,
                                unsigned dimension) {
  if (dimension == 0 || state.rows() < static_cast<int>(dimension) ||
      state.rows() != tangent.rows() || state.cols() != tangent.cols() ||
      state.cols() == 0 ||
      state.cols() % static_cast<int>(dimension + 1u) != 0 ||
      !state.allFinite() || !tangent.allFinite()) {
    throw std::invalid_argument(
        "lower SE gauge projection state/tangent dimensions mismatch");
  }

  const unsigned blockWidth = dimension + 1u;
  const unsigned rank = static_cast<unsigned>(state.rows());
  const std::size_t numBlocks =
      static_cast<std::size_t>(state.cols()) / blockWidth;
  std::vector<Matrix> gaugeBasis;
  gaugeBasis.reserve(
      rank + rank * (rank - 1u) / 2u);
  const auto appendOrthonormalGaugeDirection =
      [&](Matrix direction) {
        for (const Matrix &basis : gaugeBasis) {
          const double coefficient =
              (direction.array() * basis.array()).sum();
          direction.noalias() -= coefficient * basis;
        }
        const double norm = direction.norm();
        if (std::isfinite(norm) && norm > 1e-12) {
          gaugeBasis.push_back(direction / norm);
        }
      };

  for (unsigned axis = 0; axis < rank; ++axis) {
    Matrix direction = Matrix::Zero(state.rows(), state.cols());
    for (std::size_t block = 0; block < numBlocks; ++block) {
      const int translationCol =
          static_cast<int>(block * blockWidth + dimension);
      direction(static_cast<int>(axis), translationCol) = 1.0;
    }
    appendOrthonormalGaugeDirection(std::move(direction));
  }
  for (unsigned first = 0; first < rank; ++first) {
    for (unsigned second = first + 1u; second < rank; ++second) {
      Matrix generator = Matrix::Zero(rank, rank);
      generator(static_cast<int>(first), static_cast<int>(second)) = -1.0;
      generator(static_cast<int>(second), static_cast<int>(first)) = 1.0;
      Matrix direction = Matrix::Zero(state.rows(), state.cols());
      for (std::size_t block = 0; block < numBlocks; ++block) {
        const int col = static_cast<int>(block * blockWidth);
        direction.block(0, col, rank, dimension).noalias() =
            generator * state.block(0, col, rank, dimension);
        direction.col(col + static_cast<int>(dimension)).noalias() =
            generator * state.col(col + static_cast<int>(dimension));
      }
      appendOrthonormalGaugeDirection(std::move(direction));
    }
  }

  Matrix gaugeComponent = Matrix::Zero(tangent.rows(), tangent.cols());
  for (const Matrix &basis : gaugeBasis) {
    const double coefficient = (tangent.array() * basis.array()).sum();
    gaugeComponent.noalias() += coefficient * basis;
  }
  return tangent - gaugeComponent;
}

LowerValidationAdjointResult solveLowerValidationAdjoint(
    const QuadraticProblem &lowerProblem, const Matrix &state,
    const Matrix &validationGradient, unsigned dimension,
    const LowerValidationAdjointOptions &options) {
  if (state.rows() != validationGradient.rows() ||
      state.cols() != validationGradient.cols() || !state.allFinite() ||
      !validationGradient.allFinite() ||
      lowerProblem.dimension() != dimension ||
      lowerProblem.relaxation_rank() !=
          static_cast<unsigned>(state.rows()) ||
      lowerProblem.num_poses() * (dimension + 1u) !=
          static_cast<unsigned>(state.cols())) {
    throw std::invalid_argument(
        "lower validation adjoint dimensions mismatch");
  }
  if (!std::isfinite(options.relativeTolerance) ||
      options.relativeTolerance <= 0.0) {
    throw std::invalid_argument(
        "lower validation adjoint tolerance must be finite and positive");
  }

  LowerValidationAdjointResult result;
  result.adjoint = Matrix::Zero(state.rows(), state.cols());
  const auto project = [&](const Matrix &value) {
    return options.projectGauge
               ? projectLowerGaugeTangent(state, value, dimension)
               : value;
  };
  Matrix residual = project(validationGradient);
  result.initialResidualNorm = residual.norm();
  result.finalResidualNorm = result.initialResidualNorm;
  if (result.initialResidualNorm == 0.0) {
    result.converged = true;
    return result;
  }

  double operatorScale = 1.0;
  const SparseMatrix &Q = lowerProblem.getQRef();
  for (int index = 0; index < Q.rows(); ++index) {
    operatorScale =
        std::max(operatorScale, std::abs(Q.coeff(index, index)));
  }
  result.damping =
      std::sqrt(std::numeric_limits<double>::epsilon()) * operatorScale;
  const auto applyOperator = [&](const Matrix &eta) {
    const Matrix hessianEta = options.useGaussNewtonLinearization
                                  ? lowerProblem.RieGaussNewtonHessianEta(
                                        state, eta)
                                  : lowerProblem.RieHessianEta(state, eta);
    return project(hessianEta + result.damping * eta);
  };

  const Matrix firstOperatorResidual = applyOperator(residual);
  const Matrix secondOperatorResidual =
      applyOperator(firstOperatorResidual);
  const double symmetryFirst =
      (residual.array() * secondOperatorResidual.array()).sum();
  const double symmetrySecond =
      firstOperatorResidual.squaredNorm();
  result.operatorSymmetryError =
      std::abs(symmetryFirst - symmetrySecond) /
      std::max({1.0, std::abs(symmetryFirst), std::abs(symmetrySecond)});

  using ColMajorSparseMatrix =
      Eigen::SparseMatrix<double, Eigen::ColMajor>;
  ColMajorSparseMatrix preconditionerMatrix = Q;
  for (int index = 0; index < preconditionerMatrix.rows(); ++index) {
    preconditionerMatrix.coeffRef(index, index) += result.damping;
  }
  preconditionerMatrix.makeCompressed();
  Eigen::SimplicialLDLT<ColMajorSparseMatrix> sparsePreconditioner;
  sparsePreconditioner.compute(preconditionerMatrix);
  const bool sparsePreconditionerValid =
      sparsePreconditioner.info() == Eigen::Success;
  const auto applyPreconditioner = [&](const Matrix &value) {
    Matrix preconditioned;
    if (sparsePreconditionerValid) {
      preconditioned =
          sparsePreconditioner.solve(value.transpose()).transpose();
    }
    if (!sparsePreconditionerValid ||
        sparsePreconditioner.info() != Eigen::Success ||
        preconditioned.rows() != value.rows() ||
        preconditioned.cols() != value.cols() ||
        !preconditioned.allFinite()) {
      preconditioned = value;
      for (int col = 0; col < preconditioned.cols(); ++col) {
        const double diagonal =
            std::abs(Q.coeff(col, col)) + result.damping;
        if (std::isfinite(diagonal) && diagonal > 0.0) {
          preconditioned.col(col) /= diagonal;
        }
      }
    }
    return project(preconditioned);
  };
  Matrix preconditionedResidual = applyPreconditioner(residual);
  Matrix direction = preconditionedResidual;
  double residualPreconditionedInnerProduct =
      (residual.array() * preconditionedResidual.array()).sum();
  const double targetResidual =
      options.relativeTolerance * result.initialResidualNorm;
  for (std::size_t iteration = 0; iteration < options.maxIterations;
       ++iteration) {
    const Matrix operatorDirection = applyOperator(direction);
    const double curvature =
        (direction.array() * operatorDirection.array()).sum();
    const double directionSquaredNorm = direction.squaredNorm();
    if (std::isfinite(curvature) && directionSquaredNorm > 0.0) {
      const double rayleigh = curvature / directionSquaredNorm;
      result.minimumRayleighQuotient =
          std::min(result.minimumRayleighQuotient, rayleigh);
      result.maximumRayleighQuotient =
          std::max(result.maximumRayleighQuotient, rayleigh);
    }
    if (!std::isfinite(curvature) || curvature <= 0.0) {
      result.negativeCurvature = true;
      break;
    }
    if (!std::isfinite(residualPreconditionedInnerProduct) ||
        residualPreconditionedInnerProduct <= 0.0) {
      result.negativeCurvature = true;
      break;
    }
    const double alpha = residualPreconditionedInnerProduct / curvature;
    result.adjoint.noalias() += alpha * direction;
    residual.noalias() -= alpha * operatorDirection;
    result.iterations = iteration + 1u;
    result.finalResidualNorm = residual.norm();
    if (result.finalResidualNorm <= targetResidual) {
      result.converged = true;
      break;
    }
    Matrix nextPreconditionedResidual = applyPreconditioner(residual);
    const double nextResidualPreconditionedInnerProduct =
        (residual.array() * nextPreconditionedResidual.array()).sum();
    if (!std::isfinite(nextResidualPreconditionedInnerProduct) ||
        nextResidualPreconditionedInnerProduct <= 0.0) {
      break;
    }
    const double beta = nextResidualPreconditionedInnerProduct /
                        residualPreconditionedInnerProduct;
    direction = nextPreconditionedResidual + beta * direction;
    residualPreconditionedInnerProduct =
        nextResidualPreconditionedInnerProduct;
  }
  result.adjoint = project(result.adjoint);
  return result;
}

double implicitLowerValidationHypergradient(
    const Matrix &adjoint, const Matrix &factorGradient) {
  if (adjoint.rows() != factorGradient.rows() ||
      adjoint.cols() != factorGradient.cols() || !adjoint.allFinite() ||
      !factorGradient.allFinite()) {
    throw std::invalid_argument(
        "lower validation hypergradient dimensions mismatch");
  }
  return -(adjoint.array() * factorGradient.array()).sum();
}

SoftAssignmentOptimizer::SoftAssignmentOptimizer(
    PartialSinkhornOptions options)
    : options_(options) {
  validateSinkhornOptions(options_);
}

double reciprocalFiniteCostMean(double firstCost, double secondCost) {
  if (!std::isfinite(firstCost) || !std::isfinite(secondCost)) {
    return std::numeric_limits<double>::infinity();
  }
  if ((firstCost >= 0.0 && secondCost >= 0.0) ||
      (firstCost <= 0.0 && secondCost <= 0.0)) {
    return firstCost / 2.0 + secondCost / 2.0;
  }
  return (firstCost + secondCost) / 2.0;
}

std::vector<SparseAssignmentCostEntry> normalizeSparseAssignmentCostEntries(
    int rows, const std::vector<SparseAssignmentCostEntry> &entries,
    std::size_t rowTopK, double targetRms) {
  if (rows < 0) {
    throw std::invalid_argument("sparse assignment row count must be >= 0");
  }
  if (!std::isfinite(targetRms) || targetRms <= 0.0) {
    throw std::invalid_argument(
        "sparse assignment normalization target RMS must be positive");
  }

  std::vector<std::vector<std::pair<int, double>>> rowEntries(
      static_cast<std::size_t>(rows));
  for (const SparseAssignmentCostEntry &entry : entries) {
    if (entry.row < 0 || entry.row >= rows || entry.col < 0) {
      throw std::invalid_argument(
          "sparse assignment normalization entry is outside its row shape");
    }
    if (std::isfinite(entry.cost)) {
      rowEntries[static_cast<std::size_t>(entry.row)].push_back(
          {entry.col, entry.cost});
    }
  }

  double squaredNorm = 0.0;
  std::size_t count = 0;
  for (std::vector<std::pair<int, double>> &row : rowEntries) {
    std::sort(row.begin(), row.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });
    std::vector<double> costs;
    costs.reserve(row.size());
    for (const auto &entry : row) {
      costs.push_back(entry.second);
    }
    if (rowTopK > 0 && costs.size() > rowTopK) {
      std::nth_element(costs.begin(),
                       costs.begin() + static_cast<long>(rowTopK),
                       costs.end());
      costs.resize(rowTopK);
    }
    for (const double cost : costs) {
      squaredNorm += cost * cost;
      ++count;
    }
  }
  const double norm = std::sqrt(squaredNorm);
  if (count == 0 || !std::isfinite(norm) || norm <= 0.0) {
    return entries;
  }

  const double scale =
      targetRms * std::sqrt(static_cast<double>(count)) / norm;
  std::vector<SparseAssignmentCostEntry> normalized = entries;
  for (SparseAssignmentCostEntry &entry : normalized) {
    entry.cost *= scale;
  }
  return normalized;
}

Matrix SoftAssignmentOptimizer::updateFromCosts(const Matrix &costs) const {
  return applyRegularizationGradients(
      projectPartial(initializeWeights(costs, nullptr)), nullptr);
}

Matrix SoftAssignmentOptimizer::updateFromCosts(
    const Matrix &costs, const Matrix &previous) const {
  if (previous.rows() != costs.rows() || previous.cols() != costs.cols()) {
    throw std::invalid_argument(
        "previous assignment matrix shape must match costs");
  }
  if (options_.temporalBlend > 0.0) {
    const bool preservePreviousSupport = options_.restrictToPreviousSupport;
    const Matrix proposal = projectPartial(initializeWeights(costs, nullptr));
    const Matrix projectedPrevious =
        projectPartial(previous, !preservePreviousSupport);
    Matrix blended = (1.0 - options_.temporalBlend) * proposal +
                     options_.temporalBlend * projectedPrevious;
    Matrix updated =
        applyRegularizationGradients(
            projectPartial(blended, !preservePreviousSupport),
            &projectedPrevious, !preservePreviousSupport);
    if (options_.restrictToPreviousSupport) {
      updated = applyPreviousSupportMask(updated, projectedPrevious);
      updated = projectPartial(updated, false);
    }
    return updated;
  }
  Matrix updated = applyRegularizationGradients(
      projectPartial(initializeWeights(costs, &previous),
                     !options_.restrictToPreviousSupport),
      &previous, !options_.restrictToPreviousSupport);
  if (options_.restrictToPreviousSupport) {
    updated = applyPreviousSupportMask(updated, previous);
    updated = projectPartial(updated, false);
  }
  return updated;
}

Matrix SoftAssignmentOptimizer::updateFromCostsAndGradient(
    const Matrix &costs, const Matrix &extraGradient,
    const Matrix &previous) const {
  if (extraGradient.rows() != costs.rows() ||
      extraGradient.cols() != costs.cols()) {
    throw std::invalid_argument(
        "extra gradient matrix shape must match costs");
  }
  if (std::isfinite(options_.unmatchedCost)) {
    return updateFromCosts(costs + extraGradient, previous);
  }
  Matrix proposal = updateFromCosts(costs, previous);
  proposal.noalias() -= options_.stepSize * extraGradient;
  if (options_.restrictToPreviousSupport) {
    proposal = applyPreviousSupportMask(proposal, previous);
  }
  Matrix updated = projectPartial(proposal, !options_.restrictToPreviousSupport);
  if (options_.restrictToPreviousSupport) {
    updated = applyPreviousSupportMask(updated, previous);
    updated = projectPartial(updated, false);
  }
  return updated;
}

Matrix SoftAssignmentOptimizer::updateFromCostEntries(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries) const {
  if (options_.temporalBlend > 0.0 ||
      options_.temporalKlWeight > 0.0 ||
      options_.entropyRegularizationWeight > 0.0 ||
      options_.restrictToPreviousSupport) {
    return updateFromCosts(materializeSparseCosts(rows, cols, entries));
  }
  return projectPartialEntries(
      rows, cols, initializeWeightEntries(rows, cols, entries, nullptr,
                                          nullptr));
}

Matrix SoftAssignmentOptimizer::updateFromCostEntries(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries,
    const Matrix &previous) const {
  if (previous.rows() != rows || previous.cols() != cols) {
    throw std::invalid_argument(
        "previous assignment matrix shape must match sparse costs");
  }
  if (options_.temporalBlend > 0.0 ||
      options_.temporalKlWeight > 0.0 ||
      options_.entropyRegularizationWeight > 0.0 ||
      options_.restrictToPreviousSupport) {
    return updateFromCosts(materializeSparseCosts(rows, cols, entries),
                           previous);
  }
  return projectPartialEntries(
      rows, cols, initializeWeightEntries(rows, cols, entries, &previous,
                                          nullptr));
}

Matrix SoftAssignmentOptimizer::updateFromCostEntriesAndGradient(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries,
    const Matrix &extraGradient, const Matrix &previous) const {
  if (extraGradient.rows() != rows || extraGradient.cols() != cols) {
    throw std::invalid_argument(
        "extra gradient matrix shape must match sparse costs");
  }
  if (previous.rows() != rows || previous.cols() != cols) {
    throw std::invalid_argument(
        "previous assignment matrix shape must match sparse costs");
  }
  if (!std::isfinite(options_.unmatchedCost) ||
      options_.temporalBlend > 0.0 ||
      options_.temporalKlWeight > 0.0 ||
      options_.entropyRegularizationWeight > 0.0 ||
      options_.restrictToPreviousSupport) {
    return updateFromCostsAndGradient(
        materializeSparseCosts(rows, cols, entries), extraGradient, previous);
  }
  return projectPartialEntries(
      rows, cols, initializeWeightEntries(rows, cols, entries, &previous,
                                          &extraGradient));
}

FreshUnbalancedTransportResult
SoftAssignmentOptimizer::solveFreshUnbalancedTransport(
    const Matrix &costs) const {
  std::vector<SparseAssignmentCostEntry> entries;
  entries.reserve(static_cast<std::size_t>(costs.size()));
  for (int row = 0; row < costs.rows(); ++row) {
    for (int col = 0; col < costs.cols(); ++col) {
      if (std::isfinite(costs(row, col))) {
        entries.push_back(SparseAssignmentCostEntry{row, col, costs(row, col)});
      }
    }
  }
  return solveFreshUnbalancedTransport(costs.rows(), costs.cols(), entries);
}

FreshUnbalancedTransportResult
SoftAssignmentOptimizer::solveFreshUnbalancedTransport(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries) const {
  return solveFreshUnbalancedTransport(rows, cols, entries, nullptr);
}

FreshUnbalancedTransportResult
SoftAssignmentOptimizer::solveFreshUnbalancedTransport(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries,
    std::vector<FreshUnbalancedTransportEdgeDiagnostic> *edgeDiagnostics)
    const {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("fresh transport has invalid dimensions");
  }
  if (!std::isfinite(options_.unmatchedCost)) {
    throw std::invalid_argument(
        "fresh transport unmatchedCost must be finite");
  }

  FreshUnbalancedTransportResult result;
  result.assignment = Matrix::Zero(rows, cols);
  result.rowUnmatchedMass = Vector::Zero(rows);
  result.columnUnmatchedMass = Vector::Zero(cols);
  if (edgeDiagnostics != nullptr) {
    edgeDiagnostics->clear();
  }

  if (rows == 0 && cols == 0) {
    return result;
  }

  std::vector<SparseAssignmentCostEntry> candidates;
  candidates.reserve(entries.size());
  for (const SparseAssignmentCostEntry &entry : entries) {
    if (entry.row < 0 || entry.col < 0 || entry.row >= rows ||
        entry.col >= cols) {
      throw std::invalid_argument(
          "fresh transport sparse cost entry is outside matrix shape");
    }
    if (std::isfinite(entry.cost)) {
      candidates.push_back(entry);
    }
  }

  const double scale = options_.stepSize / options_.temperature;
  struct TransportEdge {
    int row;
    int col;
    double value;
    double rowCorrection;
    double columnCorrection;
  };
  std::vector<TransportEdge> edges;
  edges.reserve(candidates.size());
  std::vector<std::vector<int>> edgesByRow(static_cast<std::size_t>(rows));
  std::vector<std::vector<int>> edgesByCol(static_cast<std::size_t>(cols));
  for (const SparseAssignmentCostEntry &candidate : candidates) {
    const int index = static_cast<int>(edges.size());
    const double score =
        finiteNonnegative(scale * (options_.unmatchedCost - candidate.cost));
    edges.push_back(
        TransportEdge{candidate.row, candidate.col, score, 0.0, 0.0});
    edgesByRow[static_cast<std::size_t>(candidate.row)].push_back(index);
    edgesByCol[static_cast<std::size_t>(candidate.col)].push_back(index);
  }
  const auto projectCappedSimplex = [](const std::vector<double> &input,
                                       double capacity) {
    std::vector<double> projected(input.size(), 0.0);
    if (capacity <= 0.0 || input.empty()) {
      return projected;
    }
    double positiveSum = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index) {
      projected[index] = finiteNonnegative(input[index]);
      positiveSum += projected[index];
    }
    if (positiveSum <= capacity) {
      return projected;
    }
    std::vector<double> sorted = input;
    std::sort(sorted.begin(), sorted.end(),
              [](double lhs, double rhs) { return lhs > rhs; });
    double prefix = 0.0;
    double threshold = 0.0;
    for (std::size_t index = 0; index < sorted.size(); ++index) {
      prefix += sorted[index];
      const double candidate =
          (prefix - capacity) / static_cast<double>(index + 1u);
      if (sorted[index] > candidate) {
        threshold = candidate;
      }
    }
    for (std::size_t index = 0; index < input.size(); ++index) {
      projected[index] = std::max(0.0, input[index] - threshold);
    }
    return projected;
  };
  const auto projectGroups = [&](const std::vector<std::vector<int>> &groups,
                                 double capacity, bool rowProjection) {
    for (const std::vector<int> &group : groups) {
      std::vector<double> shifted;
      shifted.reserve(group.size());
      for (int edgeIndex : group) {
        const TransportEdge &edge =
            edges[static_cast<std::size_t>(edgeIndex)];
        shifted.push_back(
            edge.value +
            (rowProjection ? edge.rowCorrection : edge.columnCorrection));
      }
      const std::vector<double> projected =
          projectCappedSimplex(shifted, capacity);
      for (std::size_t index = 0; index < group.size(); ++index) {
        TransportEdge &edge =
            edges[static_cast<std::size_t>(group[index])];
        double &correction = rowProjection ? edge.rowCorrection
                                           : edge.columnCorrection;
        correction = shifted[index] - projected[index];
        edge.value = projected[index];
      }
    }
  };
  const std::size_t projectionIterations =
      std::max<std::size_t>(1u, options_.iterations);
  for (std::size_t iteration = 0; iteration < projectionIterations;
       ++iteration) {
    projectGroups(edgesByRow, options_.rowCapacity, true);
    projectGroups(edgesByCol, options_.columnCapacity, false);
  }

  std::vector<double> rowMass(static_cast<std::size_t>(rows), 0.0);
  std::vector<double> columnMass(static_cast<std::size_t>(cols), 0.0);
  for (const TransportEdge &edge : edges) {
    const double mass = finiteNonnegative(edge.value);
    rowMass[static_cast<std::size_t>(edge.row)] += mass;
    columnMass[static_cast<std::size_t>(edge.col)] += mass;
    result.assignment(edge.row, edge.col) += mass;
  }
  for (int row = 0; row < rows; ++row) {
    result.rowUnmatchedMass(row) = std::max(
        0.0, options_.rowCapacity - rowMass[static_cast<std::size_t>(row)]);
    result.marginalResidualNorm = std::max(
        result.marginalResidualNorm,
        std::abs(rowMass[static_cast<std::size_t>(row)] +
                     result.rowUnmatchedMass(row) -
                 options_.rowCapacity));
  }
  for (int col = 0; col < cols; ++col) {
    result.columnUnmatchedMass(col) = std::max(
        0.0,
        options_.columnCapacity - columnMass[static_cast<std::size_t>(col)]);
    result.marginalResidualNorm = std::max(
        result.marginalResidualNorm,
        std::abs(columnMass[static_cast<std::size_t>(col)] +
                     result.columnUnmatchedMass(col) -
                 options_.columnCapacity));
  }
  if (edgeDiagnostics != nullptr) {
    edgeDiagnostics->reserve(edges.size());
    for (std::size_t index = 0; index < edges.size(); ++index) {
      const TransportEdge &edge = edges[index];
      const SparseAssignmentCostEntry &candidate = candidates[index];
      edgeDiagnostics->push_back(FreshUnbalancedTransportEdgeDiagnostic{
          edge.row,
          edge.col,
          candidate.cost,
          finiteNonnegative(
              scale * (options_.unmatchedCost - candidate.cost)),
          finiteNonnegative(edge.value),
          rowMass[static_cast<std::size_t>(edge.row)],
          columnMass[static_cast<std::size_t>(edge.col)]});
    }
  }
  result.activeCandidateCount = candidates.size();
  return result;
}

Matrix SoftAssignmentOptimizer::updateFromCostsAndReferenceTrust(
    const Matrix &costs, const Matrix &reference, double referenceWeight,
    const Matrix &previous) const {
  Matrix proposal = updateFromCosts(costs, previous);
  return applyReferenceTrust(proposal, reference, referenceWeight, previous);
}

Matrix SoftAssignmentOptimizer::updateFromCostsGradientAndReferenceTrust(
    const Matrix &costs, const Matrix &extraGradient,
    const Matrix &reference, double referenceWeight,
    const Matrix &previous) const {
  Matrix proposal = updateFromCostsAndGradient(costs, extraGradient, previous);
  return applyReferenceTrust(proposal, reference, referenceWeight, previous);
}

Matrix SoftAssignmentOptimizer::applyRegularizationGradients(
    const Matrix &proposal, const Matrix *previous,
    bool applySparsity) const {
  if (options_.temporalKlWeight <= 0.0 &&
      options_.entropyRegularizationWeight <= 0.0) {
    return proposal;
  }
  Matrix gradient = Matrix::Zero(proposal.rows(), proposal.cols());
  const double eps = options_.temporalKlEpsilon;
  if (previous != nullptr && options_.temporalKlWeight > 0.0) {
    if (previous->rows() != proposal.rows() ||
        previous->cols() != proposal.cols()) {
      throw std::invalid_argument(
          "temporal KL reference matrix shape must match proposal");
    }
    for (int row = 0; row < proposal.rows(); ++row) {
      for (int col = 0; col < proposal.cols(); ++col) {
        const double current =
            std::max(0.0, std::isfinite(proposal(row, col))
                              ? proposal(row, col)
                              : 0.0);
        const double reference =
            std::max(0.0, std::isfinite((*previous)(row, col))
                              ? (*previous)(row, col)
                              : 0.0);
        gradient(row, col) +=
            options_.temporalKlWeight *
            (std::log(current + eps) - std::log(reference + eps) + 1.0);
      }
    }
  }
  if (options_.entropyRegularizationWeight > 0.0) {
    for (int row = 0; row < proposal.rows(); ++row) {
      for (int col = 0; col < proposal.cols(); ++col) {
        const double current =
            std::max(0.0, std::isfinite(proposal(row, col))
                              ? proposal(row, col)
                              : 0.0);
        gradient(row, col) += options_.entropyRegularizationWeight *
                              (std::log(current + eps) + 1.0);
      }
    }
  }
  Matrix updated = proposal - options_.stepSize * gradient;
  return projectPartial(updated, applySparsity);
}

Matrix SoftAssignmentOptimizer::applyReferenceTrust(
    const Matrix &proposal, const Matrix &reference, double referenceWeight,
    const Matrix &previous) const {
  if (reference.rows() != proposal.rows() ||
      reference.cols() != proposal.cols() ||
      previous.rows() != proposal.rows() ||
      previous.cols() != proposal.cols()) {
    throw std::invalid_argument(
        "reference trust matrices must match proposal shape");
  }
  if (!std::isfinite(referenceWeight) || referenceWeight < 0.0) {
    throw std::invalid_argument(
        "reference trust weight must be finite and nonnegative");
  }
  if (referenceWeight == 0.0) {
    return proposal;
  }
  if (!reference.allFinite()) {
    throw std::invalid_argument("reference trust matrix must be finite");
  }
  const double proximalWeight = options_.stepSize * referenceWeight;
  if (!std::isfinite(proximalWeight) || proximalWeight < 0.0) {
    throw std::invalid_argument(
        "reference trust proximal weight must be finite and nonnegative");
  }

  Matrix trusted =
      (proposal + proximalWeight * reference) / (1.0 + proximalWeight);
  if (options_.restrictToPreviousSupport) {
    trusted = applyPreviousSupportMask(trusted, previous);
  }
  Matrix updated = projectPartial(trusted, !options_.restrictToPreviousSupport);
  if (options_.restrictToPreviousSupport) {
    updated = applyPreviousSupportMask(updated, previous);
    updated = projectPartial(updated, false);
  }
  return updated;
}

Matrix SoftAssignmentOptimizer::applyReferenceMassTrust(
    const Matrix &proposal, double referenceMass, double referenceWeight,
    const Matrix &previous) const {
  if (previous.rows() != proposal.rows() ||
      previous.cols() != proposal.cols()) {
    throw std::invalid_argument(
        "reference mass trust previous matrix must match proposal shape");
  }
  if (!proposal.allFinite()) {
    throw std::invalid_argument(
        "reference mass trust proposal matrix must be finite");
  }
  if (!std::isfinite(referenceMass) || referenceMass < 0.0) {
    throw std::invalid_argument(
        "reference mass must be finite and nonnegative");
  }
  if (!std::isfinite(referenceWeight) || referenceWeight < 0.0) {
    throw std::invalid_argument(
        "reference mass trust weight must be finite and nonnegative");
  }
  if (referenceWeight == 0.0) {
    return proposal;
  }

  double activeMass = 0.0;
  std::size_t activeCount = 0;
  for (int row = 0; row < proposal.rows(); ++row) {
    for (int col = 0; col < proposal.cols(); ++col) {
      if (options_.restrictToPreviousSupport &&
          (!std::isfinite(previous(row, col)) ||
           previous(row, col) <= options_.previousSupportThreshold)) {
        continue;
      }
      activeMass += proposal(row, col);
      ++activeCount;
    }
  }
  if (activeCount == 0) {
    return proposal;
  }

  const double correction =
      referenceWeight * (activeMass - referenceMass) /
      (1.0 + referenceWeight * static_cast<double>(activeCount));
  Matrix trusted = proposal;
  for (int row = 0; row < trusted.rows(); ++row) {
    for (int col = 0; col < trusted.cols(); ++col) {
      if (options_.restrictToPreviousSupport &&
          (!std::isfinite(previous(row, col)) ||
           previous(row, col) <= options_.previousSupportThreshold)) {
        trusted(row, col) = 0.0;
        continue;
      }
      trusted(row, col) -= correction;
    }
  }

  Matrix updated = projectPartial(trusted, !options_.restrictToPreviousSupport);
  if (options_.restrictToPreviousSupport) {
    updated = applyPreviousSupportMask(updated, previous);
    updated = projectPartial(updated, false);
  }
  return updated;
}

Matrix SoftAssignmentOptimizer::applyConfidenceWeightedReferenceMassTrust(
    const Matrix &proposal, const Matrix &reference,
    const Matrix &confidence, double referenceWeight,
    const Matrix &previous) const {
  if (reference.rows() != proposal.rows() ||
      reference.cols() != proposal.cols() ||
      confidence.rows() != proposal.rows() ||
      confidence.cols() != proposal.cols() ||
      previous.rows() != proposal.rows() ||
      previous.cols() != proposal.cols()) {
    throw std::invalid_argument(
        "confidence-weighted reference mass trust matrices must match "
        "proposal shape");
  }
  if (!proposal.allFinite() || !reference.allFinite() ||
      !confidence.allFinite()) {
    throw std::invalid_argument(
        "confidence-weighted reference mass trust inputs must be finite");
  }
  if (!std::isfinite(referenceWeight) || referenceWeight < 0.0) {
    throw std::invalid_argument(
        "confidence-weighted reference mass trust weight must be finite and "
        "nonnegative");
  }
  if (referenceWeight == 0.0 || proposal.size() == 0) {
    return proposal;
  }

  double weightedMass = 0.0;
  double referenceWeightedMass = 0.0;
  double confidenceSquaredNorm = 0.0;
  for (int row = 0; row < proposal.rows(); ++row) {
    for (int col = 0; col < proposal.cols(); ++col) {
      if (options_.restrictToPreviousSupport &&
          (!std::isfinite(previous(row, col)) ||
           previous(row, col) <= options_.previousSupportThreshold)) {
        continue;
      }
      const double gate = std::clamp(confidence(row, col), 0.0, 1.0);
      if (gate <= 0.0) {
        continue;
      }
      weightedMass += gate * proposal(row, col);
      referenceWeightedMass += gate * reference(row, col);
      confidenceSquaredNorm += gate * gate;
    }
  }
  if (confidenceSquaredNorm <= 0.0) {
    return proposal;
  }

  const double correction =
      referenceWeight * (weightedMass - referenceWeightedMass) /
      (1.0 + referenceWeight * confidenceSquaredNorm);
  Matrix trusted = proposal;
  for (int row = 0; row < trusted.rows(); ++row) {
    for (int col = 0; col < trusted.cols(); ++col) {
      if (options_.restrictToPreviousSupport &&
          (!std::isfinite(previous(row, col)) ||
           previous(row, col) <= options_.previousSupportThreshold)) {
        trusted(row, col) = 0.0;
        continue;
      }
      const double gate = std::clamp(confidence(row, col), 0.0, 1.0);
      trusted(row, col) -= correction * gate;
    }
  }

  Matrix updated = projectPartial(trusted, !options_.restrictToPreviousSupport);
  if (options_.restrictToPreviousSupport) {
    updated = applyPreviousSupportMask(updated, previous);
    updated = projectPartial(updated, false);
  }
  return updated;
}

Matrix SoftAssignmentOptimizer::applyConfidenceWeightedReferenceMassTrustOnEntries(
    const Matrix &proposal, const Matrix &reference,
    const Matrix &confidence,
    const std::vector<SparseAssignmentCostEntry> &supportEntries,
    double referenceWeight, const Matrix &previous) const {
  if (options_.restrictToPreviousSupport) {
    return applyConfidenceWeightedReferenceMassTrust(
        proposal, reference, confidence, referenceWeight, previous);
  }
  if (reference.rows() != proposal.rows() ||
      reference.cols() != proposal.cols() ||
      confidence.rows() != proposal.rows() ||
      confidence.cols() != proposal.cols() ||
      previous.rows() != proposal.rows() ||
      previous.cols() != proposal.cols()) {
    throw std::invalid_argument(
        "sparse confidence-weighted reference mass trust matrices must match "
        "proposal shape");
  }
  if (!std::isfinite(referenceWeight) || referenceWeight < 0.0) {
    throw std::invalid_argument(
        "sparse confidence-weighted reference mass trust weight must be finite "
        "and nonnegative");
  }
  if (referenceWeight == 0.0 || proposal.size() == 0 ||
      supportEntries.empty()) {
    return proposal;
  }

  double weightedMass = 0.0;
  double referenceWeightedMass = 0.0;
  double confidenceSquaredNorm = 0.0;
  for (const SparseAssignmentCostEntry &entry : supportEntries) {
    if (entry.row < 0 || entry.col < 0 ||
        entry.row >= proposal.rows() || entry.col >= proposal.cols()) {
      throw std::invalid_argument(
          "sparse reference mass trust support entry is outside matrix shape");
    }
    const double proposalValue = proposal(entry.row, entry.col);
    const double referenceValue = reference(entry.row, entry.col);
    const double rawGate = confidence(entry.row, entry.col);
    if (!std::isfinite(proposalValue) || !std::isfinite(referenceValue) ||
        !std::isfinite(rawGate)) {
      throw std::invalid_argument(
          "sparse confidence-weighted reference mass trust support values must "
          "be finite");
    }
    const double gate = std::clamp(rawGate, 0.0, 1.0);
    if (gate <= 0.0) {
      continue;
    }
    weightedMass += gate * proposalValue;
    referenceWeightedMass += gate * referenceValue;
    confidenceSquaredNorm += gate * gate;
  }
  if (confidenceSquaredNorm <= 0.0) {
    return proposal;
  }

  const double correction =
      referenceWeight * (weightedMass - referenceWeightedMass) /
      (1.0 + referenceWeight * confidenceSquaredNorm);
  std::vector<SparseAssignmentWeightEntry> trustedEntries;
  trustedEntries.reserve(supportEntries.size());
  for (const SparseAssignmentCostEntry &entry : supportEntries) {
    const double gate = std::clamp(confidence(entry.row, entry.col), 0.0, 1.0);
    double value = proposal(entry.row, entry.col) - correction * gate;
    value = finiteNonnegative(value);
    if (value > 0.0) {
      trustedEntries.push_back(
          SparseAssignmentWeightEntry{entry.row, entry.col, value});
    }
  }
  return projectPartialEntries(proposal.rows(), proposal.cols(),
                               std::move(trustedEntries),
                               !options_.restrictToPreviousSupport);
}

Matrix SoftAssignmentOptimizer::projectWeights(
    const Matrix &weights, bool applySparsity) const {
  return projectPartial(weights, applySparsity);
}

Matrix SoftAssignmentOptimizer::projectWeightsSparseSupport(
    const Matrix &weights, bool applySparsity) const {
  return projectPartialSparseSupport(weights, applySparsity);
}

Matrix SoftAssignmentOptimizer::initializeWeights(
    const Matrix &costs, const Matrix *previous) const {
  if (costs.rows() < 0 || costs.cols() < 0) {
    throw std::invalid_argument("cost matrix has invalid dimensions");
  }

  double minCost = std::numeric_limits<double>::infinity();
  for (int row = 0; row < costs.rows(); ++row) {
    for (int col = 0; col < costs.cols(); ++col) {
      const double cost = costs(row, col);
      if (std::isfinite(cost)) {
        minCost = std::min(minCost, cost);
      }
    }
  }
  if (!std::isfinite(minCost)) {
    minCost = 0.0;
  }

  Matrix weights = Matrix::Zero(costs.rows(), costs.cols());
  const double scale = options_.stepSize / options_.temperature;
  const double costAnchor =
      std::isfinite(options_.unmatchedCost) ? options_.unmatchedCost : minCost;
  for (int row = 0; row < costs.rows(); ++row) {
    for (int col = 0; col < costs.cols(); ++col) {
      const double cost = costs(row, col);
      if (!std::isfinite(cost)) {
        continue;
      }
      const double exponent =
          std::max(-745.0, std::min(50.0, -scale * (cost - costAnchor)));
      double value = std::exp(exponent);
      if (previous != nullptr) {
        value *= finiteNonnegative((*previous)(row, col));
      }
      weights(row, col) = finiteNonnegative(value);
    }
  }
  return weights;
}

Matrix SoftAssignmentOptimizer::applyPreviousSupportMask(
    const Matrix &weights, const Matrix &previous) const {
  if (previous.rows() != weights.rows() || previous.cols() != weights.cols()) {
    throw std::invalid_argument(
        "previous assignment matrix shape must match weights");
  }
  Matrix masked = weights;
  for (int row = 0; row < masked.rows(); ++row) {
    for (int col = 0; col < masked.cols(); ++col) {
      if (!std::isfinite(previous(row, col)) ||
          previous(row, col) <= options_.previousSupportThreshold) {
        masked(row, col) = 0.0;
      }
    }
  }
  return masked;
}

Matrix SoftAssignmentOptimizer::materializeSparseCosts(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries) const {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("sparse cost matrix has invalid dimensions");
  }
  Matrix costs =
      Matrix::Constant(rows, cols, std::numeric_limits<double>::infinity());
  for (const SparseAssignmentCostEntry &entry : entries) {
    if (entry.row < 0 || entry.col < 0 || entry.row >= rows ||
        entry.col >= cols) {
      throw std::invalid_argument(
          "sparse cost entry index is outside matrix shape");
    }
    if (!std::isfinite(entry.cost)) {
      continue;
    }
    costs(entry.row, entry.col) = entry.cost;
  }
  return costs;
}

std::vector<SparseAssignmentWeightEntry>
SoftAssignmentOptimizer::initializeWeightEntries(
    int rows, int cols,
    const std::vector<SparseAssignmentCostEntry> &entries,
    const Matrix *previous, const Matrix *extraGradient) const {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("sparse cost matrix has invalid dimensions");
  }
  if (previous != nullptr &&
      (previous->rows() != rows || previous->cols() != cols)) {
    throw std::invalid_argument(
        "previous assignment matrix shape must match sparse costs");
  }
  if (extraGradient != nullptr &&
      (extraGradient->rows() != rows || extraGradient->cols() != cols)) {
    throw std::invalid_argument(
        "extra gradient matrix shape must match sparse costs");
  }

  double minCost = std::numeric_limits<double>::infinity();
  for (const SparseAssignmentCostEntry &entry : entries) {
    if (entry.row < 0 || entry.col < 0 || entry.row >= rows ||
        entry.col >= cols) {
      throw std::invalid_argument(
          "sparse cost entry index is outside matrix shape");
    }
    double cost = entry.cost;
    if (extraGradient != nullptr && std::isfinite(cost)) {
      cost += (*extraGradient)(entry.row, entry.col);
    }
    if (std::isfinite(cost)) {
      minCost = std::min(minCost, cost);
    }
  }
  if (!std::isfinite(minCost)) {
    minCost = 0.0;
  }

  std::vector<SparseAssignmentWeightEntry> weights;
  weights.reserve(entries.size());
  const double scale = options_.stepSize / options_.temperature;
  const double costAnchor =
      std::isfinite(options_.unmatchedCost) ? options_.unmatchedCost : minCost;
  for (const SparseAssignmentCostEntry &entry : entries) {
    double cost = entry.cost;
    if (extraGradient != nullptr && std::isfinite(cost)) {
      cost += (*extraGradient)(entry.row, entry.col);
    }
    if (!std::isfinite(cost)) {
      continue;
    }
    const double exponent =
        std::max(-745.0, std::min(50.0, -scale * (cost - costAnchor)));
    double value = std::exp(exponent);
    if (previous != nullptr) {
      value *= finiteNonnegative((*previous)(entry.row, entry.col));
    }
    value = finiteNonnegative(value);
    if (value > 0.0) {
      weights.push_back(
          SparseAssignmentWeightEntry{entry.row, entry.col, value});
    }
  }
  return weights;
}

bool SoftAssignmentOptimizer::shouldUseSparseSupportProjection(
    const Matrix &weights) const {
  if (weights.size() < 256) {
    return false;
  }
  std::size_t activeCount = 0;
  for (int row = 0; row < weights.rows(); ++row) {
    for (int col = 0; col < weights.cols(); ++col) {
      if (finiteNonnegative(weights(row, col)) > 0.0) {
        ++activeCount;
      }
    }
  }
  return activeCount == 0 ||
         activeCount * 4 < static_cast<std::size_t>(weights.size());
}

Matrix SoftAssignmentOptimizer::projectPartialEntries(
    int rows, int cols, std::vector<SparseAssignmentWeightEntry> entries,
    bool applySparsity) const {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("sparse weight matrix has invalid dimensions");
  }
  Matrix projected = Matrix::Zero(rows, cols);
  if (rows == 0 || cols == 0 || entries.empty()) {
    return projected;
  }

  entries.erase(
      std::remove_if(entries.begin(), entries.end(),
                     [&](const SparseAssignmentWeightEntry &entry) {
                       if (entry.row < 0 || entry.col < 0 ||
                           entry.row >= rows || entry.col >= cols) {
                         throw std::invalid_argument(
                             "sparse weight entry index is outside matrix "
                             "shape");
                       }
                       return finiteNonnegative(entry.value) <= 0.0;
                     }),
      entries.end());
  if (entries.empty()) {
    return projected;
  }
  for (SparseAssignmentWeightEntry &entry : entries) {
    entry.value = finiteNonnegative(entry.value);
  }

  std::vector<std::vector<std::size_t>> rowEntries(
      static_cast<std::size_t>(rows));
  std::vector<std::vector<std::size_t>> colEntries(
      static_cast<std::size_t>(cols));
  auto rebuildIndex = [&]() {
    for (std::vector<std::size_t> &indices : rowEntries) {
      indices.clear();
    }
    for (std::vector<std::size_t> &indices : colEntries) {
      indices.clear();
    }
    for (std::size_t index = 0; index < entries.size(); ++index) {
      rowEntries[static_cast<std::size_t>(entries[index].row)].push_back(
          index);
      colEntries[static_cast<std::size_t>(entries[index].col)].push_back(
          index);
    }
  };
  auto sortRowMajor = [&]() {
    std::sort(entries.begin(), entries.end(),
              [](const SparseAssignmentWeightEntry &a,
                 const SparseAssignmentWeightEntry &b) {
                if (a.row != b.row) {
                  return a.row < b.row;
                }
                return a.col < b.col;
              });
  };
  auto dropZeroEntries = [&]() {
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                       [](const SparseAssignmentWeightEntry &entry) {
                         return !std::isfinite(entry.value) ||
                                entry.value <= 0.0;
                       }),
        entries.end());
  };
  auto scaleRows = [&](bool positiveRows) {
    if (options_.rowCapacity < 0.0) {
      return;
    }
    for (std::size_t row = 0; row < rowEntries.size(); ++row) {
      double sum = 0.0;
      for (std::size_t index : rowEntries[row]) {
        sum += entries[index].value;
      }
      const bool shouldScale =
          positiveRows ? sum > 0.0
                       : (sum > options_.rowCapacity && sum > 0.0);
      if (!shouldScale) {
        continue;
      }
      const double scale = options_.rowCapacity / sum;
      for (std::size_t index : rowEntries[row]) {
        entries[index].value *= scale;
      }
    }
  };
  auto scaleColumns = [&](bool positiveColumns) {
    if (options_.columnCapacity < 0.0) {
      return;
    }
    for (std::size_t col = 0; col < colEntries.size(); ++col) {
      double sum = 0.0;
      for (std::size_t index : colEntries[col]) {
        sum += entries[index].value;
      }
      const bool shouldScale =
          positiveColumns
              ? sum > 0.0
              : (sum > options_.columnCapacity && sum > 0.0);
      if (!shouldScale) {
        continue;
      }
      const double scale = options_.columnCapacity / sum;
      for (std::size_t index : colEntries[col]) {
        entries[index].value *= scale;
      }
    }
  };
  auto pruneLargest = [&]() {
    if (options_.maxNonzeros == 0 ||
        entries.size() <= options_.maxNonzeros) {
      return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const SparseAssignmentWeightEntry &a,
                 const SparseAssignmentWeightEntry &b) {
                if (a.value != b.value) {
                  return a.value > b.value;
                }
                if (a.row != b.row) {
                  return a.row < b.row;
                }
                return a.col < b.col;
              });
    entries.resize(options_.maxNonzeros);
    sortRowMajor();
    rebuildIndex();
  };

  rebuildIndex();
  for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
    scaleRows(false);
    scaleColumns(false);
  }
  scaleRows(false);
  scaleColumns(false);
  if (applySparsity) {
    dropZeroEntries();
    sortRowMajor();
    rebuildIndex();
    pruneLargest();
  }
  if (applySparsity && options_.renormalizeAfterSparsify &&
      options_.maxNonzeros > 0) {
    for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
      scaleRows(true);
      scaleColumns(true);
    }
    scaleRows(false);
    scaleColumns(false);
  }

  for (const SparseAssignmentWeightEntry &entry : entries) {
    projected(entry.row, entry.col) = finiteNonnegative(entry.value);
  }
  return projected;
}

Matrix SoftAssignmentOptimizer::projectPartial(
    const Matrix &weights, bool applySparsity) const {
  if (shouldUseSparseSupportProjection(weights)) {
    return projectPartialSparseSupport(weights, applySparsity);
  }
  Matrix projected = weights;
  for (int row = 0; row < projected.rows(); ++row) {
    for (int col = 0; col < projected.cols(); ++col) {
      projected(row, col) = finiteNonnegative(projected(row, col));
    }
  }

  for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
    scaleRowsToCapacity(projected, options_.rowCapacity);
    scaleColumnsToCapacity(projected, options_.columnCapacity);
  }
  scaleRowsToCapacity(projected, options_.rowCapacity);
  scaleColumnsToCapacity(projected, options_.columnCapacity);
  if (applySparsity) {
    keepLargestNonzeros(projected, options_.maxNonzeros);
  }
  if (applySparsity && options_.renormalizeAfterSparsify &&
      options_.maxNonzeros > 0) {
    for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
      scalePositiveRowsToCapacity(projected, options_.rowCapacity);
      scalePositiveColumnsToCapacity(projected, options_.columnCapacity);
    }
    scaleRowsToCapacity(projected, options_.rowCapacity);
    scaleColumnsToCapacity(projected, options_.columnCapacity);
  }
  return projected;
}

Matrix SoftAssignmentOptimizer::projectPartialSparseSupport(
    const Matrix &weights, bool applySparsity) const {
  Matrix projected = Matrix::Zero(weights.rows(), weights.cols());
  if (weights.size() == 0) {
    return projected;
  }

  std::vector<Entry> entries;
  entries.reserve(static_cast<std::size_t>(weights.size()));
  for (int row = 0; row < weights.rows(); ++row) {
    for (int col = 0; col < weights.cols(); ++col) {
      const double value = finiteNonnegative(weights(row, col));
      if (value > 0.0) {
        entries.push_back(Entry{row, col, value});
      }
    }
  }
  if (entries.empty()) {
    return projected;
  }

  std::vector<std::vector<std::size_t>> rowEntries(
      static_cast<std::size_t>(weights.rows()));
  std::vector<std::vector<std::size_t>> colEntries(
      static_cast<std::size_t>(weights.cols()));
  auto rebuildIndex = [&]() {
    for (std::vector<std::size_t> &indices : rowEntries) {
      indices.clear();
    }
    for (std::vector<std::size_t> &indices : colEntries) {
      indices.clear();
    }
    for (std::size_t index = 0; index < entries.size(); ++index) {
      rowEntries[static_cast<std::size_t>(entries[index].row)].push_back(
          index);
      colEntries[static_cast<std::size_t>(entries[index].col)].push_back(
          index);
    }
  };
  auto sortRowMajor = [&]() {
    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) {
                if (a.row != b.row) {
                  return a.row < b.row;
                }
                return a.col < b.col;
              });
  };
  auto dropZeroEntries = [&]() {
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                       [](const Entry &entry) {
                         return !std::isfinite(entry.value) ||
                                entry.value <= 0.0;
                       }),
        entries.end());
  };
  auto scaleRows = [&](bool positiveRows) {
    if (options_.rowCapacity < 0.0) {
      return;
    }
    for (std::size_t row = 0; row < rowEntries.size(); ++row) {
      double sum = 0.0;
      for (std::size_t index : rowEntries[row]) {
        sum += entries[index].value;
      }
      const bool shouldScale =
          positiveRows ? sum > 0.0
                       : (sum > options_.rowCapacity && sum > 0.0);
      if (!shouldScale) {
        continue;
      }
      const double scale = options_.rowCapacity / sum;
      for (std::size_t index : rowEntries[row]) {
        entries[index].value *= scale;
      }
    }
  };
  auto scaleColumns = [&](bool positiveColumns) {
    if (options_.columnCapacity < 0.0) {
      return;
    }
    for (std::size_t col = 0; col < colEntries.size(); ++col) {
      double sum = 0.0;
      for (std::size_t index : colEntries[col]) {
        sum += entries[index].value;
      }
      const bool shouldScale =
          positiveColumns
              ? sum > 0.0
              : (sum > options_.columnCapacity && sum > 0.0);
      if (!shouldScale) {
        continue;
      }
      const double scale = options_.columnCapacity / sum;
      for (std::size_t index : colEntries[col]) {
        entries[index].value *= scale;
      }
    }
  };
  auto pruneLargest = [&]() {
    if (options_.maxNonzeros == 0 ||
        entries.size() <= options_.maxNonzeros) {
      return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) {
                if (a.value != b.value) {
                  return a.value > b.value;
                }
                if (a.row != b.row) {
                  return a.row < b.row;
                }
                return a.col < b.col;
              });
    entries.resize(options_.maxNonzeros);
    sortRowMajor();
    rebuildIndex();
  };

  rebuildIndex();
  for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
    scaleRows(false);
    scaleColumns(false);
  }
  scaleRows(false);
  scaleColumns(false);
  if (applySparsity) {
    dropZeroEntries();
    sortRowMajor();
    rebuildIndex();
    pruneLargest();
  }
  if (applySparsity && options_.renormalizeAfterSparsify &&
      options_.maxNonzeros > 0) {
    for (std::size_t iter = 0; iter < options_.iterations; ++iter) {
      scaleRows(true);
      scaleColumns(true);
    }
    scaleRows(false);
    scaleColumns(false);
  }

  for (const Entry &entry : entries) {
    projected(entry.row, entry.col) = finiteNonnegative(entry.value);
  }
  return projected;
}

std::size_t applyObjectCopySharedBarycenter(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    std::vector<ObjectCopyConsistencyTerm> &terms) {
  if (dimension == 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  if (terms.empty()) {
    return 0u;
  }
  const unsigned blockWidth = dimension + 1;
  if (localState.rows() != static_cast<int>(dimension) ||
      localState.cols() % static_cast<int>(blockWidth) != 0) {
    throw std::invalid_argument("localState dimensions must be block aligned");
  }

  std::size_t maxLocalObjectId = 0u;
  for (const ObjectCopyConsistencyTerm &term : terms) {
    maxLocalObjectId = std::max(maxLocalObjectId, term.localObjectId);
  }
  const std::size_t numGroups = maxLocalObjectId + 1u;
  std::vector<double> groupWeight(numGroups, 0.0);
  std::vector<std::size_t> groupCount(numGroups, 0u);
  std::vector<Matrix> groupTargetSum(
      numGroups, Matrix::Zero(static_cast<int>(dimension),
                              static_cast<int>(blockWidth)));

  const auto effectiveTermWeight =
      [](const ObjectCopyConsistencyTerm &term) -> double {
    const double reliability =
        std::clamp(finiteNonnegative(term.reliability), 0.0, 1.0);
    return finiteNonnegative(term.weight) * reliability *
           finiteNonnegative(term.targetPrecisionScale);
  };

  for (const ObjectCopyConsistencyTerm &term : terms) {
    const double weight = effectiveTermWeight(term);
    if (weight <= 1e-12 || term.localObjectId >= numGroups ||
        !term.targetFrameVerified) {
      continue;
    }
    validatePoseBlock(term.targetPose, dimension);
    const std::size_t localPoseIndex = objectOffset + term.localObjectId;
    const std::size_t colOffset = localPoseIndex * blockWidth;
    if (colOffset + blockWidth >
        static_cast<std::size_t>(localState.cols())) {
      continue;
    }
    const Matrix localPose =
        localState.block(0, static_cast<int>(colOffset), dimension,
                         blockWidth);
    const Matrix alignedTarget =
        closestYawSymmetryPose(localPose, term.targetPose, dimension,
                               term.yawSymmetryOrder);
    groupTargetSum[term.localObjectId].noalias() += weight * alignedTarget;
    groupWeight[term.localObjectId] += weight;
    ++groupCount[term.localObjectId];
  }

  std::size_t fusedGroups = 0u;
  for (std::size_t group = 0; group < numGroups; ++group) {
    if (groupCount[group] < 2u || groupWeight[group] <= 1e-12) {
      continue;
    }
    const Matrix groupMean = groupTargetSum[group] / groupWeight[group];
    for (ObjectCopyConsistencyTerm &term : terms) {
      if (term.localObjectId != group || !term.targetFrameVerified) {
        continue;
      }
      term.targetPose = groupMean;
    }
    ++fusedGroups;
  }
  return fusedGroups;
}

std::size_t normalizeObjectCopyConsistencyTermsBySupportDegree(
    std::vector<ObjectCopyConsistencyTerm> &terms) {
  if (terms.empty()) {
    return 0u;
  }

  std::size_t maxLocalObjectId = 0u;
  for (const ObjectCopyConsistencyTerm &term : terms) {
    maxLocalObjectId = std::max(maxLocalObjectId, term.localObjectId);
  }
  std::vector<double> supportDegree(maxLocalObjectId + 1u, 0.0);
  std::vector<double> localPrecision(maxLocalObjectId + 1u, 0.0);
  for (const ObjectCopyConsistencyTerm &term : terms) {
    const double reliability =
        std::clamp(finiteNonnegative(term.reliability), 0.0, 1.0) *
        std::clamp(finiteNonnegative(term.supportReliabilityScale), 0.0,
                   1.0);
    const double supportWeight = finiteNonnegative(term.weight) * reliability;
    const double effectiveWeight =
        supportWeight * finiteNonnegative(term.targetPrecisionScale);
    if (effectiveWeight > 0.0 && term.localObjectId < supportDegree.size()) {
      supportDegree[term.localObjectId] += effectiveWeight;
    }
    const double precision = finiteNonnegative(term.localPrecisionScale);
    if (precision > localPrecision[term.localObjectId]) {
      localPrecision[term.localObjectId] = precision;
    }
  }

  std::size_t normalizedObjects = 0u;
  for (const double degree : supportDegree) {
    if (degree > 0.0) {
      ++normalizedObjects;
    }
  }
  for (ObjectCopyConsistencyTerm &term : terms) {
    if (term.localObjectId >= supportDegree.size()) {
      continue;
    }
    const double degree = supportDegree[term.localObjectId];
    if (degree <= 0.0 || finiteNonnegative(term.weight) <= 0.0) {
      continue;
    }
    const double precision = localPrecision[term.localObjectId];
    if (precision > 0.0) {
      term.weight *= precision / (precision + degree);
    } else {
      term.weight /= 1.0 + degree;
    }
  }
  return normalizedObjects;
}

ObjectCopyChannelInformation independentRelativeObjectCopyInformation(
    const ObjectCopyChannelInformation &local,
    const ObjectCopyChannelInformation &target) {
  const auto fuse = [](double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || lhs <= 0.0 ||
        rhs <= 0.0) {
      return 0.0;
    }
    const double denominator = lhs + rhs;
    return std::isfinite(denominator) && denominator > 0.0
               ? lhs * rhs / denominator
               : 0.0;
  };
  return ObjectCopyChannelInformation{fuse(local.rotation, target.rotation),
                                      fuse(local.translation,
                                           target.translation)};
}

ObjectCopyChannelInformation
marginalizeObjectCopyAlignmentTranslationVariance(
    const ObjectCopyChannelInformation &information,
    double alignmentTranslationVariance) {
  ObjectCopyChannelInformation marginalized = information;
  if (std::isfinite(marginalized.translation) &&
      marginalized.translation > 0.0 &&
      std::isfinite(alignmentTranslationVariance) &&
      alignmentTranslationVariance > 0.0) {
    marginalized.translation /=
        1.0 + marginalized.translation * alignmentTranslationVariance;
  }
  return marginalized;
}

void configureExactSharedObjectConsensusPenalty(
    ObjectCopyConsistencyTerm &term) {
  ObjectCopyChannelInformation information =
      independentRelativeObjectCopyInformation(term.localInformation,
                                               term.targetInformation);
  if (!std::isfinite(information.rotation) || information.rotation <= 0.0) {
    information.rotation = 1.0;
  }
  if (!std::isfinite(information.translation) ||
      information.translation <= 0.0) {
    information.translation = 1.0;
  }
  term.weight = 1.0;
  term.reliability = 1.0;
  term.supportReliabilityScale = 1.0;
  term.localPrecisionScale = 1.0;
  term.targetPrecisionScale = 1.0;
  term.targetInformation = information;
  term.useChannelInformation = true;
  term.robustResidualScale = 0.0;
  term.robustMinWeight = 0.0;
  term.robustLatentTarget = 1.0;
  term.robustLatentValue = 1.0;
}

void configureWeightedSharedObjectConsensusPenalty(
    ObjectCopyConsistencyTerm &term) {
  const double penaltyScale =
      std::isfinite(term.consensusPenaltyScale) &&
              term.consensusPenaltyScale > 0.0
          ? term.consensusPenaltyScale
          : 1.0;
  ObjectCopyChannelInformation metric =
      independentRelativeObjectCopyInformation(term.localInformation,
                                               term.targetInformation);
  if (!std::isfinite(metric.rotation) || metric.rotation <= 0.0) {
    metric.rotation = 1.0;
  }
  if (!std::isfinite(metric.translation) || metric.translation <= 0.0) {
    metric.translation = 1.0;
  }
  term.weight = penaltyScale;
  term.associationProbability =
      std::isfinite(term.associationProbability)
          ? std::clamp(term.associationProbability, 0.0, 1.0)
          : 0.0;
  term.reliability = term.associationProbability;
  term.supportReliabilityScale = 1.0;
  term.localPrecisionScale = 1.0;
  term.targetPrecisionScale = 1.0;
  term.targetInformation = metric;
  term.useChannelInformation = true;
  term.robustResidualScale = 0.0;
  term.robustMinWeight = 0.0;
  term.robustLatentTarget = 1.0;
  term.robustLatentValue = 1.0;
}

void configureDirectEdgeDualConsensusProximalFactor(
    ObjectCopyConsistencyTerm &term) {
  configureWeightedSharedObjectConsensusPenalty(term);
}

void configureExactDirectEdgeDualConsensusProximalFactor(
    ObjectCopyConsistencyTerm &term) {
  if (!std::isfinite(term.consensusPenaltyScale) ||
      term.consensusPenaltyScale <= 0.0) {
    throw std::invalid_argument(
        "direct-edge consensus penalty scale must be finite and positive");
  }
  term.consensusPenaltyScale *= 0.5;
  configureWeightedSharedObjectConsensusPenalty(term);
}

bool sharedObjectConsensusHypothesisIsActive(double associationWeight,
                                             bool boundOnly) {
  if (!std::isfinite(associationWeight) || associationWeight <= 0.0) {
    return false;
  }
  return !boundOnly || associationWeight == 1.0;
}

double whitenedObjectCopyInnovationSquared(
    double rotationResidualSquared, double translationResidualSquared,
    const ObjectCopyChannelInformation &relativeInformation) {
  const double rotationResidual =
      std::isfinite(rotationResidualSquared) && rotationResidualSquared > 0.0
          ? rotationResidualSquared
          : 0.0;
  const double translationResidual =
      std::isfinite(translationResidualSquared) &&
              translationResidualSquared > 0.0
          ? translationResidualSquared
          : 0.0;
  const double rotationInformation =
      std::isfinite(relativeInformation.rotation) &&
              relativeInformation.rotation > 0.0
          ? relativeInformation.rotation
          : 0.0;
  const double translationInformation =
      std::isfinite(relativeInformation.translation) &&
              relativeInformation.translation > 0.0
          ? relativeInformation.translation
          : 0.0;
  return rotationInformation * rotationResidual +
         translationInformation * translationResidual;
}

double dimensionCalibratedCauchyLatent(double whitenedInnovationSquared,
                                       double residualDimension) {
  if (!std::isfinite(residualDimension) || residualDimension <= 0.0) {
    throw std::invalid_argument("residual dimension must be positive");
  }
  const double innovation =
      std::isfinite(whitenedInnovationSquared) &&
              whitenedInnovationSquared > 0.0
          ? whitenedInnovationSquared
          : 0.0;
  return residualDimension / (residualDimension + innovation);
}

double dimensionCalibratedStudentTAlignmentCost(
    double translationResidualSquared, double translationRmseSquared,
    double residualDimension) {
  if (!std::isfinite(residualDimension) || residualDimension <= 0.0) {
    throw std::invalid_argument("residual dimension must be positive");
  }
  const double residual =
      std::isfinite(translationResidualSquared) &&
              translationResidualSquared > 0.0
          ? translationResidualSquared
          : 0.0;
  const double scale =
      std::isfinite(translationRmseSquared) &&
              translationRmseSquared > 0.0
          ? translationRmseSquared
          : std::numeric_limits<double>::epsilon();
  return 0.5 * (residualDimension + 1.0) *
         std::log1p(residual / scale);
}

double configureExpectedObjectCopyFactor(ObjectCopyConsistencyTerm &term) {
  const double probability =
      std::isfinite(term.associationProbability)
          ? std::clamp(term.associationProbability, 0.0, 1.0)
          : 0.0;
  term.associationProbability = probability;
  term.weight = 1.0;
  term.reliability = probability;
  term.supportReliabilityScale = 1.0;
  term.localPrecisionScale = 1.0;
  term.targetPrecisionScale = 1.0;
  term.targetInformation = independentRelativeObjectCopyInformation(
      term.localInformation, term.targetInformation);
  term.useChannelInformation = true;
  term.robustResidualScale = 0.0;
  term.robustMinWeight = 0.0;
  term.robustLatentTarget = probability;
  term.robustLatentValue = probability;
  return probability;
}

void configurePhysicalObjectCopyMajorizationFactor(
    ObjectCopyConsistencyTerm &term) {
  ObjectCopyChannelInformation information =
      independentRelativeObjectCopyInformation(term.localInformation,
                                               term.targetInformation);
  if (!std::isfinite(information.rotation) || information.rotation <= 0.0) {
    information.rotation = 1.0;
  }
  if (!std::isfinite(information.translation) ||
      information.translation <= 0.0) {
    information.translation = 1.0;
  }
  term.localPrecisionScale = 1.0;
  term.targetPrecisionScale = 1.0;
  term.localInformation = information;
  term.targetInformation = information;
  term.useChannelInformation = true;
  term.robustResidualScale = 0.0;
  term.robustMinWeight = 0.0;
  term.robustLatentTarget = 1.0;
  term.robustLatentValue = 1.0;
}

double objectCopyMetropolisDegreeScale(double localDegree,
                                       double targetDegree) {
  const auto finiteDegree = [](double degree) {
    return std::isfinite(degree) && degree > 0.0 ? degree : 0.0;
  };
  return 1.0 /
         std::max({1.0, finiteDegree(localDegree), finiteDegree(targetDegree)});
}

double objectCopyRobustLatentFixedPointResidual(
    const std::vector<ObjectCopyConsistencyTerm> &terms) {
  double squaredResidual = 0.0;
  double latentWeightSum = 0.0;
  for (const ObjectCopyConsistencyTerm &term : terms) {
    const double latentWeight =
        finiteNonnegative(term.weight) *
        finiteNonnegative(term.targetPrecisionScale);
    if (latentWeight <= 0.0 || !std::isfinite(term.robustLatentTarget) ||
        !std::isfinite(term.robustLatentValue)) {
      continue;
    }
    const double target = std::clamp(term.robustLatentTarget, 0.0, 1.0);
    const double value = std::clamp(term.robustLatentValue, 0.0, 1.0);
    const double residual = value - target;
    squaredResidual += latentWeight * residual * residual;
    latentWeightSum += latentWeight;
  }
  return latentWeightSum > 0.0
             ? std::sqrt(squaredResidual / latentWeightSum)
             : 0.0;
}

namespace {

template <typename AccumulateTerm>
ObjectCopyConsistencyStats forEachWeightedObjectCopyConsistencyTerm(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms,
    std::size_t numColumns, AccumulateTerm &&accumulateTerm) {
  if (dimension == 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  const unsigned blockWidth = dimension + 1;
  const int rank = localState.rows();
  if (rank < static_cast<int>(dimension) ||
      static_cast<std::size_t>(localState.cols()) != numColumns ||
      numColumns % blockWidth != 0u) {
    throw std::invalid_argument("localState dimensions must be block aligned");
  }

  ObjectCopyConsistencyStats stats;
  double activeRmsRobustScale = 0.0;
  {
    double squaredResidualSum = 0.0;
    std::size_t residualCount = 0u;
    for (const ObjectCopyConsistencyTerm &term : terms) {
      const double reliability =
          std::clamp(finiteNonnegative(term.reliability), 0.0, 1.0);
      const double targetPrecisionScale =
          finiteNonnegative(term.targetPrecisionScale);
      const double weight =
          finiteNonnegative(term.weight) * reliability * targetPrecisionScale;
      if (weight <= 0.0 || !(std::isfinite(term.robustResidualScale) &&
                             term.robustResidualScale < 0.0)) {
        continue;
      }
      validateLiftedPoseBlock(term.targetPose, rank, dimension);
      const std::size_t localPoseIndex = objectOffset + term.localObjectId;
      const std::size_t colOffset = localPoseIndex * blockWidth;
      if (colOffset + blockWidth > numColumns) {
        throw std::out_of_range(
            "ObjectCopyConsistencyTerm.localObjectId is out of range");
      }
      const Matrix localPose =
          localState.block(0, static_cast<int>(colOffset), rank,
                           blockWidth);
      const Matrix targetPose =
          closestYawSymmetryPose(localPose, term.targetPose, dimension,
                                 term.yawSymmetryOrder);
      const Matrix residual = localPose - targetPose;
      squaredResidualSum += residual.squaredNorm();
      ++residualCount;
    }
    if (residualCount > 1u && squaredResidualSum > 0.0) {
      activeRmsRobustScale =
          std::sqrt(squaredResidualSum / static_cast<double>(residualCount));
    }
  }
  for (const ObjectCopyConsistencyTerm &term : terms) {
    const double reliability =
        std::clamp(finiteNonnegative(term.reliability), 0.0, 1.0);
    const double targetPrecisionScale =
        finiteNonnegative(term.targetPrecisionScale);
    double weight =
        finiteNonnegative(term.weight) * reliability * targetPrecisionScale;
    if (weight <= 0.0) {
      continue;
    }
    validateLiftedPoseBlock(term.targetPose, rank, dimension);
    const std::size_t localPoseIndex = objectOffset + term.localObjectId;
    const std::size_t colOffset = localPoseIndex * blockWidth;
    if (colOffset + blockWidth > numColumns) {
      throw std::out_of_range(
          "ObjectCopyConsistencyTerm.localObjectId is out of range");
    }
    const Matrix localPose =
        localState.block(0, static_cast<int>(colOffset), rank,
                         blockWidth);
    const Matrix targetPose =
        closestYawSymmetryPose(localPose, term.targetPose, dimension,
                               term.yawSymmetryOrder);
    const Matrix residual = localPose - targetPose;
    double robustScale = finiteNonnegative(term.robustResidualScale);
    if (std::isfinite(term.robustResidualScale) &&
        term.robustResidualScale < 0.0) {
      robustScale = activeRmsRobustScale;
    }
    if (robustScale > 0.0) {
      const double normalizedResidual =
          residual.norm() / std::max(robustScale, 1e-12);
      const double lineProcess =
          1.0 / (1.0 + normalizedResidual * normalizedResidual);
      const double minWeight =
          std::clamp(finiteNonnegative(term.robustMinWeight), 0.0, 1.0);
      weight *= std::max(minWeight, lineProcess);
      if (weight <= 0.0) {
        continue;
      }
    }

    double rotationWeight = weight;
    double translationWeight = weight;
    if (term.useChannelInformation) {
      rotationWeight *= finiteNonnegative(term.targetInformation.rotation);
      translationWeight *=
          finiteNonnegative(term.targetInformation.translation);
    }
    if (rotationWeight <= 0.0 && translationWeight <= 0.0) {
      continue;
    }

    accumulateTerm(colOffset, rotationWeight, translationWeight, targetPose,
                   residual);
    stats.cost +=
        0.5 * (rotationWeight *
                   residual.leftCols(static_cast<int>(dimension)).squaredNorm() +
               translationWeight *
                   residual.col(static_cast<int>(dimension)).squaredNorm());
    stats.totalWeight +=
        (static_cast<double>(dimension) * rotationWeight +
         translationWeight) /
        static_cast<double>(blockWidth);
    ++stats.activeTerms;
  }

  return stats;
}

}  // namespace

ObjectCopyConsistencyStats addSoftObjectCopyConsistencyTerms(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms, SparseMatrix &Q,
    SparseMatrix &G) {
  const unsigned blockWidth = dimension + 1;
  if (Q.rows() != Q.cols()) {
    throw std::invalid_argument("Q must be square");
  }
  if (dimension == 0 ||
      Q.cols() % static_cast<int>(blockWidth) != 0) {
    throw std::invalid_argument("Q dimensions must be block aligned");
  }
  const int rank = localState.rows();
  if (rank < static_cast<int>(dimension)) {
    throw std::invalid_argument("localState rank must be at least dimension");
  }
  Matrix denseG;
  if (G.rows() == 0 || G.cols() == 0) {
    denseG = Matrix::Zero(rank, Q.cols());
  } else {
    if (G.rows() != rank || G.cols() != Q.cols()) {
      throw std::invalid_argument("G dimensions must match localState and Q");
    }
    denseG = Matrix(G);
  }
  const ObjectCopyConsistencyStats stats =
      forEachWeightedObjectCopyConsistencyTerm(
          objectOffset, dimension, localState, terms,
          static_cast<std::size_t>(Q.cols()),
          [&](std::size_t colOffset, double rotationWeight,
              double translationWeight,
              const Matrix &targetPose, const Matrix &) {
            for (unsigned c = 0; c < dimension; ++c) {
              Q.coeffRef(static_cast<int>(colOffset + c),
                         static_cast<int>(colOffset + c)) += rotationWeight;
            }
            Q.coeffRef(static_cast<int>(colOffset + dimension),
                       static_cast<int>(colOffset + dimension)) +=
                translationWeight;
            denseG.block(0, static_cast<int>(colOffset), rank,
                         dimension)
                .noalias() +=
                -rotationWeight *
                targetPose.leftCols(static_cast<int>(dimension));
            denseG.col(static_cast<int>(colOffset + dimension)).noalias() +=
                -translationWeight *
                targetPose.col(static_cast<int>(dimension));
          });
  Q.makeCompressed();
  G = denseG.sparseView();
  return stats;
}

void addSoftObjectCopyJacobiMajorizationTerms(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms, SparseMatrix &Q,
    SparseMatrix &G) {
  const unsigned blockWidth = dimension + 1;
  if (Q.rows() != Q.cols()) {
    throw std::invalid_argument("Q must be square");
  }
  const int rank = localState.rows();
  if (dimension == 0 ||
      Q.cols() % static_cast<int>(blockWidth) != 0 ||
      rank < static_cast<int>(dimension) ||
      localState.cols() != Q.cols()) {
    throw std::invalid_argument(
        "majorization state and Q must be block aligned");
  }
  Matrix denseG;
  if (G.rows() == 0 || G.cols() == 0) {
    denseG = Matrix::Zero(rank, Q.cols());
  } else {
    if (G.rows() != rank || G.cols() != Q.cols()) {
      throw std::invalid_argument("G dimensions must match localState and Q");
    }
    denseG = Matrix(G);
  }
  (void)forEachWeightedObjectCopyConsistencyTerm(
      objectOffset, dimension, localState, terms,
      static_cast<std::size_t>(Q.cols()),
      [&](std::size_t colOffset, double rotationWeight,
          double translationWeight, const Matrix &, const Matrix &) {
        const Matrix anchor = localState.block(
            0, static_cast<int>(colOffset), rank, blockWidth);
        for (unsigned c = 0; c < dimension; ++c) {
          Q.coeffRef(static_cast<int>(colOffset + c),
                     static_cast<int>(colOffset + c)) += rotationWeight;
        }
        Q.coeffRef(static_cast<int>(colOffset + dimension),
                   static_cast<int>(colOffset + dimension)) +=
            translationWeight;
        denseG.block(0, static_cast<int>(colOffset), rank, dimension)
            .noalias() +=
            -rotationWeight * anchor.leftCols(static_cast<int>(dimension));
        denseG.col(static_cast<int>(colOffset + dimension)).noalias() +=
            -translationWeight * anchor.col(static_cast<int>(dimension));
      });
  Q.makeCompressed();
  G = denseG.sparseView();
}

ObjectCopyConsistencyStats
accumulateSoftObjectCopyConsistencyEuclideanGradient(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms,
    Matrix &euclideanGradient) {
  if (euclideanGradient.rows() != localState.rows() ||
      euclideanGradient.cols() != localState.cols()) {
    throw std::invalid_argument(
        "euclideanGradient dimensions must match localState");
  }
  return forEachWeightedObjectCopyConsistencyTerm(
      objectOffset, dimension, localState, terms,
      static_cast<std::size_t>(localState.cols()),
      [&](std::size_t colOffset, double rotationWeight,
          double translationWeight, const Matrix &,
          const Matrix &residual) {
        euclideanGradient
            .block(0, static_cast<int>(colOffset), localState.rows(),
                   dimension)
            .noalias() +=
            rotationWeight *
            residual.leftCols(static_cast<int>(dimension));
        euclideanGradient.col(
            static_cast<int>(colOffset + dimension)).noalias() +=
            translationWeight * residual.col(static_cast<int>(dimension));
      });
}

}  // namespace DPGO
