/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/SyntheticObjectPGO.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DPGO {
namespace {

struct Pose2 {
  double x{0.0};
  double y{0.0};
  double theta{0.0};
};

struct Pose3 {
  Eigen::Vector3d t{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

struct LocalObjectSlot {
  std::size_t localSlotId{0};
  int trueObjectId{-1};
  std::string kind{"primary"};
};

constexpr double kPi = 3.14159265358979323846;

double deterministicNoise(std::size_t a, std::size_t b, std::size_t c,
                          std::size_t seed);
double deterministicUnit(std::size_t a, std::size_t b, std::size_t c,
                         std::size_t seed);

std::size_t duplicateSlotCount(const SyntheticObjectPGOOptions &options) {
  if (options.duplicateObjectRatio <= 0.0) {
    return 0;
  }
  return static_cast<std::size_t>(
      std::ceil(static_cast<double>(options.numObjects) *
                options.duplicateObjectRatio));
}

std::vector<std::size_t> permutedTrueObjectIds(
    std::size_t robotId, const SyntheticObjectPGOOptions &options) {
  std::vector<std::size_t> ids(options.numObjects);
  for (std::size_t objectId = 0; objectId < options.numObjects; ++objectId) {
    ids[objectId] = objectId;
  }
  if (!options.permuteLocalObjectIds) {
    return ids;
  }
  std::sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
    const double ka = deterministicNoise(robotId + 101, a + 17, 901,
                                         options.instanceSeed);
    const double kb = deterministicNoise(robotId + 101, b + 17, 901,
                                         options.instanceSeed);
    if (ka != kb) {
      return ka < kb;
    }
    return a < b;
  });
  return ids;
}

std::vector<LocalObjectSlot> localObjectSlotsForRobot(
    std::size_t robotId, const SyntheticObjectPGOOptions &options) {
  std::vector<LocalObjectSlot> slots;
  slots.reserve(syntheticLocalObjectSlotCount(options));
  const std::vector<std::size_t> primaryIds =
      permutedTrueObjectIds(robotId, options);
  for (std::size_t slotId = 0; slotId < primaryIds.size(); ++slotId) {
    slots.push_back(
        LocalObjectSlot{slotId, static_cast<int>(primaryIds[slotId]),
                        "primary"});
  }
  const std::size_t duplicateCount = duplicateSlotCount(options);
  for (std::size_t duplicateId = 0; duplicateId < duplicateCount;
       ++duplicateId) {
    const std::size_t trueObjectId =
        (robotId * 11 + duplicateId * 7 + 3 +
         options.instanceSeed * 23) %
        options.numObjects;
    slots.push_back(LocalObjectSlot{
        slots.size(), static_cast<int>(trueObjectId), "duplicate"});
  }
  for (std::size_t falseId = 0; falseId < options.falsePositiveObjectsPerRobot;
       ++falseId) {
    slots.push_back(LocalObjectSlot{slots.size(), -1, "false_positive"});
  }
  return slots;
}

void validateOptions(const SyntheticObjectPGOOptions &options) {
  if (options.dimension != 2 && options.dimension != 3) {
    throw std::invalid_argument(
        "SyntheticObjectPGOOptions.dimension must be 2 or 3");
  }
  if (options.numRobots == 0) {
    throw std::invalid_argument("SyntheticObjectPGOOptions.numRobots must be "
                                "positive");
  }
  if (options.numObjects == 0) {
    throw std::invalid_argument("SyntheticObjectPGOOptions.numObjects must be "
                                "positive");
  }
  if (options.posesPerRobot < 2) {
    throw std::invalid_argument(
        "SyntheticObjectPGOOptions.posesPerRobot must be at least 2");
  }
  if (!std::isfinite(options.poseSpacing) || options.poseSpacing <= 0.0 ||
      !std::isfinite(options.robotSpacing) || options.robotSpacing < 0.0 ||
      !std::isfinite(options.objectSpacing) || options.objectSpacing <= 0.0 ||
      !std::isfinite(options.objectYOffset) ||
      !std::isfinite(options.worldLength) || options.worldLength <= 0.0 ||
      !std::isfinite(options.worldWidth) || options.worldWidth <= 0.0 ||
      !std::isfinite(options.worldHeight) || options.worldHeight <= 0.0 ||
      !std::isfinite(options.detectionRange) ||
      options.detectionRange <= 0.0 ||
      !std::isfinite(options.horizontalFovDeg) ||
      options.horizontalFovDeg <= 0.0 ||
      options.horizontalFovDeg > 360.0 ||
      !std::isfinite(options.verticalFovDeg) ||
      options.verticalFovDeg <= 0.0 || options.verticalFovDeg > 180.0 ||
      !std::isfinite(options.missDetectionRate) ||
      options.missDetectionRate < 0.0 || options.missDetectionRate >= 1.0 ||
      !std::isfinite(options.observationTranslationNoise) ||
      options.observationTranslationNoise < 0.0 ||
      !std::isfinite(options.observationRotationNoise) ||
      options.observationRotationNoise < 0.0 ||
      options.objectObservationsPerObject == 0 ||
      !std::isfinite(options.duplicateObjectRatio) ||
      options.duplicateObjectRatio < 0.0 ||
      !std::isfinite(options.robotDriftPerRobot) ||
      !std::isfinite(options.objectInitialDriftPerRobot) ||
      !std::isfinite(options.semanticAccuracy) ||
      options.semanticAccuracy < 0.0 || options.semanticAccuracy > 1.0 ||
      !std::isfinite(options.descriptorNoise) ||
      options.descriptorNoise < 0.0 ||
      !std::isfinite(options.shapeNoise) || options.shapeNoise < 0.0 ||
      !std::isfinite(options.symmetryYawAmbiguityRate) ||
      options.symmetryYawAmbiguityRate < 0.0 ||
      options.symmetryYawAmbiguityRate > 1.0) {
    throw std::invalid_argument(
        "SyntheticObjectPGOOptions contains invalid geometric parameters");
  }
  if (options.numCategories == 0 || options.descriptorDim == 0) {
    throw std::invalid_argument(
        "SyntheticObjectPGOOptions metadata dimensions must be positive");
  }
}

Pose2 composeInverse(const Pose2 &a, const Pose2 &b) {
  const double c = std::cos(a.theta);
  const double s = std::sin(a.theta);
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  Pose2 relative;
  relative.x = c * dx + s * dy;
  relative.y = -s * dx + c * dy;
  relative.theta = b.theta - a.theta;
  return relative;
}

Eigen::Quaterniond eulerQuaternion(double roll, double pitch, double yaw) {
  Eigen::AngleAxisd yawRot(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchRot(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rollRot(roll, Eigen::Vector3d::UnitX());
  return Eigen::Quaterniond(yawRot * pitchRot * rollRot).normalized();
}

Pose3 composeInverse(const Pose3 &a, const Pose3 &b) {
  Pose3 relative;
  const Eigen::Quaterniond aq = a.q.normalized();
  const Eigen::Quaterniond bq = b.q.normalized();
  relative.q = (aq.conjugate() * bq).normalized();
  relative.t = aq.conjugate() * (b.t - a.t);
  return relative;
}

Pose3 compose(const Pose3 &a, const Pose3 &b) {
  Pose3 composed;
  composed.q = (a.q.normalized() * b.q.normalized()).normalized();
  composed.t = a.t + a.q.normalized() * b.t;
  return composed;
}

Pose2 trajectoryPose(std::size_t robotId, std::size_t poseId,
                     const SyntheticObjectPGOOptions &options) {
  Pose2 pose;
  pose.x = static_cast<double>(poseId) * options.poseSpacing;
  pose.y = static_cast<double>(robotId) * options.robotSpacing;
  return pose;
}

Pose3 trajectoryPose3(std::size_t robotId, std::size_t poseId,
                      const SyntheticObjectPGOOptions &options) {
  const double denom =
      static_cast<double>(std::max<std::size_t>(1, options.posesPerRobot - 1));
  const double u = static_cast<double>(poseId) / denom;
  const double lane =
      (static_cast<double>(robotId) + 0.5) /
      static_cast<double>(std::max<std::size_t>(1, options.numRobots));
  const double phase = 0.73 * static_cast<double>(robotId);
  Pose3 pose;
  pose.t.x() = 0.06 * options.worldLength + 0.88 * options.worldLength * u;
  pose.t.y() =
      std::min(options.worldWidth - 0.5,
               std::max(0.5, lane * options.worldWidth +
                                  0.12 * options.worldWidth *
                                      std::sin(2.0 * kPi * u + phase)));
  pose.t.z() =
      std::min(options.worldHeight - 0.2,
               std::max(0.4, 1.2 + 0.35 *
                                      std::sin(4.0 * kPi * u + 0.5 * phase)));

  const double dx = 0.88 * options.worldLength / denom;
  const double dy = 0.12 * options.worldWidth *
                    std::cos(2.0 * kPi * u + phase) * 2.0 * kPi / denom;
  const double dz = 0.35 * std::cos(4.0 * kPi * u + 0.5 * phase) *
                    4.0 * kPi / denom;
  const double yaw = std::atan2(dy, dx);
  const double pitch = std::atan2(dz, std::sqrt(dx * dx + dy * dy));
  const double roll = 0.03 * std::sin(2.0 * kPi * u + phase);
  pose.q = eulerQuaternion(roll, pitch, yaw);
  return pose;
}

Pose2 objectPose(std::size_t objectId,
                 const SyntheticObjectPGOOptions &options) {
  Pose2 pose;
  pose.x = (static_cast<double>(objectId) + 0.5) * options.objectSpacing;
  pose.y = options.objectYOffset;
  return pose;
}

Pose3 objectPose3(std::size_t objectId,
                  const SyntheticObjectPGOOptions &options) {
  const std::size_t cols = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(std::sqrt(
             static_cast<double>(options.numObjects) *
             options.worldLength / std::max(1.0, options.worldWidth)))));
  const std::size_t rows =
      std::max<std::size_t>(1,
                            (options.numObjects + cols - 1) / cols);
  const std::size_t col = objectId % cols;
  const std::size_t row = objectId / cols;
  const double jitterX =
      0.18 * options.worldLength / static_cast<double>(cols) *
      deterministicNoise(objectId, 3, 1, options.instanceSeed);
  const double jitterY =
      0.18 * options.worldWidth / static_cast<double>(rows) *
      deterministicNoise(objectId, 5, 2, options.instanceSeed);

  Pose3 pose;
  pose.t.x() =
      std::min(options.worldLength - 0.4,
               std::max(0.4, options.worldLength *
                                  (static_cast<double>(col) + 0.5) /
                                  static_cast<double>(cols) +
                              jitterX));
  pose.t.y() =
      std::min(options.worldWidth - 0.4,
               std::max(0.4, options.worldWidth *
                                  (static_cast<double>(row) + 0.5) /
                                  static_cast<double>(rows) +
                              jitterY));
  pose.t.z() =
      std::min(options.worldHeight - 0.2,
               std::max(0.15, 0.25 + std::fmod(0.73 *
                                                   static_cast<double>(
                                                       objectId + 1),
                                               std::max(0.5,
                                                        options.worldHeight -
                                                            0.5))));
  pose.q = eulerQuaternion(
      0.02 * deterministicNoise(objectId, 11, 0, options.instanceSeed),
      0.02 * deterministicNoise(objectId, 13, 0, options.instanceSeed),
      0.35 * static_cast<double>(objectId % 10));
  return pose;
}

Pose3 falsePositivePose3(std::size_t robotId, std::size_t falseId,
                         const SyntheticObjectPGOOptions &options) {
  const double u =
      deterministicUnit(robotId + 211, falseId + 17, 1,
                        options.instanceSeed);
  Pose3 pose;
  pose.t.x() = options.worldLength * (0.08 + 0.84 * u);
  pose.t.y() = options.worldWidth *
               (0.08 + 0.84 *
                           deterministicUnit(robotId + 211, falseId + 17, 2,
                                             options.instanceSeed));
  pose.t.z() = std::min(
      options.worldHeight - 0.2,
      std::max(0.2, options.worldHeight *
                        (0.1 + 0.8 * deterministicUnit(
                                         robotId + 211, falseId + 17, 3,
                                         options.instanceSeed))));
  pose.q = eulerQuaternion(
      0.03 * deterministicNoise(robotId, falseId, 401,
                                options.instanceSeed),
      0.03 * deterministicNoise(robotId, falseId, 402,
                                options.instanceSeed),
      2.0 * kPi * deterministicUnit(robotId, falseId, 403,
                                    options.instanceSeed));
  return pose;
}

Pose2 initialPoseWithRobotDrift(const Pose2 &pose, std::size_t robotId,
                                double drift) {
  Pose2 initial = pose;
  initial.x += drift * static_cast<double>(robotId);
  initial.y += 0.5 * drift * static_cast<double>(robotId);
  return initial;
}

Pose3 initialPoseWithRobotDrift(const Pose3 &pose, std::size_t robotId,
                                double drift) {
  Pose3 initial = pose;
  const double rid = static_cast<double>(robotId);
  initial.t += Eigen::Vector3d(drift * rid, 0.45 * drift * rid,
                               0.12 * drift * rid);
  initial.q =
      (eulerQuaternion(0.0, 0.01 * drift * rid, 0.03 * drift * rid) *
       initial.q)
          .normalized();
  return initial;
}

void writeVertex(std::ofstream &out, std::size_t id, const Pose2 &pose) {
  out << "VERTEX_SE2 " << id << " " << pose.x << " " << pose.y << " "
      << pose.theta << "\n";
}

void writeVertex(std::ofstream &out, std::size_t id, const Pose3 &pose) {
  const Eigen::Quaterniond q = pose.q.normalized();
  out << "VERTEX_SE3:QUAT " << id << " " << pose.t.x() << " "
      << pose.t.y() << " " << pose.t.z() << " " << q.x() << " "
      << q.y() << " " << q.z() << " " << q.w() << "\n";
}

void writeEdge(std::ofstream &out, std::size_t from, std::size_t to,
               const Pose2 &relative) {
  out << "EDGE_SE2 " << from << " " << to << " " << relative.x << " "
      << relative.y << " " << relative.theta
      << " 1 0 0 1 0 1\n";
}

void writeEdge(std::ofstream &out, std::size_t from, std::size_t to,
               const Pose3 &relative) {
  const Eigen::Quaterniond q = relative.q.normalized();
  out << "EDGE_SE3:QUAT " << from << " " << to << " " << relative.t.x()
      << " " << relative.t.y() << " " << relative.t.z() << " " << q.x()
      << " " << q.y() << " " << q.z() << " " << q.w()
      << " 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1\n";
}

Matrix poseBlock(const Pose2 &pose) {
  Matrix block = Matrix::Zero(2, 3);
  const double c = std::cos(pose.theta);
  const double s = std::sin(pose.theta);
  block(0, 0) = c;
  block(0, 1) = -s;
  block(1, 0) = s;
  block(1, 1) = c;
  block(0, 2) = pose.x;
  block(1, 2) = pose.y;
  return block;
}

Matrix poseBlock(const Pose3 &pose) {
  Matrix block = Matrix::Zero(3, 4);
  block.leftCols(3) = pose.q.normalized().toRotationMatrix();
  block.col(3) = pose.t;
  return block;
}

Matrix blockPose(const Matrix &state, std::size_t poseId, std::size_t d) {
  return state.block(0, static_cast<int>(poseId * (d + 1)),
                     static_cast<int>(d), static_cast<int>(d + 1));
}

Matrix invertPoseBlock(const Matrix &pose) {
  const int d = pose.rows();
  Matrix inverse = Matrix::Zero(d, d + 1);
  const Matrix rotation = pose.leftCols(d);
  const Matrix translation = pose.rightCols(1);
  inverse.leftCols(d) = rotation.transpose();
  inverse.rightCols(1) = -rotation.transpose() * translation;
  return inverse;
}

Matrix composePoseBlocks(const Matrix &a, const Matrix &b) {
  const int d = a.rows();
  Matrix composed = Matrix::Zero(d, d + 1);
  composed.leftCols(d) = a.leftCols(d) * b.leftCols(d);
  composed.rightCols(1) =
      a.leftCols(d) * b.rightCols(1) + a.rightCols(1);
  return composed;
}

void accumulatePoseError(const Matrix &estimate, const Matrix &truth,
                         double &translationSquared,
                         double &rotationSquared) {
  translationSquared +=
      (estimate.rightCols(1) - truth.rightCols(1)).squaredNorm();
  rotationSquared +=
      (estimate.leftCols(2) - truth.leftCols(2)).squaredNorm();
}

SyntheticObjectSymmetry symmetryForObject(std::size_t objectId) {
  switch (objectId % 3) {
    case 1:
      return SyntheticObjectSymmetry::C2;
    case 2:
      return SyntheticObjectSymmetry::C4;
    default:
      return SyntheticObjectSymmetry::Identity;
  }
}

double deterministicNoise(std::size_t a, std::size_t b, std::size_t c,
                          std::size_t seed) {
  const double value =
      std::sin(12.9898 *
                   static_cast<double>(a + 1 + 104729 * seed) +
               78.233 *
                   static_cast<double>(b + 1 + 130363 * seed) +
               37.719 *
                   static_cast<double>(c + 1 + 155921 * seed));
  return value;
}

Vector baseShapeForObject(std::size_t objectId,
                          const SyntheticObjectPGOOptions &options) {
  Vector shape(3);
  const double repeated =
      options.repeatedObjectRatio > 0.0
          ? std::floor(static_cast<double>(objectId) *
                       std::max(0.0, std::min(1.0,
                                              options.repeatedObjectRatio)))
          : static_cast<double>(objectId);
  shape << 0.6 + 0.05 * static_cast<double>(objectId % 5),
      0.4 + 0.04 * std::fmod(repeated, 7.0),
      0.8 + 0.03 * static_cast<double>((objectId + 2) % 6);
  return shape;
}

Vector baseDescriptorForObject(std::size_t objectId,
                               const SyntheticObjectPGOOptions &options) {
  Vector descriptor(options.descriptorDim);
  const double repeated =
      options.repeatedObjectRatio > 0.0
          ? std::floor(static_cast<double>(objectId) *
                       std::max(0.0, std::min(1.0,
                                              options.repeatedObjectRatio)))
          : static_cast<double>(objectId);
  for (std::size_t dim = 0; dim < options.descriptorDim; ++dim) {
    descriptor(static_cast<int>(dim)) =
        std::cos(0.37 * static_cast<double>(dim + 1) *
                 static_cast<double>(static_cast<std::size_t>(repeated) + 1));
  }
  const double norm = descriptor.norm();
  if (norm > 0.0) {
    descriptor /= norm;
  }
  return descriptor;
}

Vector perturbVector(const Vector &base, double scale, std::size_t robotId,
                     std::size_t objectId, std::size_t seed) {
  Vector perturbed = base;
  for (int idx = 0; idx < perturbed.rows(); ++idx) {
    perturbed(idx) += scale * deterministicNoise(robotId, objectId,
                                                 static_cast<std::size_t>(idx),
                                                 seed);
  }
  return perturbed;
}

std::size_t observedCategory(std::size_t category, std::size_t robotId,
                             std::size_t objectId,
                             const SyntheticObjectPGOOptions &options) {
  if (options.semanticAccuracy >= 1.0) {
    return category;
  }
  const double score =
      0.5 + 0.5 * deterministicNoise(robotId, objectId, 17,
                                     options.instanceSeed);
  if (score <= options.semanticAccuracy) {
    return category;
  }
  return (category + 1 + (robotId + objectId) %
                         std::max<std::size_t>(1, options.numCategories - 1)) %
         options.numCategories;
}

double deterministicUnit(std::size_t a, std::size_t b, std::size_t c,
                         std::size_t seed) {
  return 0.5 + 0.5 * deterministicNoise(a, b, c, seed);
}

const char *symmetryName(SyntheticObjectSymmetry symmetry) {
  switch (symmetry) {
    case SyntheticObjectSymmetry::Identity:
      return "identity";
    case SyntheticObjectSymmetry::C2:
      return "c2";
    case SyntheticObjectSymmetry::C4:
      return "c4";
  }
  return "identity";
}

std::size_t symmetryOrder(SyntheticObjectSymmetry symmetry) {
  switch (symmetry) {
    case SyntheticObjectSymmetry::C2:
      return 2;
    case SyntheticObjectSymmetry::C4:
      return 4;
    case SyntheticObjectSymmetry::Identity:
      return 1;
  }
  return 1;
}

Matrix yawSymmetryRotationMatrix(unsigned dimension, std::size_t branch,
                                 std::size_t order) {
  Matrix rotation = Matrix::Identity(dimension, dimension);
  if (order <= 1 || dimension < 2) {
    return rotation;
  }
  const double theta =
      2.0 * kPi * static_cast<double>(branch % order) /
      static_cast<double>(order);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  rotation(0, 0) = c;
  rotation(0, 1) = -s;
  rotation(1, 0) = s;
  rotation(1, 1) = c;
  return rotation;
}

double quotientRotationSquaredError(const Matrix &estimate,
                                    const Matrix &truth,
                                    unsigned dimension,
                                    SyntheticObjectSymmetry symmetry) {
  const Matrix estimateRotation = estimate.leftCols(dimension);
  const Matrix truthRotation = truth.leftCols(dimension);
  const std::size_t order = symmetryOrder(symmetry);
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t branch = 0; branch < order; ++branch) {
    const Matrix equivalentTruth =
        truthRotation * yawSymmetryRotationMatrix(dimension, branch, order);
    best = std::min(best, (estimateRotation - equivalentTruth).squaredNorm());
  }
  return best;
}

Pose3 applySymmetryYawAmbiguity(Pose3 relative,
                                SyntheticObjectSymmetry symmetry,
                                std::size_t robotId,
                                std::size_t objectId,
                                std::size_t poseId,
                                const SyntheticObjectPGOOptions &options) {
  const std::size_t order = symmetryOrder(symmetry);
  if (order <= 1 || options.symmetryYawAmbiguityRate <= 0.0) {
    return relative;
  }
  if (deterministicUnit(robotId, objectId, poseId + 307,
                        options.instanceSeed) >
      options.symmetryYawAmbiguityRate) {
    return relative;
  }
  const std::size_t branch =
      1 + ((31 * robotId + 17 * objectId + 13 * poseId +
            19 * options.instanceSeed) %
           (order - 1));
  const double theta =
      2.0 * kPi * static_cast<double>(branch) /
      static_cast<double>(order);
  const Eigen::Quaterniond yawBranch(
      Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));
  relative.q = (relative.q * yawBranch).normalized();
  return relative;
}

bool visibleFromPose(const Pose3 &camera, const Pose3 &object,
                     const SyntheticObjectPGOOptions &options,
                     double *distanceOut) {
  const Eigen::Vector3d delta = object.t - camera.t;
  const double distance = delta.norm();
  if (distanceOut) {
    *distanceOut = distance;
  }
  if (distance <= 1e-12 || distance > options.detectionRange) {
    return false;
  }
  const Eigen::Vector3d local = camera.q.conjugate() * delta;
  if (local.x() <= 0.0) {
    return false;
  }
  const double horizontal =
      std::abs(std::atan2(local.y(), local.x())) * 180.0 / kPi;
  const double vertical =
      std::abs(std::atan2(local.z(),
                          std::sqrt(local.x() * local.x() +
                                    local.y() * local.y()))) *
      180.0 / kPi;
  return horizontal <= 0.5 * options.horizontalFovDeg &&
         vertical <= 0.5 * options.verticalFovDeg;
}

Pose3 slotWorldPose3(const LocalObjectSlot &slot, std::size_t robotId,
                     const SyntheticObjectPGOOptions &options) {
  if (slot.trueObjectId >= 0) {
    return objectPose3(static_cast<std::size_t>(slot.trueObjectId), options);
  }
  const std::size_t falseId =
      slot.localSlotId >= options.numObjects + duplicateSlotCount(options)
          ? slot.localSlotId - options.numObjects - duplicateSlotCount(options)
          : slot.localSlotId;
  return falsePositivePose3(robotId, falseId, options);
}

std::vector<std::pair<std::size_t, double>> selectObservationAnchorPoses3(
    std::size_t robotId, const LocalObjectSlot &slot,
    const SyntheticObjectPGOOptions &options) {
  std::vector<std::pair<std::size_t, double>> selected;
  if (!options.observeAllObjects &&
      deterministicUnit(robotId, slot.localSlotId, 31,
                        options.instanceSeed) <
          options.missDetectionRate) {
    return selected;
  }

  const Pose3 object = slotWorldPose3(slot, robotId, options);
  std::vector<std::pair<double, std::size_t>> visible;
  for (std::size_t poseId = 0; poseId < options.posesPerRobot; ++poseId) {
    double distance = 0.0;
    if (options.observeAllObjects ||
        visibleFromPose(trajectoryPose3(robotId, poseId, options), object,
                        options, &distance)) {
      if (options.observeAllObjects) {
        distance =
            (object.t - trajectoryPose3(robotId, poseId, options).t).norm();
      }
      visible.emplace_back(distance, poseId);
    }
  }
  std::sort(visible.begin(), visible.end());
  const std::size_t maxObservations =
      std::min(options.objectObservationsPerObject, visible.size());
  for (std::size_t idx = 0; idx < maxObservations; ++idx) {
    selected.emplace_back(visible[idx].second, visible[idx].first);
  }
  std::sort(selected.begin(), selected.end());
  return selected;
}

Pose3 noisyObservation(const Pose3 &relative, std::size_t robotId,
                       std::size_t objectId, std::size_t poseId,
                       const SyntheticObjectPGOOptions &options,
                       SyntheticObjectSymmetry symmetry =
                           SyntheticObjectSymmetry::Identity) {
  Pose3 noisy = relative;
  const double tNoise = options.observationTranslationNoise;
  noisy.t.x() += tNoise * deterministicNoise(
                              robotId, objectId, poseId + 101,
                              options.instanceSeed);
  noisy.t.y() += tNoise * deterministicNoise(
                              robotId, objectId, poseId + 102,
                              options.instanceSeed);
  noisy.t.z() += tNoise * deterministicNoise(
                              robotId, objectId, poseId + 103,
                              options.instanceSeed);

  const double rNoise = options.observationRotationNoise;
  const Eigen::Quaterniond perturb =
      eulerQuaternion(rNoise * deterministicNoise(robotId, objectId,
                                                  poseId + 201,
                                                  options.instanceSeed),
                      rNoise * deterministicNoise(robotId, objectId,
                                                  poseId + 202,
                                                  options.instanceSeed),
                      rNoise * deterministicNoise(robotId, objectId,
                                                  poseId + 203,
                                                  options.instanceSeed));
  noisy.q = (perturb * noisy.q).normalized();
  return applySymmetryYawAmbiguity(noisy, symmetry, robotId, objectId,
                                   poseId, options);
}

void writeGroundTruthAndMetadataFiles(
    const std::filesystem::path &dirPath,
    const SyntheticObjectPGOOptions &options,
    const SyntheticObjectMetadataSet &metadata) {
  std::ofstream posesOut(dirPath / "ground_truth_poses.csv");
  std::ofstream objectsOut(dirPath / "ground_truth_objects.csv");
  std::ofstream metadataOut(dirPath / "metadata.csv");
  if (!posesOut || !objectsOut || !metadataOut) {
    throw std::runtime_error("Could not write synthetic ground-truth CSV files");
  }

  posesOut << std::setprecision(20);
  objectsOut << std::setprecision(20);
  metadataOut << std::setprecision(20);
  posesOut << "robot_id,pose_id,x,y,z,qx,qy,qz,qw\n";
  objectsOut << "object_id,x,y,z,qx,qy,qz,qw,category,symmetry\n";
  metadataOut << "object_id,category,symmetry,shape_x,shape_y,shape_z";
  for (std::size_t dim = 0; dim < options.descriptorDim; ++dim) {
    metadataOut << ",descriptor_" << dim;
  }
  metadataOut << "\n";

  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    for (std::size_t poseId = 0; poseId < options.posesPerRobot; ++poseId) {
      if (options.dimension == 3) {
        const Pose3 pose = trajectoryPose3(robotId, poseId, options);
        const Eigen::Quaterniond q = pose.q.normalized();
        posesOut << robotId << "," << poseId << "," << pose.t.x() << ","
                 << pose.t.y() << "," << pose.t.z() << "," << q.x() << ","
                 << q.y() << "," << q.z() << "," << q.w() << "\n";
      } else {
        const Pose2 pose = trajectoryPose(robotId, poseId, options);
        const Eigen::Quaterniond q =
            Eigen::Quaterniond(Eigen::AngleAxisd(pose.theta,
                                                 Eigen::Vector3d::UnitZ()));
        posesOut << robotId << "," << poseId << "," << pose.x << ","
                 << pose.y << ",0," << q.x() << "," << q.y() << ","
                 << q.z() << "," << q.w() << "\n";
      }
    }
  }

  for (std::size_t objectId = 0; objectId < options.numObjects; ++objectId) {
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (options.dimension == 3) {
      const Pose3 pose = objectPose3(objectId, options);
      t = pose.t;
      q = pose.q.normalized();
    } else {
      const Pose2 pose = objectPose(objectId, options);
      t = Eigen::Vector3d(pose.x, pose.y, 0.0);
      q = Eigen::Quaterniond(
          Eigen::AngleAxisd(pose.theta, Eigen::Vector3d::UnitZ()));
    }
    const SyntheticObjectMetadata &object = metadata.objects[objectId];
    objectsOut << objectId << "," << t.x() << "," << t.y() << ","
               << t.z() << "," << q.x() << "," << q.y() << "," << q.z()
               << "," << q.w() << "," << object.category << ","
               << symmetryName(object.symmetry) << "\n";

    metadataOut << objectId << "," << object.category << ","
                << symmetryName(object.symmetry) << "," << object.shape(0)
                << "," << object.shape(1) << "," << object.shape(2);
    for (int dim = 0; dim < object.descriptor.rows(); ++dim) {
      metadataOut << "," << object.descriptor(dim);
    }
    metadataOut << "\n";
  }
}

void writeAssociationAndDifficultyFiles(
    const std::filesystem::path &dirPath,
    const SyntheticObjectPGOOptions &options,
    const SyntheticObjectMetadataSet &metadata) {
  std::ofstream mapOut(dirPath / "object_id_map.csv");
  std::ofstream gtOut(dirPath / "gt_associations.csv");
  std::ofstream localObjectsOut(dirPath / "local_objects.csv");
  std::ofstream localMetadataOut(dirPath / "local_object_metadata.csv");
  std::ofstream difficultyOut(dirPath / "difficulty_report.csv");
  if (!mapOut || !gtOut || !localObjectsOut || !localMetadataOut ||
      !difficultyOut) {
    throw std::runtime_error("Could not write synthetic association CSV files");
  }

  mapOut << "robot_id,local_slot_id,local_vertex_id,true_object_id,slot_kind\n";
  gtOut << "receiver_robot,sender_robot,receiver_object,sender_object,weight,"
           "true_object_id\n";
  localObjectsOut << std::setprecision(20);
  localObjectsOut << "robot_id,local_slot_id,true_object_id,slot_kind,x,y,z,"
                     "category\n";
  localMetadataOut << std::setprecision(20);
  localMetadataOut << "robot_id,local_slot_id,true_object_id,slot_kind,"
                      "category,symmetry,semantic_confidence,shape_x,"
                      "shape_y,shape_z";
  for (std::size_t dim = 0; dim < options.descriptorDim; ++dim) {
    localMetadataOut << ",descriptor_" << dim;
  }
  localMetadataOut << "\n";

  std::vector<std::vector<LocalObjectSlot>> allSlots(options.numRobots);
  std::vector<std::vector<std::size_t>> trueObjectSlotCounts(
      options.numRobots, std::vector<std::size_t>(options.numObjects, 0));
  std::size_t falsePositiveSlots = 0;
  std::size_t duplicateSlots = 0;
  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    allSlots[robotId] = localObjectSlotsForRobot(robotId, options);
    for (const LocalObjectSlot &slot : allSlots[robotId]) {
      const std::size_t localVertexId = options.posesPerRobot + slot.localSlotId;
      mapOut << robotId << "," << slot.localSlotId << "," << localVertexId
             << "," << slot.trueObjectId << "," << slot.kind << "\n";
      if (slot.kind == "duplicate") {
        ++duplicateSlots;
      } else if (slot.kind == "false_positive") {
        ++falsePositiveSlots;
      }
      if (slot.trueObjectId >= 0 &&
          static_cast<std::size_t>(slot.trueObjectId) < options.numObjects) {
        ++trueObjectSlotCounts[robotId]
                              [static_cast<std::size_t>(slot.trueObjectId)];
      }

      const Pose3 pose = slotWorldPose3(slot, robotId, options);
      const SyntheticObjectObservationMetadata &observation =
          metadata.observations[robotId][slot.localSlotId];
      localObjectsOut << robotId << "," << slot.localSlotId << ","
                      << slot.trueObjectId << "," << slot.kind << ","
                      << pose.t.x() << "," << pose.t.y() << ","
                      << pose.t.z() << "," << observation.category << "\n";
      localMetadataOut << robotId << "," << slot.localSlotId << ","
                       << slot.trueObjectId << "," << slot.kind << ","
                       << observation.category << ","
                       << symmetryName(observation.symmetry) << ","
                       << observation.semanticConfidence << ","
                       << observation.shape(0) << ","
                       << observation.shape(1) << ","
                       << observation.shape(2);
      for (int dim = 0; dim < observation.descriptor.rows(); ++dim) {
        localMetadataOut << "," << observation.descriptor(dim);
      }
      localMetadataOut << "\n";
    }
  }

  std::size_t gtPairCount = 0;
  gtOut << std::setprecision(20);
  for (std::size_t receiver = 0; receiver < options.numRobots; ++receiver) {
    for (std::size_t sender = 0; sender < options.numRobots; ++sender) {
      if (receiver == sender) {
        continue;
      }
      for (const LocalObjectSlot &receiverSlot : allSlots[receiver]) {
        if (receiverSlot.trueObjectId < 0) {
          continue;
        }
        for (const LocalObjectSlot &senderSlot : allSlots[sender]) {
          if (senderSlot.trueObjectId == receiverSlot.trueObjectId) {
            const std::size_t trueObjectId =
                static_cast<std::size_t>(receiverSlot.trueObjectId);
            const std::size_t senderSlotCount =
                trueObjectSlotCounts[sender][trueObjectId];
            const double weight =
                senderSlotCount > 0
                    ? 1.0 / static_cast<double>(senderSlotCount)
                    : 0.0;
            gtOut << receiver << "," << sender << ","
                  << receiverSlot.localSlotId << ","
                  << senderSlot.localSlotId << "," << weight << ","
                  << receiverSlot.trueObjectId << "\n";
            ++gtPairCount;
          }
        }
      }
    }
  }

  difficultyOut << "key,value\n";
  difficultyOut << "instance_seed," << options.instanceSeed << "\n";
  difficultyOut << "dimension," << options.dimension << "\n";
  difficultyOut << "num_robots," << options.numRobots << "\n";
  difficultyOut << "true_objects," << options.numObjects << "\n";
  difficultyOut << "local_slots_per_robot,"
                << syntheticLocalObjectSlotCount(options) << "\n";
  difficultyOut << "primary_slots_total,"
                << options.numRobots * options.numObjects << "\n";
  difficultyOut << "duplicate_slots_total," << duplicateSlots << "\n";
  difficultyOut << "false_positive_slots_total," << falsePositiveSlots
                << "\n";
  difficultyOut << "gt_association_pairs," << gtPairCount << "\n";
  difficultyOut << "object_observations_per_object,"
                << options.objectObservationsPerObject << "\n";
  difficultyOut << "permute_local_object_ids,"
                << (options.permuteLocalObjectIds ? 1 : 0) << "\n";
  difficultyOut << "miss_detection_rate," << options.missDetectionRate
                << "\n";
  difficultyOut << "semantic_accuracy," << options.semanticAccuracy << "\n";
  difficultyOut << "descriptor_noise," << options.descriptorNoise << "\n";
  difficultyOut << "shape_noise," << options.shapeNoise << "\n";
  difficultyOut << "repeated_object_ratio," << options.repeatedObjectRatio
                << "\n";
  difficultyOut << "symmetry_yaw_ambiguity_rate,"
                << options.symmetryYawAmbiguityRate << "\n";
}

}  // namespace

std::size_t syntheticLocalObjectSlotCount(
    const SyntheticObjectPGOOptions &options) {
  return options.numObjects + duplicateSlotCount(options) +
         options.falsePositiveObjectsPerRobot;
}

void writeSyntheticObjectPGODirectory(
    const std::string &directory, const SyntheticObjectPGOOptions &options) {
  validateOptions(options);
  const std::filesystem::path dirPath(directory);
  std::filesystem::create_directories(dirPath);
  const SyntheticObjectMetadataSet metadata =
      generateSyntheticObjectMetadata(options);
  writeGroundTruthAndMetadataFiles(dirPath, options, metadata);
  writeAssociationAndDifficultyFiles(dirPath, options, metadata);

  std::ofstream observationsOut(dirPath / "observations.csv");
  if (!observationsOut) {
    throw std::runtime_error("Could not write synthetic observations.csv");
  }
  observationsOut << std::setprecision(20);
  observationsOut << "robot_id,pose_id,local_object_id,true_object_id,"
                     "slot_kind,distance,category,semantic_confidence\n";

  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    const std::filesystem::path file =
        dirPath / ("rob_" + std::to_string(robotId) + ".g2o");
    std::ofstream out(file);
    if (!out.is_open()) {
      throw std::runtime_error("Could not write synthetic g2o file: " +
                               file.string());
    }

    if (options.dimension == 3) {
      const std::vector<LocalObjectSlot> slots =
          localObjectSlotsForRobot(robotId, options);
      std::vector<std::vector<std::pair<std::size_t, double>>> slotObservations(
          slots.size());
      for (const LocalObjectSlot &slot : slots) {
        slotObservations[slot.localSlotId] =
            selectObservationAnchorPoses3(robotId, slot, options);
      }

      for (std::size_t poseId = 0; poseId < options.posesPerRobot; ++poseId) {
        writeVertex(out, poseId,
                    initialPoseWithRobotDrift(
                        trajectoryPose3(robotId, poseId, options), robotId,
                        options.robotDriftPerRobot));
      }

      for (const LocalObjectSlot &slot : slots) {
        const std::size_t objectVertexId =
            options.posesPerRobot + slot.localSlotId;
        Pose3 initial;
        if (!slotObservations[slot.localSlotId].empty()) {
          const std::size_t anchorPoseId =
              slotObservations[slot.localSlotId].front().first;
          const SyntheticObjectObservationMetadata &observation =
              metadata.observations[robotId][slot.localSlotId];
          const Pose3 relative = noisyObservation(
              composeInverse(trajectoryPose3(robotId, anchorPoseId, options),
                             slotWorldPose3(slot, robotId, options)),
              robotId, slot.localSlotId, anchorPoseId, options,
              observation.symmetry);
          initial = compose(initialPoseWithRobotDrift(
                                trajectoryPose3(robotId, anchorPoseId,
                                                options),
                                robotId, options.robotDriftPerRobot),
                            relative);
        } else if (slot.trueObjectId < 0) {
          initial = initialPoseWithRobotDrift(
              slotWorldPose3(slot, robotId, options), robotId,
              options.objectInitialDriftPerRobot);
        } else {
          initial = initialPoseWithRobotDrift(
              falsePositivePose3(robotId, slot.localSlotId + 37, options),
              robotId, options.objectInitialDriftPerRobot);
        }
        writeVertex(out, objectVertexId, initial);
      }

      for (std::size_t poseId = 1; poseId < options.posesPerRobot; ++poseId) {
        const Pose3 previous = trajectoryPose3(robotId, poseId - 1, options);
        const Pose3 current = trajectoryPose3(robotId, poseId, options);
        writeEdge(out, poseId - 1, poseId,
                  composeInverse(previous, current));
      }

      for (const LocalObjectSlot &slot : slots) {
        const std::size_t objectVertexId =
            options.posesPerRobot + slot.localSlotId;
        const SyntheticObjectObservationMetadata &observation =
            metadata.observations[robotId][slot.localSlotId];
        for (const std::pair<std::size_t, double> &anchor :
             slotObservations[slot.localSlotId]) {
          const std::size_t anchorPoseId = anchor.first;
          const double distance = anchor.second;
          const Pose3 relative = composeInverse(
              trajectoryPose3(robotId, anchorPoseId, options),
              slotWorldPose3(slot, robotId, options));
          writeEdge(out, anchorPoseId, objectVertexId,
                    noisyObservation(relative, robotId, slot.localSlotId,
                                     anchorPoseId, options,
                                     observation.symmetry));
          observationsOut << robotId << "," << anchorPoseId << ","
                          << slot.localSlotId << "," << slot.trueObjectId
                          << "," << slot.kind << "," << distance << ","
                          << observation.category << ","
                          << observation.semanticConfidence << "\n";
        }
      }
    } else {
      for (std::size_t poseId = 0; poseId < options.posesPerRobot; ++poseId) {
        writeVertex(out, poseId,
                    initialPoseWithRobotDrift(
                        trajectoryPose(robotId, poseId, options), robotId,
                        options.robotDriftPerRobot));
      }

      for (std::size_t objectId = 0; objectId < options.numObjects;
           ++objectId) {
        const std::size_t objectVertexId = options.posesPerRobot + objectId;
        writeVertex(out, objectVertexId,
                    initialPoseWithRobotDrift(
                        objectPose(objectId, options), robotId,
                        options.objectInitialDriftPerRobot));
      }

      for (std::size_t poseId = 1; poseId < options.posesPerRobot; ++poseId) {
        const Pose2 previous = trajectoryPose(robotId, poseId - 1, options);
        const Pose2 current = trajectoryPose(robotId, poseId, options);
        writeEdge(out, poseId - 1, poseId,
                  composeInverse(previous, current));
      }

      for (std::size_t objectId = 0; objectId < options.numObjects;
           ++objectId) {
        if (!options.observeAllObjects &&
            (objectId + robotId) % options.numRobots != 0) {
          continue;
        }
        const std::size_t anchorPoseId =
            std::min(objectId, options.posesPerRobot - 1);
        const std::size_t objectVertexId = options.posesPerRobot + objectId;
        writeEdge(out, anchorPoseId, objectVertexId,
                  composeInverse(
                      trajectoryPose(robotId, anchorPoseId, options),
                      objectPose(objectId, options)));
        const SyntheticObjectObservationMetadata &observation =
            metadata.observations[robotId][objectId];
        observationsOut << robotId << "," << anchorPoseId << ","
                        << objectId << "," << static_cast<int>(objectId)
                        << ",primary,0," << observation.category << ","
                        << observation.semanticConfidence << "\n";
      }
    }
  }
}

SyntheticObjectMetadataSet generateSyntheticObjectMetadata(
    const SyntheticObjectPGOOptions &options) {
  validateOptions(options);
  SyntheticObjectMetadataSet metadata;
  metadata.objects.resize(options.numObjects);
  metadata.observations.resize(options.numRobots);

  for (std::size_t objectId = 0; objectId < options.numObjects; ++objectId) {
    SyntheticObjectMetadata object;
    const double repeatClamp =
        std::max(0.0, std::min(1.0, options.repeatedObjectRatio));
    const std::size_t categorySource =
        repeatClamp > 0.0
            ? static_cast<std::size_t>(
                  std::floor(static_cast<double>(objectId) * repeatClamp))
            : objectId;
    object.category = categorySource % options.numCategories;
    object.symmetry = symmetryForObject(objectId);
    object.shape = baseShapeForObject(objectId, options);
    object.descriptor = baseDescriptorForObject(objectId, options);
    metadata.objects[objectId] = object;
  }

  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    const std::vector<LocalObjectSlot> slots =
        localObjectSlotsForRobot(robotId, options);
    metadata.observations[robotId].reserve(slots.size());
    for (const LocalObjectSlot &slot : slots) {
      const std::size_t metadataObjectId =
          slot.trueObjectId >= 0
              ? static_cast<std::size_t>(slot.trueObjectId)
              : options.numObjects + robotId *
                                         options.falsePositiveObjectsPerRobot +
                    slot.localSlotId;
      const SyntheticObjectMetadata object =
          slot.trueObjectId >= 0
              ? metadata.objects[static_cast<std::size_t>(slot.trueObjectId)]
              : SyntheticObjectMetadata{
                    (robotId + slot.localSlotId * 3) % options.numCategories,
                    SyntheticObjectSymmetry::Identity,
                    baseShapeForObject(metadataObjectId, options),
                    baseDescriptorForObject(metadataObjectId, options)};
      SyntheticObjectObservationMetadata observation;
      observation.robotId = robotId;
      observation.objectId = slot.localSlotId;
      observation.category =
          observedCategory(object.category, robotId, slot.localSlotId,
                           options);
      observation.symmetry = object.symmetry;
      observation.semanticConfidence =
          observation.category == object.category
              ? std::max(0.0, std::min(1.0, options.semanticAccuracy))
              : 1.0 - std::max(0.0, std::min(1.0,
                                             options.semanticAccuracy));
      observation.shape =
          perturbVector(object.shape, options.shapeNoise, robotId,
                        slot.localSlotId, options.instanceSeed);
      observation.descriptor = perturbVector(
          object.descriptor, options.descriptorNoise, robotId,
          slot.localSlotId, options.instanceSeed);
      const double norm = observation.descriptor.norm();
      if (norm > 0.0) {
        observation.descriptor /= norm;
      }
      metadata.observations[robotId].push_back(std::move(observation));
    }
  }
  return metadata;
}

SyntheticObjectPGOEvaluation evaluateSyntheticObjectPGOEstimates(
    const SyntheticObjectPGOOptions &options,
    const std::vector<Matrix> &estimates) {
  validateOptions(options);
  if (estimates.size() < options.numRobots) {
    throw std::invalid_argument(
        "Synthetic estimate vector has fewer entries than numRobots");
  }

  SyntheticObjectPGOEvaluation evaluation;
  double trajectoryTranslationSquared = 0.0;
  double trajectoryRotationSquared = 0.0;
  double objectTranslationSquared = 0.0;
  double objectRotationSquared = 0.0;
  double objectQuotientRotationSquared = 0.0;
  double observedObjectTranslationSquared = 0.0;
  double observedObjectRotationSquared = 0.0;
  double observedObjectQuotientRotationSquared = 0.0;
  double globalTrajectoryTranslationSquared = 0.0;
  double globalTrajectoryRotationSquared = 0.0;
  double globalObjectTranslationSquared = 0.0;
  double globalObjectRotationSquared = 0.0;
  double globalObjectQuotientRotationSquared = 0.0;
  double observedGlobalObjectTranslationSquared = 0.0;
  double observedGlobalObjectRotationSquared = 0.0;
  double observedGlobalObjectQuotientRotationSquared = 0.0;
  std::vector<std::vector<Matrix>> globalObjectTranslations(
      options.numObjects);

  const std::size_t localPoseCount =
      options.posesPerRobot + syntheticLocalObjectSlotCount(options);
  const std::size_t blockWidth = options.dimension + 1;
  const std::size_t expectedCols = localPoseCount * blockWidth;
  if (estimates[0].rows() != static_cast<int>(options.dimension) ||
      estimates[0].cols() != static_cast<int>(expectedCols) ||
      !estimates[0].allFinite()) {
    throw std::invalid_argument(
        "Synthetic estimate state has wrong shape or non-finite values");
  }

  const Matrix globalGtAnchor =
      options.dimension == 3
          ? poseBlock(trajectoryPose3(0, 0, options))
          : poseBlock(trajectoryPose(0, 0, options));
  const Matrix globalEstAnchor = blockPose(estimates[0], 0, options.dimension);
  const Matrix globalAlignment =
      composePoseBlocks(globalGtAnchor, invertPoseBlock(globalEstAnchor));

  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    const Matrix &state = estimates[robotId];
    if (state.rows() != static_cast<int>(options.dimension) ||
        state.cols() != static_cast<int>(expectedCols) ||
        !state.allFinite()) {
      throw std::invalid_argument(
          "Synthetic estimate state has wrong shape or non-finite values");
    }

    const Matrix gtAnchor =
        options.dimension == 3
            ? poseBlock(trajectoryPose3(robotId, 0, options))
            : poseBlock(trajectoryPose(robotId, 0, options));
    const Matrix estAnchor = blockPose(state, 0, options.dimension);
    const Matrix alignment =
        composePoseBlocks(gtAnchor, invertPoseBlock(estAnchor));

    for (std::size_t poseId = 0; poseId < options.posesPerRobot; ++poseId) {
      const Matrix alignedEstimate =
          composePoseBlocks(alignment,
                            blockPose(state, poseId, options.dimension));
      const Matrix truth =
          options.dimension == 3
              ? poseBlock(trajectoryPose3(robotId, poseId, options))
              : poseBlock(trajectoryPose(robotId, poseId, options));
      accumulatePoseError(alignedEstimate, truth,
                          trajectoryTranslationSquared,
                          trajectoryRotationSquared);
      const Matrix globalAlignedEstimate =
          composePoseBlocks(globalAlignment,
                            blockPose(state, poseId, options.dimension));
      accumulatePoseError(globalAlignedEstimate, truth,
                          globalTrajectoryTranslationSquared,
                          globalTrajectoryRotationSquared);
      ++evaluation.trajectoryPoseCount;
    }

    const std::vector<LocalObjectSlot> slots =
        localObjectSlotsForRobot(robotId, options);
    for (const LocalObjectSlot &slot : slots) {
      if (slot.trueObjectId < 0) {
        continue;
      }
      const Matrix alignedEstimate = composePoseBlocks(
          alignment, blockPose(state, options.posesPerRobot + slot.localSlotId,
                               options.dimension));
      const Matrix truth =
          options.dimension == 3
              ? poseBlock(objectPose3(
                    static_cast<std::size_t>(slot.trueObjectId), options))
              : poseBlock(objectPose(
                    static_cast<std::size_t>(slot.trueObjectId), options));
      const SyntheticObjectSymmetry symmetry =
          symmetryForObject(static_cast<std::size_t>(slot.trueObjectId));
      accumulatePoseError(alignedEstimate, truth, objectTranslationSquared,
                          objectRotationSquared);
      objectQuotientRotationSquared += quotientRotationSquaredError(
          alignedEstimate, truth, options.dimension, symmetry);
      const Matrix globalAlignedEstimate = composePoseBlocks(
          globalAlignment, blockPose(state, options.posesPerRobot + slot.localSlotId,
                                     options.dimension));
      accumulatePoseError(globalAlignedEstimate, truth,
                          globalObjectTranslationSquared,
                          globalObjectRotationSquared);
      globalObjectQuotientRotationSquared += quotientRotationSquaredError(
          globalAlignedEstimate, truth, options.dimension, symmetry);
      const bool observedCopy =
          options.dimension == 3
              ? !selectObservationAnchorPoses3(robotId, slot, options).empty()
              : true;
      if (observedCopy) {
        accumulatePoseError(alignedEstimate, truth,
                            observedObjectTranslationSquared,
                            observedObjectRotationSquared);
        observedObjectQuotientRotationSquared +=
            quotientRotationSquaredError(alignedEstimate, truth,
                                         options.dimension, symmetry);
        accumulatePoseError(globalAlignedEstimate, truth,
                            observedGlobalObjectTranslationSquared,
                            observedGlobalObjectRotationSquared);
        observedGlobalObjectQuotientRotationSquared +=
            quotientRotationSquaredError(globalAlignedEstimate, truth,
                                         options.dimension, symmetry);
        ++evaluation.observedObjectCopyCount;
      }
      globalObjectTranslations[static_cast<std::size_t>(slot.trueObjectId)]
          .push_back(globalAlignedEstimate.rightCols(1));
      ++evaluation.objectCopyCount;
    }
  }

  if (evaluation.trajectoryPoseCount > 0) {
    evaluation.trajectoryTranslationRmse =
        std::sqrt(trajectoryTranslationSquared /
                  static_cast<double>(evaluation.trajectoryPoseCount));
    evaluation.trajectoryRotationFrobeniusRmse =
        std::sqrt(trajectoryRotationSquared /
                  static_cast<double>(evaluation.trajectoryPoseCount));
    evaluation.globalTrajectoryTranslationRmse =
        std::sqrt(globalTrajectoryTranslationSquared /
                  static_cast<double>(evaluation.trajectoryPoseCount));
    evaluation.globalTrajectoryRotationFrobeniusRmse =
        std::sqrt(globalTrajectoryRotationSquared /
                  static_cast<double>(evaluation.trajectoryPoseCount));
  }
  if (evaluation.objectCopyCount > 0) {
    evaluation.objectTranslationRmse =
        std::sqrt(objectTranslationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
    evaluation.objectRotationFrobeniusRmse =
        std::sqrt(objectRotationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
    evaluation.objectQuotientRotationFrobeniusRmse =
        std::sqrt(objectQuotientRotationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
    evaluation.globalObjectTranslationRmse =
        std::sqrt(globalObjectTranslationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
    evaluation.globalObjectRotationFrobeniusRmse =
        std::sqrt(globalObjectRotationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
    evaluation.globalObjectQuotientRotationFrobeniusRmse =
        std::sqrt(globalObjectQuotientRotationSquared /
                  static_cast<double>(evaluation.objectCopyCount));
  }
  if (evaluation.observedObjectCopyCount > 0) {
    evaluation.observedObjectTranslationRmse =
        std::sqrt(observedObjectTranslationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
    evaluation.observedObjectRotationFrobeniusRmse =
        std::sqrt(observedObjectRotationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
    evaluation.observedObjectQuotientRotationFrobeniusRmse =
        std::sqrt(observedObjectQuotientRotationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
    evaluation.observedGlobalObjectTranslationRmse =
        std::sqrt(observedGlobalObjectTranslationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
    evaluation.observedGlobalObjectRotationFrobeniusRmse =
        std::sqrt(observedGlobalObjectRotationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
    evaluation.observedGlobalObjectQuotientRotationFrobeniusRmse =
        std::sqrt(observedGlobalObjectQuotientRotationSquared /
                  static_cast<double>(evaluation.observedObjectCopyCount));
  }

  double objectSpreadSquared = 0.0;
  for (const std::vector<Matrix> &translations : globalObjectTranslations) {
    if (translations.size() < 2) {
      continue;
    }
    Matrix mean = Matrix::Zero(options.dimension, 1);
    for (const Matrix &translation : translations) {
      mean.noalias() += translation;
    }
    mean /= static_cast<double>(translations.size());
    for (const Matrix &translation : translations) {
      objectSpreadSquared += (translation - mean).squaredNorm();
      ++evaluation.objectCopySpreadCount;
    }
  }
  if (evaluation.objectCopySpreadCount > 0) {
    evaluation.objectCopyTranslationSpreadRmse =
        std::sqrt(objectSpreadSquared /
                  static_cast<double>(evaluation.objectCopySpreadCount));
  }
  return evaluation;
}

}  // namespace DPGO
