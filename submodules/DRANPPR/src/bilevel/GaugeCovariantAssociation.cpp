#include <DPGO/GaugeCovariantAssociation.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/GeodesicSE3.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace DPGO {

namespace {

bool supportedDimension(int dimension) {
  return dimension == 2 || dimension == 3;
}

bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

Matrix planarGenerator() {
  Matrix result = Matrix::Zero(2, 2);
  result(0, 1) = -1.0;
  result(1, 0) = 1.0;
  return result;
}

bool validRotation(const Matrix &rotation, int dimension) {
  if (!supportedDimension(dimension) || rotation.rows() != dimension ||
      rotation.cols() != dimension || !rotation.allFinite()) {
    return false;
  }
  const Matrix orthogonality =
      rotation.transpose() * rotation - Matrix::Identity(dimension, dimension);
  const double orthogonalityNorm = orthogonality.norm();
  const double scale = std::max(1.0, rotation.squaredNorm());
  const double determinant = rotation.determinant();
  return std::isfinite(orthogonalityNorm) && std::isfinite(scale) &&
         std::isfinite(determinant) && orthogonalityNorm <= 1e-7 * scale &&
         determinant > 0.0;
}

bool validCompactTransform(const Matrix &transform, int dimension) {
  return supportedDimension(dimension) &&
         transform.rows() == dimension && transform.cols() == dimension + 1 &&
         transform.allFinite() &&
         validRotation(transform.leftCols(dimension), dimension);
}

Matrix compactTransform(const Matrix &rotation, const Vector &translation) {
  const int dimension = static_cast<int>(rotation.rows());
  Matrix result = Matrix::Zero(dimension, dimension + 1);
  result.leftCols(dimension) = rotation;
  result.col(dimension) = translation;
  return result;
}

Matrix inverseCompactTransform(const Matrix &transform) {
  const int dimension = transform.rows();
  const Matrix rotation = transform.leftCols(dimension);
  Matrix result = Matrix::Zero(dimension, dimension + 1);
  result.leftCols(dimension) = rotation.transpose();
  result.col(dimension).noalias() =
      -rotation.transpose() * transform.col(dimension);
  return result;
}

Matrix composeCompactTransforms(const Matrix &first, const Matrix &second) {
  const int dimension = first.rows();
  Matrix result = Matrix::Zero(dimension, dimension + 1);
  result.leftCols(dimension).noalias() =
      first.leftCols(dimension) * second.leftCols(dimension);
  result.col(dimension).noalias() =
      first.leftCols(dimension) * second.col(dimension) +
      first.col(dimension);
  result.leftCols(dimension) =
      projectToRotationGroup(result.leftCols(dimension));
  return result;
}

Matrix identityCompactTransform(int dimension) {
  Matrix result = Matrix::Zero(dimension, dimension + 1);
  result.leftCols(dimension).setIdentity();
  return result;
}

Matrix pointJacobian(const Matrix &rotation, const Vector &point) {
  const int dimension = static_cast<int>(rotation.rows());
  const int tangentDimension = dimension == 3 ? 6 : 3;
  Matrix jacobian = Matrix::Zero(dimension, tangentDimension);
  if (dimension == 3) {
    jacobian.leftCols(3).noalias() = -rotation * skew3(point);
    jacobian.rightCols(3) = rotation;
  } else {
    jacobian.col(0).noalias() = rotation * planarGenerator() * point;
    jacobian.rightCols(2) = rotation;
  }
  return jacobian;
}

int tangentDimension(int dimension) {
  return dimension == 3 ? 6 : 3;
}

bool numericallyPositiveDefinite(const Matrix &matrix, int dimension) {
  const int tangent = tangentDimension(dimension);
  if (matrix.rows() != tangent || matrix.cols() != tangent ||
      !matrix.allFinite()) {
    return false;
  }
  const Matrix symmetric = 0.5 * (matrix + matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric);
  if (solver.info() != Eigen::Success ||
      solver.eigenvalues().size() != tangent ||
      !solver.eigenvalues().allFinite()) {
    return false;
  }
  const double largest = solver.eigenvalues().maxCoeff();
  const double smallest = solver.eigenvalues().minCoeff();
  if (!std::isfinite(largest) || !std::isfinite(smallest) ||
      largest <= 0.0) {
    return false;
  }
  const double threshold =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(tangent) * std::max(1.0, largest);
  return smallest > threshold;
}

struct PointLayout {
  int dimension{0};
  int receiverCount{0};
  int senderCount{0};
  bool pointsAreRows{true};
};

bool determinePointLayout(const Matrix &receiverPoints,
                          const Matrix &senderPoints, PointLayout &layout) {
  if (receiverPoints.cols() == senderPoints.cols() &&
      supportedDimension(static_cast<int>(receiverPoints.cols()))) {
    layout.dimension = static_cast<int>(receiverPoints.cols());
    layout.receiverCount = static_cast<int>(receiverPoints.rows());
    layout.senderCount = static_cast<int>(senderPoints.rows());
    layout.pointsAreRows = true;
    return true;
  }
  if (receiverPoints.rows() == senderPoints.rows() &&
      supportedDimension(static_cast<int>(receiverPoints.rows()))) {
    layout.dimension = static_cast<int>(receiverPoints.rows());
    layout.receiverCount = static_cast<int>(receiverPoints.cols());
    layout.senderCount = static_cast<int>(senderPoints.cols());
    layout.pointsAreRows = false;
    return true;
  }
  return false;
}

Vector pointAt(const Matrix &points, int index, const PointLayout &layout) {
  if (layout.pointsAreRows) {
    return points.row(index).transpose();
  }
  return points.col(index);
}

struct AlignmentMatch {
  Vector receiverPoint;
  Vector senderPoint;
  double associationWeight{0.0};
  double translationWeight{0.0};
};

Matrix seAdjoint(const Matrix &transform) {
  const int dimension = transform.rows();
  Matrix adjoint = Matrix::Zero(tangentDimension(dimension),
                                tangentDimension(dimension));
  const Matrix rotation = transform.leftCols(dimension);
  const Vector translation = transform.col(dimension);
  if (dimension == 3) {
    adjoint.topLeftCorner(3, 3) = rotation;
    adjoint.bottomLeftCorner(3, 3) = skew3(translation) * rotation;
    adjoint.bottomRightCorner(3, 3) = rotation;
  } else {
    const Matrix generator = planarGenerator();
    adjoint(0, 0) = 1.0;
    adjoint.block(1, 0, 2, 1) = -generator * translation;
    adjoint.bottomRightCorner(2, 2) = rotation;
  }
  return adjoint;
}

bool seLog(const Matrix &transform, Vector &log) {
  const int dimension = transform.rows();
  if (!validCompactTransform(transform, dimension)) {
    return false;
  }
  const Vector translation = transform.col(dimension);
  if (dimension == 3) {
    const Eigen::Vector3d rotationLog =
        so3Log(transform.leftCols(3));
    const Vector translationLog =
        so3LeftJacobianInverse(rotationLog) * translation;
    log.resize(6);
    log.head(3) = rotationLog;
    log.tail(3) = translationLog;
  } else {
    const Matrix &rotation = transform.leftCols(2);
    const double theta = std::atan2(rotation(1, 0), rotation(0, 0));
    double a = 1.0;
    double b = 0.0;
    if (std::abs(theta) < 1e-8) {
      const double thetaSquared = theta * theta;
      a = 1.0 - thetaSquared / 6.0;
      b = theta / 2.0 - theta * thetaSquared / 24.0;
    } else {
      a = std::sin(theta) / theta;
      b = (1.0 - std::cos(theta)) / theta;
    }
    const double denominator = a * a + b * b;
    if (!std::isfinite(denominator) || denominator <= 0.0) {
      return false;
    }
    const Matrix generator = planarGenerator();
    const Matrix inverseLeftJacobian =
        (a * Matrix::Identity(2, 2) - b * generator) / denominator;
    log.resize(3);
    log(0) = theta;
    log.tail(2) = inverseLeftJacobian * translation;
  }
  return log.allFinite();
}

bool validInformationShape(const Matrix &information, int dimension) {
  const int tangent = tangentDimension(dimension);
  return information.rows() == tangent && information.cols() == tangent &&
         information.allFinite();
}

struct ActiveSynchronizationEdge {
  std::size_t inputIndex{0};
  std::size_t firstRobotId{0};
  std::size_t secondRobotId{0};
  Matrix transform;
  Matrix inverseTransform;
  double precision{0.0};
};

void addBytes(std::size_t &total, std::size_t bytes) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (maximum - total < bytes) {
    total = maximum;
  } else {
    total += bytes;
  }
}

std::size_t matrixPayloadBytes(std::size_t rows, std::size_t cols) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (rows != 0 && cols > maximum / rows) {
    return maximum;
  }
  const std::size_t scalars = rows * cols;
  if (scalars > maximum / sizeof(double)) {
    return maximum;
  }
  return scalars * sizeof(double);
}

std::size_t phaseRoundCap(std::size_t numRobots) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const std::size_t factor = maximum / 64;
  if (numRobots == 0 || numRobots > factor / numRobots) {
    return maximum;
  }
  const std::size_t squared = numRobots * numRobots;
  return std::max<std::size_t>(64, 64 * squared);
}

double reliabilityFromDisagreement(double disagreement) {
  if (!std::isfinite(disagreement) || disagreement < 0.0) {
    return 0.0;
  }
  const double reliability = 1.0 / (1.0 + disagreement);
  return std::isfinite(reliability)
             ? std::clamp(reliability, 0.0, 1.0)
             : 0.0;
}

}  // namespace

GaugeCovariantTransformEstimate estimateGaugeCovariantTransform(
    const Matrix &receiverPoints, const Matrix &senderPoints,
    const std::vector<SparsePoseAlignmentWeightEntry> &matches) {
  GaugeCovariantTransformEstimate result;
  PointLayout layout;
  if (!determinePointLayout(receiverPoints, senderPoints, layout) ||
      layout.receiverCount <= 0 || layout.senderCount <= 0) {
    return result;
  }

  std::vector<AlignmentMatch> validMatches;
  validMatches.reserve(matches.size());
  for (const SparsePoseAlignmentWeightEntry &match : matches) {
    if (match.row < 0 || match.col < 0 ||
        match.row >= layout.receiverCount ||
        match.col >= layout.senderCount ||
        !finitePositive(match.associationWeight) ||
        !finitePositive(match.translationInformation)) {
      continue;
    }
    const Vector receiverPoint = pointAt(receiverPoints, match.row, layout);
    const Vector senderPoint = pointAt(senderPoints, match.col, layout);
    if (!receiverPoint.allFinite() || !senderPoint.allFinite()) {
      continue;
    }
    const double translationWeight =
        match.associationWeight * match.translationInformation;
    if (!finitePositive(translationWeight)) {
      continue;
    }
    validMatches.push_back(
        {receiverPoint, senderPoint, match.associationWeight,
         translationWeight});
  }
  if (validMatches.empty()) {
    return result;
  }

  const int dimension = layout.dimension;
  Vector receiverMean = Vector::Zero(dimension);
  Vector senderMean = Vector::Zero(dimension);
  double translationMass = 0.0;
  double associationMass = 0.0;
  for (const AlignmentMatch &match : validMatches) {
    translationMass += match.translationWeight;
    associationMass += match.associationWeight;
    receiverMean.noalias() +=
        match.translationWeight * match.receiverPoint;
    senderMean.noalias() += match.translationWeight * match.senderPoint;
  }
  if (!std::isfinite(translationMass) || !finitePositive(translationMass) ||
      !std::isfinite(associationMass) || !finitePositive(associationMass) ||
      !receiverMean.allFinite() || !senderMean.allFinite()) {
    return result;
  }
  receiverMean /= translationMass;
  senderMean /= translationMass;

  Matrix crossCovariance = Matrix::Zero(dimension, dimension);
  for (const AlignmentMatch &match : validMatches) {
    const Vector receiverCentered = match.receiverPoint - receiverMean;
    const Vector senderCentered = match.senderPoint - senderMean;
    crossCovariance.noalias() +=
        match.translationWeight * receiverCentered * senderCentered.transpose();
  }
  if (!crossCovariance.allFinite()) {
    return result;
  }
  const Matrix rotation = projectToRotationGroup(crossCovariance);
  if (!validRotation(rotation, dimension)) {
    return result;
  }
  const Vector translation = receiverMean - rotation * senderMean;
  if (!translation.allFinite()) {
    return result;
  }

  Matrix information =
      Matrix::Zero(tangentDimension(dimension), tangentDimension(dimension));
  double squaredResidualMass = 0.0;
  for (const AlignmentMatch &match : validMatches) {
    const Matrix jacobian = pointJacobian(rotation, match.senderPoint);
    information.noalias() +=
        match.translationWeight * jacobian.transpose() * jacobian;
    const Vector residual =
        match.receiverPoint - (rotation * match.senderPoint + translation);
    const double squaredResidual = residual.squaredNorm();
    squaredResidualMass += match.associationWeight * squaredResidual;
  }
  information = 0.5 * (information + information.transpose());
  if (!information.allFinite() || !std::isfinite(squaredResidualMass) ||
      !numericallyPositiveDefinite(information, dimension)) {
    return result;
  }

  Eigen::LDLT<Matrix> ldlt(information);
  if (ldlt.info() != Eigen::Success) {
    return result;
  }
  const Matrix covariance = ldlt.solve(
      Matrix::Identity(tangentDimension(dimension), tangentDimension(dimension)));
  if (ldlt.info() != Eigen::Success || !covariance.allFinite()) {
    return result;
  }

  result.valid = true;
  result.transform = compactTransform(rotation, translation);
  result.information = information;
  result.covariance = 0.5 * (covariance + covariance.transpose());
  result.fitTranslationRmse =
      std::sqrt(std::max(0.0, squaredResidualMass / associationMass));
  if (!std::isfinite(result.fitTranslationRmse)) {
    return GaugeCovariantTransformEstimate();
  }
  return result;
}

GaugeCovariantTransformEstimate reciprocalGaugeCovariantTransform(
    const GaugeCovariantTransformEstimate &estimate) {
  GaugeCovariantTransformEstimate result;
  const int dimension = static_cast<int>(estimate.transform.rows());
  if (!estimate.valid || !validCompactTransform(estimate.transform, dimension) ||
      !validInformationShape(estimate.information, dimension) ||
      !validInformationShape(estimate.covariance, dimension)) {
    return result;
  }

  const Matrix inverse = inverseCompactTransform(estimate.transform);
  const Matrix adjoint = seAdjoint(estimate.transform);
  const Matrix inverseAdjoint = seAdjoint(inverse);
  result.transform = inverse;
  result.information =
      0.5 * (inverseAdjoint.transpose() * estimate.information *
                  inverseAdjoint +
              (inverseAdjoint.transpose() * estimate.information *
               inverseAdjoint)
                  .transpose());
  result.covariance = 0.5 *
                      (adjoint * estimate.covariance * adjoint.transpose() +
                       (adjoint * estimate.covariance * adjoint.transpose())
                           .transpose());
  result.fitTranslationRmse = estimate.fitTranslationRmse;
  if (!result.information.allFinite() || !result.covariance.allFinite() ||
      !std::isfinite(result.fitTranslationRmse)) {
    return GaugeCovariantTransformEstimate();
  }
  result.valid = true;
  return result;
}

double gaugeCovariantPredictiveTranslationVariance(
    const GaugeCovariantTransformEstimate &estimate,
    const Vector &senderPoint) {
  const int dimension = static_cast<int>(estimate.transform.rows());
  if (!estimate.valid || !validCompactTransform(estimate.transform, dimension) ||
      !validInformationShape(estimate.covariance, dimension) ||
      senderPoint.size() != dimension || !senderPoint.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  const Matrix jacobian =
      pointJacobian(estimate.transform.leftCols(dimension), senderPoint);
  const double variance =
      (jacobian * estimate.covariance * jacobian.transpose()).trace() /
      static_cast<double>(dimension);
  if (!std::isfinite(variance)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, variance);
}

double gaugeCovariantNormalizedDisagreement(
    const GaugeCovariantTransformEstimate &directFirstFromSecond,
    const Matrix &firstGauge, const Matrix &secondGauge) {
  const int dimension =
      static_cast<int>(directFirstFromSecond.transform.rows());
  if (!directFirstFromSecond.valid ||
      !validCompactTransform(directFirstFromSecond.transform, dimension) ||
      !validInformationShape(directFirstFromSecond.information, dimension) ||
      firstGauge.rows() != dimension || firstGauge.cols() != dimension + 1 ||
      secondGauge.rows() != dimension ||
      secondGauge.cols() != dimension + 1 ||
      !validCompactTransform(firstGauge, dimension) ||
      !validCompactTransform(secondGauge, dimension)) {
    return std::numeric_limits<double>::infinity();
  }

  const Matrix network = composeCompactTransforms(
      inverseCompactTransform(firstGauge), secondGauge);
  const Matrix error = composeCompactTransforms(
      inverseCompactTransform(directFirstFromSecond.transform), network);
  Vector errorLog;
  if (!seLog(error, errorLog)) {
    return std::numeric_limits<double>::infinity();
  }
  const Matrix symmetricInformation =
      0.5 * (directFirstFromSecond.information +
             directFirstFromSecond.information.transpose());
  const double disagreement =
      errorLog.dot(symmetricInformation * errorLog) /
      static_cast<double>(tangentDimension(dimension));
  if (!std::isfinite(disagreement)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, disagreement);
}

Matrix gaugeCovariantLowerViewTransform(
    const GaugeCovariantAssociationEdgeState &edge,
    bool preserveLocalStates) {
  const int dimension = edge.network.transform.rows();
  if (!edge.network.valid ||
      !validCompactTransform(edge.network.transform, dimension)) {
    return Matrix();
  }
  return preserveLocalStates ? edge.network.transform
                             : identityCompactTransform(dimension);
}

GaugeCovariantSynchronizationResult synchronizeGaugeCovariantRobotGraphImpl(
    std::size_t numRobots,
    const std::vector<GaugeCovariantSynchronizationEdge> &edges,
    const std::vector<Matrix> &warmStartGauges) {
  GaugeCovariantSynchronizationResult result;
  result.gauges.resize(numRobots);
  result.edges.resize(edges.size());
  for (std::size_t index = 0; index < edges.size(); ++index) {
    result.edges[index].direct = edges[index].directFirstFromSecond;
  }

  int dimension = 0;
  for (const GaugeCovariantSynchronizationEdge &edge : edges) {
    const Matrix &transform = edge.directFirstFromSecond.transform;
    if (transform.rows() == transform.cols() - 1 &&
        supportedDimension(static_cast<int>(transform.rows()))) {
      dimension = static_cast<int>(transform.rows());
      break;
    }
  }
  if (dimension == 0) {
    for (const Matrix &gauge : warmStartGauges) {
      if (gauge.rows() == gauge.cols() - 1 &&
          supportedDimension(static_cast<int>(gauge.rows()))) {
        dimension = static_cast<int>(gauge.rows());
        break;
      }
    }
  }

  if (dimension == 0) {
    return result;
  }
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    result.gauges[robotId] = identityCompactTransform(dimension);
  }

  std::vector<ActiveSynchronizationEdge> activeEdges;
  activeEdges.reserve(edges.size());
  std::vector<std::size_t> activeDegree(numRobots, 0u);
  std::vector<std::size_t> parent(numRobots);
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    parent[robotId] = robotId;
  }
  auto findRoot = [&](std::size_t robotId) {
    std::size_t root = robotId;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[robotId] != robotId) {
      const std::size_t next = parent[robotId];
      parent[robotId] = root;
      robotId = next;
    }
    return root;
  };
  auto unionRobots = [&](std::size_t first, std::size_t second) {
    const std::size_t firstRoot = findRoot(first);
    const std::size_t secondRoot = findRoot(second);
    if (firstRoot != secondRoot) {
      parent[secondRoot] = firstRoot;
    }
  };

  for (std::size_t index = 0; index < edges.size(); ++index) {
    const GaugeCovariantSynchronizationEdge &edge = edges[index];
    const GaugeCovariantTransformEstimate &direct =
        edge.directFirstFromSecond;
    const bool dimensionMatches =
        direct.transform.rows() == 0 ||
        direct.transform.rows() == dimension;
    if (edge.firstRobotId >= numRobots || edge.secondRobotId >= numRobots ||
        edge.firstRobotId == edge.secondRobotId ||
        !dimensionMatches || !std::isfinite(edge.topologyWeight) ||
        edge.topologyWeight <= 0.0 ||
        !direct.valid || !validCompactTransform(direct.transform, dimension) ||
        !validInformationShape(direct.information, dimension) ||
        !numericallyPositiveDefinite(direct.information, dimension)) {
      continue;
    }
    const double translationInformation =
        direct.information
            .bottomRightCorner(dimension, dimension)
            .trace() /
        static_cast<double>(dimension);
    const double fitVariance =
        std::max(direct.fitTranslationRmse *
                     direct.fitTranslationRmse,
                 std::numeric_limits<double>::epsilon());
    const double precision =
        edge.topologyWeight * translationInformation / fitVariance;
    if (!finitePositive(translationInformation) ||
        !finitePositive(fitVariance) ||
        !finitePositive(precision)) {
      continue;
    }
    ActiveSynchronizationEdge active;
    active.inputIndex = index;
    active.firstRobotId = edge.firstRobotId;
    active.secondRobotId = edge.secondRobotId;
    active.transform = direct.transform;
    active.inverseTransform = inverseCompactTransform(direct.transform);
    active.precision = precision;
    activeEdges.push_back(std::move(active));
    ++activeDegree[edge.firstRobotId];
    ++activeDegree[edge.secondRobotId];
    unionRobots(edge.firstRobotId, edge.secondRobotId);
  }

  std::vector<std::size_t> anchors(numRobots);
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    anchors[robotId] = numRobots;
  }
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    const std::size_t root = findRoot(robotId);
    anchors[root] = std::min(anchors[root], robotId);
  }
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    anchors[robotId] = anchors[findRoot(robotId)];
  }

  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    if (robotId < warmStartGauges.size() &&
        validCompactTransform(warmStartGauges[robotId], dimension)) {
      result.gauges[robotId] = warmStartGauges[robotId];
    }
  }
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    if (activeDegree[robotId] == 0u) {
      continue;
    }
    if (anchors[robotId] != robotId) {
      continue;
    }
    const Matrix anchorInverse =
        inverseCompactTransform(result.gauges[robotId]);
    for (std::size_t other = 0; other < numRobots; ++other) {
      if (anchors[other] != robotId) {
        continue;
      }
      result.gauges[other] = composeCompactTransforms(
          anchorInverse, result.gauges[other]);
    }
    result.gauges[robotId] = identityCompactTransform(dimension);
  }

  std::vector<double> previousReliability(activeEdges.size(), 1.0);
  const double convergenceTolerance =
      std::sqrt(std::numeric_limits<double>::epsilon());
  bool informativeWarmStart = false;
  const Matrix identity = identityCompactTransform(dimension);
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    if (activeDegree[robotId] > 0u &&
        (result.gauges[robotId] - identity).norm() >
            convergenceTolerance) {
      informativeWarmStart = true;
      break;
    }
  }
  const std::size_t maximumPhaseRounds = phaseRoundCap(numRobots);
  auto runPhase = [&](bool robust) {
    std::size_t phaseRounds = 0;
    if (activeEdges.empty()) {
      return phaseRounds;
    }
    for (std::size_t iteration = 0; iteration < maximumPhaseRounds;
         ++iteration) {
      std::vector<Matrix> rotationSums(
          numRobots, Matrix::Zero(dimension, dimension));
      std::vector<Vector> translationSums(
          numRobots, Vector::Zero(dimension));
      std::vector<double> weightSums(numRobots, 0.0);

      for (std::size_t edgeIndex = 0; edgeIndex < activeEdges.size();
           ++edgeIndex) {
        const ActiveSynchronizationEdge &edge = activeEdges[edgeIndex];
        double weight = edge.precision;
        if (robust) {
          weight *= previousReliability[edgeIndex];
        }
        if (!finitePositive(weight)) {
          continue;
        }
        const Matrix proposalForFirst = composeCompactTransforms(
            result.gauges[edge.secondRobotId], edge.inverseTransform);
        const Matrix proposalForSecond = composeCompactTransforms(
            result.gauges[edge.firstRobotId], edge.transform);
        if (!validCompactTransform(proposalForFirst, dimension) ||
            !validCompactTransform(proposalForSecond, dimension)) {
          continue;
        }
        if (anchors[edge.firstRobotId] != edge.firstRobotId) {
          const double nextWeight = weightSums[edge.firstRobotId] + weight;
          if (std::isfinite(nextWeight)) {
            rotationSums[edge.firstRobotId].noalias() +=
                weight * proposalForFirst.leftCols(dimension);
            translationSums[edge.firstRobotId].noalias() +=
                weight * proposalForFirst.col(dimension);
            weightSums[edge.firstRobotId] = nextWeight;
          }
        }
        if (anchors[edge.secondRobotId] != edge.secondRobotId) {
          const double nextWeight = weightSums[edge.secondRobotId] + weight;
          if (std::isfinite(nextWeight)) {
            rotationSums[edge.secondRobotId].noalias() +=
                weight * proposalForSecond.leftCols(dimension);
            translationSums[edge.secondRobotId].noalias() +=
                weight * proposalForSecond.col(dimension);
            weightSums[edge.secondRobotId] = nextWeight;
          }
        }
      }

      std::vector<Matrix> nextGauges = result.gauges;
      double maximumRelativeChange = 0.0;
      for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
        if (activeDegree[robotId] == 0u) {
          continue;
        }
        if (anchors[robotId] == robotId ||
            !finitePositive(weightSums[robotId])) {
          nextGauges[robotId] = anchors[robotId] == robotId
                                    ? identityCompactTransform(dimension)
                                    : result.gauges[robotId];
          continue;
        }
        Matrix candidate = Matrix::Zero(dimension, dimension + 1);
        candidate.leftCols(dimension) =
            projectToRotationGroup(rotationSums[robotId]);
        candidate.col(dimension) =
            translationSums[robotId] / weightSums[robotId];
        if (!validCompactTransform(candidate, dimension)) {
          continue;
        }
        const double scale = std::max(
            {1.0, result.gauges[robotId].norm(), candidate.norm()});
        maximumRelativeChange = std::max(
            maximumRelativeChange,
            (candidate - result.gauges[robotId]).norm() / scale);
        nextGauges[robotId] = std::move(candidate);
      }
      result.gauges = std::move(nextGauges);
      ++phaseRounds;

      if (robust) {
        for (std::size_t edgeIndex = 0; edgeIndex < activeEdges.size();
             ++edgeIndex) {
          const ActiveSynchronizationEdge &edge = activeEdges[edgeIndex];
          previousReliability[edgeIndex] = reliabilityFromDisagreement(
              gaugeCovariantNormalizedDisagreement(
                  edges[edge.inputIndex].directFirstFromSecond,
                  result.gauges[edge.firstRobotId],
                  result.gauges[edge.secondRobotId]));
        }
      }
      if (maximumRelativeChange <= convergenceTolerance) {
        break;
      }
    }
    return phaseRounds;
  };

  result.rounds = informativeWarmStart ? 0u : runPhase(false);
  for (std::size_t edgeIndex = 0; edgeIndex < activeEdges.size();
       ++edgeIndex) {
    const ActiveSynchronizationEdge &edge = activeEdges[edgeIndex];
    previousReliability[edgeIndex] = reliabilityFromDisagreement(
        gaugeCovariantNormalizedDisagreement(
            edges[edge.inputIndex].directFirstFromSecond,
            result.gauges[edge.firstRobotId], result.gauges[edge.secondRobotId]));
  }
  addBytes(result.rounds, runPhase(true));

  for (const GaugeCovariantSynchronizationEdge &edge : edges) {
    const GaugeCovariantTransformEstimate &direct =
        edge.directFirstFromSecond;
    if (direct.valid && validCompactTransform(direct.transform, dimension) &&
        validInformationShape(direct.information, dimension)) {
      addBytes(result.payloadBytes,
               matrixPayloadBytes(static_cast<std::size_t>(dimension),
                                  static_cast<std::size_t>(dimension + 1)));
      addBytes(result.payloadBytes,
               matrixPayloadBytes(
                   static_cast<std::size_t>(tangentDimension(dimension)),
                   static_cast<std::size_t>(tangentDimension(dimension))));
    }
  }
  const std::size_t directedGaugeBytes =
      matrixPayloadBytes(static_cast<std::size_t>(dimension),
                         static_cast<std::size_t>(dimension + 1));
  if (!activeEdges.empty() &&
      directedGaugeBytes != std::numeric_limits<std::size_t>::max()) {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    bool payloadFits = activeEdges.size() <= maximum / 2;
    std::size_t directedTraversalsPerRound =
        payloadFits ? 2 * activeEdges.size() : 0;
    payloadFits = payloadFits &&
                  directedGaugeBytes <=
                      maximum / directedTraversalsPerRound;
    if (payloadFits) {
      directedTraversalsPerRound *= directedGaugeBytes;
      payloadFits = result.rounds <=
                    maximum / directedTraversalsPerRound;
    }
    if (payloadFits) {
      addBytes(result.payloadBytes,
               result.rounds * directedTraversalsPerRound);
    } else if (result.rounds > 0) {
      result.payloadBytes = maximum;
    }
  }

  for (const ActiveSynchronizationEdge &edge : activeEdges) {
    GaugeCovariantAssociationEdgeState &state =
        result.edges[edge.inputIndex];
    state.network = edges[edge.inputIndex].directFirstFromSecond;
    state.network.transform = composeCompactTransforms(
        inverseCompactTransform(result.gauges[edge.firstRobotId]),
        result.gauges[edge.secondRobotId]);
    state.normalizedDisagreement = gaugeCovariantNormalizedDisagreement(
        state.direct, result.gauges[edge.firstRobotId],
        result.gauges[edge.secondRobotId]);
    state.reliability =
        reliabilityFromDisagreement(state.normalizedDisagreement);
  }
  return result;
}

GaugeCovariantSynchronizationResult synchronizeGaugeCovariantRobotGraph(
    std::size_t numRobots,
    const std::vector<GaugeCovariantSynchronizationEdge> &edges,
    const std::vector<Matrix> &warmStartGauges) {
  return synchronizeGaugeCovariantRobotGraphImpl(
      numRobots, edges, warmStartGauges);
}

}  // namespace DPGO
