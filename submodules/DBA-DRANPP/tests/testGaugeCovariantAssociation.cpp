#include <DPGO/GaugeCovariantAssociation.h>

#include <DPGO/GeodesicSE3.h>

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

namespace {

using DPGO::GaugeCovariantTransformEstimate;
using DPGO::Matrix;
using DPGO::Vector;

Matrix compactTransform(const Matrix &rotation, const Vector &translation) {
  Matrix transform = Matrix::Zero(rotation.rows(), rotation.cols() + 1);
  transform.leftCols(rotation.cols()) = rotation;
  transform.col(rotation.cols()) = translation;
  return transform;
}

Matrix composeTransform(const Matrix &first, const Matrix &second) {
  const int dimension = first.rows();
  Matrix result = Matrix::Zero(dimension, dimension + 1);
  result.leftCols(dimension).noalias() =
      first.leftCols(dimension) * second.leftCols(dimension);
  result.col(dimension).noalias() =
      first.leftCols(dimension) * second.col(dimension) +
      first.col(dimension);
  return result;
}

Matrix inverseTransform(const Matrix &transform) {
  const int dimension = transform.rows();
  const Matrix rotation = transform.leftCols(dimension);
  Matrix inverse = Matrix::Zero(dimension, dimension + 1);
  inverse.leftCols(dimension) = rotation.transpose();
  inverse.col(dimension).noalias() =
      -rotation.transpose() * transform.col(dimension);
  return inverse;
}

GaugeCovariantTransformEstimate makeSynchronizationEstimate(
    const Vector &translation, double informationScale = 1.0) {
  GaugeCovariantTransformEstimate estimate;
  estimate.valid = true;
  estimate.transform = compactTransform(Matrix::Identity(3, 3), translation);
  estimate.information = informationScale * Matrix::Identity(6, 6);
  estimate.covariance =
      (1.0 / informationScale) * Matrix::Identity(6, 6);
  estimate.fitTranslationRmse = 0.0;
  return estimate;
}

Matrix se3Adjoint(const Matrix &transform) {
  Matrix adjoint = Matrix::Zero(6, 6);
  const Matrix rotation = transform.leftCols(3);
  const Vector translation = transform.col(3);
  adjoint.topLeftCorner(3, 3) = rotation;
  adjoint.bottomLeftCorner(3, 3) = DPGO::skew3(translation) * rotation;
  adjoint.bottomRightCorner(3, 3) = rotation;
  return adjoint;
}

}  // namespace

TEST(testDPGO, GaugeCovariantFullRankEstimateAndReciprocal) {
  const Matrix sender = (Matrix(6, 3) << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                        1.0, 1.0, 0.0,
                        1.0, 0.0, 1.0)
                           .finished();
  const Matrix rotation =
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, -0.1, 0.15).finished());
  const Vector translation = (Vector(3) << 0.4, -0.3, 0.7).finished();
  Matrix receiver(sender.rows() + 1, sender.cols());
  for (int row = 0; row < sender.rows(); ++row) {
    receiver.row(row) =
        (rotation * sender.row(row).transpose() + translation).transpose();
  }
  receiver.row(sender.rows()) << 10.0, -4.0, 2.0;

  std::vector<DPGO::SparsePoseAlignmentWeightEntry> matches;
  for (int row = 0; row < sender.rows(); ++row) {
    matches.push_back({row, row, 1.0 + 0.1 * row, 1.0, 2.0});
  }

  const GaugeCovariantTransformEstimate estimate =
      DPGO::estimateGaugeCovariantTransform(receiver, sender, matches);

  ASSERT_TRUE(estimate.valid);
  EXPECT_NEAR((estimate.transform.leftCols(3) - rotation).norm(), 0.0,
              1e-10);
  EXPECT_NEAR((estimate.transform.col(3) - translation).norm(), 0.0, 1e-10);
  EXPECT_EQ(estimate.information.rows(), 6);
  EXPECT_EQ(estimate.information.cols(), 6);
  EXPECT_TRUE(estimate.information.allFinite());
  EXPECT_TRUE(estimate.covariance.allFinite());
  EXPECT_TRUE(std::isfinite(estimate.fitTranslationRmse));

  const GaugeCovariantTransformEstimate reciprocal =
      DPGO::reciprocalGaugeCovariantTransform(estimate);
  ASSERT_TRUE(reciprocal.valid);
  const Matrix composed = composeTransform(estimate.transform,
                                           reciprocal.transform);
  EXPECT_NEAR((composed.leftCols(3) - Matrix::Identity(3, 3)).norm(), 0.0,
              1e-10);
  EXPECT_NEAR(composed.col(3).norm(), 0.0, 1e-10);

  const GaugeCovariantTransformEstimate roundTrip =
      DPGO::reciprocalGaugeCovariantTransform(reciprocal);
  EXPECT_NEAR((roundTrip.information - estimate.information).norm(), 0.0,
              1e-8);
  EXPECT_NEAR((roundTrip.covariance - estimate.covariance).norm(), 0.0,
              1e-8);

  GaugeCovariantTransformEstimate overflowingRotation;
  overflowingRotation.valid = true;
  overflowingRotation.transform = Matrix::Zero(3, 4);
  overflowingRotation.transform.leftCols(3) =
      1e200 * Matrix::Identity(3, 3);
  overflowingRotation.information = Matrix::Zero(6, 6);
  overflowingRotation.covariance = Matrix::Zero(6, 6);
  overflowingRotation.fitTranslationRmse = 0.0;
  EXPECT_FALSE(
      DPGO::reciprocalGaugeCovariantTransform(overflowingRotation).valid);

  const Vector senderPoint = (Vector(3) << 0.3, -0.4, 0.8).finished();
  EXPECT_GT(DPGO::gaugeCovariantPredictiveTranslationVariance(
                estimate, senderPoint),
            0.0);

  const Matrix sender2d = (Matrix(4, 2) << 0.2, -0.1,
                           1.1, 0.3,
                           -0.4, 1.2,
                           0.7, 1.5)
                              .finished();
  const double angle2d = 0.35;
  Matrix rotation2d(2, 2);
  rotation2d << std::cos(angle2d), -std::sin(angle2d),
      std::sin(angle2d), std::cos(angle2d);
  const Matrix receiver2d = sender2d * rotation2d.transpose();
  const std::vector<DPGO::SparsePoseAlignmentWeightEntry> matches2d{
      {0, 0, 1.0, 1.0, 1.0}, {1, 1, 1.0, 1.0, 1.0},
      {2, 2, 1.0, 1.0, 1.0}, {3, 3, 1.0, 1.0, 1.0}};
  const GaugeCovariantTransformEstimate estimate2d =
      DPGO::estimateGaugeCovariantTransform(receiver2d, sender2d, matches2d);
  ASSERT_TRUE(estimate2d.valid);

  Matrix finiteDifferenceInformation = Matrix::Zero(3, 3);
  constexpr double step = 1e-6;
  Matrix incrementalPlus(2, 2);
  incrementalPlus << std::cos(step), -std::sin(step), std::sin(step),
      std::cos(step);
  const Matrix incrementalMinus = incrementalPlus.transpose();
  for (int row = 0; row < sender2d.rows(); ++row) {
    const Vector point = sender2d.row(row).transpose();
    Matrix jacobian = Matrix::Zero(2, 3);
    jacobian.col(0).noalias() =
        rotation2d * (incrementalPlus - incrementalMinus) * point /
        (2.0 * step);
    jacobian.rightCols(2) = rotation2d;
    finiteDifferenceInformation.noalias() += jacobian.transpose() * jacobian;
  }
  EXPECT_NEAR((estimate2d.information - finiteDifferenceInformation).norm(),
              0.0, 1e-7);
}

TEST(testDPGO, GaugeCovariantRejectsCollinearThreeDimensionalEvidence) {
  const Matrix sender = (Matrix(5, 3) << 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0,
                        2.0, 0.0, 0.0,
                        3.0, 0.0, 0.0,
                        4.0, 0.0, 0.0)
                           .finished();
  const Matrix receiver = sender.rowwise() +
                          (Vector(3) << 0.5, -0.2, 0.9).finished().transpose();
  std::vector<DPGO::SparsePoseAlignmentWeightEntry> matches;
  for (int row = 0; row < sender.rows(); ++row) {
    matches.push_back({row, row, 1.0, 1.0, 1.0});
  }

  const GaugeCovariantTransformEstimate estimate =
      DPGO::estimateGaugeCovariantTransform(receiver, sender, matches);

  EXPECT_FALSE(estimate.valid);
  EXPECT_TRUE(std::isinf(estimate.fitTranslationRmse));
}

TEST(testDPGO, GaugeCovariantSynchronizationScoresWeakCycle) {
  const Vector unitX = (Vector(3) << 1.0, 0.0, 0.0).finished();
  const Vector fiveX = (Vector(3) << 5.0, 0.0, 0.0).finished();

  const std::vector<DPGO::GaugeCovariantSynchronizationEdge> tree{
      {0, 1, 1.0, makeSynchronizationEstimate(unitX)},
      {1, 2, 1.0, makeSynchronizationEstimate(unitX)}};
  const auto treeResult =
      DPGO::synchronizeGaugeCovariantRobotGraph(3, tree);

  ASSERT_EQ(treeResult.gauges.size(), 3u);
  ASSERT_EQ(treeResult.edges.size(), tree.size());
  for (const auto &edge : treeResult.edges) {
    EXPECT_TRUE(edge.network.valid);
    EXPECT_NEAR(edge.reliability, 1.0, 1e-8);
  }

  const std::vector<DPGO::GaugeCovariantSynchronizationEdge> triangle{
      {0, 1, 1.0, makeSynchronizationEstimate(unitX)},
      {1, 2, 1.0, makeSynchronizationEstimate(unitX)},
      {0, 2, 0.001, makeSynchronizationEstimate(fiveX)}};
  const auto triangleResult =
      DPGO::synchronizeGaugeCovariantRobotGraph(3, triangle);

  ASSERT_EQ(triangleResult.edges.size(), triangle.size());
  EXPECT_TRUE(triangleResult.edges[0].network.valid);
  EXPECT_TRUE(triangleResult.edges[1].network.valid);
  EXPECT_TRUE(triangleResult.edges[2].network.valid);
  EXPECT_GT(triangleResult.edges[0].reliability,
            triangleResult.edges[2].reliability);
  EXPECT_GT(triangleResult.edges[1].reliability,
            triangleResult.edges[2].reliability);
  EXPECT_LT(triangleResult.edges[2].reliability, 0.8);
  EXPECT_NEAR(triangleResult.edges[2].network.transform(0, 3), 2.0, 5e-2);
  EXPECT_GT(triangleResult.payloadBytes, 0u);

  const std::vector<DPGO::GaugeCovariantSynchronizationEdge> inactiveEdges{
      {0, 1, 0.0, makeSynchronizationEstimate(unitX)},
      {0, 0, 1.0, makeSynchronizationEstimate(unitX)}};
  const auto inactiveResult =
      DPGO::synchronizeGaugeCovariantRobotGraph(2, inactiveEdges);
  ASSERT_EQ(inactiveResult.edges.size(), inactiveEdges.size());
  EXPECT_FALSE(inactiveResult.edges[0].network.valid);
  EXPECT_FALSE(inactiveResult.edges[1].network.valid);
}

TEST(testDPGO, GaugeCovariantLowerViewUsesNetworkTransformOnlyForLocalStates) {
  DPGO::GaugeCovariantAssociationEdgeState edge;
  edge.network = makeSynchronizationEstimate(
      (Vector(3) << 1.25, -0.5, 0.75).finished());

  const Matrix localView =
      DPGO::gaugeCovariantLowerViewTransform(edge, true);
  const Matrix commonView =
      DPGO::gaugeCovariantLowerViewTransform(edge, false);

  ASSERT_EQ(localView.rows(), 3);
  ASSERT_EQ(localView.cols(), 4);
  EXPECT_TRUE(localView.isApprox(edge.network.transform, 1e-12));
  ASSERT_EQ(commonView.rows(), 3);
  ASSERT_EQ(commonView.cols(), 4);
  EXPECT_TRUE(commonView.leftCols(3).isApprox(Matrix::Identity(3, 3), 1e-12));
  EXPECT_TRUE(commonView.col(3).isZero(1e-12));
}

TEST(testDPGO, GaugeCovariantLowerViewRejectsMissingNetworkTransform) {
  const DPGO::GaugeCovariantAssociationEdgeState edge;

  EXPECT_EQ(DPGO::gaugeCovariantLowerViewTransform(edge, true).size(), 0);
  EXPECT_EQ(DPGO::gaugeCovariantLowerViewTransform(edge, false).size(), 0);
}

TEST(testDPGO, GaugeCovariantReliabilityIsCoordinateInvariant) {
  const Matrix directTransform = compactTransform(
      DPGO::so3Exp((Eigen::Vector3d() << 0.15, -0.12, 0.08).finished()),
      (Vector(3) << 0.7, -0.4, 0.2).finished());
  GaugeCovariantTransformEstimate direct;
  direct.valid = true;
  direct.transform = directTransform;
  direct.information = Matrix::Identity(6, 6);
  direct.information.diagonal() << 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
  direct.covariance = direct.information.inverse();

  const Matrix firstGauge = compactTransform(
      DPGO::so3Exp((Eigen::Vector3d() << -0.1, 0.05, 0.07).finished()),
      (Vector(3) << -0.2, 0.4, 0.1).finished());
  const Matrix secondGauge = compactTransform(
      DPGO::so3Exp((Eigen::Vector3d() << 0.04, -0.08, 0.06).finished()),
      (Vector(3) << 1.1, -0.3, 0.5).finished());
  const double q = DPGO::gaugeCovariantNormalizedDisagreement(
      direct, firstGauge, secondGauge);

  const Matrix firstChange = compactTransform(
      DPGO::so3Exp((Eigen::Vector3d() << 0.2, 0.03, -0.04).finished()),
      (Vector(3) << 0.5, -0.1, 0.3).finished());
  const Matrix secondChange = compactTransform(
      DPGO::so3Exp((Eigen::Vector3d() << -0.06, 0.09, 0.02).finished()),
      (Vector(3) << -0.4, 0.2, 0.6).finished());
  GaugeCovariantTransformEstimate changed = direct;
  changed.transform = composeTransform(
      firstChange,
      composeTransform(directTransform, inverseTransform(secondChange)));
  const Matrix senderAdjoint = se3Adjoint(secondChange);
  const Matrix senderAdjointInverse = senderAdjoint.inverse();
  changed.information = senderAdjointInverse.transpose() *
                        direct.information * senderAdjointInverse;
  changed.covariance = changed.information.inverse();

  const double changedQ = DPGO::gaugeCovariantNormalizedDisagreement(
      changed, composeTransform(firstGauge, inverseTransform(firstChange)),
      composeTransform(secondGauge, inverseTransform(secondChange)));
  EXPECT_NEAR(changedQ, q, 1e-8);
  EXPECT_NEAR(1.0 / (1.0 + changedQ), 1.0 / (1.0 + q), 1e-12);

}
