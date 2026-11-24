#include <DPGO/BilevelAssociation.h>
#include <DPGO/QuadraticProblem.h>

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace DPGO;

TEST(testDPGO, TrackEmpiricalInformationFallsBackForUnidentifiableTracks) {
  const TrackEmpiricalInformationEstimate singleton =
      estimateTrackEmpiricalInformation({Vector::Constant(2, 3.0)}, {4.0});
  EXPECT_EQ(singleton.sampleCount, 1u);
  EXPECT_DOUBLE_EQ(singleton.informationScale, 1.0);
  EXPECT_DOUBLE_EQ(singleton.overdispersion, 1.0);
  EXPECT_DOUBLE_EQ(singleton.effectiveSampleSize, 1.0);
  EXPECT_DOUBLE_EQ(singleton.effectiveSampleRatio, 1.0);
  EXPECT_FALSE(singleton.empirical);

  const std::vector<Vector> identical(3, Vector::Constant(2, -2.0));
  const TrackEmpiricalInformationEstimate zeroEnergy =
      estimateTrackEmpiricalInformation(identical, {1.0, 2.0, 3.0});
  EXPECT_EQ(zeroEnergy.sampleCount, 3u);
  EXPECT_DOUBLE_EQ(zeroEnergy.informationScale, 1.0);
  EXPECT_DOUBLE_EQ(zeroEnergy.overdispersion, 1.0);
  EXPECT_DOUBLE_EQ(zeroEnergy.effectiveSampleSize, 3.0);
  EXPECT_DOUBLE_EQ(zeroEnergy.effectiveSampleRatio, 1.0);
  EXPECT_FALSE(zeroEnergy.empirical);
}

TEST(testDPGO, TrackEmpiricalInformationDoesNotPenalizeNegativeCorrelation) {
  const std::vector<Vector> centers{
      Vector::Constant(1, -0.1), Vector::Constant(1, 0.1),
      Vector::Constant(1, -0.1), Vector::Constant(1, 0.1)};
  const TrackEmpiricalInformationEstimate estimate =
      estimateTrackEmpiricalInformation(centers, {1.0, 1.0, 1.0, 1.0});

  EXPECT_TRUE(estimate.empirical);
  EXPECT_DOUBLE_EQ(estimate.overdispersion, 1.0);
  EXPECT_DOUBLE_EQ(estimate.effectiveSampleSize, 4.0);
  EXPECT_DOUBLE_EQ(estimate.effectiveSampleRatio, 1.0);
  EXPECT_DOUBLE_EQ(estimate.informationScale, 1.0);
}

TEST(testDPGO, TrackEmpiricalInformationReducesPositiveSerialCorrelation) {
  const std::vector<Vector> centers{
      Vector::Constant(1, -0.2), Vector::Constant(1, -0.1),
      Vector::Constant(1, -0.1), Vector::Constant(1, 0.1),
      Vector::Constant(1, 0.1), Vector::Constant(1, 0.2)};
  const TrackEmpiricalInformationEstimate estimate =
      estimateTrackEmpiricalInformation(
          centers, std::vector<double>(centers.size(), 1.0));

  EXPECT_TRUE(estimate.empirical);
  EXPECT_DOUBLE_EQ(estimate.overdispersion, 1.0);
  EXPECT_GT(estimate.effectiveSampleSize, 1.0);
  EXPECT_LT(estimate.effectiveSampleSize,
            static_cast<double>(centers.size()));
  EXPECT_GT(estimate.informationScale,
            1.0 / static_cast<double>(centers.size()));
  EXPECT_LT(estimate.informationScale, 1.0);
}

TEST(testDPGO, TrackEmpiricalInformationReducesOverdispersedTracks) {
  const std::vector<Vector> centers{
      Vector::Constant(1, -10.0), Vector::Constant(1, 10.0),
      Vector::Constant(1, -10.0), Vector::Constant(1, 10.0)};
  const TrackEmpiricalInformationEstimate estimate =
      estimateTrackEmpiricalInformation(
          centers, std::vector<double>(centers.size(), 1.0));

  EXPECT_TRUE(estimate.empirical);
  EXPECT_GT(estimate.overdispersion, 1.0);
  EXPECT_DOUBLE_EQ(estimate.effectiveSampleRatio, 1.0);
  EXPECT_LT(estimate.informationScale, 0.01);
}

TEST(testDPGO, TrackEmpiricalInformationIsRigidTransformInvariant) {
  std::vector<Vector> centers(6, Vector(2));
  centers[0] << -0.2, 0.0;
  centers[1] << -0.1, 0.0;
  centers[2] << -0.1, 0.0;
  centers[3] << 0.1, 0.0;
  centers[4] << 0.1, 0.0;
  centers[5] << 0.2, 0.0;
  const std::vector<double> information{1.0, 2.0, 1.5, 0.5, 2.5, 1.0};
  const TrackEmpiricalInformationEstimate reference =
      estimateTrackEmpiricalInformation(centers, information);

  Matrix rotation(2, 2);
  rotation << 0.0, -1.0, 1.0, 0.0;
  const Vector translation = (Vector(2) << 4.0, -7.0).finished();
  std::vector<Vector> transformed;
  transformed.reserve(centers.size());
  for (const Vector &center : centers) {
    transformed.push_back(rotation * center + translation);
  }
  const TrackEmpiricalInformationEstimate moved =
      estimateTrackEmpiricalInformation(transformed, information);

  EXPECT_NEAR(moved.informationScale, reference.informationScale, 1e-12);
  EXPECT_NEAR(moved.overdispersion, reference.overdispersion, 1e-12);
  EXPECT_NEAR(moved.effectiveSampleSize, reference.effectiveSampleSize,
              1e-12);
}

TEST(testDPGO, TrackEmpiricalInformationRejectsNonFiniteInputs) {
  std::vector<Vector> centers{Vector::Zero(2), Vector::Ones(2)};
  centers[1](0) = std::numeric_limits<double>::quiet_NaN();
  const TrackEmpiricalInformationEstimate estimate =
      estimateTrackEmpiricalInformation(centers, {1.0, 1.0});

  EXPECT_EQ(estimate.sampleCount, 2u);
  EXPECT_DOUBLE_EQ(estimate.informationScale, 1.0);
  EXPECT_DOUBLE_EQ(estimate.effectiveSampleSize, 2.0);
  EXPECT_FALSE(estimate.empirical);
}

TEST(testDPGO, BilevelAssociationPartialSinkhornRespectsCapacitiesAndCosts) {
  Matrix costs(2, 2);
  costs << 0.0, 5.0, 4.0, 0.0;

  PartialSinkhornOptions options;
  options.iterations = 50;
  options.stepSize = 1.0;
  options.temperature = 1.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs);

  ASSERT_EQ(pi.rows(), 2);
  ASSERT_EQ(pi.cols(), 2);
  for (int row = 0; row < pi.rows(); ++row) {
    EXPECT_LE(pi.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < pi.cols(); ++col) {
    EXPECT_LE(pi.col(col).sum(), 1.0 + 1e-12);
  }
  EXPECT_GT(pi(0, 0), pi(0, 1));
  EXPECT_GT(pi(1, 1), pi(1, 0));
}

TEST(testDPGO, BilevelAssociationPartialSinkhornUsesPreviousBeliefs) {
  Matrix costs = Matrix::Zero(2, 2);
  Matrix previous(2, 2);
  previous << 0.9, 0.1, 0.2, 0.8;

  PartialSinkhornOptions options;
  options.iterations = 20;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs, previous);

  EXPECT_GT(pi(0, 0), pi(0, 1));
  EXPECT_GT(pi(1, 1), pi(1, 0));
  for (int row = 0; row < pi.rows(); ++row) {
    EXPECT_LE(pi.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < pi.cols(); ++col) {
    EXPECT_LE(pi.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO, BilevelAssociationPartialSinkhornCanKeepTopCandidates) {
  Matrix costs(2, 3);
  costs << 0.0, 0.5, 4.0,
           3.0, 0.1, 0.2;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.maxNonzeros = 2;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs);

  std::size_t nonzeros = 0;
  for (int row = 0; row < pi.rows(); ++row) {
    for (int col = 0; col < pi.cols(); ++col) {
      if (pi(row, col) > 1e-12) {
        ++nonzeros;
      }
    }
  }
  EXPECT_LE(nonzeros, 2u);
  EXPECT_GT(pi(0, 0), 0.0);
  EXPECT_GT(pi(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(pi(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(pi(1, 0), 0.0);
}

TEST(testDPGO,
     BilevelAssociationPartialSinkhornCanRenormalizeSparseCandidates) {
  Matrix costs(3, 3);
  costs << 0.0, 1.0, 1.0,
           1.0, 0.0, 1.0,
           1.0, 1.0, 0.0;

  PartialSinkhornOptions sparseOptions;
  sparseOptions.iterations = 30;
  sparseOptions.temperature = 10.0;
  sparseOptions.maxNonzeros = 3;
  SoftAssignmentOptimizer sparseOptimizer(sparseOptions);
  const Matrix sparse = sparseOptimizer.updateFromCosts(costs);

  PartialSinkhornOptions balancedOptions = sparseOptions;
  balancedOptions.renormalizeAfterSparsify = true;
  SoftAssignmentOptimizer balancedOptimizer(balancedOptions);
  const Matrix balanced = balancedOptimizer.updateFromCosts(costs);

  std::size_t nonzeros = 0;
  for (int row = 0; row < balanced.rows(); ++row) {
    EXPECT_LE(balanced.row(row).sum(), 1.0 + 1e-12);
    for (int col = 0; col < balanced.cols(); ++col) {
      if (balanced(row, col) > 1e-12) {
        ++nonzeros;
      }
    }
  }
  for (int col = 0; col < balanced.cols(); ++col) {
    EXPECT_LE(balanced.col(col).sum(), 1.0 + 1e-12);
  }
  EXPECT_LE(nonzeros, 3u);
  EXPECT_GT(balanced.sum(), sparse.sum() + 1.0);
  EXPECT_GT(balanced(0, 0), 0.9);
  EXPECT_GT(balanced(1, 1), 0.9);
  EXPECT_GT(balanced(2, 2), 0.9);
}

TEST(testDPGO,
     BilevelAssociationSparseSupportProjectionMatchesDenseProjection) {
  Matrix weights = Matrix::Zero(8, 8);
  weights(0, 0) = 1.7;
  weights(0, 5) = 0.3;
  weights(1, 1) = 0.8;
  weights(2, 2) = 1.2;
  weights(2, 6) = 0.5;
  weights(3, 4) = 0.4;
  weights(4, 3) = 1.4;
  weights(5, 7) = 0.9;
  weights(6, 2) = 0.6;
  weights(7, 7) = 1.1;

  PartialSinkhornOptions options;
  options.iterations = 25;
  options.rowCapacity = 0.75;
  options.columnCapacity = 0.8;
  options.maxNonzeros = 0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix dense = optimizer.projectWeights(weights);
  const Matrix sparse = optimizer.projectWeightsSparseSupport(weights);

  ASSERT_EQ(sparse.rows(), dense.rows());
  ASSERT_EQ(sparse.cols(), dense.cols());
  for (int row = 0; row < dense.rows(); ++row) {
    for (int col = 0; col < dense.cols(); ++col) {
      EXPECT_NEAR(sparse(row, col), dense(row, col), 1e-12);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationSparseCostEntriesMatchDenseGradientProjection) {
  Matrix costs = Matrix::Constant(
      8, 8, std::numeric_limits<double>::infinity());
  std::vector<SparseAssignmentCostEntry> entries;
  auto addEntry = [&](int row, int col, double cost) {
    costs(row, col) = cost;
    entries.push_back(SparseAssignmentCostEntry{row, col, cost});
  };
  addEntry(0, 0, 0.1);
  addEntry(0, 5, 1.4);
  addEntry(1, 1, 0.3);
  addEntry(2, 2, 0.2);
  addEntry(2, 6, 1.0);
  addEntry(3, 4, 0.6);
  addEntry(4, 3, 0.4);
  addEntry(5, 7, 0.7);
  addEntry(6, 2, 0.8);
  addEntry(7, 7, 0.5);

  Matrix previous = Matrix::Zero(8, 8);
  Matrix gradient = Matrix::Zero(8, 8);
  for (const SparseAssignmentCostEntry &entry : entries) {
    previous(entry.row, entry.col) = 0.2 + 0.03 * entry.row;
    gradient(entry.row, entry.col) = 0.01 * (entry.col - entry.row);
  }

  PartialSinkhornOptions options;
  options.iterations = 25;
  options.rowCapacity = 0.75;
  options.columnCapacity = 0.8;
  options.temperature = 0.5;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix dense =
      optimizer.updateFromCostsAndGradient(costs, gradient, previous);
  const Matrix sparse = optimizer.updateFromCostEntriesAndGradient(
      costs.rows(), costs.cols(), entries, gradient, previous);

  ASSERT_EQ(sparse.rows(), dense.rows());
  ASSERT_EQ(sparse.cols(), dense.cols());
  for (int row = 0; row < dense.rows(); ++row) {
    for (int col = 0; col < dense.cols(); ++col) {
      EXPECT_NEAR(sparse(row, col), dense(row, col), 1e-12);
    }
  }
}

TEST(testDPGO, BilevelAssociationPartialSinkhornUnmatchedCostSuppressesBadPairs) {
  Matrix costs(2, 2);
  costs << 10.0, 11.0,
           12.0, 13.0;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs);

  EXPECT_LT(pi.sum(), 1e-2);
  for (int row = 0; row < pi.rows(); ++row) {
    EXPECT_LE(pi.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < pi.cols(); ++col) {
    EXPECT_LE(pi.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO, BilevelAssociationPartialSinkhornUnmatchedCostKeepsGoodPairs) {
  Matrix costs(2, 2);
  costs << 0.0, 10.0,
           10.0, 0.0;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs);

  EXPECT_GT(pi(0, 0), 0.99);
  EXPECT_GT(pi(1, 1), 0.99);
  EXPECT_LT(pi(0, 1), 1e-3);
  EXPECT_LT(pi(1, 0), 1e-3);
}

TEST(testDPGO, BilevelAssociationFreshUnbalancedTransportIsStateless) {
  Matrix costs(2, 2);
  costs << -4.0, 8.0,
           8.0, -4.0;

  PartialSinkhornOptions options;
  options.iterations = 50;
  options.temperature = 1.0;
  options.stepSize = 1.0;
  options.unmatchedCost = 4.0;
  SoftAssignmentOptimizer optimizer(options);

  const FreshUnbalancedTransportResult first =
      optimizer.solveFreshUnbalancedTransport(costs);
  const FreshUnbalancedTransportResult second =
      optimizer.solveFreshUnbalancedTransport(costs);

  EXPECT_TRUE(first.assignment.isApprox(second.assignment, 1e-12));
  EXPECT_TRUE(first.rowUnmatchedMass.isApprox(second.rowUnmatchedMass,
                                               1e-12));
  EXPECT_TRUE(first.columnUnmatchedMass.isApprox(second.columnUnmatchedMass,
                                                  1e-12));
  EXPECT_EQ(first.activeCandidateCount, 4u);
  EXPECT_LT(first.marginalResidualNorm, 1e-10);
  EXPECT_GT(first.assignment.sum(), 1.99);
  for (int row = 0; row < costs.rows(); ++row) {
    EXPECT_NEAR(first.assignment.row(row).sum() + first.rowUnmatchedMass(row),
                1.0, 1e-10);
  }
  for (int col = 0; col < costs.cols(); ++col) {
    EXPECT_NEAR(first.assignment.col(col).sum() +
                    first.columnUnmatchedMass(col),
                1.0, 1e-10);
  }
}

TEST(testDPGO, BilevelAssociationReciprocalFiniteCostMeanAvoidsOverflow) {
  const double max = std::numeric_limits<double>::max();
  EXPECT_DOUBLE_EQ(reciprocalFiniteCostMean(max, max), max);
  EXPECT_TRUE(std::isfinite(reciprocalFiniteCostMean(max, max)));
  EXPECT_DOUBLE_EQ(reciprocalFiniteCostMean(2.0, 6.0), 4.0);
}

TEST(testDPGO, BilevelAssociationFreshUnbalancedTransportUsesDustbins) {
  std::vector<SparseAssignmentCostEntry> entries{
      SparseAssignmentCostEntry{0, 0, 16.0},
      SparseAssignmentCostEntry{1, 1, 16.0}};

  PartialSinkhornOptions options;
  options.iterations = 50;
  options.temperature = 1.0;
  options.stepSize = 1.0;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  const FreshUnbalancedTransportResult result =
      optimizer.solveFreshUnbalancedTransport(2, 2, entries);

  EXPECT_EQ(result.activeCandidateCount, 2u);
  EXPECT_LT(result.marginalResidualNorm, 1e-10);
  EXPECT_LT(result.assignment.sum(), 0.01);
  EXPECT_GT(result.rowUnmatchedMass.sum(), 1.99);
  EXPECT_GT(result.columnUnmatchedMass.sum(), 1.99);
}

TEST(testDPGO, BilevelAssociationSharedConsensusBoundActivationIsExact) {
  EXPECT_TRUE(sharedObjectConsensusHypothesisIsActive(1.0, true));
  EXPECT_FALSE(sharedObjectConsensusHypothesisIsActive(
      std::nextafter(1.0, 0.0), true));
  EXPECT_FALSE(sharedObjectConsensusHypothesisIsActive(0.5, true));
  EXPECT_TRUE(sharedObjectConsensusHypothesisIsActive(0.5, false));
  EXPECT_FALSE(sharedObjectConsensusHypothesisIsActive(0.0, false));
}

TEST(testDPGO, BilevelAssociationFreshTransportDiagnosticsSeparateDustbinAndCompetition) {
  const std::vector<SparseAssignmentCostEntry> entries{
      SparseAssignmentCostEntry{0, 0, 0.0},
      SparseAssignmentCostEntry{0, 1, 0.0},
      SparseAssignmentCostEntry{1, 1, 3.0}};

  PartialSinkhornOptions options;
  options.iterations = 50;
  options.temperature = 1.0;
  options.stepSize = 1.0;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  std::vector<FreshUnbalancedTransportEdgeDiagnostic> diagnostics;
  const FreshUnbalancedTransportResult result =
      optimizer.solveFreshUnbalancedTransport(2, 2, entries, &diagnostics);

  ASSERT_EQ(diagnostics.size(), entries.size());
  EXPECT_DOUBLE_EQ(diagnostics[0].rawScore, 2.0);
  EXPECT_DOUBLE_EQ(diagnostics[1].rawScore, 2.0);
  EXPECT_DOUBLE_EQ(diagnostics[2].rawScore, 0.0);
  EXPECT_NEAR(diagnostics[0].projectedWeight, 0.5, 1e-12);
  EXPECT_NEAR(diagnostics[1].projectedWeight, 0.5, 1e-12);
  EXPECT_DOUBLE_EQ(diagnostics[2].projectedWeight, 0.0);
  EXPECT_NEAR(diagnostics[0].rowMass, 1.0, 1e-12);
  EXPECT_NEAR(diagnostics[1].rowMass, 1.0, 1e-12);
  EXPECT_NEAR(diagnostics[0].columnMass, 0.5, 1e-12);
  EXPECT_NEAR(diagnostics[1].columnMass, 0.5, 1e-12);
  EXPECT_TRUE(result.assignment.isApprox(
      (Matrix(2, 2) << 0.5, 0.5, 0.0, 0.0).finished(), 1e-12));
}

TEST(testDPGO, BilevelAssociationSparseCostNormalizationMatchesDenseRows) {
  const std::vector<SparseAssignmentCostEntry> entries{
      {0, 0, 3.0}, {0, 2, 4.0}, {0, 3, 12.0},
      {1, 1, 5.0}, {1, 3, 12.0}};

  const std::vector<SparseAssignmentCostEntry> normalized =
      normalizeSparseAssignmentCostEntries(2, entries, 2, 1.0);

  ASSERT_EQ(normalized.size(), entries.size());
  const double expectedScale = std::sqrt(4.0 / (9.0 + 16.0 + 25.0 + 144.0));
  for (std::size_t index = 0; index < entries.size(); ++index) {
    EXPECT_EQ(normalized[index].row, entries[index].row);
    EXPECT_EQ(normalized[index].col, entries[index].col);
    EXPECT_NEAR(normalized[index].cost,
                expectedScale * entries[index].cost, 1e-15);
  }
}

TEST(testDPGO,
     BilevelAssociationPartialSinkhornUnmatchedCostSuppressesBadPairsWithGradient) {
  Matrix costs(2, 2);
  costs << 10.0, 11.0,
           12.0, 13.0;
  Matrix previous = Matrix::Ones(2, 2);
  Matrix attractiveGradient = Matrix::Constant(2, 2, -1.0);

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.unmatchedCost = 2.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi =
      optimizer.updateFromCostsAndGradient(costs, attractiveGradient, previous);

  EXPECT_LT(pi.sum(), 1e-2);
}

TEST(testDPGO, BilevelAssociationPartialSinkhornProjectsExternalWeights) {
  Matrix weights(2, 2);
  weights << 2.0, 1.0,
             1.5, 0.5;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.rowCapacity = 1.0;
  options.columnCapacity = 1.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix projected = optimizer.projectWeights(weights);

  for (int row = 0; row < projected.rows(); ++row) {
    EXPECT_LE(projected.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < projected.cols(); ++col) {
    EXPECT_LE(projected.col(col).sum(), 1.0 + 1e-12);
  }
  EXPECT_GT(projected(0, 0), 0.0);
}

TEST(testDPGO, BilevelAssociationBoundedResidualReliabilityBlendIsConservative) {
  EXPECT_DOUBLE_EQ(associationBoundedResidualReliabilityBlend(0.0, 1.0),
                   0.0);
  EXPECT_DOUBLE_EQ(associationBoundedResidualReliabilityBlend(0.8, 0.9),
                   0.8);

  const double attenuated =
      associationBoundedResidualReliabilityBlend(0.8, 0.2);
  EXPECT_LT(attenuated, 0.8);
  EXPECT_GT(attenuated, 0.4);
  EXPECT_DOUBLE_EQ(attenuated, 0.8 / 1.6);

  const double nonfiniteShared =
      associationBoundedResidualReliabilityBlend(
          std::numeric_limits<double>::quiet_NaN(), 0.8);
  EXPECT_DOUBLE_EQ(nonfiniteShared, 0.0);
  const double nonfiniteCurrent =
      associationBoundedResidualReliabilityBlend(
          0.75, std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(nonfiniteCurrent, 0.75);
}

TEST(testDPGO, BilevelAssociationSupportConfidenceReliabilityRewardsUniqueReciprocalMass) {
  const double strong =
      associationSupportConfidenceReliability(0.9, 0.8, 0.02, 0.03);
  const double ambiguous =
      associationSupportConfidenceReliability(0.9, 0.8, 0.7, 0.6);
  const double weak =
      associationSupportConfidenceReliability(0.2, 0.15, 0.02, 0.03);

  EXPECT_GT(strong, 0.75);
  EXPECT_LT(ambiguous, strong);
  EXPECT_LT(weak, ambiguous);
  EXPECT_DOUBLE_EQ(associationSupportConfidenceReliability(
                       std::numeric_limits<double>::quiet_NaN(), 0.8, 0.0,
                       0.0),
                   0.0);
}

TEST(testDPGO, BilevelAssociationTrajectoryFeedbackTrustIsConservative) {
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackTrustScale(1.0), 1.0);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackTrustScale(0.5), 0.25);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackTrustScale(-1.0), 0.0);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackTrustScale(
                       std::numeric_limits<double>::quiet_NaN()),
                   0.0);
}

TEST(testDPGO, BilevelAssociationTrajectoryFeedbackMassNormalizerDoesNotAmplify) {
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackObjectMassNormalizer(0.0), 1.0);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackObjectMassNormalizer(0.5), 1.0);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackObjectMassNormalizer(2.0), 0.5);
  EXPECT_DOUBLE_EQ(associationTrajectoryFeedbackObjectMassNormalizer(
                       std::numeric_limits<double>::infinity()),
                   0.0);
}

TEST(testDPGO,
     BilevelAssociationLowerProjectionPowerSharpensWhilePreservingMass) {
  Matrix assignment(2, 2);
  assignment << 0.8, 0.1,
                0.1, 0.8;

  const Matrix identity =
      projectAssociationForLowerGraph(assignment, 1.0);
  EXPECT_TRUE(identity.isApprox(assignment, 1e-12));

  const Matrix sharpened =
      projectAssociationForLowerGraph(assignment, 8.0);
  ASSERT_EQ(sharpened.rows(), assignment.rows());
  ASSERT_EQ(sharpened.cols(), assignment.cols());
  EXPECT_GT(sharpened(0, 0), assignment(0, 0));
  EXPECT_GT(sharpened(1, 1), assignment(1, 1));
  EXPECT_LT(sharpened(0, 1), assignment(0, 1));
  EXPECT_LT(sharpened(1, 0), assignment(1, 0));
  EXPECT_NEAR(sharpened.row(0).sum(), assignment.row(0).sum(), 1e-9);
  EXPECT_NEAR(sharpened.row(1).sum(), assignment.row(1).sum(), 1e-9);
  EXPECT_LE(sharpened.col(0).sum(), 1.0 + 1e-9);
  EXPECT_LE(sharpened.col(1).sum(), 1.0 + 1e-9);
}

TEST(testDPGO, BilevelAssociationHardLowerProjectionKeepsGreedyMatchingMass) {
  Matrix assignment(3, 3);
  assignment << 0.8, 0.1, 0.0,
                0.2, 0.7, 0.1,
                0.0, 0.3, 0.6;

  const Matrix projected = projectAssociationForLowerGraphHard(assignment);
  ASSERT_EQ(projected.rows(), assignment.rows());
  ASSERT_EQ(projected.cols(), assignment.cols());
  EXPECT_DOUBLE_EQ(projected(0, 0), 0.8);
  EXPECT_DOUBLE_EQ(projected(1, 1), 0.7);
  EXPECT_DOUBLE_EQ(projected(2, 2), 0.6);
  EXPECT_DOUBLE_EQ(projected(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(projected(1, 0), 0.0);
  for (int row = 0; row < projected.rows(); ++row) {
    EXPECT_LE(projected.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < projected.cols(); ++col) {
    EXPECT_LE(projected.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO, BilevelAssociationPartialSinkhornCanBlendPreviousBeliefs) {
  Matrix costs(2, 2);
  costs << 0.0, 8.0,
           8.0, 0.0;
  Matrix previous(2, 2);
  previous << 0.1, 0.9,
              0.8, 0.2;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 0.5;
  options.temporalBlend = 0.75;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs, previous);

  EXPECT_GT(pi(0, 1), 0.5);
  EXPECT_GT(pi(1, 0), 0.5);
  EXPECT_GT(pi(0, 0), 0.0);
  for (int row = 0; row < pi.rows(); ++row) {
    EXPECT_LE(pi.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < pi.cols(); ++col) {
    EXPECT_LE(pi.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO, BilevelAssociationPartialSinkhornUsesExtraGradient) {
  Matrix costs = Matrix::Zero(2, 3);
  Matrix previous = Matrix::Zero(2, 3);
  previous(1, 2) = 1.0;
  Matrix gradient = Matrix::Zero(2, 3);
  gradient(1, 2) = 1.0;
  gradient(1, 1) = -1.0;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi =
      optimizer.updateFromCostsAndGradient(costs, gradient, previous);

  EXPECT_LT(pi(1, 2), previous(1, 2));
  EXPECT_GT(pi(1, 1), previous(1, 1));
  EXPECT_LE(pi.row(1).sum(), 1.0 + 1e-12);
}

TEST(testDPGO, BilevelAssociationDistributedUpperTrackerMixesNeighbors) {
  DistributedUpperTrackingState state;
  state.initialized = true;
  state.tracker = Matrix::Zero(2, 2);
  state.tracker << 1.0, 0.0,
                   0.0, -1.0;
  state.previousLocalGradient = state.tracker;
  const Matrix previousGradient = state.previousLocalGradient;

  Matrix localGradient(2, 2);
  localGradient << 2.0, 0.0,
                   0.0, -2.0;
  Matrix neighborTracker(2, 2);
  neighborTracker << -1.0, 0.0,
                     0.0, 1.0;

  DistributedUpperTrackingOptions options;
  options.consensusWeight = 0.5;
  options.correctionWeight = 1.0;
  options.centerRows = false;
  options.centerColumns = false;

  const DistributedUpperTrackingResult result =
      updateDistributedUpperTracker(state, localGradient, {neighborTracker},
                                    options);

  Matrix expectedTracker(2, 2);
  expectedTracker << 0.0, 0.0,
                     0.0, 0.0;
  expectedTracker += localGradient - previousGradient;
  EXPECT_LT((state.tracker - expectedTracker).norm(), 1e-12);
  EXPECT_LT((result.trackedGradient - state.tracker).norm(), 1e-12);
  EXPECT_LT((result.correction -
             (expectedTracker - localGradient)).norm(),
            1e-12);
  EXPECT_GT(result.consensusResidualNorm, 0.0);
  EXPECT_GT(result.localGradientDeltaNorm, 0.0);
}

TEST(testDPGO, BilevelAssociationDistributedUpperTrackerSkipsEmptyGradient) {
  DistributedUpperTrackingState state;
  Matrix localGradient(0, 2);

  DistributedUpperTrackingOptions options;
  options.consensusWeight = 0.5;
  options.correctionWeight = 1.0;

  const DistributedUpperTrackingResult result =
      updateDistributedUpperTracker(state, localGradient, {}, options);

  EXPECT_FALSE(state.initialized);
  EXPECT_EQ(result.trackedGradient.size(), 0);
  EXPECT_EQ(result.correction.size(), 0);
  EXPECT_DOUBLE_EQ(result.trackerNorm, 0.0);
}

TEST(testDPGO,
     BilevelAssociationDistributedUpperTrackerMasksInactiveGradientEntries) {
  DistributedUpperTrackingState state;
  Matrix initialGradient(2, 2);
  initialGradient << 1.0, std::numeric_limits<double>::infinity(),
                     0.0, -1.0;

  DistributedUpperTrackingOptions options;
  options.consensusWeight = 0.5;
  options.correctionWeight = 1.0;
  options.centerRows = true;
  options.centerColumns = true;

  const DistributedUpperTrackingResult initial =
      updateDistributedUpperTracker(state, initialGradient, {}, options);
  EXPECT_TRUE(state.initialized);
  EXPECT_TRUE(initial.correction.allFinite());
  EXPECT_DOUBLE_EQ(initial.correction(0, 1), 0.0);

  Matrix localGradient(2, 2);
  localGradient << 2.0, std::numeric_limits<double>::infinity(),
                   0.0, -2.0;
  Matrix neighborTracker = Matrix::Zero(2, 2);

  const DistributedUpperTrackingResult updated =
      updateDistributedUpperTracker(state, localGradient, {neighborTracker},
                                    options);

  EXPECT_TRUE(updated.correction.allFinite());
  EXPECT_DOUBLE_EQ(updated.correction(0, 1), 0.0);
  EXPECT_GT(updated.localGradientDeltaNorm, 0.0);
}

TEST(testDPGO, BilevelAssociationGaugeStationarityRemovesTranslationGauge) {
  Matrix gradient = Matrix::Zero(2, 6);
  gradient.col(2) << 1.0, -2.0;
  gradient.col(5) << 1.0, -2.0;

  const LowerGaugeStationarityDiagnostics diagnostics =
      evaluateLowerGaugeStationarity(gradient, 2);

  EXPECT_GT(diagnostics.rawGradientNorm, 0.0);
  EXPECT_LT(diagnostics.quotientGradientNorm, 1e-12);
  EXPECT_NEAR(diagnostics.gaugeComponentNorm,
              diagnostics.rawGradientNorm, 1e-12);
  EXPECT_NEAR(diagnostics.gaugeComponentRatio, 1.0, 1e-12);
}

TEST(testDPGO, BilevelAssociationGaugeStationarityKeepsNonGaugeResidual) {
  Matrix gradient = Matrix::Zero(2, 6);
  gradient.col(2) << 1.0, 0.0;
  gradient.col(5) << -1.0, 0.0;

  const LowerGaugeStationarityDiagnostics diagnostics =
      evaluateLowerGaugeStationarity(gradient, 2);

  EXPECT_GT(diagnostics.rawGradientNorm, 0.0);
  EXPECT_NEAR(diagnostics.gaugeComponentNorm, 0.0, 1e-12);
  EXPECT_NEAR(diagnostics.quotientGradientNorm,
              diagnostics.rawGradientNorm, 1e-12);
  EXPECT_NEAR(diagnostics.gaugeComponentRatio, 0.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationGaugeStationarityRemovesFullSeGaugeTangent) {
  constexpr unsigned d = 2;
  constexpr unsigned blockWidth = d + 1;
  constexpr std::size_t blocks = 3;
  Matrix state = Matrix::Zero(d, blocks * blockWidth);
  const std::array<double, blocks> angles{0.0, 0.4, -0.7};
  const std::array<Vector, blocks> translations{
      (Vector(d) << 0.0, 0.0).finished(),
      (Vector(d) << 2.0, -1.0).finished(),
      (Vector(d) << -0.5, 3.0).finished()};
  Matrix omega(d, d);
  omega << 0.0, -1.0,
           1.0, 0.0;
  const Vector commonTranslation =
      (Vector(d) << 0.3, -0.2).finished();
  Matrix gradient = Matrix::Zero(state.rows(), state.cols());
  for (std::size_t block = 0; block < blocks; ++block) {
    const double c = std::cos(angles[block]);
    const double s = std::sin(angles[block]);
    Matrix rotation(d, d);
    rotation << c, -s,
                s, c;
    const int col = static_cast<int>(block * blockWidth);
    state.block(0, col, d, d) = rotation;
    state.col(col + d) = translations[block];
    gradient.block(0, col, d, d) = omega * rotation;
    gradient.col(col + d) =
        omega * translations[block] + commonTranslation;
  }

  const LowerGaugeStationarityDiagnostics translationOnly =
      evaluateLowerGaugeStationarity(gradient, d);
  const LowerGaugeStationarityDiagnostics fullGauge =
      evaluateLowerGaugeStationarity(state, gradient, d);

  EXPECT_GT(translationOnly.quotientGradientNorm, 0.1);
  EXPECT_GT(fullGauge.rawGradientNorm, 0.0);
  EXPECT_LT(fullGauge.quotientGradientNorm, 1e-12);
  EXPECT_NEAR(fullGauge.gaugeComponentNorm,
              fullGauge.rawGradientNorm, 1e-12);
  EXPECT_NEAR(fullGauge.gaugeComponentRatio, 1.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationGaugeStationarityIsCovariantAtLiftedRank) {
  constexpr unsigned d = 3;
  constexpr unsigned rank = d + 2;
  constexpr unsigned blockWidth = d + 1;
  Matrix state = Matrix::Zero(rank, 2 * blockWidth);
  state.block(0, 0, rank, d) = Matrix::Identity(rank, d);
  state.block(0, blockWidth, rank, d) = Matrix::Identity(rank, d);
  state.col(d) << 0.2, -0.4, 0.8, 0.1, -0.3;
  state.col(2 * blockWidth - 1) << 1.0, 0.7, -0.2, 0.4, 0.9;

  Matrix omega = Matrix::Zero(rank, rank);
  omega(0, 3) = -0.7;
  omega(3, 0) = 0.7;
  omega(2, 4) = 0.4;
  omega(4, 2) = -0.4;
  const Vector translationGauge =
      (Vector(rank) << 0.3, -0.1, 0.2, 0.5, -0.4).finished();
  Matrix gradient = Matrix::Zero(rank, state.cols());
  for (int block = 0; block < 2; ++block) {
    const int col = block * blockWidth;
    gradient.block(0, col, rank, d) =
        omega * state.block(0, col, rank, d);
    gradient.col(col + d) = omega * state.col(col + d) + translationGauge;
  }

  Matrix gauge = Matrix::Identity(rank, rank);
  const double c = std::cos(0.37);
  const double s = std::sin(0.37);
  gauge(1, 1) = c;
  gauge(1, 4) = -s;
  gauge(4, 1) = s;
  gauge(4, 4) = c;
  const LowerGaugeStationarityDiagnostics original =
      evaluateLowerGaugeStationarity(state, gradient, d);
  const LowerGaugeStationarityDiagnostics transformed =
      evaluateLowerGaugeStationarity(gauge * state, gauge * gradient, d);

  EXPECT_LT(original.quotientGradientNorm, 1e-12);
  EXPECT_NEAR(transformed.rawGradientNorm, original.rawGradientNorm, 1e-12);
  EXPECT_NEAR(transformed.quotientGradientNorm,
              original.quotientGradientNorm, 1e-12);
  EXPECT_NEAR(transformed.gaugeComponentNorm,
              original.gaugeComponentNorm, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationPartialSinkhornCanTrustPreviousSupport) {
  Matrix costs = Matrix::Zero(2, 3);
  Matrix previous = Matrix::Zero(2, 3);
  previous(0, 0) = 1.0;
  previous(1, 2) = 1.0;
  Matrix gradient = Matrix::Zero(2, 3);
  gradient(0, 1) = -100.0;
  gradient(1, 1) = -100.0;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.restrictToPreviousSupport = true;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi =
      optimizer.updateFromCostsAndGradient(costs, gradient, previous);

  EXPECT_GT(pi(0, 0), 0.0);
  EXPECT_GT(pi(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(pi(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(pi(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(pi(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(pi(1, 0), 0.0);
  for (int row = 0; row < pi.rows(); ++row) {
    EXPECT_LE(pi.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < pi.cols(); ++col) {
    EXPECT_LE(pi.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationPartialSinkhornSupportTrustPreservesPreviousSupport) {
  Matrix costs = Matrix::Zero(3, 3);
  Matrix previous = Matrix::Zero(3, 3);
  previous(0, 0) = 1.0;
  previous(1, 1) = 1.0;
  previous(2, 2) = 1.0;

  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.maxNonzeros = 1;
  options.restrictToPreviousSupport = true;
  SoftAssignmentOptimizer optimizer(options);

  const Matrix pi = optimizer.updateFromCosts(costs, previous);

  EXPECT_GT(pi(0, 0), 0.0);
  EXPECT_GT(pi(1, 1), 0.0);
  EXPECT_GT(pi(2, 2), 0.0);
  EXPECT_DOUBLE_EQ(pi(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(pi(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(pi(2, 0), 0.0);
}

TEST(testDPGO,
     BilevelAssociationRelationalGeometryObjectiveMatchesFiniteDifference) {
  Matrix sourceDistances(3, 3);
  sourceDistances << 0.0, 1.0, 2.5,
                     1.0, 0.0, 1.5,
                     2.5, 1.5, 0.0;
  Matrix targetDistances(3, 3);
  targetDistances << 0.0, 1.2, 2.2,
                     1.2, 0.0, 1.7,
                     2.2, 1.7, 0.0;
  Matrix assignment(3, 3);
  assignment << 0.55, 0.20, 0.05,
                0.10, 0.60, 0.15,
                0.20, 0.05, 0.50;

  AssociationRelationalGeometryOptions options;
  options.weight = 0.7;
  options.distanceScale = 2.0;
  const AssociationRelationalGeometryResult result =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, assignment, options);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost = evaluateAssociationRelationalGeometryObjective(
                                  sourceDistances, targetDistances, plus,
                                  options)
                                  .cost;
      const double minusCost = evaluateAssociationRelationalGeometryObjective(
                                   sourceDistances, targetDistances, minus,
                                   options)
                                   .cost;
      const double numerical = (plusCost - minusCost) / (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-6);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationRelationalGeometryIsExtensiveInAssignmentMass) {
  Matrix sourceDistances(3, 3);
  sourceDistances << 0.0, 1.0, 2.5,
                     1.0, 0.0, 1.5,
                     2.5, 1.5, 0.0;
  Matrix targetDistances(3, 3);
  targetDistances << 0.0, 1.2, 2.2,
                     1.2, 0.0, 1.7,
                     2.2, 1.7, 0.0;
  Matrix assignment(3, 3);
  assignment << 0.55, 0.20, 0.05,
                0.10, 0.60, 0.15,
                0.20, 0.05, 0.50;

  AssociationRelationalGeometryOptions options;
  options.weight = 0.7;
  options.distanceScale = 2.0;
  const AssociationRelationalGeometryResult base =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, assignment, options);
  constexpr double scale = 0.4;
  const AssociationRelationalGeometryResult scaled =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, scale * assignment, options);

  EXPECT_NEAR(scaled.cost, scale * base.cost, 1e-12);
  EXPECT_LT((scaled.gradient - base.gradient).norm(), 1e-12);
}

TEST(testDPGO,
     BilevelAssociationRelationalGeometryCanBeScaleInvariantInAssignmentMass) {
  Matrix sourceDistances(3, 3);
  sourceDistances << 0.0, 1.0, 2.5,
                     1.0, 0.0, 1.5,
                     2.5, 1.5, 0.0;
  Matrix targetDistances(3, 3);
  targetDistances << 0.0, 1.2, 2.2,
                     1.2, 0.0, 1.7,
                     2.2, 1.7, 0.0;
  Matrix assignment(3, 3);
  assignment << 0.55, 0.20, 0.05,
                0.10, 0.60, 0.15,
                0.20, 0.05, 0.50;

  AssociationRelationalGeometryOptions options;
  options.weight = 0.7;
  options.distanceScale = 2.0;
  options.normalizeBySquaredAssignmentMass = true;
  const AssociationRelationalGeometryResult base =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, assignment, options);
  constexpr double scale = 0.4;
  const AssociationRelationalGeometryResult scaled =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, scale * assignment, options);

  EXPECT_NEAR(scaled.cost, base.cost, 1e-12);
  EXPECT_LT((scale * scaled.gradient - base.gradient).norm(), 1e-12);
  EXPECT_NEAR((base.gradient.array() * assignment.array()).sum(), 0.0,
              1e-12);
}

TEST(testDPGO,
     BilevelAssociationRelationalAutoScaleIgnoresInactiveFarObjects) {
  Matrix sourceDistances(3, 3);
  sourceDistances << 0.0, 1.0, 4.0,
                     1.0, 0.0, 3.0,
                     4.0, 3.0, 0.0;
  Matrix targetDistances(3, 3);
  targetDistances << 0.0, 1.2, 5.0,
                     1.2, 0.0, 2.0,
                     5.0, 2.0, 0.0;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 1.0;
  assignment(1, 1) = 1.0;

  AssociationRelationalGeometryOptions options;
  options.weight = 1.0;
  options.distanceScale = 0.0;
  const AssociationRelationalGeometryResult base =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, assignment, options);

  sourceDistances(0, 2) = sourceDistances(2, 0) = 4000.0;
  sourceDistances(1, 2) = sourceDistances(2, 1) = 3000.0;
  targetDistances(0, 2) = targetDistances(2, 0) = 5000.0;
  targetDistances(1, 2) = targetDistances(2, 1) = 2000.0;
  const AssociationRelationalGeometryResult withFarInactive =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, assignment, options);

  EXPECT_NEAR(base.cost, withFarInactive.cost, 1e-12);
  EXPECT_NEAR(base.gradient.block(0, 0, 2, 2).norm(),
              withFarInactive.gradient.block(0, 0, 2, 2).norm(), 1e-12);
}

TEST(testDPGO, BilevelAssociationSupportCouplingPersistsSelectedGraphEvidence) {
  EXPECT_DOUBLE_EQ(
      partialAssignmentSupportCoupling(0.0, 0.0, 0.0, 1.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(
      partialAssignmentSupportCoupling(1.0, 1.0, 0.0, 1.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(
      partialAssignmentSupportCoupling(1.0, 1.0, 1.0, 1.0, 1.0), 1.0);
  EXPECT_NEAR(
      partialAssignmentSupportCoupling(0.75, 0.36, 0.10, 1.0, 1.0),
      std::sqrt(0.35 * 0.74), 1e-12);
}

TEST(testDPGO,
     BilevelAssociationRelationalGeometryObjectivePrefersConsistentPermutation) {
  Matrix sourceDistances(3, 3);
  sourceDistances << 0.0, 1.0, 3.0,
                     1.0, 0.0, 2.0,
                     3.0, 2.0, 0.0;
  Matrix targetDistances(3, 3);
  targetDistances << 0.0, 1.0, 2.0,
                     1.0, 0.0, 3.0,
                     2.0, 3.0, 0.0;
  Matrix consistent = Matrix::Zero(3, 3);
  consistent(0, 1) = 1.0;
  consistent(1, 0) = 1.0;
  consistent(2, 2) = 1.0;
  Matrix inconsistent = Matrix::Identity(3, 3);

  AssociationRelationalGeometryOptions options;
  options.weight = 1.0;
  options.distanceScale = 1.0;
  const double consistentCost =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, consistent, options)
          .cost;
  const double inconsistentCost =
      evaluateAssociationRelationalGeometryObjective(
          sourceDistances, targetDistances, inconsistent, options)
          .cost;

  EXPECT_NEAR(consistentCost, 0.0, 1e-12);
  EXPECT_NEAR(inconsistentCost, 2.0 / 3.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationRelationalNullEvidenceRejectsCyclicMismatch) {
  constexpr int objectCount = 10;
  Vector sourcePoints(objectCount);
  sourcePoints << 0.0, 1.1, 2.7, 4.8, 7.4, 10.5, 14.1, 18.2, 22.8,
      27.9;
  Vector targetPoints = sourcePoints;
  targetPoints << 0.01, 1.08, 2.73, 4.79, 7.42, 10.47, 14.12, 18.19,
      22.82, 27.88;
  Matrix sourceDistances(objectCount, objectCount);
  Matrix targetDistances(objectCount, objectCount);
  for (int row = 0; row < objectCount; ++row) {
    for (int col = 0; col < objectCount; ++col) {
      sourceDistances(row, col) =
          std::abs(sourcePoints(row) - sourcePoints(col));
      targetDistances(row, col) =
          std::abs(targetPoints(row) - targetPoints(col));
    }
  }
  Matrix assignment = Matrix::Zero(objectCount, objectCount);
  for (int index = 0; index < 8; ++index) {
    assignment(index, index) = 0.9;
  }
  Matrix support = Matrix::Zero(objectCount, objectCount);
  std::vector<std::size_t> sourceCategories(objectCount, 10u);
  std::vector<std::size_t> targetCategories(objectCount, 10u);

  const AssociationRelationalNullEvidenceResult result =
      evaluateAssociationRelationalNullEvidence(
          sourceDistances, targetDistances, assignment, support,
          sourceCategories, targetCategories);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.anchorCount, 8u);
  EXPECT_EQ(result.nullSampleCount, 56u);
  EXPECT_LT(result.costOffset(8, 8), 0.0);
  EXPECT_LT(result.costOffset(8, 8), result.costOffset(8, 9));
  EXPECT_LT(result.costOffset(9, 9), result.costOffset(9, 8));

  const AssociationRelationalNullEvidenceResult reverse =
      evaluateAssociationRelationalNullEvidence(
          targetDistances, sourceDistances, assignment.transpose(),
          support.transpose(), targetCategories, sourceCategories);
  ASSERT_TRUE(reverse.valid);
  EXPECT_LT((result.costOffset - reverse.costOffset.transpose()).norm(),
            1e-12);
}

TEST(testDPGO,
     BilevelAssociationRelationalNullEvidenceNeedsEnoughAnchors) {
  Matrix distances(4, 4);
  distances << 0.0, 1.0, 2.0, 3.0,
               1.0, 0.0, 1.0, 2.0,
               2.0, 1.0, 0.0, 1.0,
               3.0, 2.0, 1.0, 0.0;
  Matrix assignment = Matrix::Identity(4, 4);
  Matrix support = Matrix::Zero(4, 4);
  std::vector<std::size_t> categories(4, 0u);

  const AssociationRelationalNullEvidenceResult result =
      evaluateAssociationRelationalNullEvidence(
          distances, distances, assignment, support, categories, categories);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.anchorCount, 4u);
  EXPECT_DOUBLE_EQ(result.costOffset.norm(), 0.0);
}

TEST(testDPGO,
     BilevelAssociationAdaptiveRelationalNullUsesDustbinReciprocalAnchors) {
  constexpr int objectCount = 7;
  Vector sourcePoints(objectCount);
  sourcePoints << 0.0, 1.1, 2.7, 4.8, 7.4, 10.5, 14.1;
  Vector targetPoints(objectCount);
  targetPoints << 0.01, 1.08, 2.73, 4.79, 7.42, 10.47, 14.12;
  Matrix sourceDistances(objectCount, objectCount);
  Matrix targetDistances(objectCount, objectCount);
  for (int row = 0; row < objectCount; ++row) {
    for (int col = 0; col < objectCount; ++col) {
      sourceDistances(row, col) =
          std::abs(sourcePoints(row) - sourcePoints(col));
      targetDistances(row, col) =
          std::abs(targetPoints(row) - targetPoints(col));
    }
  }

  Matrix assignment = Matrix::Zero(objectCount, objectCount);
  for (int index = 0; index < 5; ++index) {
    assignment(index, index) = 0.45;
    assignment(index, (index + 1) % 5) = 0.20;
  }
  assignment(5, 0) = 0.46;
  assignment(5, 5) = 0.80;
  Matrix support = Matrix::Zero(objectCount, objectCount);
  std::vector<std::size_t> categories(objectCount, 10u);

  AssociationRelationalNullEvidenceOptions options;
  options.calibration =
      AssociationRelationalNullCalibration::AdaptiveEmpiricalRank;
  options.geometricDimension = 3;
  const AssociationRelationalNullEvidenceResult result =
      evaluateAssociationRelationalNullEvidence(
          sourceDistances, targetDistances, assignment, support, categories,
          categories, options);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.anchorCount, 5u);
  EXPECT_GT(result.nullSampleCount, 0u);
}

TEST(testDPGO,
     BilevelAssociationAdaptiveRelationalNullCentersRanksAndTransposes) {
  constexpr int objectCount = 7;
  Vector sourcePoints(objectCount);
  sourcePoints << 0.0, 1.1, 2.7, 4.8, 7.4, 10.5, 14.1;
  Vector targetPoints(objectCount);
  targetPoints << 0.01, 1.08, 2.73, 4.79, 7.42, 10.47, 14.12;
  Matrix sourceDistances(objectCount, objectCount);
  Matrix targetDistances(objectCount, objectCount);
  for (int row = 0; row < objectCount; ++row) {
    for (int col = 0; col < objectCount; ++col) {
      sourceDistances(row, col) =
          std::abs(sourcePoints(row) - sourcePoints(col));
      targetDistances(row, col) =
          std::abs(targetPoints(row) - targetPoints(col));
    }
  }

  Matrix assignment = Matrix::Zero(objectCount, objectCount);
  for (int index = 0; index < 5; ++index) {
    assignment(index, index) = 0.45;
    assignment(index, (index + 1) % 5) = 0.20;
  }
  Matrix support = Matrix::Zero(objectCount, objectCount);
  std::vector<std::size_t> categories(objectCount, 10u);

  AssociationRelationalNullEvidenceOptions options;
  options.calibration =
      AssociationRelationalNullCalibration::AdaptiveEmpiricalRank;
  options.geometricDimension = 3;
  const AssociationRelationalNullEvidenceResult result =
      evaluateAssociationRelationalNullEvidence(
          sourceDistances, targetDistances, assignment, support, categories,
          categories, options);

  ASSERT_TRUE(result.valid);
  const double denominator =
      static_cast<double>(result.nullSampleCount + 1u);
  double expectedNullLogMean = 0.0;
  for (std::size_t rank = 1; rank <= result.nullSampleCount + 1u; ++rank) {
    expectedNullLogMean +=
        std::log(static_cast<double>(rank) / denominator);
  }
  expectedNullLogMean /= denominator;
  EXPECT_NEAR(result.centeredNullLogMean, expectedNullLogMean, 1e-12);
  EXPECT_NEAR(result.empiricalRankResolution, 1.0 / denominator, 1e-12);
  EXPECT_LT(result.costOffset(5, 5), 0.0);
  EXPECT_GT(result.costOffset(5, 0), 0.0);

  const AssociationRelationalNullEvidenceResult reverse =
      evaluateAssociationRelationalNullEvidence(
          targetDistances, sourceDistances, assignment.transpose(), support,
          categories, categories, options);
  ASSERT_TRUE(reverse.valid);
  EXPECT_LT((result.costOffset - reverse.costOffset.transpose()).norm(),
            1e-12);
}

TEST(testDPGO,
     BilevelAssociationRotoTranslationAlignmentIsRigidTransformInvariant) {
  Matrix sourcePoints(3, 2);
  sourcePoints << 0.0, 0.0,
                  2.0, 0.0,
                  0.0, 1.0;
  const double theta = 0.7;
  Matrix rotation(2, 2);
  rotation << std::cos(theta), -std::sin(theta),
              std::sin(theta), std::cos(theta);
  Vector translation(2);
  translation << 4.0, -1.5;
  Matrix targetPoints(3, 2);
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    targetPoints.row(row) =
        (rotation * sourcePoints.row(row).transpose() + translation)
            .transpose();
  }

  Matrix consistent = Matrix::Identity(3, 3);
  Matrix inconsistent = Matrix::Zero(3, 3);
  inconsistent(0, 1) = 1.0;
  inconsistent(1, 0) = 1.0;
  inconsistent(2, 2) = 1.0;

  AssociationRotoTranslationAlignmentOptions options;
  options.weight = 1.0;
  options.distanceScale = 1.0;
  const double consistentCost =
      evaluateAssociationRotoTranslationAlignmentObjective(
          sourcePoints, targetPoints, consistent, options)
          .cost;
  const double inconsistentCost =
      evaluateAssociationRotoTranslationAlignmentObjective(
          sourcePoints, targetPoints, inconsistent, options)
          .cost;

  EXPECT_NEAR(consistentCost, 0.0, 1e-10);
  EXPECT_GT(inconsistentCost, 0.1);
}

TEST(testDPGO,
     BilevelAssociationSparseRotoTranslationEstimateMatchesDenseAlignment) {
  Matrix sourcePoints(4, 3);
  sourcePoints << 0.0, 0.0, 0.0,
                  1.0, 0.0, 0.2,
                  0.0, 2.0, -0.1,
                  1.0, 1.0, 0.5;
  Matrix targetPoints(4, 3);
  targetPoints << 0.3, -0.2, 0.1,
                  1.1, 0.4, 0.2,
                 -0.5, 1.8, -0.2,
                  0.8, 1.2, 0.6;
  Matrix assignment = Matrix::Zero(4, 4);
  assignment(0, 0) = 0.8;
  assignment(0, 1) = 0.1;
  assignment(1, 1) = 0.7;
  assignment(2, 2) = 0.9;
  assignment(3, 3) = 0.6;
  assignment(3, 2) = 0.2;
  std::vector<SparseAssignmentWeightEntry> entries;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      if (assignment(row, col) > 0.0) {
        entries.push_back({row, col, assignment(row, col)});
      }
    }
  }

  AssociationRotoTranslationAlignmentOptions options;
  options.weight = 1.0;
  options.distanceScale = 0.0;
  const AssociationRotoTranslationAlignmentResult dense =
      evaluateAssociationRotoTranslationAlignmentObjective(
          sourcePoints, targetPoints, assignment, options);
  const AssociationRotoTranslationEstimate sparse =
      estimateSparseAssociationRotoTranslationAlignment(
          sourcePoints, targetPoints, entries);

  EXPECT_NEAR(sparse.mass, dense.mass, 1e-12);
  EXPECT_LT((sparse.rotation - dense.rotation).norm(), 1e-12);
  EXPECT_LT((sparse.translation - dense.translation).norm(), 1e-12);
}

TEST(testDPGO,
     BilevelAssociationPoseAlignmentResolvesCollinearCenterRotation) {
  constexpr int d = 3;
  constexpr int count = 4;
  const double angle = 0.75;
  Matrix expectedRotation = Matrix::Identity(d, d);
  expectedRotation(1, 1) = std::cos(angle);
  expectedRotation(1, 2) = -std::sin(angle);
  expectedRotation(2, 1) = std::sin(angle);
  expectedRotation(2, 2) = std::cos(angle);
  const Vector expectedTranslation =
      (Vector(d) << 2.0, -3.0, 1.0).finished();

  std::vector<Matrix> sourcePoses;
  std::vector<Matrix> targetPoses;
  std::vector<SparsePoseAlignmentWeightEntry> entries;
  Matrix sourcePoints(count, d);
  Matrix targetPoints(count, d);
  const std::array<double, count> coordinates{0.0, 2.0, 5.0, 9.0};
  for (int index = 0; index < count; ++index) {
    Matrix targetPose = Matrix::Zero(d, d + 1);
    targetPose.leftCols(d) = Matrix::Identity(d, d);
    targetPose(0, d) = coordinates[static_cast<std::size_t>(index)];
    Matrix sourcePose = Matrix::Zero(d, d + 1);
    sourcePose.leftCols(d) = expectedRotation;
    sourcePose.col(d) =
        expectedRotation * targetPose.col(d) + expectedTranslation;
    targetPoses.push_back(targetPose);
    sourcePoses.push_back(sourcePose);
    targetPoints.row(index) = targetPose.col(d).transpose();
    sourcePoints.row(index) = sourcePose.col(d).transpose();
    entries.push_back(
        SparsePoseAlignmentWeightEntry{index, index, 1.0, 1.0, 1.0});
  }

  std::vector<SparseAssignmentWeightEntry> pointEntries;
  for (int index = 0; index < count; ++index) {
    pointEntries.push_back({index, index, 1.0});
  }
  const AssociationRotoTranslationEstimate pointOnly =
      estimateSparseAssociationRotoTranslationAlignment(
          sourcePoints, targetPoints, pointEntries);
  const AssociationRotoTranslationEstimate poseAware =
      estimateSparseAssociationPoseAlignment(
          sourcePoses, targetPoses, entries);

  EXPECT_GT((pointOnly.rotation - expectedRotation).norm(), 0.1);
  EXPECT_LT((poseAware.rotation - expectedRotation).norm(), 1e-12);
  EXPECT_LT((poseAware.translation - expectedTranslation).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(poseAware.mass, static_cast<double>(count));
}

TEST(testDPGO,
     BilevelAssociationRotoTranslationAlignmentMatchesFiniteDifference) {
  Matrix sourcePoints(3, 2);
  sourcePoints << -0.2, 0.1,
                   1.3, -0.4,
                   0.4, 1.7;
  Matrix targetPoints(3, 2);
  targetPoints << 0.5, -0.1,
                  1.2, 0.9,
                 -0.6, 1.1;
  Matrix assignment(3, 3);
  assignment << 0.60, 0.15, 0.05,
                0.10, 0.55, 0.20,
                0.05, 0.25, 0.50;

  AssociationRotoTranslationAlignmentOptions options;
  options.weight = 0.8;
  options.distanceScale = 1.7;
  const AssociationRotoTranslationAlignmentResult result =
      evaluateAssociationRotoTranslationAlignmentObjective(
          sourcePoints, targetPoints, assignment, options);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost =
          evaluateAssociationRotoTranslationAlignmentObjective(
              sourcePoints, targetPoints, plus, options)
              .cost;
      const double minusCost =
          evaluateAssociationRotoTranslationAlignmentObjective(
              sourcePoints, targetPoints, minus, options)
              .cost;
      const double numerical = (plusCost - minusCost) / (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-5);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationRotoTranslationHypothesisMixtureSelectsCorrectMode) {
  Matrix sourcePoints(3, 2);
  sourcePoints << 0.0, 0.0,
                  2.0, 0.0,
                  0.0, 1.0;
  const double theta = 0.8;
  Matrix trueRotation(2, 2);
  trueRotation << std::cos(theta), -std::sin(theta),
                  std::sin(theta), std::cos(theta);
  Vector trueTranslation(2);
  trueTranslation << 1.5, -0.4;
  Matrix targetPoints(3, 2);
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    targetPoints.row(row) =
        (trueRotation.transpose() *
         (sourcePoints.row(row).transpose() - trueTranslation))
            .transpose();
  }

  Matrix consistent = Matrix::Identity(3, 3);
  Matrix inconsistent = Matrix::Zero(3, 3);
  inconsistent(0, 1) = 1.0;
  inconsistent(1, 0) = 1.0;
  inconsistent(2, 2) = 1.0;

  std::vector<Matrix> rotations;
  std::vector<Vector> translations;
  rotations.push_back(Matrix::Identity(2, 2));
  translations.push_back(Vector::Zero(2));
  rotations.push_back(trueRotation);
  translations.push_back(trueTranslation);

  AssociationRotoTranslationHypothesisMixtureOptions options;
  options.weight = 1.0;
  options.distanceScale = 1.0;
  options.softminTemperature = 0.02;

  const AssociationRotoTranslationHypothesisMixtureResult consistentResult =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, consistent, rotations, translations,
          options);
  const AssociationRotoTranslationHypothesisMixtureResult inconsistentResult =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, inconsistent, rotations, translations,
          options);

  ASSERT_EQ(consistentResult.hypothesisWeights.size(), 2);
  EXPECT_GT(consistentResult.hypothesisWeights(1), 0.99);
  EXPECT_LT(consistentResult.cost, 0.02);
  EXPECT_GT(inconsistentResult.cost, consistentResult.cost + 0.1);
}

TEST(testDPGO,
     BilevelAssociationRotoTranslationHypothesisMixtureMatchesFiniteDifference) {
  Matrix sourcePoints(3, 2);
  sourcePoints << -0.2, 0.1,
                   1.3, -0.4,
                   0.4, 1.7;
  Matrix targetPoints(3, 2);
  targetPoints << 0.5, -0.1,
                  1.2, 0.9,
                 -0.6, 1.1;
  Matrix assignment(3, 3);
  assignment << 0.60, 0.15, 0.05,
                0.10, 0.55, 0.20,
                0.05, 0.25, 0.50;

  Matrix rotationA = Matrix::Identity(2, 2);
  Vector translationA(2);
  translationA << 0.2, -0.1;
  const double theta = 0.35;
  Matrix rotationB(2, 2);
  rotationB << std::cos(theta), -std::sin(theta),
               std::sin(theta), std::cos(theta);
  Vector translationB(2);
  translationB << -0.4, 0.6;
  std::vector<Matrix> rotations{rotationA, rotationB};
  std::vector<Vector> translations{translationA, translationB};

  AssociationRotoTranslationHypothesisMixtureOptions options;
  options.weight = 0.7;
  options.distanceScale = 1.5;
  options.softminTemperature = 0.4;

  const AssociationRotoTranslationHypothesisMixtureResult result =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, assignment, rotations, translations,
          options);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost =
          evaluateAssociationRotoTranslationHypothesisMixtureObjective(
              sourcePoints, targetPoints, plus, rotations, translations,
              options)
              .cost;
      const double minusCost =
          evaluateAssociationRotoTranslationHypothesisMixtureObjective(
              sourcePoints, targetPoints, minus, rotations, translations,
              options)
              .cost;
      const double numerical = (plusCost - minusCost) / (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-6);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationRotoTranslationHypothesisMixtureCanUseFiniteGradientSupport) {
  Matrix sourcePoints(3, 2);
  sourcePoints << -0.2, 0.1,
                   1.3, -0.4,
                   0.4, 1.7;
  Matrix targetPoints(3, 2);
  targetPoints << 0.5, -0.1,
                  1.2, 0.9,
                 -0.6, 1.1;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 0.6;
  assignment(1, 1) = 0.5;
  assignment(2, 2) = 0.4;

  Matrix gradientSupport =
      Matrix::Constant(3, 3, std::numeric_limits<double>::infinity());
  gradientSupport(0, 0) = 0.0;
  gradientSupport(1, 1) = 0.0;
  gradientSupport(2, 2) = 0.0;
  gradientSupport(0, 2) = 0.0;

  Matrix rotationA = Matrix::Identity(2, 2);
  Vector translationA(2);
  translationA << 0.2, -0.1;
  const double theta = 0.35;
  Matrix rotationB(2, 2);
  rotationB << std::cos(theta), -std::sin(theta),
               std::sin(theta), std::cos(theta);
  Vector translationB(2);
  translationB << -0.4, 0.6;
  std::vector<Matrix> rotations{rotationA, rotationB};
  std::vector<Vector> translations{translationA, translationB};

  AssociationRotoTranslationHypothesisMixtureOptions options;
  options.weight = 0.7;
  options.distanceScale = 1.5;
  options.softminTemperature = 0.4;

  const AssociationRotoTranslationHypothesisMixtureResult dense =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, assignment, rotations, translations,
          options);
  const AssociationRotoTranslationHypothesisMixtureResult supported =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, assignment, gradientSupport, rotations,
          translations, options);

  EXPECT_NEAR(supported.cost, dense.cost, 1e-12);
  EXPECT_NEAR(supported.gradient(0, 0), dense.gradient(0, 0), 1e-12);
  EXPECT_NEAR(supported.gradient(1, 1), dense.gradient(1, 1), 1e-12);
  EXPECT_NEAR(supported.gradient(2, 2), dense.gradient(2, 2), 1e-12);
  EXPECT_NEAR(supported.gradient(0, 2), dense.gradient(0, 2), 1e-12);
  EXPECT_DOUBLE_EQ(supported.gradient(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(supported.gradient(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(supported.gradient(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(supported.gradient(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(supported.gradient(2, 1), 0.0);
}

TEST(testDPGO,
     BilevelAssociationSparseRotoHypothesesMatchDenseOnZeroOutsideSupport) {
  Matrix sourcePoints(3, 2);
  sourcePoints << -0.2, 0.1,
                   1.3, -0.4,
                   0.4, 1.7;
  Matrix targetPoints(3, 2);
  targetPoints << 0.5, -0.1,
                  1.2, 0.9,
                 -0.6, 1.1;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 0.6;
  assignment(1, 1) = 0.5;
  assignment(2, 2) = 0.4;
  std::vector<SparseAssignmentCostEntry> support = {
      {0, 0, 0.0}, {1, 1, 0.0}, {2, 2, 0.0}, {0, 2, 0.0}};

  AssociationRotoTranslationRingHypothesisOptions ringOptions;
  ringOptions.maxHypotheses = 3;
  ringOptions.angleBins = 16;
  const AssociationRotoTranslationRingHypothesisResult denseHypotheses =
      generateAssociationRotoTranslationRingHypotheses(
          sourcePoints, targetPoints, assignment, ringOptions);
  const AssociationRotoTranslationRingHypothesisResult sparseHypotheses =
      generateAssociationRotoTranslationRingHypotheses(
          sourcePoints, targetPoints, assignment, support, ringOptions);

  ASSERT_EQ(sparseHypotheses.rotations.size(),
            denseHypotheses.rotations.size());
  ASSERT_EQ(sparseHypotheses.translations.size(),
            denseHypotheses.translations.size());
  EXPECT_NEAR((sparseHypotheses.scores - denseHypotheses.scores).norm(),
              0.0, 1e-12);
  for (std::size_t idx = 0; idx < sparseHypotheses.rotations.size(); ++idx) {
    EXPECT_NEAR((sparseHypotheses.rotations[idx] -
                 denseHypotheses.rotations[idx]).norm(),
                0.0, 1e-12);
    EXPECT_NEAR((sparseHypotheses.translations[idx] -
                 denseHypotheses.translations[idx]).norm(),
                0.0, 1e-12);
  }

  AssociationRotoTranslationHypothesisMixtureOptions mixtureOptions;
  mixtureOptions.weight = 0.7;
  mixtureOptions.distanceScale = 1.5;
  mixtureOptions.softminTemperature = 0.4;
  const AssociationRotoTranslationHypothesisMixtureResult dense =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, assignment,
          denseHypotheses.rotations, denseHypotheses.translations,
          mixtureOptions);
  const AssociationRotoTranslationHypothesisMixtureResult sparse =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, assignment, support,
          sparseHypotheses.rotations, sparseHypotheses.translations,
          mixtureOptions);

  EXPECT_NEAR(sparse.cost, dense.cost, 1e-12);
  EXPECT_NEAR(sparse.gradient(0, 0), dense.gradient(0, 0), 1e-12);
  EXPECT_NEAR(sparse.gradient(1, 1), dense.gradient(1, 1), 1e-12);
  EXPECT_NEAR(sparse.gradient(2, 2), dense.gradient(2, 2), 1e-12);
  EXPECT_NEAR(sparse.gradient(0, 2), dense.gradient(0, 2), 1e-12);
  EXPECT_DOUBLE_EQ(sparse.gradient(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(sparse.gradient(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(sparse.gradient(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(sparse.gradient(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(sparse.gradient(2, 1), 0.0);
}

TEST(testDPGO,
     BilevelAssociationPairwiseTransformValidationMatchesFiniteDifference) {
  Matrix source(3, 2);
  source << 0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0;
  Matrix target(3, 2);
  target << 0.1, 0.0,
            1.1, 0.2,
            0.0, 0.9;
  Matrix assignment(3, 3);
  assignment << 0.55, 0.20, 0.05,
                0.10, 0.60, 0.15,
                0.20, 0.05, 0.50;
  Matrix rotation = Matrix::Identity(2, 2);

  AssociationPairwiseTransformValidationOptions options;
  options.weight = 0.8;
  options.distanceScale = 1.5;
  const AssociationPairwiseTransformValidationResult result =
      evaluateAssociationPairwiseTransformValidationObjective(
          source, target, assignment, rotation, options);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost =
          evaluateAssociationPairwiseTransformValidationObjective(
              source, target, plus, rotation, options)
              .cost;
      const double minusCost =
          evaluateAssociationPairwiseTransformValidationObjective(
              source, target, minus, rotation, options)
              .cost;
      EXPECT_NEAR(result.gradient(row, col),
                  (plusCost - minusCost) / (2.0 * eps), 1e-5);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationPairwiseTransformValidationPenalizesAliasCandidate) {
  Matrix source(3, 2);
  source << 0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0;
  Matrix target = source;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 1.0;
  assignment(1, 1) = 1.0;
  assignment(2, 2) = 1.0;
  Matrix rotation = Matrix::Identity(2, 2);

  AssociationPairwiseTransformValidationOptions options;
  options.weight = 1.0;
  options.distanceScale = 1.0;
  const AssociationPairwiseTransformValidationResult result =
      evaluateAssociationPairwiseTransformValidationObjective(
          source, target, assignment, rotation, options);

  EXPECT_NEAR(result.cost, 0.0, 1e-12);
  EXPECT_LT(result.gradient(0, 0), 1e-12);
  EXPECT_GT(result.gradient(0, 1), 1.0);
  EXPECT_GT(result.gradient(1, 0), 1.0);
}

TEST(testDPGO,
     BilevelAssociationPairwiseTransformValidationCanLimitGradientSupport) {
  Matrix source(3, 2);
  source << 0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0;
  Matrix target = source;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 1.0;
  assignment(1, 1) = 1.0;
  assignment(2, 2) = 1.0;
  Matrix rotation = Matrix::Identity(2, 2);
  Matrix support = Matrix::Constant(3, 3,
                                    std::numeric_limits<double>::infinity());
  support(0, 0) = 0.0;
  support(0, 1) = 0.0;
  support(1, 1) = 0.0;
  support(2, 2) = 0.0;

  AssociationPairwiseTransformValidationOptions options;
  const AssociationPairwiseTransformValidationResult result =
      evaluateAssociationPairwiseTransformValidationObjective(
          source, target, assignment, rotation, support, options);

  EXPECT_GT(result.gradient(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(result.gradient(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(result.gradient(2, 0), 0.0);
}

TEST(testDPGO,
     BilevelAssociationRingHypothesesFeedMixtureWithoutHardThresholds) {
  Matrix sourcePoints(4, 2);
  sourcePoints << 0.0, 0.0,
                  2.0, 0.0,
                  0.0, 1.0,
                  1.0, 2.0;
  const double theta = 0.5 * M_PI;
  Matrix trueRotation(2, 2);
  trueRotation << std::cos(theta), -std::sin(theta),
                  std::sin(theta), std::cos(theta);
  Vector trueTranslation(2);
  trueTranslation << 3.0, -1.0;
  Matrix targetPoints(4, 2);
  for (int row = 0; row < sourcePoints.rows(); ++row) {
    targetPoints.row(row) =
        (trueRotation.transpose() *
         (sourcePoints.row(row).transpose() - trueTranslation))
            .transpose();
  }

  Matrix consistent = Matrix::Identity(4, 4);
  Matrix inconsistent = Matrix::Zero(4, 4);
  inconsistent(0, 1) = 1.0;
  inconsistent(1, 0) = 1.0;
  inconsistent(2, 3) = 1.0;
  inconsistent(3, 2) = 1.0;

  AssociationRotoTranslationRingHypothesisOptions generatorOptions;
  generatorOptions.maxHypotheses = 2;
  generatorOptions.angleBins = 4;
  const AssociationRotoTranslationRingHypothesisResult hypotheses =
      generateAssociationRotoTranslationRingHypotheses(
          sourcePoints, targetPoints, consistent, generatorOptions);

  ASSERT_EQ(hypotheses.rotations.size(), 2u);
  ASSERT_EQ(hypotheses.translations.size(), 2u);
  ASSERT_EQ(hypotheses.scores.size(), 2);
  double bestRotationError = std::numeric_limits<double>::infinity();
  for (const Matrix &rotation : hypotheses.rotations) {
    bestRotationError =
        std::min(bestRotationError, (rotation - trueRotation).norm());
  }
  EXPECT_LT(bestRotationError, 1e-12);

  AssociationRotoTranslationHypothesisMixtureOptions mixtureOptions;
  mixtureOptions.weight = 1.0;
  mixtureOptions.distanceScale = 1.0;
  mixtureOptions.softminTemperature = 0.05;
  const AssociationRotoTranslationHypothesisMixtureResult consistentResult =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, consistent, hypotheses.rotations,
          hypotheses.translations, mixtureOptions);
  const AssociationRotoTranslationHypothesisMixtureResult inconsistentResult =
      evaluateAssociationRotoTranslationHypothesisMixtureObjective(
          sourcePoints, targetPoints, inconsistent, hypotheses.rotations,
          hypotheses.translations, mixtureOptions);

  EXPECT_LT(consistentResult.cost, 0.04);
  EXPECT_GT(inconsistentResult.cost, consistentResult.cost + 0.5);
  EXPECT_GT(consistentResult.hypothesisWeights.maxCoeff(), 0.5);
}

TEST(testDPGO, BilevelAssociationCanCenterGradientRowsOnFiniteSupport) {
  Matrix gradient(2, 3);
  gradient << 2.0, 5.0, 100.0,
              -1.0, 3.0, 7.0;
  Matrix support(2, 3);
  support << 0.0, 1.0, std::numeric_limits<double>::infinity(),
             2.0, std::numeric_limits<double>::infinity(), 4.0;

  const Matrix centered =
      centerAssignmentGradientRowsOnFiniteSupport(gradient, support);

  EXPECT_NEAR(centered(0, 0) + centered(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(centered(1, 0) + centered(1, 2), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(centered(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(centered(1, 1), 0.0);
  EXPECT_LT(centered(0, 0), 0.0);
  EXPECT_GT(centered(0, 1), 0.0);
}

TEST(testDPGO,
     BilevelAssociationCanCenterAndNormalizeGradientOnFiniteSupport) {
  Matrix gradient(2, 2);
  gradient << 1.0, 4.0,
              6.0, 3.0;
  Matrix support = Matrix::Zero(2, 2);

  const Matrix normalized = centerAndNormalizeAssignmentGradientOnFiniteSupport(
      gradient, support, true, true, 2.0);

  for (int row = 0; row < normalized.rows(); ++row) {
    EXPECT_NEAR(normalized.row(row).sum(), 0.0, 1e-12);
  }
  for (int col = 0; col < normalized.cols(); ++col) {
    EXPECT_NEAR(normalized.col(col).sum(), 0.0, 1e-12);
  }
  const double rms =
      std::sqrt(normalized.array().square().sum() /
                static_cast<double>(normalized.size()));
  EXPECT_NEAR(rms, 2.0, 1e-12);
  EXPECT_LT(normalized(0, 0), 0.0);
  EXPECT_GT(normalized(0, 1), 0.0);
  EXPECT_GT(normalized(1, 0), 0.0);
  EXPECT_LT(normalized(1, 1), 0.0);
}

TEST(testDPGO,
     BilevelAssociationCanCenterOnCandidateSupportAndNormalizeOnActiveSupport) {
  Matrix gradient(2, 3);
  gradient << 1.0, 4.0, 7.0,
              2.0, 5.0, 11.0;
  Matrix candidateSupport = Matrix::Zero(2, 3);
  Matrix activeSupport =
      Matrix::Constant(2, 3, std::numeric_limits<double>::infinity());
  activeSupport(0, 0) = 0.0;
  activeSupport(1, 1) = 0.0;

  const Matrix normalized =
      centerAndNormalizeAssignmentGradientOnFiniteSupport(
          gradient, candidateSupport, activeSupport, true, false, 2.0);

  for (int row = 0; row < normalized.rows(); ++row) {
    EXPECT_NEAR(normalized.row(row).sum(), 0.0, 1e-12);
  }
  EXPECT_GT(std::abs(normalized(0, 0)), 0.0);
  EXPECT_GT(std::abs(normalized(1, 1)), 0.0);
  const double activeRms =
      std::sqrt((normalized(0, 0) * normalized(0, 0) +
                 normalized(1, 1) * normalized(1, 1)) /
                2.0);
  EXPECT_NEAR(activeRms, 2.0, 1e-12);
}

TEST(testDPGO, BilevelAssociationMeritGateContinuouslySuppressesConflicts) {
  EXPECT_DOUBLE_EQ(associationMeritGateFromDecrease(10.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(associationMeritGateFromDecrease(10.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(associationMeritGateFromDecrease(0.0, 1.0), 1.0);
  EXPECT_NEAR(associationMeritGateFromDecrease(-1.0, 1.0), 0.5, 1e-12);
  EXPECT_NEAR(associationMeritGateFromDecrease(-2.0, 1.0), 0.2, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationFreshSupportStateValidationMeritGateSuppressesWorseCandidate) {
  EXPECT_DOUBLE_EQ(
      associationFreshSupportStateValidationMeritGate(10.0, 9.0, 1.0),
      1.0);
  EXPECT_DOUBLE_EQ(
      associationFreshSupportStateValidationMeritGate(10.0, 10.0, 1.0),
      1.0);
  EXPECT_NEAR(
      associationFreshSupportStateValidationMeritGate(10.0, 11.0, 1.0),
      0.5, 1e-12);
  EXPECT_NEAR(
      associationFreshSupportStateValidationMeritGate(10.0, 12.0, 1.0),
      0.2, 1e-12);
  EXPECT_DOUBLE_EQ(
      associationFreshSupportStateValidationMeritGate(10.0, 12.0, 0.0),
      1.0);
}

TEST(testDPGO,
     BilevelAssociationStateValidationConfidencePenaltyIsContinuous) {
  EXPECT_DOUBLE_EQ(stateValidationTransformConfidencePenalty(1.0), 0.0);
  EXPECT_NEAR(stateValidationTransformConfidencePenalty(0.5), 0.25, 1e-12);
  EXPECT_DOUBLE_EQ(stateValidationTransformConfidencePenalty(0.0), 1.0);
  EXPECT_DOUBLE_EQ(stateValidationTransformConfidencePenalty(-1.0), 1.0);
  EXPECT_DOUBLE_EQ(stateValidationTransformConfidencePenalty(2.0), 0.0);
  EXPECT_DOUBLE_EQ(
      stateValidationTransformConfidencePenalty(
          std::numeric_limits<double>::quiet_NaN()),
      1.0);
}

TEST(testDPGO,
     BilevelAssociationCoveragePriorMatchesFiniteDifferenceAndRestoresMass) {
  Matrix assignment(2, 3);
  assignment << 0.3, 0.1, 0.0,
                0.0, 0.2, 0.1;
  Matrix reference(2, 3);
  reference << 0.6, 0.2, 0.0,
               0.0, 0.4, 0.3;

  AssociationCoveragePriorOptions options;
  options.weight = 0.8;
  const AssociationCoveragePriorResult result =
      evaluateAssociationCoveragePrior(assignment, reference, options);

  const double expectedGradient =
      options.weight * (assignment.sum() - reference.sum());
  EXPECT_NEAR(result.gradient(0, 0), expectedGradient, 1e-12);
  EXPECT_NEAR(result.gradient(0, 0), result.gradient(1, 2), 1e-12);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost =
          evaluateAssociationCoveragePrior(plus, reference, options).cost;
      const double minusCost =
          evaluateAssociationCoveragePrior(minus, reference, options).cost;
      const double numerical = (plusCost - minusCost) / (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-6);
    }
  }

  EXPECT_LT(result.gradient(0, 0), 0.0);
  EXPECT_LT(result.gradient(1, 2), 0.0);
  const Matrix restored = assignment - 0.25 * result.gradient;
  EXPECT_GT(restored.sum(), assignment.sum());
}

TEST(testDPGO,
     BilevelAssociationCoveragePriorStepMovesTowardReferenceMass) {
  Matrix costs = Matrix::Zero(2, 3);
  Matrix previous(2, 3);
  previous << 0.2, 0.0, 0.0,
              0.0, 0.2, 0.0;
  Matrix reference(2, 3);
  reference << 0.7, 0.0, 0.0,
               0.0, 0.7, 0.0;

  AssociationCoveragePriorOptions coverageOptions;
  coverageOptions.weight = 0.5;
  const AssociationCoveragePriorResult coverage =
      evaluateAssociationCoveragePrior(previous, reference, coverageOptions);

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 20;
  sinkhornOptions.stepSize = 1.0;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);

  const Matrix withoutPrior = optimizer.updateFromCosts(costs, previous);
  const Matrix withPrior =
      optimizer.updateFromCostsAndGradient(costs, coverage.gradient, previous);

  EXPECT_GT(withPrior.sum(), withoutPrior.sum());
  for (int row = 0; row < withPrior.rows(); ++row) {
    EXPECT_LE(withPrior.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < withPrior.cols(); ++col) {
    EXPECT_LE(withPrior.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationCoveragePriorCanNormalizeByReferenceMass) {
  Matrix smallAssignment = Matrix::Zero(1, 4);
  Matrix smallReference = Matrix::Zero(1, 4);
  smallReference(0, 0) = 1.0;

  Matrix largeAssignment = Matrix::Zero(1, 4);
  Matrix largeReference = Matrix::Ones(1, 4);

  AssociationCoveragePriorOptions options;
  options.weight = 1.0;
  options.normalizeByReferenceMass = true;

  const AssociationCoveragePriorResult small =
      evaluateAssociationCoveragePrior(smallAssignment, smallReference,
                                       options);
  const AssociationCoveragePriorResult large =
      evaluateAssociationCoveragePrior(largeAssignment, largeReference,
                                       options);

  EXPECT_NEAR(small.gradient(0, 0), large.gradient(0, 0), 1e-12);
  EXPECT_LT(small.gradient(0, 0), 0.0);
  EXPECT_LT(large.gradient(0, 0), 0.0);
}

TEST(testDPGO,
     BilevelAssociationCoveragePriorNormalizationAvoidsTinyReferenceBlowup) {
  Matrix assignment = Matrix::Zero(2, 2);
  assignment(0, 0) = 1.0;
  assignment(1, 1) = 1.0;
  Matrix tinyReference = Matrix::Constant(2, 2, 1e-12);

  AssociationCoveragePriorOptions options;
  options.weight = 1.0;
  options.matchRowColumnMarginals = true;
  options.normalizeByReferenceMass = true;

  const AssociationCoveragePriorResult result =
      evaluateAssociationCoveragePrior(assignment, tinyReference, options);

  EXPECT_LT(result.gradient.cwiseAbs().maxCoeff(), 2.0);
}

TEST(testDPGO,
     BilevelAssociationCoveragePriorCanMatchRowColumnMarginals) {
  Matrix assignment(2, 3);
  assignment << 0.35, 0.05, 0.10,
                0.00, 0.30, 0.05;
  Matrix reference(2, 3);
  reference << 0.60, 0.00, 0.15,
               0.00, 0.45, 0.20;

  AssociationCoveragePriorOptions options;
  options.weight = 0.8;
  options.matchRowColumnMarginals = true;
  const AssociationCoveragePriorResult result =
      evaluateAssociationCoveragePrior(assignment, reference, options);

  const Eigen::VectorXd rowDiff =
      assignment.rowwise().sum() - reference.rowwise().sum();
  const Eigen::RowVectorXd colDiff =
      assignment.colwise().sum() - reference.colwise().sum();
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      EXPECT_NEAR(result.gradient(row, col),
                  options.weight * (rowDiff(row) + colDiff(col)), 1e-12);
    }
  }

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double plusCost =
          evaluateAssociationCoveragePrior(plus, reference, options).cost;
      const double minusCost =
          evaluateAssociationCoveragePrior(minus, reference, options).cost;
      const double numerical = (plusCost - minusCost) / (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-6);
    }
  }

  const double rowBefore = rowDiff.norm();
  const double colBefore = colDiff.norm();
  const Matrix restored = assignment - 0.2 * result.gradient;
  const double rowAfter =
      (restored.rowwise().sum() - reference.rowwise().sum()).norm();
  const double colAfter =
      (restored.colwise().sum() - reference.colwise().sum()).norm();
  EXPECT_LT(rowAfter, rowBefore);
  EXPECT_LT(colAfter, colBefore);
}

TEST(testDPGO,
     BilevelAssociationSignedFreshSupportGateUsesNegativeGradientAndConfidence) {
  Matrix signedGradient(2, 3);
  signedGradient << -2.0, 0.0, 2.0,
                    -1.0, 1.0, -0.5;
  Matrix confidence(2, 3);
  confidence << 1.0, 0.5, 1.0,
                0.25, 0.75, 0.0;

  const Matrix gate = associationConfidenceGatedSignedRecoveryGate(
      signedGradient, confidence, 1.0, 0.5);

  ASSERT_EQ(gate.rows(), signedGradient.rows());
  ASSERT_EQ(gate.cols(), signedGradient.cols());
  EXPECT_NEAR(gate(0, 0), 0.5 / (1.0 + std::exp(-2.0)), 1e-12);
  EXPECT_NEAR(gate(0, 1), 0.125, 1e-12);
  EXPECT_NEAR(gate(0, 2), 0.5 / (1.0 + std::exp(2.0)), 1e-12);
  EXPECT_NEAR(gate(1, 0), 0.125 / (1.0 + std::exp(-1.0)), 1e-12);
  EXPECT_NEAR(gate(1, 1), 0.375 / (1.0 + std::exp(1.0)), 1e-12);
  EXPECT_DOUBLE_EQ(gate(1, 2), 0.0);
}

TEST(testDPGO,
     BilevelAssociationReciprocalFreshSupportGateUsesOnlyPositiveResidual) {
  Matrix local(2, 3);
  local << 0.2, 0.7, 0.0,
           0.5, 0.1, 0.9;
  Matrix reverseTransposed(2, 3);
  reverseTransposed << 0.8, 0.6, 0.0,
                       0.2, 0.4, 1.5;
  Matrix confidence(2, 3);
  confidence << 1.0, 1.0, 0.5,
                0.25, 0.5, 2.0;

  const Matrix gate = associationReciprocalFreshSupportGate(
      local, reverseTransposed, confidence, 0.5);

  ASSERT_EQ(gate.rows(), local.rows());
  ASSERT_EQ(gate.cols(), local.cols());
  EXPECT_NEAR(gate(0, 0), 0.5 * 1.0 * (0.8 - 0.2), 1e-12);
  EXPECT_DOUBLE_EQ(gate(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(gate(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(gate(1, 0), 0.0);
  EXPECT_NEAR(gate(1, 1), 0.5 * 0.5 * (0.4 - 0.1), 1e-12);
  EXPECT_NEAR(gate(1, 2), 0.5 * 1.0 * (1.0 - 0.9), 1e-12);
}

TEST(testDPGO,
     BilevelAssociationReciprocalUniquenessConfidenceIsContinuous) {
  Matrix local(2, 3);
  local << 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0;
  Matrix reverseTransposed(2, 3);
  reverseTransposed << 1.0, 0.0, 0.0,
                       0.5, 0.5, 0.0;

  const Matrix confidence =
      associationReciprocalUniquenessConfidence(local, reverseTransposed, 1.0);

  ASSERT_EQ(confidence.rows(), reverseTransposed.rows());
  ASSERT_EQ(confidence.cols(), reverseTransposed.cols());
  EXPECT_GT(confidence(0, 0), confidence(1, 0));
  EXPECT_GT(confidence(0, 0), 0.66);
  EXPECT_LT(confidence(1, 0), 0.5);
  EXPECT_LT(confidence(1, 1), 1.0);

  const Matrix disabled =
      associationReciprocalUniquenessConfidence(local, reverseTransposed, 0.0);
  EXPECT_TRUE(disabled.isApprox(Matrix::Ones(2, 3), 1e-12));
}

TEST(testDPGO, BilevelAssociationObjectCopyResidualReliabilityIsContinuous) {
  EXPECT_DOUBLE_EQ(objectCopyResidualReliability(0.0, 2.0, 0.1), 1.0);
  EXPECT_NEAR(objectCopyResidualReliability(2.0, 2.0, 0.1),
              0.1 + 0.9 / 2.0, 1e-12);
  EXPECT_NEAR(objectCopyResidualReliability(6.0, 2.0, 0.1),
              0.1 + 0.9 / 10.0, 1e-12);
  EXPECT_DOUBLE_EQ(objectCopyResidualReliability(6.0, 0.0, 0.1), 1.0);
}

TEST(testDPGO, BilevelAssociationObjectCopyReliabilityUpperCostIsLogPenalty) {
  EXPECT_DOUBLE_EQ(objectCopyReliabilityUpperCost(1.0, 3.0), 0.0);
  EXPECT_DOUBLE_EQ(objectCopyReliabilityUpperCost(0.5, 0.0), 0.0);

  const double medium = objectCopyReliabilityUpperCost(0.5, 2.0);
  const double low = objectCopyReliabilityUpperCost(0.25, 2.0);

  EXPECT_NEAR(medium, 2.0 * std::log(2.0), 1e-12);
  EXPECT_GT(low, medium);
  EXPECT_DOUBLE_EQ(objectCopyReliabilityUpperCost(2.0, 2.0), 0.0);
}

TEST(testDPGO, BilevelAssociationSupportSharpnessMatchesFiniteDifference) {
  Matrix assignment(2, 3);
  assignment << 0.8, 0.15, 0.0,
                0.05, 0.55, 0.25;

  AssociationSupportSharpnessOptions options;
  options.weight = 0.7;
  const AssociationSupportSharpnessResult result =
      evaluateAssociationSupportSharpnessObjective(assignment, options);

  ASSERT_EQ(result.gradient.rows(), assignment.rows());
  ASSERT_EQ(result.gradient.cols(), assignment.cols());
  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double numerical =
          (evaluateAssociationSupportSharpnessObjective(plus, options).cost -
           evaluateAssociationSupportSharpnessObjective(minus, options).cost) /
          (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-7);
    }
  }
}

TEST(testDPGO, BilevelAssociationSupportSharpnessPushesSoftTailDown) {
  Matrix assignment(1, 3);
  assignment << 0.75, 0.2, 0.05;

  AssociationSupportSharpnessOptions options;
  options.weight = 0.4;
  const AssociationSupportSharpnessResult result =
      evaluateAssociationSupportSharpnessObjective(assignment, options);

  EXPECT_LT(result.gradient(0, 0), 0.0);
  EXPECT_GT(result.gradient(0, 1), 0.0);
  EXPECT_GT(result.gradient(0, 2), 0.0);

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 20;
  sinkhornOptions.stepSize = 1.0;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);
  const Matrix costs = Matrix::Zero(1, 3);
  const Matrix updated =
      optimizer.updateFromCostsAndGradient(costs, result.gradient,
                                           assignment);

  EXPECT_GT(updated(0, 0), assignment(0, 0));
  EXPECT_LT(updated(0, 1), assignment(0, 1));
  EXPECT_LT(updated(0, 2), assignment(0, 2));
}

TEST(testDPGO,
     BilevelAssociationSupportSharpnessAdaptiveAlphaKeepsFalseTailScaling) {
  AssociationSupportSharpnessAdaptiveOptions options;
  options.maxWeight = 0.1;
  options.targetGradientRatio = 3e-8;

  const double unitSharpnessGradientNorm = 60.0;
  const double localGradientNorm = 5.5e7;
  const double exactTargetAlpha =
      options.targetGradientRatio * localGradientNorm /
      unitSharpnessGradientNorm;
  const double alpha = adaptiveAssociationSupportSharpnessWeight(
      unitSharpnessGradientNorm, localGradientNorm, options);

  EXPECT_GT(alpha, exactTargetAlpha);
  EXPECT_LT(alpha, options.maxWeight);
  EXPECT_GT(alpha, 0.05);
  EXPECT_DOUBLE_EQ(
      adaptiveAssociationSupportSharpnessWeight(
          unitSharpnessGradientNorm, localGradientNorm,
          AssociationSupportSharpnessAdaptiveOptions{options.maxWeight, 0.0}),
      options.maxWeight);
}

TEST(testDPGO,
     BilevelAssociationStructuralExpansionRewardsConsistentNewSupport) {
  Matrix sourcePoints(2, 2);
  sourcePoints << 0.0, 0.0,
                  5.0, 0.0;
  Matrix targetPoints(2, 2);
  targetPoints << 0.0, 0.0,
                  5.0, 0.0;
  Matrix assignment(2, 2);
  assignment << 0.1, 0.01,
                0.01, 0.2;
  Matrix reference = Matrix::Zero(2, 2);
  reference(0, 0) = 1.0;
  Matrix rotation = Matrix::Identity(2, 2);
  Vector translation = Vector::Zero(2);

  AssociationStructuralExpansionOptions options;
  options.weight = 0.7;
  options.distanceScale = 0.5;
  const AssociationStructuralExpansionResult result =
      evaluateAssociationStructuralExpansionObjective(
          sourcePoints, targetPoints, assignment, reference, rotation,
          translation, options);

  ASSERT_EQ(result.gradient.rows(), assignment.rows());
  ASSERT_EQ(result.gradient.cols(), assignment.cols());
  EXPECT_NEAR(result.gradient(0, 0), 0.0, 1e-12);
  EXPECT_LT(result.gradient(1, 1), -0.65);
  EXPECT_GT(result.gradient(1, 0), result.gradient(1, 1));
  EXPECT_LT(result.cost, 0.0);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double numerical =
          (evaluateAssociationStructuralExpansionObjective(
               sourcePoints, targetPoints, plus, reference, rotation,
               translation, options)
               .cost -
           evaluateAssociationStructuralExpansionObjective(
               sourcePoints, targetPoints, minus, reference, rotation,
               translation, options)
               .cost) /
          (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-8);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationStructuralExpansionSuppressesAmbiguousNearMatches) {
  Matrix sourcePoints(2, 2);
  sourcePoints << 0.0, 0.0,
                  1.0, 0.0;
  Matrix targetPoints(2, 2);
  targetPoints << 0.0, 0.0,
                  1.0, 0.0;
  Matrix assignment = Matrix::Constant(2, 2, 0.1);
  Matrix reference = Matrix::Zero(2, 2);
  Matrix rotation = Matrix::Identity(2, 2);
  Vector translation = Vector::Zero(2);

  AssociationStructuralExpansionOptions options;
  options.weight = 0.7;
  options.distanceScale = 1.0;
  const AssociationStructuralExpansionResult result =
      evaluateAssociationStructuralExpansionObjective(
          sourcePoints, targetPoints, assignment, reference, rotation,
          translation, options);

  EXPECT_LT(result.gradient(0, 0), 0.0);
  EXPECT_LT(result.gradient(1, 1), 0.0);
  EXPECT_GT(result.gradient(0, 1), 0.0);
  EXPECT_GT(result.gradient(1, 0), 0.0);
  EXPECT_NEAR(result.gradient(0, 0), result.gradient(1, 1), 1e-12);
  EXPECT_NEAR(result.gradient(0, 1), result.gradient(1, 0), 1e-12);
}

TEST(testDPGO, BilevelAssociationProposalSupportGateIsContinuousMassTrust) {
  Matrix proposal(2, 3);
  proposal << 0.0, 0.04, 0.81,
              -0.5, 0.25, 1.44;

  const Matrix gate = associationProposalSupportGate(proposal);

  ASSERT_EQ(gate.rows(), proposal.rows());
  ASSERT_EQ(gate.cols(), proposal.cols());
  EXPECT_DOUBLE_EQ(gate(0, 0), 0.0);
  EXPECT_NEAR(gate(0, 1), std::sqrt(0.04 / std::sqrt(0.81 * 0.25)),
              1e-12);
  EXPECT_NEAR(gate(0, 2), std::sqrt(0.81 / std::sqrt(0.81 * 1.0)),
              1e-12);
  EXPECT_DOUBLE_EQ(gate(1, 0), 0.0);
  EXPECT_NEAR(gate(1, 1), std::sqrt(0.25 / std::sqrt(1.0 * 0.25)),
              1e-12);
  EXPECT_DOUBLE_EQ(gate(1, 2), 1.0);
}

TEST(testDPGO, BilevelAssociationUnionSupportGateKeepsEitherActiveSupport) {
  Matrix current(2, 3);
  current << 0.0, 0.81, 0.0,
             0.0, 0.0, 0.0;
  Matrix reference(2, 3);
  reference << 0.64, 0.0, 0.0,
               0.0, 0.0, 0.49;

  const Matrix gate = associationUnionSupportGate(current, reference);

  ASSERT_EQ(gate.rows(), current.rows());
  ASSERT_EQ(gate.cols(), current.cols());
  EXPECT_GT(gate(0, 0), 0.0);
  EXPECT_GT(gate(0, 1), 0.0);
  EXPECT_GT(gate(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(gate(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(gate(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(gate(1, 1), 0.0);
}

TEST(testDPGO, BilevelAssociationResidualConfidenceGateRewardsAlignedPairs) {
  Matrix sourcePoints(2, 2);
  sourcePoints << 0.0, 0.0,
                  3.0, 0.0;
  Matrix targetPoints(2, 2);
  targetPoints << 0.0, 0.0,
                  3.0, 0.0;
  Matrix rotation = Matrix::Identity(2, 2);
  Vector translation = Vector::Zero(2);

  const Matrix gate = associationResidualConfidenceGate(
      sourcePoints, targetPoints, rotation, translation, 0.5);

  ASSERT_EQ(gate.rows(), 2);
  ASSERT_EQ(gate.cols(), 2);
  EXPECT_DOUBLE_EQ(gate(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(gate(1, 1), 1.0);
  EXPECT_LT(gate(0, 1), 1e-6);
  EXPECT_LT(gate(1, 0), 1e-6);

  const Matrix automaticGate = associationResidualConfidenceGate(
      sourcePoints, targetPoints, rotation, translation, 0.0);
  EXPECT_TRUE(automaticGate.allFinite());
  EXPECT_GT(automaticGate(0, 0), automaticGate(0, 1));
}

TEST(testDPGO,
     BilevelAssociationRelativeResidualConfidenceGatePreservesBestPairs) {
  Matrix sourcePoints(2, 2);
  sourcePoints << 0.0, 0.0,
                  1.0, 0.0;
  Matrix targetPoints(2, 2);
  targetPoints << 0.2, 0.0,
                  1.2, 0.0;
  Matrix rotation = Matrix::Identity(2, 2);
  Vector translation = Vector::Zero(2);

  const Matrix rawGate = associationResidualConfidenceGate(
      sourcePoints, targetPoints, rotation, translation, 0.2);
  const Matrix relativeGate = associationRelativeResidualConfidenceGate(
      sourcePoints, targetPoints, rotation, translation, 0.2);

  EXPECT_LT(rawGate(0, 0), 0.7);
  EXPECT_LT(rawGate(1, 1), 0.7);
  EXPECT_DOUBLE_EQ(relativeGate(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(relativeGate(1, 1), 1.0);
  EXPECT_LT(relativeGate(0, 1), 0.1);
  EXPECT_LT(relativeGate(1, 0), 0.1);
}

TEST(testDPGO,
     BilevelAssociationRelativeResidualConfidenceGateCanUseSparseSupport) {
  Matrix sourcePoints(3, 2);
  sourcePoints << 0.0, 0.0,
                  1.0, 0.0,
                  5.0, 0.0;
  Matrix targetPoints(3, 2);
  targetPoints << 0.1, 0.0,
                  1.1, 0.0,
                  -5.0, 0.0;
  Matrix rotation = Matrix::Identity(2, 2);
  Vector translation = Vector::Zero(2);
  Matrix support = Matrix::Zero(3, 3);
  support(0, 0) = 1.0;
  support(1, 1) = 1.0;

  const Matrix dense = associationRelativeResidualConfidenceGate(
      sourcePoints, targetPoints, rotation, translation, 0.2);
  const Matrix sparse = associationRelativeResidualConfidenceGateOnSupport(
      sourcePoints, targetPoints, rotation, translation, 0.2, support);

  ASSERT_EQ(sparse.rows(), dense.rows());
  ASSERT_EQ(sparse.cols(), dense.cols());
  EXPECT_NEAR(sparse(0, 0), dense(0, 0), 1e-12);
  EXPECT_NEAR(sparse(1, 1), dense(1, 1), 1e-12);
  EXPECT_DOUBLE_EQ(sparse(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(sparse(2, 2), 0.0);
}

TEST(testDPGO,
     BilevelAssociationReferenceTrustMatchesFiniteDifferenceAndRestoresMass) {
  Matrix assignment(2, 3);
  assignment << 0.30, 0.10, 0.00,
                0.05, 0.35, 0.00;
  Matrix reference(2, 3);
  reference << 0.60, 0.05, 0.00,
               0.00, 0.70, 0.00;

  AssociationReferenceTrustOptions trustOptions;
  trustOptions.weight = 2.5;
  const AssociationReferenceTrustResult trust =
      evaluateAssociationReferenceTrust(assignment, reference, trustOptions);

  const Matrix expectedGradient =
      trustOptions.weight * (assignment - reference);
  EXPECT_NEAR((trust.gradient - expectedGradient).norm(), 0.0, 1e-12);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double numerical =
          (evaluateAssociationReferenceTrust(plus, reference, trustOptions)
               .cost -
           evaluateAssociationReferenceTrust(minus, reference, trustOptions)
               .cost) /
          (2.0 * eps);
      EXPECT_NEAR(trust.gradient(row, col), numerical, 1e-6);
    }
  }

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 30;
  sinkhornOptions.temperature = 1.0;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);
  const Matrix costs = Matrix::Zero(2, 3);
  const Matrix updated =
      optimizer.updateFromCostsAndGradient(costs, trust.gradient, assignment);

  EXPECT_GT(updated(0, 0), assignment(0, 0));
  EXPECT_GT(updated(1, 1), assignment(1, 1));
}

TEST(testDPGO,
     BilevelAssociationReferenceSupportSharpeningPushesReferenceSupportToOne) {
  Matrix assignment(2, 3);
  assignment << 0.70, 0.10, 0.00,
                0.02, 0.55, 0.15;
  Matrix reference(2, 3);
  reference << 0.80, 0.00, 0.00,
               0.00, 0.60, 0.00;

  AssociationReferenceSupportSharpeningOptions options;
  options.weight = 0.4;
  options.target = 1.0;
  const AssociationReferenceSupportSharpeningResult result =
      evaluateAssociationReferenceSupportSharpening(
          assignment, reference, options);

  ASSERT_EQ(result.gradient.rows(), assignment.rows());
  ASSERT_EQ(result.gradient.cols(), assignment.cols());
  EXPECT_LT(result.gradient(0, 0), 0.0);
  EXPECT_LT(result.gradient(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(result.gradient(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(result.gradient(1, 2), 0.0);
  EXPECT_LT(result.cost, 0.05);

  const double eps = 1e-6;
  for (int row = 0; row < assignment.rows(); ++row) {
    for (int col = 0; col < assignment.cols(); ++col) {
      Matrix plus = assignment;
      Matrix minus = assignment;
      plus(row, col) += eps;
      minus(row, col) -= eps;
      const double numerical =
          (evaluateAssociationReferenceSupportSharpening(
               plus, reference, options)
               .cost -
           evaluateAssociationReferenceSupportSharpening(
               minus, reference, options)
               .cost) /
          (2.0 * eps);
      EXPECT_NEAR(result.gradient(row, col), numerical, 1e-8);
    }
  }
}

TEST(testDPGO,
     BilevelAssociationReferenceTrustProximalStepAffectsFirstUpdate) {
  Matrix costs(2, 2);
  costs << 6.0, 0.0,
           0.0, 6.0;
  Matrix reference(2, 2);
  reference << 0.95, 0.05,
               0.05, 0.95;

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 30;
  sinkhornOptions.temperature = 1.0;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);

  const Matrix withoutTrust = optimizer.updateFromCosts(costs, reference);
  const Matrix withTrust =
      optimizer.updateFromCostsAndReferenceTrust(costs, reference, 8.0,
                                                reference);

  const double referenceMassWithout =
      (withoutTrust.array() * reference.array()).sum();
  const double referenceMassWith =
      (withTrust.array() * reference.array()).sum();
  EXPECT_GT(referenceMassWith, referenceMassWithout);
  for (int row = 0; row < withTrust.rows(); ++row) {
    EXPECT_LE(withTrust.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < withTrust.cols(); ++col) {
    EXPECT_LE(withTrust.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO, BilevelAssociationReferenceTrustWeightScalesWithCostRms) {
  Matrix costs(2, 3);
  costs << 3.0, 4.0, std::numeric_limits<double>::infinity(),
           -5.0, std::numeric_limits<double>::quiet_NaN(), 0.0;

  const double scaled =
      scaleAssociationReferenceTrustWeightByCostRms(0.2, costs, 1.0);
  const double expectedRms = std::sqrt((9.0 + 16.0 + 25.0) / 4.0);
  EXPECT_NEAR(scaled, 0.2 * expectedRms, 1e-12);

  Matrix tiny = Matrix::Constant(2, 2, 1e-3);
  EXPECT_DOUBLE_EQ(scaleAssociationReferenceTrustWeightByCostRms(0.2, tiny,
                                                                1.0),
                   0.2);
  EXPECT_DOUBLE_EQ(scaleAssociationReferenceTrustWeightByCostRms(0.0, costs,
                                                                1.0),
                   0.0);
}

TEST(testDPGO,
     BilevelAssociationReferenceMassTrustProximalStepRestoresMass) {
  Matrix costs(2, 2);
  costs << 0.0, std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::infinity(), 8.0;
  Matrix reference(2, 2);
  reference << 0.8, 0.0,
               0.0, 0.8;

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 30;
  sinkhornOptions.temperature = 1.0;
  sinkhornOptions.restrictToPreviousSupport = true;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);

  const Matrix withoutTrust = optimizer.updateFromCosts(costs, reference);
  const Matrix withTrust = optimizer.applyReferenceMassTrust(
      withoutTrust, reference.sum(), 4.0, reference);

  EXPECT_GT(withTrust.sum(), withoutTrust.sum());
  EXPECT_NEAR(withTrust(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(withTrust(1, 0), 0.0, 1e-12);
  for (int row = 0; row < withTrust.rows(); ++row) {
    EXPECT_LE(withTrust.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < withTrust.cols(); ++col) {
    EXPECT_LE(withTrust.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationConfidenceWeightedReferenceMassTrustIgnoresUntrustedSupport) {
  Matrix proposal(2, 2);
  proposal << 0.02, 0.70,
              0.00, 0.02;
  Matrix reference(2, 2);
  reference << 0.80, 0.70,
               0.00, 0.80;
  Matrix confidence(2, 2);
  confidence << 1.00, 0.00,
                0.00, 1.00;

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 30;
  sinkhornOptions.temperature = 1.0;
  sinkhornOptions.restrictToPreviousSupport = true;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);

  const Matrix unweighted = optimizer.applyReferenceMassTrust(
      proposal, reference.sum(), 4.0, reference);
  const Matrix weighted = optimizer.applyConfidenceWeightedReferenceMassTrust(
      proposal, reference, confidence, 4.0, reference);
  const Matrix allOnes = optimizer.applyConfidenceWeightedReferenceMassTrust(
      proposal, reference, Matrix::Ones(2, 2), 4.0, reference);

  EXPECT_NEAR((allOnes - unweighted).norm(), 0.0, 1e-12);
  EXPECT_LT(weighted(0, 1), unweighted(0, 1));
  EXPECT_LE(weighted(0, 1), proposal(0, 1) + 1e-12);
  EXPECT_GT(weighted(0, 0), proposal(0, 0));
  EXPECT_GT(weighted(1, 1), proposal(1, 1));
  EXPECT_GT(weighted(0, 0) + weighted(1, 1),
            proposal(0, 0) + proposal(1, 1));
  for (int row = 0; row < weighted.rows(); ++row) {
    EXPECT_LE(weighted.row(row).sum(), 1.0 + 1e-12);
  }
  for (int col = 0; col < weighted.cols(); ++col) {
    EXPECT_LE(weighted.col(col).sum(), 1.0 + 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationSparseReferenceMassTrustMatchesDenseOnSupport) {
  Matrix proposal = Matrix::Zero(3, 3);
  proposal(0, 0) = 0.02;
  proposal(0, 2) = 0.70;
  proposal(2, 1) = 0.02;
  Matrix reference = Matrix::Zero(3, 3);
  reference(0, 0) = 0.80;
  reference(0, 2) = 0.70;
  reference(2, 1) = 0.80;
  Matrix confidence = Matrix::Zero(3, 3);
  confidence(0, 0) = 1.00;
  confidence(0, 2) = 0.00;
  confidence(2, 1) = 1.00;
  std::vector<SparseAssignmentCostEntry> support = {
      {0, 0, 0.0}, {0, 2, 0.0}, {2, 1, 0.0}};

  PartialSinkhornOptions sinkhornOptions;
  sinkhornOptions.iterations = 30;
  sinkhornOptions.temperature = 1.0;
  SoftAssignmentOptimizer optimizer(sinkhornOptions);

  const Matrix dense = optimizer.applyConfidenceWeightedReferenceMassTrust(
      proposal, reference, confidence, 4.0, reference);
  const Matrix sparse =
      optimizer.applyConfidenceWeightedReferenceMassTrustOnEntries(
          proposal, reference, confidence, support, 4.0, reference);

  EXPECT_NEAR((sparse - dense).norm(), 0.0, 1e-12);
  EXPECT_GT(sparse(0, 0), proposal(0, 0));
  EXPECT_GT(sparse(2, 1), proposal(2, 1));
  EXPECT_LE(sparse(0, 2), proposal(0, 2) + 1e-12);
}

TEST(testDPGO, BilevelAssociationConsensusWeightPowerSuppressesSoftTail) {
  EXPECT_DOUBLE_EQ(associationConsensusWeightPower(1.0, 2.0), 1.0);
  EXPECT_DOUBLE_EQ(associationConsensusWeightPower(0.0, 2.0), 0.0);
  EXPECT_NEAR(associationConsensusWeightPower(0.5, 2.0), 0.25, 1e-12);
  EXPECT_NEAR(associationConsensusWeightPower(0.25, 0.5), 0.5, 1e-12);
  EXPECT_DOUBLE_EQ(associationConsensusWeightPower(0.37, 1.0), 0.37);
}

TEST(testDPGO,
     BilevelAssociationHighConfidenceAnchorScaleIsContinuousAndConservative) {
  EXPECT_NEAR(
      associationHighConfidenceAnchorScale(1.0, 1.0, 0.0, 0.0, 1.0, 2.0),
      3.0, 1e-12);
  EXPECT_DOUBLE_EQ(
      associationHighConfidenceAnchorScale(1.0, 1.0, 0.0, 0.0, 0.0, 2.0),
      1.0);
  const double ambiguous =
      associationHighConfidenceAnchorScale(0.8, 0.9, 0.6, 0.4, 1.0, 2.0);
  EXPECT_GT(ambiguous, 1.0);
  EXPECT_LT(ambiguous,
            associationHighConfidenceAnchorScale(0.8, 0.9, 0.0, 0.0, 1.0,
                                                 2.0));
  EXPECT_DOUBLE_EQ(
      associationHighConfidenceAnchorScale(0.8, 0.9, 0.0, 0.0, 1.0, 0.0),
      1.0);
}

TEST(testDPGO, BilevelAssociationContinuousSupportGateSuppressesSoftTail) {
  EXPECT_DOUBLE_EQ(
      associationContinuousSupportGate(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
      1.0);
  EXPECT_DOUBLE_EQ(
      associationContinuousSupportGate(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
      0.0);
  EXPECT_DOUBLE_EQ(
      associationContinuousSupportGate(0.8, 0.9, 0.0, 0.0, 1.0, 1.0, 0.0),
      1.0);
  const double ambiguous =
      associationContinuousSupportGate(0.8, 0.9, 0.6, 0.4, 1.0, 0.5, 1.0);
  EXPECT_GT(ambiguous, 0.0);
  EXPECT_LT(ambiguous,
            associationContinuousSupportGate(0.8, 0.9, 0.0, 0.0, 1.0, 1.0,
                                             1.0));
  EXPECT_LT(associationContinuousSupportGate(0.8, 0.9, 0.6, 0.4, 1.0, 0.5,
                                             2.0),
            ambiguous);
}

TEST(testDPGO, BilevelAssociationPartialSinkhornTemporalKlReducesChurn) {
  Matrix costs(2, 2);
  costs << 8.0, 0.0,
           0.0, 8.0;
  Matrix previous(2, 2);
  previous << 0.9, 0.1,
              0.1, 0.9;

  PartialSinkhornOptions baseOptions;
  baseOptions.iterations = 30;
  baseOptions.temperature = 1.0;
  SoftAssignmentOptimizer baseOptimizer(baseOptions);
  const Matrix base = baseOptimizer.updateFromCosts(costs, previous);

  PartialSinkhornOptions klOptions = baseOptions;
  klOptions.temporalKlWeight = 5.0;
  SoftAssignmentOptimizer klOptimizer(klOptions);
  const Matrix kl = klOptimizer.updateFromCosts(costs, previous);

  const double baseChurn = (base - previous).cwiseAbs().sum();
  const double klChurn = (kl - previous).cwiseAbs().sum();
  EXPECT_LT(klChurn, baseChurn);
  EXPECT_GT(kl(0, 0), base(0, 0));
  EXPECT_GT(kl(1, 1), base(1, 1));
}

TEST(testDPGO, BilevelAssociationVerifiesTransformAwareObjectMatches) {
  std::vector<TransformAssociationCandidate> candidates;
  const double c = std::cos(M_PI / 2.0);
  const double s = std::sin(M_PI / 2.0);
  auto senderFromReceiver = [&](double x, double y) {
    Vector point(2);
    const double tx = 4.0;
    const double ty = -2.0;
    point(0) = c * x - s * y + tx;
    point(1) = s * x + c * y + ty;
    return point;
  };
  auto add = [&](std::size_t receiverObject, std::size_t senderObject,
                 double x, double y, double score) {
    Vector receiver(2);
    receiver << x, y;
    candidates.push_back(TransformAssociationCandidate{
        receiverObject, senderObject, receiver, senderFromReceiver(x, y),
        score});
  };
  add(2, 4, 0.0, 0.0, 1.0);
  add(0, 1, 1.0, 0.0, 0.9);
  add(3, 2, 1.0, 1.0, 0.8);
  add(1, 3, 0.0, 1.0, 0.7);
  Vector outlierReceiver(2);
  outlierReceiver << 100.0, 100.0;
  Vector outlierSender(2);
  outlierSender << -100.0, 20.0;
  candidates.push_back(
      TransformAssociationCandidate{4, 0, outlierReceiver, outlierSender,
                                    1.0});

  TransformAssociationVerificationOptions options;
  options.dimension = 2;
  options.minInliers = 4;
  options.maxResidual = 1e-6;
  options.minScore = 0.0;
  options.weightScale = 1.0;

  const TransformAssociationVerificationResult result =
      verifyTransformAwareAssociation(5, 5, candidates, options);

  ASSERT_TRUE(result.accepted);
  EXPECT_EQ(result.inlierCount, 4u);
  EXPECT_LT(result.rmse, 1e-9);
  ASSERT_EQ(result.weights.rows(), 5);
  ASSERT_EQ(result.weights.cols(), 5);
  EXPECT_NEAR(result.weights(2, 4), 1.0, 1e-12);
  EXPECT_NEAR(result.weights(0, 1), 0.9, 1e-12);
  EXPECT_NEAR(result.weights(3, 2), 0.8, 1e-12);
  EXPECT_NEAR(result.weights(1, 3), 0.7, 1e-12);
  EXPECT_DOUBLE_EQ(result.weights(4, 0), 0.0);
}

TEST(testDPGO, BilevelAssociationVerifiesFull3DTransformAwareObjectMatches) {
  std::vector<TransformAssociationCandidate> candidates;
  const double c = std::cos(M_PI / 2.0);
  const double s = std::sin(M_PI / 2.0);
  auto senderFromReceiver = [&](double x, double y, double z) {
    Vector point(3);
    const double tx = 10.0;
    const double ty = -25.0;
    const double tz = 20.0;
    point(0) = c * x - s * z + tx;
    point(1) = y + ty;
    point(2) = s * x + c * z + tz;
    return point;
  };
  auto add = [&](std::size_t receiverObject, std::size_t senderObject,
                 double x, double y, double z, double score) {
    Vector receiver(3);
    receiver << x, y, z;
    candidates.push_back(TransformAssociationCandidate{
        receiverObject, senderObject, receiver,
        senderFromReceiver(x, y, z), score});
  };
  add(0, 10, 0.0, 0.0, 0.0, 1.0);
  add(1, 11, 4.0, 0.5, 0.0, 0.9);
  add(2, 12, 0.0, 1.0, 3.0, 0.8);
  add(3, 13, 5.0, 1.5, 4.0, 0.7);
  Vector outlierReceiver(3);
  outlierReceiver << 100.0, 100.0, 100.0;
  Vector outlierSender(3);
  outlierSender << -100.0, 20.0, -40.0;
  candidates.push_back(TransformAssociationCandidate{
      4, 14, outlierReceiver, outlierSender, 1.0});

  TransformAssociationVerificationOptions options;
  options.dimension = 3;
  options.minInliers = 4;
  options.maxResidual = 1e-6;
  options.minScore = 0.0;
  options.weightScale = 1.0;

  const TransformAssociationVerificationResult result =
      verifyTransformAwareAssociation(5, 15, candidates, options);

  ASSERT_TRUE(result.accepted);
  EXPECT_EQ(result.inlierCount, 4u);
  EXPECT_LT(result.rmse, 1e-9);
  ASSERT_EQ(result.transform.rows(), 3);
  ASSERT_EQ(result.transform.cols(), 4);
  EXPECT_NEAR(result.weights(0, 10), 1.0, 1e-12);
  EXPECT_NEAR(result.weights(1, 11), 0.9, 1e-12);
  EXPECT_NEAR(result.weights(2, 12), 0.8, 1e-12);
  EXPECT_NEAR(result.weights(3, 13), 0.7, 1e-12);
  EXPECT_DOUBLE_EQ(result.weights(4, 14), 0.0);
}

TEST(testDPGO,
     BilevelAssociationSkipsRigidInconsistentTransformSeedHypotheses) {
  std::vector<TransformAssociationCandidate> candidates;
  auto add = [&](std::size_t receiverObject, std::size_t senderObject,
                 const Vector &receiver, const Vector &sender,
                 double score) {
    candidates.push_back(TransformAssociationCandidate{
        receiverObject, senderObject, receiver, sender, score});
  };

  for (std::size_t idx = 0; idx < 10; ++idx) {
    Vector receiver(3);
    receiver << static_cast<double>(idx), 10.0 + idx, -3.0 * idx;
    Vector sender(3);
    sender << 100.0 + 11.0 * idx, -50.0 + 7.0 * idx, 3.0 + idx * idx;
    add(idx, idx, receiver, sender, 1.0);
  }

  const Vector translation = (Vector(3) << 2.0, -4.0, 7.0).finished();
  for (std::size_t idx = 0; idx < 4; ++idx) {
    Vector receiver(3);
    receiver << static_cast<double>(idx % 2),
        static_cast<double>(idx / 2),
        static_cast<double>(idx);
    add(10 + idx, 20 + idx, receiver, receiver + translation, 0.8);
  }

  TransformAssociationVerificationOptions options;
  options.dimension = 3;
  options.minInliers = 4;
  options.maxResidual = 1e-6;
  options.minScore = 0.0;
  options.weightScale = 1.0;
  options.maxSeedCandidates = 30;

  const TransformAssociationVerificationResult result =
      verifyTransformAwareAssociation(20, 30, candidates, options);

  ASSERT_TRUE(result.accepted);
  EXPECT_EQ(result.inlierCount, 4u);
  EXPECT_GT(result.seedHypothesesRejectedByDistance, 0u);
  EXPECT_GT(result.seedHypothesesEvaluated, 0u);
}

TEST(testDPGO,
     BilevelAssociationRanksGeometricallyCompatible3DSeedCandidates) {
  std::vector<TransformAssociationCandidate> candidates;
  for (std::size_t idx = 0; idx < 36; ++idx) {
    Vector receiver(3);
    receiver << static_cast<double>(idx),
        0.1 * static_cast<double>(idx * idx),
        static_cast<double>(idx % 5);
    Vector sender(3);
    sender << 1000.0 + 13.0 * static_cast<double>(idx),
        -500.0 + 0.7 * static_cast<double>(idx * idx),
        200.0 + 17.0 * static_cast<double>(idx);
    candidates.push_back(TransformAssociationCandidate{
        idx, idx, receiver, sender, 1.0});
  }

  const Vector translation = (Vector(3) << 7.0, -3.0, 2.0).finished();
  const std::array<Vector, 4> inlierPoints{
      (Vector(3) << 0.0, 0.0, 0.0).finished(),
      (Vector(3) << 3.0, 0.0, 0.0).finished(),
      (Vector(3) << 0.0, 4.0, 0.0).finished(),
      (Vector(3) << 0.0, 0.0, 5.0).finished()};
  for (std::size_t idx = 0; idx < inlierPoints.size(); ++idx) {
    candidates.push_back(TransformAssociationCandidate{
        36 + idx, 36 + idx, inlierPoints[idx],
        inlierPoints[idx] + translation, 0.5});
  }

  TransformAssociationVerificationOptions options;
  options.dimension = 3;
  options.minInliers = 4;
  options.maxResidual = 1e-6;
  options.minScore = 0.0;
  options.maxSeedCandidates = 80;

  const TransformAssociationVerificationResult result =
      verifyTransformAwareAssociation(40, 40, candidates, options);

  ASSERT_TRUE(result.accepted);
  EXPECT_EQ(result.inlierCount, 4u);
  EXPECT_LT(result.rmse, 1e-9);
}

TEST(testDPGO, BilevelAssociationRejectsTransformWhenInliersAreInsufficient) {
  std::vector<TransformAssociationCandidate> candidates;
  for (std::size_t idx = 0; idx < 3; ++idx) {
    Vector receiver(2);
    receiver << static_cast<double>(idx), 0.0;
    Vector sender(2);
    sender << 10.0 + static_cast<double>(idx) * 20.0,
        -5.0 + static_cast<double>(idx) * 3.0;
    candidates.push_back(
        TransformAssociationCandidate{idx, idx, receiver, sender, 1.0});
  }

  TransformAssociationVerificationOptions options;
  options.dimension = 2;
  options.minInliers = 4;
  options.maxResidual = 0.5;

  const TransformAssociationVerificationResult result =
      verifyTransformAwareAssociation(3, 3, candidates, options);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.inlierCount, 0u);
  EXPECT_EQ(result.weights.rows(), 3);
  EXPECT_EQ(result.weights.cols(), 3);
  EXPECT_DOUBLE_EQ(result.weights.sum(), 0.0);
}

TEST(testDPGO,
     BilevelAssociationPairSupportGateSuppressesSingleMatchContinuously) {
  Matrix single = Matrix::Zero(5, 5);
  single(2, 3) = 1.0;

  AssociationPairSupportGateOptions options;
  options.minEffectiveSupport = 4.0;
  options.power = 2.0;

  const AssociationPairSupportGateResult singleGate =
      applyAssociationPairSupportGate(single, options);

  EXPECT_DOUBLE_EQ(singleGate.effectiveSupport, 1.0);
  EXPECT_NEAR(singleGate.gate, 1.0 / 16.0, 1e-12);
  EXPECT_NEAR(singleGate.weights(2, 3), 1.0 / 16.0, 1e-12);

  Matrix supported = Matrix::Zero(5, 5);
  for (int index = 0; index < 4; ++index) {
    supported(index, index) = 1.0;
  }
  const AssociationPairSupportGateResult supportedGate =
      applyAssociationPairSupportGate(supported, options);

  EXPECT_DOUBLE_EQ(supportedGate.effectiveSupport, 4.0);
  EXPECT_DOUBLE_EQ(supportedGate.gate, 1.0);
  EXPECT_NEAR((supportedGate.weights - supported).norm(), 0.0, 1e-12);
}

TEST(testDPGO, BilevelAssociationPairSupportConfidenceNeedsMassAndDiversity) {
  AssociationPairSupportGateOptions options;
  options.minEffectiveSupport = 4.0;

  Matrix single = Matrix::Zero(5, 5);
  single(2, 3) = 1.0;
  EXPECT_NEAR(associationPairSupportConfidence(single, options), 1.0 / 16.0,
              1e-12);

  Matrix tinyButDiverse = Matrix::Zero(5, 5);
  for (int index = 0; index < 4; ++index) {
    tinyButDiverse(index, index) = 0.05;
  }
  EXPECT_NEAR(associationPairSupportConfidence(tinyButDiverse, options), 0.05,
              1e-12);

  Matrix supported = Matrix::Zero(5, 5);
  for (int index = 0; index < 4; ++index) {
    supported(index, index) = 1.0;
  }
  EXPECT_DOUBLE_EQ(associationPairSupportConfidence(supported, options), 1.0);
}

TEST(testDPGO,
     BilevelAssociationPairSupportLatentContinuationRestoresTrustedSupport) {
  AssociationPairSupportGateOptions options;
  options.minEffectiveSupport = 4.0;

  Matrix reference = Matrix::Zero(5, 5);
  for (int index = 0; index < 4; ++index) {
    reference(index, index) = 1.0;
  }
  Matrix collapsed = Matrix::Zero(5, 5);
  collapsed(0, 0) = 0.05;

  const AssociationPairSupportLatentContinuationResult trusted =
      applyAssociationPairSupportLatentContinuation(collapsed, reference, 1.0,
                                                   options);
  EXPECT_GT(trusted.blend, 0.9);
  EXPECT_GT(trusted.weights(1, 1), 0.9);
  EXPECT_GT(associationPairSupportConfidence(trusted.weights, options), 0.9);

  Matrix singleReference = Matrix::Zero(5, 5);
  singleReference(2, 3) = 1.0;
  const double singleConfidence =
      associationPairSupportConfidence(singleReference, options);
  const AssociationPairSupportLatentContinuationResult single =
      applyAssociationPairSupportLatentContinuation(collapsed, singleReference,
                                                   singleConfidence, options);
  EXPECT_LT(single.blend, 0.1);
  EXPECT_LT(single.weights(2, 3), 0.1);

  Matrix lowerCoverage = Matrix::Zero(5, 5);
  lowerCoverage(0, 0) = 1.0;
  lowerCoverage(1, 1) = 1.0;
  const AssociationPairSupportLatentContinuationResult coverage =
      applyAssociationPairSupportLatentContinuation(lowerCoverage, reference,
                                                   1.0, options);
  EXPECT_GT(coverage.blend, 0.4);
  EXPECT_GT(coverage.weights(2, 2), 0.4);

  AssociationPairSupportGateOptions saturatedOptions;
  saturatedOptions.minEffectiveSupport = 2.0;
  const AssociationPairSupportLatentContinuationResult saturated =
      applyAssociationPairSupportLatentContinuation(
          lowerCoverage, reference, 1.0, saturatedOptions);
  EXPECT_GT(saturated.blend, 0.4);
  EXPECT_GT(saturated.weights(2, 2), 0.4);
}

TEST(testDPGO,
     BilevelAssociationPairSupportReferenceUpdatePreservesCoverageMass) {
  AssociationPairSupportGateOptions options;
  options.minEffectiveSupport = 2.0;

  Matrix full = Matrix::Zero(5, 5);
  for (int index = 0; index < 4; ++index) {
    full(index, index) = 1.0;
  }
  Matrix lowerMass = Matrix::Zero(5, 5);
  lowerMass(0, 0) = 1.0;
  lowerMass(1, 1) = 1.0;

  EXPECT_FALSE(associationPairSupportReferenceShouldUpdate(lowerMass, full,
                                                           options));
  EXPECT_TRUE(
      associationPairSupportReferenceShouldUpdate(full, lowerMass, options));

  Matrix weak = Matrix::Zero(5, 5);
  weak(0, 0) = 0.2;
  EXPECT_TRUE(
      associationPairSupportReferenceShouldUpdate(lowerMass, weak, options));
}

TEST(testDPGO, BilevelAssociationPrunesCycleInconsistentAssociation) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Zero(3, 3)));
  weights[0][1](0, 0) = 0.9;
  weights[0][1](1, 2) = 0.8;
  weights[0][2](0, 0) = 0.9;
  weights[0][2](1, 1) = 0.9;
  weights[2][1](0, 0) = 0.9;
  weights[2][1](1, 1) = 0.9;

  AssociationCycleConsistencyOptions options;
  options.minDirectWeight = 0.5;
  options.minComposedWeight = 0.2;
  options.minTwoHopSupport = 1;
  options.keepPairsWithoutTwoHopEvidence = true;

  const AssociationCycleConsistencyStats stats =
      pruneAssociationCycleInconsistencies(weights, options);

  EXPECT_EQ(stats.prunedEntries, 1u);
  EXPECT_NEAR(weights[0][1](0, 0), 0.9, 1e-12);
  EXPECT_DOUBLE_EQ(weights[0][1](1, 2), 0.0);
  EXPECT_NEAR(weights[0][2](1, 1), 0.9, 1e-12);
  EXPECT_NEAR(weights[2][1](1, 1), 0.9, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationKeepsSparseEdgesWithoutTwoHopEvidenceByDefault) {
  std::vector<std::vector<Matrix>> weights(
      2, std::vector<Matrix>(2, Matrix::Zero(2, 2)));
  weights[0][1](0, 1) = 0.7;

  AssociationCycleConsistencyOptions options;
  options.minDirectWeight = 0.5;
  options.keepPairsWithoutTwoHopEvidence = true;

  AssociationCycleConsistencyStats stats =
      pruneAssociationCycleInconsistencies(weights, options);
  EXPECT_EQ(stats.prunedEntries, 0u);
  EXPECT_NEAR(weights[0][1](0, 1), 0.7, 1e-12);

  options.keepPairsWithoutTwoHopEvidence = false;
  stats = pruneAssociationCycleInconsistencies(weights, options);
  EXPECT_EQ(stats.prunedEntries, 1u);
  EXPECT_DOUBLE_EQ(weights[0][1](0, 1), 0.0);
}

TEST(testDPGO,
     BilevelAssociationCycleObjectiveReturnsDistributedGradientBlocks) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Zero(3, 3)));
  weights[0][1](0, 0) = 1.0;
  weights[0][1](1, 2) = 1.0;
  weights[0][2](0, 0) = 1.0;
  weights[0][2](1, 1) = 1.0;
  weights[2][1](0, 0) = 1.0;
  weights[2][1](1, 1) = 1.0;

  AssociationCycleObjectiveOptions options;
  options.weight = 1.0;
  options.minEdgeMass = 1e-12;

  const AssociationCycleObjectiveResult result =
      evaluateAssociationCycleObjective(weights, options);

  EXPECT_EQ(result.activeTriplets, 1u);
  EXPECT_DOUBLE_EQ(result.cost, 1.0);
  ASSERT_EQ(result.gradient.size(), 3u);
  ASSERT_EQ(result.gradient[0].size(), 3u);
  EXPECT_NEAR(result.gradient[0][1](1, 2), 1.0, 1e-12);
  EXPECT_NEAR(result.gradient[0][1](1, 1), -1.0, 1e-12);
  EXPECT_NEAR(result.gradient[0][2](1, 1), 1.0, 1e-12);
  EXPECT_NEAR(result.gradient[2][1](1, 1), 1.0, 1e-12);

  const Matrix descentDirect =
      weights[0][1] - 0.25 * result.gradient[0][1];
  EXPECT_LT(descentDirect(1, 2), weights[0][1](1, 2));
  EXPECT_GT(descentDirect(1, 1), weights[0][1](1, 1));
}

TEST(testDPGO,
     BilevelAssociationCycleObjectiveReportsValidBlockMajorizer) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Zero(2, 2)));
  weights[0][1] << 0.7, 0.1, 0.2, 0.6;
  weights[1][0] = weights[0][1].transpose();
  weights[0][2] << 0.6, 0.2, 0.1, 0.5;
  weights[2][0] = weights[0][2].transpose();
  weights[1][2] << 0.5, 0.1, 0.2, 0.7;
  weights[2][1] = weights[1][2].transpose();

  AssociationCycleObjectiveOptions options;
  options.weight = 1.0;
  options.minEdgeMass = 0.0;
  const AssociationCycleObjectiveResult initial =
      evaluateAssociationCycleObjective(weights, options);

  ASSERT_EQ(initial.blockLipschitz.size(), 3u);
  ASSERT_EQ(initial.blockLipschitz[0].size(), 3u);
  const double majorizer = initial.blockLipschitz[0][1];
  ASSERT_TRUE(std::isfinite(majorizer));
  ASSERT_GT(majorizer, 0.0);

  Matrix perturbation(2, 2);
  perturbation << 0.02, -0.01, 0.01, -0.015;
  auto perturbedWeights = weights;
  perturbedWeights[0][1] += perturbation;
  const AssociationCycleObjectiveResult perturbed =
      evaluateAssociationCycleObjective(perturbedWeights, options);
  const double gradientChange =
      (perturbed.gradient[0][1] - initial.gradient[0][1]).norm();
  EXPECT_LE(gradientChange,
            majorizer * perturbation.norm() + 1e-12);

  auto majorizedWeights = weights;
  majorizedWeights[0][1] -= initial.gradient[0][1] / majorizer;
  const AssociationCycleObjectiveResult majorized =
      evaluateAssociationCycleObjective(majorizedWeights, options);
  EXPECT_LT(majorized.cost, initial.cost);
}

TEST(testDPGO,
     BilevelAssociationPartialCycleAffinityDoesNotPenalizeMissingWitness) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Zero(3, 3)));
  weights[0][1](0, 0) = 1.0;
  weights[1][0] = weights[0][1].transpose();
  weights[1][2](0, 0) = 1.0;
  weights[2][1] = weights[1][2].transpose();
  weights[0][2](0, 0) = 0.5;
  weights[0][2](1, 1) = 1.0;
  weights[2][0] = weights[0][2].transpose();

  AssociationCycleObjectiveOptions options;
  options.weight = 1.0;
  options.minEdgeMass = 1e-12;
  const AssociationCycleObjectiveResult result =
      evaluateAssociationPartialCycleAffinityObjective(weights, options);

  EXPECT_EQ(result.activeTriplets, 1u);
  EXPECT_DOUBLE_EQ(result.cost, -0.5);
  EXPECT_DOUBLE_EQ(result.gradient[0][2](0, 0), -1.0);
  EXPECT_DOUBLE_EQ(result.gradient[0][2](1, 1), 0.0);
  EXPECT_LT((result.gradient[0][2] -
             result.gradient[2][0].transpose()).norm(),
            1e-12);

  auto descentWeights = weights;
  descentWeights[0][2] -= 0.1 * result.gradient[0][2];
  descentWeights[2][0] = descentWeights[0][2].transpose();
  const AssociationCycleObjectiveResult descent =
      evaluateAssociationPartialCycleAffinityObjective(descentWeights,
                                                        options);
  EXPECT_LT(descent.cost, result.cost);
}

TEST(testDPGO,
     BilevelAssociationPartialCycleClosureOnlyRepairsMissingComposition) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Zero(3, 3)));
  weights[0][1](0, 0) = 1.0;
  weights[1][0] = weights[0][1].transpose();
  weights[1][2](0, 0) = 1.0;
  weights[2][1] = weights[1][2].transpose();
  weights[0][2](0, 0) = 0.5;
  weights[0][2](1, 1) = 1.0;
  weights[2][0] = weights[0][2].transpose();

  AssociationCycleObjectiveOptions options;
  options.weight = 1.0;
  options.minEdgeMass = 1e-12;
  const AssociationCycleObjectiveResult result =
      evaluateAssociationPartialCycleClosureObjective(weights, options);

  EXPECT_EQ(result.activeTriplets, 1u);
  EXPECT_DOUBLE_EQ(result.cost, 0.125);
  EXPECT_DOUBLE_EQ(result.gradient[0][2](0, 0), -0.5);
  EXPECT_DOUBLE_EQ(result.gradient[0][2](1, 1), 0.0);
  EXPECT_DOUBLE_EQ(result.gradient[0][1](0, 0), 0.5);
  EXPECT_DOUBLE_EQ(result.gradient[1][2](0, 0), 0.5);

  weights[0][2](0, 0) = 1.0;
  weights[2][0] = weights[0][2].transpose();
  const AssociationCycleObjectiveResult closed =
      evaluateAssociationPartialCycleClosureObjective(weights, options);
  EXPECT_DOUBLE_EQ(closed.cost, 0.0);
  EXPECT_DOUBLE_EQ(closed.gradient[0][2](0, 0), 0.0);
  EXPECT_DOUBLE_EQ(closed.gradient[0][2](1, 1), 0.0);
}

TEST(testDPGO, BilevelAssociationBackendFeedbackSoftensUnsettledGeometryCost) {
  BackendAssociationFeedbackOptions options;
  options.gradientScale = 2.0;
  options.minTrust = 0.25;

  const double baseCost = 10.0;
  const double settled =
      backendFeedbackGeometryCost(baseCost, 0.0, 0.0, options);
  const double unsettled =
      backendFeedbackGeometryCost(baseCost, 0.5, 0.5, options);
  const double veryUnsettled =
      backendFeedbackGeometryCost(baseCost, 100.0, 100.0, options);

  EXPECT_DOUBLE_EQ(settled, baseCost);
  EXPECT_LT(unsettled, settled);
  EXPECT_GT(unsettled, options.minTrust * baseCost);
  EXPECT_DOUBLE_EQ(veryUnsettled, options.minTrust * baseCost);
}

TEST(testDPGO,
     BilevelAssociationBackendResponseRewardsDescentAlignedCandidates) {
  BackendAssociationResponseOptions options;
  options.responseGain = 1.0;
  options.curvatureDamping = 1.0;
  options.maxCostReductionFraction = 0.5;

  const double baseCost = 10.0;
  const double helpful = backendResponseGeometryCost(
      baseCost, 4.0, 2.0, 2.0, 2.0, options);
  const double conflicting = backendResponseGeometryCost(
      baseCost, -4.0, -2.0, 2.0, 2.0, options);
  const double saturated = backendResponseGeometryCost(
      baseCost, 100.0, 100.0, 1.0, 1.0, options);

  EXPECT_LT(helpful, baseCost);
  EXPECT_GT(helpful, options.maxCostReductionFraction * baseCost);
  EXPECT_DOUBLE_EQ(conflicting, baseCost);
  EXPECT_DOUBLE_EQ(saturated,
                   (1.0 - options.maxCostReductionFraction) * baseCost);
}

TEST(testDPGO, BilevelAssociationBackendAdjointAppliesSignedCostUpdate) {
  BackendAssociationAdjointOptions options;
  options.adjointGain = 1.0;
  options.curvatureDamping = 1.0;
  options.maxCostReductionFraction = 0.5;
  options.maxCostIncreaseFraction = 0.25;

  const double baseCost = 10.0;
  const double helpful = backendAdjointGeometryCost(
      baseCost, 4.0, 2.0, 2.0, 2.0, options);
  const double conflicting = backendAdjointGeometryCost(
      baseCost, -4.0, -2.0, 2.0, 2.0, options);
  const double saturatedHelpful = backendAdjointGeometryCost(
      baseCost, 100.0, 100.0, 1.0, 1.0, options);
  const double saturatedConflicting = backendAdjointGeometryCost(
      baseCost, -100.0, -100.0, 1.0, 1.0, options);

  EXPECT_LT(helpful, baseCost);
  EXPECT_GT(conflicting, baseCost);
  EXPECT_DOUBLE_EQ(saturatedHelpful,
                   (1.0 - options.maxCostReductionFraction) * baseCost);
  EXPECT_DOUBLE_EQ(saturatedConflicting,
                   (1.0 + options.maxCostIncreaseFraction) * baseCost);
}

TEST(testDPGO, BilevelAssociationBackendAdjointCanDisableCostReduction) {
  BackendAssociationAdjointOptions options;
  options.adjointGain = 1.0;
  options.curvatureDamping = 1.0;
  options.maxCostReductionFraction = 0.5;
  options.maxCostIncreaseFraction = 0.25;
  options.allowCostReduction = false;
  options.allowCostIncrease = true;

  const double baseCost = 10.0;
  const double helpful = backendAdjointGeometryCost(
      baseCost, 4.0, 2.0, 2.0, 2.0, options);
  const double conflicting = backendAdjointGeometryCost(
      baseCost, -4.0, -2.0, 2.0, 2.0, options);

  EXPECT_DOUBLE_EQ(helpful, baseCost);
  EXPECT_GT(conflicting, baseCost);
}

TEST(testDPGO, BilevelAssociationSolvesCoupledBackendAdjoint) {
  SparseMatrix hessian(2, 2);
  hessian.insert(0, 0) = 4.0;
  hessian.insert(0, 1) = 1.0;
  hessian.insert(1, 0) = 1.0;
  hessian.insert(1, 1) = 3.0;
  hessian.makeCompressed();

  Matrix gradient(1, 2);
  gradient << 1.0, 0.0;

  const Matrix adjoint =
      solveBackendCoupledAdjoint(hessian, gradient, 1.0);

  ASSERT_EQ(adjoint.rows(), 1);
  ASSERT_EQ(adjoint.cols(), 2);
  EXPECT_NEAR(adjoint(0, 0), 4.0 / 19.0, 1e-12);
  EXPECT_NEAR(adjoint(0, 1), -1.0 / 19.0, 1e-12);

  Matrix damped = Matrix(hessian);
  damped.diagonal().array() += 1.0;
  const Vector residual =
      damped * adjoint.row(0).transpose() -
      gradient.row(0).transpose();
  EXPECT_LT(residual.norm(), 1e-12);
}

TEST(testDPGO, BilevelAssociationAddsSoftObjectCopyConsistencyTerms) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  const std::size_t numLocalPoses = 3;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 1;
  term.targetObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.25;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  const std::size_t localPoseIndex = objectOffset + term.localObjectId;
  const std::size_t colOffset = localPoseIndex * blockWidth;
  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_DOUBLE_EQ(stats.totalWeight, 0.25);
  EXPECT_DOUBLE_EQ(stats.cost, 0.5 * 0.25 * term.targetPose.squaredNorm());
  for (unsigned c = 0; c < blockWidth; ++c) {
    EXPECT_DOUBLE_EQ(
        Q.coeff(static_cast<int>(colOffset + c),
                static_cast<int>(colOffset + c)),
        0.25);
  }
  const Matrix denseG = Matrix(G);
  EXPECT_TRUE(
      denseG.block(0, static_cast<int>(colOffset), d, blockWidth)
          .isApprox(-0.25 * Matrix::Ones(d, blockWidth), 1e-12));
}

TEST(testDPGO, BilevelAssociationAddsRankLiftedObjectCopyTerms) {
  const unsigned d = 3;
  const unsigned r = d + 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  Matrix state = Matrix::Zero(r, 2 * blockWidth);
  state.block(0, 0, r, d) = Matrix::Identity(r, d);
  state.block(0, blockWidth, r, d) = Matrix::Identity(r, d);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 0;
  term.targetPose = state.block(0, blockWidth, r, blockWidth);
  term.targetPose(r - 1, d) = 2.0;
  term.weight = 0.4;
  SparseMatrix Q(2 * blockWidth, 2 * blockWidth);
  SparseMatrix G;

  const ObjectCopyConsistencyStats stats =
      addSoftObjectCopyConsistencyTerms(objectOffset, d, state, {term}, Q, G);

  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_EQ(G.rows(), static_cast<int>(r));
  EXPECT_EQ(G.cols(), 2 * static_cast<int>(blockWidth));
  EXPECT_NEAR(Matrix(G)(r - 1, 2 * blockWidth - 1), -0.8, 1e-12);
}

TEST(testDPGO, BilevelAssociationRankLiftedObjectCopyGradientMatchesFiniteDifference) {
  const unsigned d = 3;
  const unsigned r = d + 2;
  const unsigned blockWidth = d + 1;
  Matrix state = Matrix::Zero(r, 2 * blockWidth);
  state.block(0, 0, r, d) = Matrix::Identity(r, d);
  state.block(0, blockWidth, r, d) = Matrix::Identity(r, d);
  state.col(blockWidth - 1) << 0.2, -0.1, 0.4, 0.3, -0.2;
  state.col(2 * blockWidth - 1) << 1.1, 0.5, -0.4, 0.2, 0.7;

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 0;
  term.targetPose = state.block(0, blockWidth, r, blockWidth);
  term.targetPose(3, 0) = 0.25;
  term.targetPose(4, d) = -0.6;
  term.weight = 0.7;
  SparseMatrix Q(2 * blockWidth, 2 * blockWidth);
  SparseMatrix G;
  addSoftObjectCopyConsistencyTerms(1, d, state, {term}, Q, G);

  Matrix direction(r, 2 * blockWidth);
  for (int row = 0; row < direction.rows(); ++row) {
    for (int col = 0; col < direction.cols(); ++col) {
      direction(row, col) =
          std::sin(0.3 * static_cast<double>(1 + row + 2 * col));
    }
  }
  const Matrix denseG = Matrix(G);
  const Matrix gradient = state * Q + denseG;
  const auto objective = [&](const Matrix &value) {
    return 0.5 * (value * Q * value.transpose()).trace() +
           (value * denseG.transpose()).trace();
  };
  const double epsilon = 1e-7;
  const double finiteDifference =
      (objective(state + epsilon * direction) -
       objective(state - epsilon * direction)) /
      (2.0 * epsilon);
  const double analytic = (gradient.array() * direction.array()).sum();

  EXPECT_NEAR(finiteDifference, analytic,
              1e-7 * std::max(1.0, std::abs(analytic)));
}

TEST(testDPGO, BilevelAssociationObjectCopyReliabilityScalesConsistencyWeight) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  const std::size_t numLocalPoses = 3;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 1;
  term.targetObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.8;
  term.reliability = 0.25;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  const double effectiveWeight = term.weight * term.reliability;
  const std::size_t localPoseIndex = objectOffset + term.localObjectId;
  const std::size_t colOffset = localPoseIndex * blockWidth;
  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_DOUBLE_EQ(stats.totalWeight, effectiveWeight);
  EXPECT_DOUBLE_EQ(stats.cost,
                   0.5 * effectiveWeight * term.targetPose.squaredNorm());
  for (unsigned c = 0; c < blockWidth; ++c) {
    EXPECT_DOUBLE_EQ(
        Q.coeff(static_cast<int>(colOffset + c),
                static_cast<int>(colOffset + c)),
        effectiveWeight);
  }
  const Matrix denseG = Matrix(G);
  EXPECT_TRUE(
      denseG.block(0, static_cast<int>(colOffset), d, blockWidth)
          .isApprox(-effectiveWeight * Matrix::Ones(d, blockWidth), 1e-12));
}

TEST(testDPGO,
     BilevelAssociationObjectCopySupportDegreeNormalizationBoundsLocalMass) {
  std::vector<ObjectCopyConsistencyTerm> terms(4);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.localObjectId = 2;
    term.weight = 1.0;
    term.reliability = 1.0;
    term.targetPrecisionScale = 1.0;
  }
  terms[0].targetObjectId = 0;
  terms[1].targetObjectId = 1;
  terms[2].targetObjectId = 2;
  terms[3].localObjectId = 5;
  terms[3].targetObjectId = 3;
  terms[3].weight = 0.0;

  const std::size_t normalized =
      normalizeObjectCopyConsistencyTermsBySupportDegree(terms);

  EXPECT_EQ(normalized, 1u);
  for (std::size_t idx = 0; idx < 3; ++idx) {
    EXPECT_NEAR(terms[idx].weight, 0.25, 1e-12);
  }
  EXPECT_DOUBLE_EQ(terms[3].weight, 0.0);
}

TEST(testDPGO,
     BilevelAssociationObjectCopySupportDegreeBalancesLocalAndTargetPrecision) {
  std::vector<ObjectCopyConsistencyTerm> terms(2);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.localObjectId = 1;
    term.weight = 1.0;
    term.reliability = 1.0;
    term.localPrecisionScale = 100.0;
    term.targetPrecisionScale = 100.0;
  }
  terms[0].targetObjectId = 0;
  terms[1].targetObjectId = 1;

  const std::size_t normalized =
      normalizeObjectCopyConsistencyTermsBySupportDegree(terms);

  EXPECT_EQ(normalized, 1u);
  EXPECT_NEAR(terms[0].weight, 1.0 / 3.0, 1e-12);
  EXPECT_NEAR(terms[1].weight, 1.0 / 3.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationObjectCopySupportDegreeUsesSupportReliabilityScale) {
  std::vector<ObjectCopyConsistencyTerm> terms(2);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.localObjectId = 1;
    term.weight = 1.0;
    term.reliability = 1.0;
    term.supportReliabilityScale = 0.25;
    term.localPrecisionScale = 1.0;
    term.targetPrecisionScale = 1.0;
  }
  terms[0].targetObjectId = 0;
  terms[1].targetObjectId = 1;

  const std::size_t normalized =
      normalizeObjectCopyConsistencyTermsBySupportDegree(terms);

  EXPECT_EQ(normalized, 1u);
  EXPECT_NEAR(terms[0].weight, 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(terms[1].weight, 2.0 / 3.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationObjectCopyAutoRmsReliabilityAttenuatesOutliers) {
  const double rms = std::sqrt((1.0 + 1.0 + 100.0) / 3.0);

  const double inlier =
      objectCopyResidualReliabilityFromActiveRms(1.0, rms, 0.0);
  const double outlier =
      objectCopyResidualReliabilityFromActiveRms(10.0, rms, 0.0);

  EXPECT_GT(inlier, 0.95);
  EXPECT_LT(outlier, 0.35);
  EXPECT_DOUBLE_EQ(objectCopyResidualReliabilityFromActiveRms(0.0, rms, 0.0),
                   1.0);
  EXPECT_DOUBLE_EQ(objectCopyResidualReliabilityFromActiveRms(
                       5.0, std::numeric_limits<double>::quiet_NaN(), 0.2),
                   1.0);
  EXPECT_GE(objectCopyResidualReliabilityFromActiveRms(10.0, rms, 0.2), 0.2);
}

TEST(testDPGO,
     BilevelAssociationObjectCopyAutoRmsReliabilityMatrixUsesActiveSupport) {
  Matrix residuals(3, 3);
  residuals << 1.0, 20.0, 20.0,
               20.0, 1.0, 20.0,
               20.0, 20.0, 10.0;
  Matrix assignment = Matrix::Zero(3, 3);
  assignment(0, 0) = 1.0;
  assignment(1, 1) = 1.0;
  assignment(2, 2) = 1.0;

  const Matrix reliability =
      objectCopyResidualReliabilityMatrixFromActiveRms(residuals, assignment,
                                                       0.0);

  EXPECT_GT(reliability(0, 0), 0.95);
  EXPECT_GT(reliability(1, 1), 0.95);
  EXPECT_LT(reliability(2, 2), 0.35);
  EXPECT_DOUBLE_EQ(reliability(0, 1), 1.0);

  Matrix singleAssignment = Matrix::Zero(3, 3);
  singleAssignment(2, 2) = 1.0;
  const Matrix singleReliability =
      objectCopyResidualReliabilityMatrixFromActiveRms(
          residuals, singleAssignment, 0.0);
  EXPECT_TRUE(singleReliability.isApprox(Matrix::Ones(3, 3), 1e-12));
}

TEST(testDPGO,
     BilevelAssociationResidualReliabilityVectorUsesActiveRmsScale) {
  const std::vector<double> residuals{1.0, 1.0, 10.0};
  const std::vector<double> reliability =
      objectResidualReliabilityVectorFromActiveRms(residuals, 0.0);

  ASSERT_EQ(reliability.size(), residuals.size());
  EXPECT_GT(reliability[0], 0.95);
  EXPECT_GT(reliability[1], 0.95);
  EXPECT_LT(reliability[2], 0.35);

  const std::vector<double> single =
      objectResidualReliabilityVectorFromActiveRms({10.0}, 0.0);
  ASSERT_EQ(single.size(), 1u);
  EXPECT_DOUBLE_EQ(single[0], 1.0);
}

TEST(testDPGO, BilevelAssociationObjectCopyRobustLineProcessScalesWeight) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  const std::size_t numLocalPoses = 3;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 1;
  term.targetObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.8;
  term.robustResidualScale = term.targetPose.norm();

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  const double effectiveWeight = 0.5 * term.weight;
  const std::size_t localPoseIndex = objectOffset + term.localObjectId;
  const std::size_t colOffset = localPoseIndex * blockWidth;
  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_NEAR(stats.totalWeight, effectiveWeight, 1e-12);
  EXPECT_NEAR(stats.cost,
              0.5 * effectiveWeight * term.targetPose.squaredNorm(),
              1e-12);
  for (unsigned c = 0; c < blockWidth; ++c) {
    EXPECT_NEAR(
        Q.coeff(static_cast<int>(colOffset + c),
                static_cast<int>(colOffset + c)),
        effectiveWeight, 1e-12);
  }
  const Matrix denseG = Matrix(G);
  EXPECT_TRUE(
      denseG.block(0, static_cast<int>(colOffset), d, blockWidth)
          .isApprox(-effectiveWeight * Matrix::Ones(d, blockWidth), 1e-12));
}

TEST(testDPGO,
     BilevelAssociationObjectCopyRobustNegativeScaleUsesActiveRms) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 0;
  const std::size_t numLocalPoses = 2;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  std::vector<ObjectCopyConsistencyTerm> terms(2);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.targetPose = Matrix::Zero(d, blockWidth);
    term.weight = 1.0;
    term.robustResidualScale = -1.0;
  }
  terms[0].localObjectId = 0;
  terms[0].targetObjectId = 0;
  terms[0].targetPose(0, static_cast<int>(blockWidth - 1)) = 1.0;
  terms[1].localObjectId = 1;
  terms[1].targetObjectId = 1;
  terms[1].targetPose(0, static_cast<int>(blockWidth - 1)) = 3.0;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, terms, Q, G);

  const double smallWeight = 5.0 / 6.0;
  const double largeWeight = 5.0 / 14.0;
  EXPECT_EQ(stats.activeTerms, 2u);
  EXPECT_NEAR(stats.totalWeight, smallWeight + largeWeight, 1e-12);
  EXPECT_NEAR(stats.cost,
              0.5 * (smallWeight * 1.0 + largeWeight * 9.0), 1e-12);
  for (unsigned c = 0; c < blockWidth; ++c) {
    EXPECT_NEAR(Q.coeff(static_cast<int>(c), static_cast<int>(c)),
                smallWeight, 1e-12);
    EXPECT_NEAR(
        Q.coeff(static_cast<int>(blockWidth + c),
                static_cast<int>(blockWidth + c)),
        largeWeight, 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationSparseObjectCopyGradientMatchesQuadraticAssembly) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 0;
  const std::size_t numLocalPoses = 2;
  Matrix localState(d, numLocalPoses * blockWidth);
  localState << 1.0, 0.0, 0.5, 1.0, 0.0, -0.25,
                0.0, 1.0, 0.2, 0.0, 1.0, 0.75;

  SparseMatrix baseQ(numLocalPoses * blockWidth,
                     numLocalPoses * blockWidth);
  for (int index = 0; index < baseQ.rows(); ++index) {
    baseQ.insert(index, index) = 0.2 + 0.1 * index;
  }
  baseQ.makeCompressed();

  std::vector<ObjectCopyConsistencyTerm> terms(2);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.targetPose = Matrix::Zero(d, blockWidth);
    term.weight = 0.8;
    term.robustResidualScale = -1.0;
  }
  terms[0].localObjectId = 0;
  terms[0].targetPose.col(blockWidth - 1) =
      (Vector(2) << 1.5, -0.2).finished();
  terms[1].localObjectId = 1;
  terms[1].targetPose.col(blockWidth - 1) =
      (Vector(2) << -1.0, 2.0).finished();

  SparseMatrix assembledQ = baseQ;
  SparseMatrix assembledG(d, numLocalPoses * blockWidth);
  const ObjectCopyConsistencyStats assembledStats =
      addSoftObjectCopyConsistencyTerms(objectOffset, d, localState, terms,
                                        assembledQ, assembledG);
  const Matrix assembledGradient =
      localState * assembledQ + Matrix(assembledG);

  Matrix sparseGradient = localState * baseQ;
  const ObjectCopyConsistencyStats sparseStats =
      accumulateSoftObjectCopyConsistencyEuclideanGradient(
          objectOffset, d, localState, terms, sparseGradient);

  EXPECT_LT((sparseGradient - assembledGradient).norm(), 1e-12);
  EXPECT_EQ(sparseStats.activeTerms, assembledStats.activeTerms);
  EXPECT_NEAR(sparseStats.totalWeight, assembledStats.totalWeight, 1e-12);
  EXPECT_NEAR(sparseStats.cost, assembledStats.cost, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationObjectCopyLatentResidualMeasuresInexactRobustBlock) {
  std::vector<ObjectCopyConsistencyTerm> terms(2);
  terms[0].weight = 3.0;
  terms[0].robustLatentTarget = 0.2;
  terms[0].robustLatentValue = 0.6;
  terms[1].weight = 1.0;
  terms[1].robustLatentTarget = 0.5;
  terms[1].robustLatentValue = 0.75;

  const double residual =
      objectCopyRobustLatentFixedPointResidual(terms);

  EXPECT_NEAR(residual,
              std::sqrt((3.0 * 0.4 * 0.4 + 0.25 * 0.25) / 4.0),
              1e-12);
}

TEST(testDPGO,
     BilevelAssociationWhitenedCopyInnovationIsSymmetricAndUnitInvariant) {
  const ObjectCopyChannelInformation local{4.0, 1.0};
  const ObjectCopyChannelInformation target{2.0, 0.5};
  const ObjectCopyChannelInformation relative =
      independentRelativeObjectCopyInformation(local, target);

  EXPECT_NEAR(relative.rotation, 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(relative.translation, 1.0 / 3.0, 1e-12);
  const ObjectCopyChannelInformation swapped =
      independentRelativeObjectCopyInformation(target, local);
  EXPECT_NEAR(relative.rotation, swapped.rotation, 1e-12);
  EXPECT_NEAR(relative.translation, swapped.translation, 1e-12);

  const double innovation = whitenedObjectCopyInnovationSquared(
      0.2, 4.0, relative);
  const ObjectCopyChannelInformation centimeterRelative{
      relative.rotation, relative.translation / 10000.0};
  const double centimeterInnovation = whitenedObjectCopyInnovationSquared(
      0.2, 40000.0, centimeterRelative);
  EXPECT_NEAR(innovation, centimeterInnovation, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationStudentTAlignmentCostIsScaleFreeAndCalibrated) {
  const double expected = 2.0 * std::log(2.0);
  EXPECT_NEAR(dimensionCalibratedStudentTAlignmentCost(0.25, 0.25, 3.0),
              expected, 1e-12);
  EXPECT_NEAR(dimensionCalibratedStudentTAlignmentCost(4.0, 4.0, 3.0),
              expected, 1e-12);
  EXPECT_GT(dimensionCalibratedStudentTAlignmentCost(4.0, 0.25, 3.0),
            2.0);
}

TEST(testDPGO,
     BilevelAssociationExactSharedConsensusUsesOneSymmetricPenaltyMetric) {
  ObjectCopyConsistencyTerm forward;
  forward.weight = 7.0;
  forward.reliability = 0.2;
  forward.targetPrecisionScale = 11.0;
  forward.localInformation = ObjectCopyChannelInformation{4.0, 1.0};
  forward.targetInformation = ObjectCopyChannelInformation{2.0, 0.5};
  forward.robustResidualScale = 3.0;
  forward.robustMinWeight = 0.1;

  ObjectCopyConsistencyTerm reverse = forward;
  std::swap(reverse.localInformation, reverse.targetInformation);

  configureExactSharedObjectConsensusPenalty(forward);
  configureExactSharedObjectConsensusPenalty(reverse);

  EXPECT_DOUBLE_EQ(forward.weight, 1.0);
  EXPECT_DOUBLE_EQ(forward.reliability, 1.0);
  EXPECT_DOUBLE_EQ(forward.targetPrecisionScale, 1.0);
  EXPECT_TRUE(forward.useChannelInformation);
  EXPECT_DOUBLE_EQ(forward.robustResidualScale, 0.0);
  EXPECT_DOUBLE_EQ(forward.robustMinWeight, 0.0);
  EXPECT_NEAR(forward.targetInformation.rotation, 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(forward.targetInformation.translation, 1.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(forward.targetInformation.rotation,
                   reverse.targetInformation.rotation);
  EXPECT_DOUBLE_EQ(forward.targetInformation.translation,
                   reverse.targetInformation.translation);
}

TEST(testDPGO,
     BilevelAssociationExpectedFactorUsesSymmetricRelativeInformation) {
  ObjectCopyConsistencyTerm term;
  term.associationProbability = 0.7;
  term.weight = 91.0;
  term.targetPrecisionScale = 17.0;
  term.localInformation = ObjectCopyChannelInformation{4.0, 1.0};
  term.targetInformation = ObjectCopyChannelInformation{2.0, 0.5};
  term.targetPose = (Matrix(1, 2) << 0.5, 3.0).finished();
  const double inlierMass = configureExpectedObjectCopyFactor(term);

  EXPECT_DOUBLE_EQ(inlierMass, 0.7);
  EXPECT_DOUBLE_EQ(term.weight, 1.0);
  EXPECT_DOUBLE_EQ(term.targetPrecisionScale, 1.0);
  EXPECT_DOUBLE_EQ(term.reliability, 0.7);
  EXPECT_NEAR(term.targetInformation.rotation, 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(term.targetInformation.translation, 1.0 / 3.0, 1e-12);
  EXPECT_TRUE(term.useChannelInformation);
  EXPECT_DOUBLE_EQ(term.robustResidualScale, 0.0);
  EXPECT_DOUBLE_EQ(term.robustLatentTarget, 0.7);
  EXPECT_DOUBLE_EQ(term.robustLatentValue, 0.7);
}

TEST(testDPGO,
     BilevelAssociationMarginalizesAlignmentTranslationVariance) {
  const ObjectCopyChannelInformation marginalized =
      marginalizeObjectCopyAlignmentTranslationVariance(
          ObjectCopyChannelInformation{3.0, 4.0}, 0.25);

  EXPECT_DOUBLE_EQ(marginalized.rotation, 3.0);
  EXPECT_DOUBLE_EQ(marginalized.translation, 2.0);
}

TEST(testDPGO,
     BilevelAssociationDirectProximalFactorUsesSymmetricCurvatureMetric) {
  ObjectCopyConsistencyTerm term;
  term.associationProbability = 0.25;
  term.weight = 91.0;
  term.reliability = 0.2;
  term.consensusPenaltyScale = 0.4;
  term.localInformation = ObjectCopyChannelInformation{4000.0, 25.0};
  term.targetInformation = ObjectCopyChannelInformation{8000.0, 50.0};
  ObjectCopyConsistencyTerm weighted = term;

  configureDirectEdgeDualConsensusProximalFactor(term);

  EXPECT_DOUBLE_EQ(term.weight, 0.4);
  EXPECT_DOUBLE_EQ(term.reliability, 0.25);
  EXPECT_NEAR(term.targetInformation.rotation, 8000.0 / 3.0, 1e-12);
  EXPECT_NEAR(term.targetInformation.translation, 50.0 / 3.0, 1e-12);
  EXPECT_TRUE(term.useChannelInformation);
  EXPECT_DOUBLE_EQ(term.robustResidualScale, 0.0);

  weighted.associationProbability = 0.6;
  weighted.consensusPenaltyScale = 0.75;
  configureWeightedSharedObjectConsensusPenalty(weighted);
  EXPECT_DOUBLE_EQ(weighted.weight, 0.75);
  EXPECT_DOUBLE_EQ(weighted.reliability, 0.6);
  EXPECT_NEAR(weighted.targetInformation.rotation, 8000.0 / 3.0, 1e-12);
  EXPECT_NEAR(weighted.targetInformation.translation, 50.0 / 3.0, 1e-12);
}

TEST(testDPGO,
     BilevelAssociationExactDirectProximalFactorUsesEliminatedCurvature) {
  ObjectCopyConsistencyTerm term;
  term.associationProbability = 0.25;
  term.consensusPenaltyScale = 0.4;
  term.localInformation = ObjectCopyChannelInformation{4000.0, 25.0};
  term.targetInformation = ObjectCopyChannelInformation{8000.0, 50.0};

  configureExactDirectEdgeDualConsensusProximalFactor(term);

  EXPECT_DOUBLE_EQ(term.consensusPenaltyScale, 0.2);
  EXPECT_DOUBLE_EQ(term.weight, 0.2);
  EXPECT_DOUBLE_EQ(term.reliability, 0.25);
  EXPECT_NEAR(term.targetInformation.rotation, 8000.0 / 3.0, 1e-12);
  EXPECT_NEAR(term.targetInformation.translation, 50.0 / 3.0, 1e-12);
  EXPECT_TRUE(term.useChannelInformation);
}

TEST(testDPGO,
     BilevelAssociationExactDirectProximalFactorRejectsInvalidScale) {
  ObjectCopyConsistencyTerm term;
  term.consensusPenaltyScale = 0.0;
  EXPECT_THROW(configureExactDirectEdgeDualConsensusProximalFactor(term),
               std::invalid_argument);
}

TEST(testDPGO,
     BilevelAssociationMetropolisCopyScaleIsSymmetricAndDegreeBounded) {
  EXPECT_DOUBLE_EQ(objectCopyMetropolisDegreeScale(0.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(objectCopyMetropolisDegreeScale(2.0, 4.0), 0.25);
  EXPECT_DOUBLE_EQ(objectCopyMetropolisDegreeScale(4.0, 2.0), 0.25);
}

TEST(testDPGO,
     BilevelAssociationObjectCopyTargetPrecisionScalesWeight) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  const std::size_t numLocalPoses = 3;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 1;
  term.targetObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.8;
  term.targetPrecisionScale = 0.25;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  const double effectiveWeight = term.weight * term.targetPrecisionScale;
  const std::size_t localPoseIndex = objectOffset + term.localObjectId;
  const std::size_t colOffset = localPoseIndex * blockWidth;
  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_NEAR(stats.totalWeight, effectiveWeight, 1e-12);
  EXPECT_NEAR(stats.cost,
              0.5 * effectiveWeight * term.targetPose.squaredNorm(),
              1e-12);
  for (unsigned c = 0; c < blockWidth; ++c) {
    EXPECT_NEAR(
        Q.coeff(static_cast<int>(colOffset + c),
                static_cast<int>(colOffset + c)),
        effectiveWeight, 1e-12);
  }
}

TEST(testDPGO,
     BilevelAssociationObjectCopyUsesRotationTranslationInformation) {
  const unsigned d = 3;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  SparseMatrix Q(2 * blockWidth, 2 * blockWidth);
  SparseMatrix G(d, 2 * blockWidth);
  Matrix localState = Matrix::Zero(d, 2 * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.5;
  term.targetInformation = ObjectCopyChannelInformation{4.0, 2.0};
  term.useChannelInformation = true;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  const std::size_t colOffset = objectOffset * blockWidth;
  for (unsigned c = 0; c < d; ++c) {
    EXPECT_DOUBLE_EQ(Q.coeff(static_cast<int>(colOffset + c),
                             static_cast<int>(colOffset + c)),
                     2.0);
  }
  EXPECT_DOUBLE_EQ(Q.coeff(static_cast<int>(colOffset + d),
                           static_cast<int>(colOffset + d)),
                   1.0);
  const Matrix denseG = Matrix(G);
  EXPECT_TRUE(denseG.block(0, static_cast<int>(colOffset), d, d)
                  .isApprox(-2.0 * Matrix::Ones(d, d), 1e-12));
  EXPECT_TRUE(denseG.col(static_cast<int>(colOffset + d))
                  .isApprox(-Vector::Ones(d), 1e-12));
  EXPECT_NEAR(stats.cost, 0.5 * (2.0 * 9.0 + 1.0 * 3.0), 1e-12);
}

TEST(testDPGO, BilevelAssociationSharedBarycenterFusesNeighborTargets) {
  const unsigned d = 3;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 0;
  Matrix localState = Matrix::Zero(d, 2 * blockWidth);

  std::vector<ObjectCopyConsistencyTerm> terms(5);
  for (ObjectCopyConsistencyTerm &term : terms) {
    term.targetPose = Matrix::Zero(d, blockWidth);
    term.weight = 1.0;
    term.reliability = 1.0;
  }
  terms[0].localObjectId = 0;
  terms[0].targetPose(0, static_cast<int>(blockWidth - 1)) = -2.0;
  terms[0].weight = 1.0;
  terms[1].localObjectId = 0;
  terms[1].targetPose(0, static_cast<int>(blockWidth - 1)) = 2.0;
  terms[1].weight = 3.0;
  terms[2].localObjectId = 1;
  terms[2].targetPose(0, static_cast<int>(blockWidth - 1)) = -0.1;
  terms[3].localObjectId = 1;
  terms[3].targetPose(0, static_cast<int>(blockWidth - 1)) = 0.1;
  terms[4].localObjectId = 0;
  terms[4].targetPose(0, static_cast<int>(blockWidth - 1)) = 100.0;
  terms[4].weight = 100.0;
  terms[4].targetFrameVerified = false;

  const std::size_t fused =
      applyObjectCopySharedBarycenter(objectOffset, d, localState, terms);

  EXPECT_EQ(fused, 2u);
  EXPECT_NEAR(terms[0].targetPose(0, static_cast<int>(blockWidth - 1)),
              1.0, 1e-12);
  EXPECT_NEAR(terms[1].targetPose(0, static_cast<int>(blockWidth - 1)),
              1.0, 1e-12);
  EXPECT_NEAR(terms[2].targetPose(0, static_cast<int>(blockWidth - 1)),
              0.0, 1e-12);
  EXPECT_NEAR(terms[3].targetPose(0, static_cast<int>(blockWidth - 1)),
              0.0, 1e-12);
  EXPECT_NEAR(terms[4].targetPose(0, static_cast<int>(blockWidth - 1)),
              100.0, 1e-12);
  for (const ObjectCopyConsistencyTerm &term : terms) {
    EXPECT_DOUBLE_EQ(term.reliability, 1.0);
  }
}

TEST(testDPGO,
     BilevelAssociationObservationUncertaintyReliabilityUsesCountAndDistance) {
  const double disabled =
      objectObservationUncertaintyReliability(1, 100.0, 0.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(disabled, 1.0);

  const double weak =
      objectObservationUncertaintyReliability(2, 30.0, 3.0, 30.0, 0.1);
  const double strong =
      objectObservationUncertaintyReliability(12, 5.0, 3.0, 30.0, 0.1);
  EXPECT_GT(strong, weak);
  EXPECT_GT(weak, 0.1);
  EXPECT_LT(weak, 0.5);
  EXPECT_LT(strong, 1.0);

  const double noObservation =
      objectObservationUncertaintyReliability(0, 0.0, 3.0, 30.0, 0.2);
  EXPECT_DOUBLE_EQ(noObservation, 0.2);
}

TEST(testDPGO, BilevelAssociationUsesYawSymmetryForObjectCopyConsistency) {
  const unsigned d = 3;
  const unsigned blockWidth = d + 1;
  SparseMatrix Q(blockWidth, blockWidth);
  SparseMatrix G(d, blockWidth);
  Matrix localState = Matrix::Zero(d, blockWidth);
  localState.leftCols(d).setIdentity();

  const double theta = M_PI / 2.0;
  Matrix target = Matrix::Zero(d, blockWidth);
  target(0, 0) = std::cos(theta);
  target(0, 1) = -std::sin(theta);
  target(1, 0) = std::sin(theta);
  target(1, 1) = std::cos(theta);
  target(2, 2) = 1.0;

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 0;
  term.targetObjectId = 0;
  term.targetPose = target;
  term.weight = 1.0;
  term.yawSymmetryOrder = 4;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      0, d, localState, {term}, Q, G);

  EXPECT_EQ(stats.activeTerms, 1u);
  EXPECT_NEAR(stats.cost, 0.0, 1e-12);
  const Matrix denseG = Matrix(G);
  EXPECT_TRUE(denseG.leftCols(d).isApprox(-Matrix::Identity(d, d), 1e-12));
  EXPECT_NEAR(denseG.col(static_cast<int>(d)).norm(), 0.0, 1e-12);
}

TEST(testDPGO, BilevelAssociationSkipsNonpositiveSoftWeights) {
  const unsigned d = 2;
  const unsigned blockWidth = d + 1;
  const std::size_t objectOffset = 1;
  const std::size_t numLocalPoses = 2;
  SparseMatrix Q(numLocalPoses * blockWidth, numLocalPoses * blockWidth);
  SparseMatrix G(d, numLocalPoses * blockWidth);
  Matrix localState = Matrix::Zero(d, numLocalPoses * blockWidth);

  ObjectCopyConsistencyTerm term;
  term.localObjectId = 0;
  term.targetPose = Matrix::Ones(d, blockWidth);
  term.weight = 0.0;

  const ObjectCopyConsistencyStats stats = addSoftObjectCopyConsistencyTerms(
      objectOffset, d, localState, {term}, Q, G);

  EXPECT_EQ(stats.activeTerms, 0u);
  EXPECT_DOUBLE_EQ(stats.totalWeight, 0.0);
  EXPECT_DOUBLE_EQ(stats.cost, 0.0);
  EXPECT_EQ(Q.nonZeros(), 0);
  EXPECT_EQ(G.nonZeros(), 0);
}

TEST(testDPGO,
     BilevelAssociationLowerValidationHypergradientMatchesFiniteDifference) {
  constexpr unsigned d = 2;
  constexpr double trainWeight = 3.0;
  constexpr double candidateWeight = 2.0;
  constexpr double validationWeight = 1.5;
  constexpr double probability = 0.4;
  constexpr double epsilon = 1e-6;
  const Vector trainTarget = (Vector(d) << 0.5, -0.2).finished();
  const Vector candidateTarget = (Vector(d) << 1.8, 0.7).finished();
  const Vector validationTarget = (Vector(d) << 0.9, -0.4).finished();

  const auto lowerTranslation = [&](double p) {
    return (trainWeight * trainTarget +
            p * candidateWeight * candidateTarget) /
           (trainWeight + p * candidateWeight);
  };
  const auto validationCost = [&](double p) {
    const Vector residual = lowerTranslation(p) - validationTarget;
    return 0.5 * validationWeight * residual.squaredNorm();
  };

  Matrix state = Matrix::Zero(d, d + 1);
  state.leftCols(d).setIdentity();
  state.col(d) = lowerTranslation(probability);

  SparseMatrix lowerQ(d + 1, d + 1);
  lowerQ.insert(d, d) = trainWeight + probability * candidateWeight;
  Matrix denseLowerG = Matrix::Zero(d, d + 1);
  denseLowerG.col(d) =
      -trainWeight * trainTarget -
      probability * candidateWeight * candidateTarget;
  QuadraticProblem lowerProblem(1, d, d);
  lowerProblem.setQ(lowerQ);
  lowerProblem.setG(denseLowerG.sparseView());

  Matrix validationGradient = Matrix::Zero(d, d + 1);
  validationGradient.col(d) =
      validationWeight * (state.col(d) - validationTarget);
  LowerValidationAdjointOptions options;
  options.projectGauge = false;
  options.useGaussNewtonLinearization = true;
  options.relativeTolerance = 1e-12;
  const LowerValidationAdjointResult adjoint = solveLowerValidationAdjoint(
      lowerProblem, state, validationGradient, d, options);

  Matrix candidateGradient = Matrix::Zero(d, d + 1);
  candidateGradient.col(d) =
      candidateWeight * (state.col(d) - candidateTarget);
  const double implicitDerivative = implicitLowerValidationHypergradient(
      adjoint.adjoint, candidateGradient);
  const double finiteDifference =
      (validationCost(probability + epsilon) -
       validationCost(probability - epsilon)) /
      (2.0 * epsilon);

  EXPECT_TRUE(adjoint.converged);
  EXPECT_FALSE(adjoint.negativeCurvature);
  EXPECT_LT(adjoint.finalResidualNorm, 1e-10);
  EXPECT_NEAR(implicitDerivative, finiteDifference, 1e-6);
}

TEST(testDPGO,
     BilevelAssociationLowerValidationAdjointUsesQuadraticPreconditioner) {
  constexpr unsigned d = 2;
  constexpr std::size_t poses = 3;
  Matrix state = Matrix::Zero(d, poses * (d + 1));
  for (std::size_t pose = 0; pose < poses; ++pose) {
    state.block(0, static_cast<int>(pose * (d + 1)), d, d).setIdentity();
  }

  SparseMatrix Q(poses * (d + 1), poses * (d + 1));
  Matrix basis(poses, poses);
  basis << 1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0), 0.0,
           1.0 / std::sqrt(6.0), 1.0 / std::sqrt(6.0),
               -2.0 / std::sqrt(6.0),
           1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0),
               1.0 / std::sqrt(3.0);
  const Vector spectrum = (Vector(poses) << 1.0, 1e4, 1e8).finished();
  const Matrix translationQ =
      basis * spectrum.asDiagonal() * basis.transpose();
  for (std::size_t row = 0; row < poses; ++row) {
    for (std::size_t col = 0; col < poses; ++col) {
      Q.insert(static_cast<int>(row * (d + 1) + d),
               static_cast<int>(col * (d + 1) + d)) =
          translationQ(row, col);
    }
  }
  QuadraticProblem lowerProblem(poses, d, d);
  lowerProblem.setQ(Q);

  Matrix gradient = Matrix::Zero(state.rows(), state.cols());
  gradient.col(d) << 1.0, 2.0;
  gradient.col(2 * (d + 1) + d) << 0.2, -0.3;
  gradient.col((d + 1) + d) << -0.5, 1.0;
  LowerValidationAdjointOptions options;
  options.projectGauge = false;
  options.useGaussNewtonLinearization = true;
  options.maxIterations = 2;
  options.relativeTolerance = 1e-10;

  const LowerValidationAdjointResult result = solveLowerValidationAdjoint(
      lowerProblem, state, gradient, d, options);

  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.iterations, 2u);
  Matrix translationGradient(d, poses);
  for (std::size_t pose = 0; pose < poses; ++pose) {
    const int col = static_cast<int>(pose * (d + 1) + d);
    translationGradient.col(static_cast<int>(pose)) = gradient.col(col);
  }
  const Matrix expected =
      (translationQ + result.damping * Matrix::Identity(poses, poses))
          .ldlt()
          .solve(translationGradient.transpose())
          .transpose();
  for (std::size_t pose = 0; pose < poses; ++pose) {
    const int col = static_cast<int>(pose * (d + 1) + d);
    EXPECT_TRUE(result.adjoint.col(col).isApprox(
        expected.col(static_cast<int>(pose)), 1e-7));
  }
}
