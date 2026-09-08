/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/DecentralizedBilevelAssociation.h>

#include <gtest/gtest.h>

#include <Eigen/Cholesky>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace DPGO {
namespace {

TEST(testDPGO, DecentralizedBilevelLowerContractFixesOnlyUpperCommand) {
  const FixedCommandLowerContract contract =
      sharedEdgeInexactBilevelLowerContract();

  EXPECT_TRUE(contract.freezeAssociationCommand);
  EXPECT_TRUE(contract.freezeAlignmentCommand);
  EXPECT_TRUE(contract.updateRobustLatent);
  EXPECT_TRUE(contract.projectGaugeRepresentative);
}

TEST(testDPGO, DecentralizedBilevelSharedEdgeCommandHasExactReciprocal) {
  SharedEdgeCommand command;
  command.firstRobotId = 2;
  command.secondRobotId = 5;
  command.association =
      (Matrix(2, 3) << 0.7, 0.2, 0.0, 0.1, 0.6, 0.3).finished();
  command.alignment = Matrix::Zero(3, 4);
  const double cosine = std::cos(0.4);
  const double sine = std::sin(0.4);
  command.alignment.leftCols(3) =
      (Matrix(3, 3) << cosine, -sine, 0.0, sine, cosine, 0.0,
       0.0, 0.0, 1.0)
          .finished();
  command.alignment.col(3) = (Vector(3) << 1.0, -2.0, 0.5).finished();

  const SharedEdgeCommand reverse = reciprocalSharedEdgeCommand(command);
  const Matrix rotation = command.alignment.leftCols(3);
  const Vector translation = command.alignment.col(3);
  const Matrix expectedRotation = rotation.transpose();
  const Vector expectedTranslation = -expectedRotation * translation;

  EXPECT_EQ(reverse.firstRobotId, command.secondRobotId);
  EXPECT_EQ(reverse.secondRobotId, command.firstRobotId);
  EXPECT_EQ(reverse.association, command.association.transpose());
  EXPECT_EQ(reverse.alignment.leftCols(3), expectedRotation);
  EXPECT_EQ(reverse.alignment.col(3), expectedTranslation);
}

TEST(testDPGO,
     DecentralizedBilevelLocalCertificateDefinesLocalContinuationOnly) {
  const LocalInexactnessCertificate certificate =
      evaluateLocalInexactnessCertificate(4.0, 3.0, 8.0, 6.0);

  EXPECT_DOUBLE_EQ(certificate.quotientStationarityResidual, 4.0);
  EXPECT_DOUBLE_EQ(certificate.robustLatentFixedPointResidual, 3.0);
  EXPECT_NEAR(certificate.normalizedResidual, std::sqrt(0.5), 1e-12);
  EXPECT_TRUE(localLowerContinuationRequired(certificate, 0.5));
  EXPECT_FALSE(localLowerContinuationRequired(certificate, 0.8));
}

TEST(testDPGO,
     DecentralizedBilevelCommandForcingComparesLowerResidualsInOneMetric) {
  const Matrix oldCommandResidual =
      (Matrix(1, 2) << 1.0, 1.0).finished();
  const Matrix newCommandResidual =
      (Matrix(1, 2) << 4.0, 5.0).finished();
  const Matrix currentResidual =
      (Matrix(1, 2) << 1.5, 2.0).finished();

  const LocalCommandForcingCertificate certificate =
      evaluateLocalCommandForcingCertificate(
          currentResidual, oldCommandResidual, newCommandResidual);

  EXPECT_DOUBLE_EQ(certificate.stationarityResidual, 2.5);
  EXPECT_DOUBLE_EQ(certificate.commandInducedResidual, 5.0);
  EXPECT_DOUBLE_EQ(certificate.forcingRatio, 0.5);
  EXPECT_FALSE(localCommandForcingContinuationRequired(certificate));

  const LocalCommandForcingCertificate scaled =
      evaluateLocalCommandForcingCertificate(
          10.0 * currentResidual, 10.0 * oldCommandResidual,
          10.0 * newCommandResidual);
  EXPECT_DOUBLE_EQ(scaled.forcingRatio, certificate.forcingRatio);
}

TEST(testDPGO,
     DecentralizedBilevelCommandForcingRequiresStationarityAtFixedCommand) {
  const Matrix fixedCommandResidual = Matrix::Zero(2, 2);
  const Matrix nonstationaryResidual = Matrix::Ones(2, 2);

  const LocalCommandForcingCertificate nonstationary =
      evaluateLocalCommandForcingCertificate(
          nonstationaryResidual, fixedCommandResidual,
          fixedCommandResidual);
  EXPECT_TRUE(std::isinf(nonstationary.forcingRatio));
  EXPECT_TRUE(localCommandForcingContinuationRequired(nonstationary));

  const LocalCommandForcingCertificate stationary =
      evaluateLocalCommandForcingCertificate(
          fixedCommandResidual, fixedCommandResidual,
          fixedCommandResidual);
  EXPECT_DOUBLE_EQ(stationary.forcingRatio, 0.0);
  EXPECT_FALSE(localCommandForcingContinuationRequired(stationary));
}

TEST(testDPGO, LocalInexactAdmmCertificateRequiresEveryIncidentResidual) {
  LocalInexactnessCertificate local;
  local.quotientStationarityResidual = 0.025;
  local.robustLatentFixedPointResidual = 0.01;

  const LocalInexactAdmmCertificate certificate =
      evaluateLocalInexactAdmmCertificate(
          local, 0.0, 0.0, 0.5, 0.75, 0.1, 0.0);
  EXPECT_DOUBLE_EQ(certificate.stationarityRatio, 0.25);
  EXPECT_DOUBLE_EQ(certificate.robustLatentRatio, 0.1);
  EXPECT_DOUBLE_EQ(certificate.incidentPrimalResidual, 0.5);
  EXPECT_DOUBLE_EQ(certificate.incidentDualResidual, 0.75);
  EXPECT_DOUBLE_EQ(certificate.normalizedResidual, 0.75);
  EXPECT_FALSE(localInexactAdmmContinuationRequired(certificate));

  const LocalInexactAdmmCertificate invalid =
      evaluateLocalInexactAdmmCertificate(
          local, 0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0,
          0.1, 0.0);
  EXPECT_TRUE(localInexactAdmmContinuationRequired(invalid));

  const LocalInexactAdmmCertificate nonstationary =
      evaluateLocalInexactAdmmCertificate(
          local, 0.0, 0.0, 0.5, 1.25, 0.1, 0.0);
  EXPECT_TRUE(localInexactAdmmContinuationRequired(nonstationary));
}

TEST(testDPGO, SharedObjectConsensusStateIsOwnedAndUpdatedByOneEdge) {
  const Matrix first = Matrix::Zero(3, 4);
  const Matrix second = Matrix::Constant(3, 4, 2.0);
  SharedObjectConsensusState state;

  const SharedObjectConsensusUpdate initialized =
      updateSharedObjectConsensusState(first, second, state);

  EXPECT_TRUE(state.initialized);
  EXPECT_EQ(state.latent, Matrix::Ones(3, 4));
  EXPECT_EQ(state.firstScaledDual, Matrix::Zero(3, 4));
  EXPECT_EQ(state.secondScaledDual, Matrix::Zero(3, 4));
  EXPECT_EQ(initialized.firstTarget, state.latent);
  EXPECT_EQ(initialized.secondTarget, state.latent);
  EXPECT_NEAR(initialized.primalResidual, std::sqrt(24.0), 1e-12);
  EXPECT_DOUBLE_EQ(initialized.dualResidual, 0.0);
  EXPECT_GT(initialized.normalizedPrimalResidual, 1.0);
  EXPECT_TRUE(std::isinf(initialized.normalizedDualResidual));

  const Matrix firstAfterLocalSolve = Matrix::Constant(3, 4, 0.5);
  const Matrix secondAfterLocalSolve = Matrix::Constant(3, 4, 1.5);
  const SharedObjectConsensusUpdate updated =
      updateSharedObjectConsensusState(firstAfterLocalSolve,
                                       secondAfterLocalSolve, state);

  EXPECT_EQ(state.latent, Matrix::Ones(3, 4));
  EXPECT_EQ(state.firstScaledDual, Matrix::Constant(3, 4, -0.5));
  EXPECT_EQ(state.secondScaledDual, Matrix::Constant(3, 4, 0.5));
  EXPECT_EQ(updated.firstTarget, Matrix::Constant(3, 4, 1.5));
  EXPECT_EQ(updated.secondTarget, Matrix::Constant(3, 4, 0.5));
  EXPECT_NEAR(updated.primalResidual, std::sqrt(6.0), 1e-12);
  EXPECT_DOUBLE_EQ(updated.dualResidual, 0.0);
  EXPECT_LT(updated.normalizedPrimalResidual,
            initialized.normalizedPrimalResidual);
  EXPECT_DOUBLE_EQ(updated.normalizedDualResidual, 0.0);

  const SharedObjectConsensusUpdate moved =
      updateSharedObjectConsensusState(Matrix::Ones(3, 4),
                                       Matrix::Constant(3, 4, 3.0), state);
  EXPECT_GT(moved.dualResidual, 0.0);
  EXPECT_GT(moved.normalizedDualResidual, 1.0);

  const SharedObjectConsensusUpdate newEpoch =
      updateSharedObjectConsensusState(
          Matrix::Ones(3, 4), Matrix::Constant(3, 4, 3.0), state, {},
          1e-3, 1e-3, 1u);
  EXPECT_TRUE(std::isinf(newEpoch.normalizedDualResidual));
}

TEST(testDPGO, SharedObjectConsensusIgnoresAnUnobservedRotationChannel) {
  Matrix first = Matrix::Zero(2, 3);
  Matrix second = Matrix::Zero(2, 3);
  first.leftCols(2) << 1.0, 0.0,
                       0.0, 1.0;
  second.leftCols(2) << -1.0, 0.0,
                        0.0, -1.0;
  second.rightCols(1).setConstant(2.0);

  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 0.0;
  penalty.translation = 4.0;
  SharedObjectConsensusState state;
  const SharedObjectConsensusUpdate initialized =
      updateSharedObjectConsensusState(first, second, state, penalty);
  const SharedObjectConsensusUpdate updated =
      updateSharedObjectConsensusState(first, second, state, penalty);

  EXPECT_NEAR(initialized.primalResidual, 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(state.firstScaledDual.leftCols(2).norm(), 0.0);
  EXPECT_DOUBLE_EQ(state.secondScaledDual.leftCols(2).norm(), 0.0);
  EXPECT_DOUBLE_EQ(updated.rotationDualResidual, 0.0);
  EXPECT_DOUBLE_EQ(updated.normalizedRotationDualResidual, 0.0);
  EXPECT_DOUBLE_EQ(updated.rotationPhysicalMultiplierNorm, 0.0);
}

TEST(testDPGO,
     DirectEdgeDualBuildsBiasedPriorsFromOnePhysicalMultiplier) {
  const Matrix first = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix second = (Matrix(1, 2) << 1.0, 1.0).finished();
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;

  const SharedObjectConsensusUpdate initialized =
      updateSharedObjectDirectEdgeDualState(first, second, state, penalty,
                                            0.1, 1e-3, 1e-3, 0u);

  ASSERT_TRUE(state.initialized);
  EXPECT_EQ(state.physicalMultiplier, Matrix::Zero(1, 2));
  EXPECT_EQ(initialized.firstTarget, second);
  EXPECT_EQ(initialized.secondTarget, first);
  EXPECT_TRUE(std::isinf(initialized.normalizedDualResidual));

  const SharedObjectConsensusUpdate update =
      updateSharedObjectDirectEdgeDualState(first, second, state, penalty,
                                            0.1, 1e-3, 1e-3, 0u);
  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.2, 1.2).finished())
                .norm(),
            1e-12);
  EXPECT_LT((update.firstTarget -
             (Matrix(1, 2) << 0.9, 0.7).finished())
                .norm(),
            1e-12);
  EXPECT_LT((update.secondTarget -
             (Matrix(1, 2) << 2.1, 4.3).finished())
                .norm(),
            1e-12);

  const Matrix firstGradient =
      (Matrix(1, 2) <<
           penalty.rotation * (first(0, 0) - update.firstTarget(0, 0)),
       penalty.translation * (first(0, 1) - update.firstTarget(0, 1)))
          .finished();
  const Matrix expectedGradient =
      state.physicalMultiplier +
      (Matrix(1, 2) << penalty.rotation * (first(0, 0) - second(0, 0)),
       penalty.translation * (first(0, 1) - second(0, 1)))
          .finished();
  EXPECT_LT((firstGradient - expectedGradient).norm(), 1e-12);

  const Matrix secondGradient =
      (Matrix(1, 2) <<
           penalty.rotation * (second(0, 0) - update.secondTarget(0, 0)),
       penalty.translation * (second(0, 1) - update.secondTarget(0, 1)))
          .finished();
  EXPECT_LT((secondGradient + expectedGradient).norm(), 1e-12);
  EXPECT_NEAR(update.primalResidual, std::sqrt(10.0), 1e-12);
  EXPECT_NEAR(update.dualResidual, std::sqrt(1.48), 1e-12);
  EXPECT_NEAR(update.rotationPhysicalMultiplierNorm, 0.2, 1e-12);
  EXPECT_NEAR(update.translationPhysicalMultiplierNorm, 1.2, 1e-12);
}

TEST(testDPGO, EdgePenaltyBalanceRaisesBetaWhenDisagreementDominates) {
  SharedObjectEdgePenaltyBalanceParameters parameters;
  parameters.enabled = true;
  parameters.initialBeta = 0.2;
  parameters.minimumBeta = 0.05;
  parameters.maximumBeta = 0.5;
  parameters.imbalanceRatio = 5.0;
  parameters.updateFactor = 1.25;
  parameters.emaDecay = 0.0;
  parameters.updatePeriod = 1;

  SharedObjectEdgePenaltyBalanceState state;
  SharedObjectConsensusPenalty information;
  information.rotation = 0.0;
  information.translation = 4.0;
  const Matrix first = (Matrix(1, 2) << 0.0, 2.0).finished();
  const Matrix second = Matrix::Zero(1, 2);

  const SharedObjectEdgePenaltyBalanceUpdate primed =
      updateSharedObjectEdgePenaltyBalance(first, second, state, parameters,
                                           information, 1.0, 0u, true);
  EXPECT_DOUBLE_EQ(primed.beta, 0.2);
  EXPECT_FALSE(primed.penaltyChanged);

  const SharedObjectEdgePenaltyBalanceUpdate update =
      updateSharedObjectEdgePenaltyBalance(first, second, state, parameters,
                                           information, 1.0, 1u, true);
  EXPECT_DOUBLE_EQ(update.beta, 0.25);
  EXPECT_TRUE(update.penaltyChanged);
  EXPECT_TRUE(update.penaltyIncreased);
  EXPECT_GT(update.normalizedPrimalResidual,
            parameters.imbalanceRatio *
                update.normalizedMidpointMotionResidual);
}

TEST(testDPGO, EdgePenaltyBalanceLowersBetaWhenCommonMotionDominates) {
  SharedObjectEdgePenaltyBalanceParameters parameters;
  parameters.enabled = true;
  parameters.initialBeta = 0.2;
  parameters.minimumBeta = 0.05;
  parameters.maximumBeta = 0.5;
  parameters.imbalanceRatio = 5.0;
  parameters.updateFactor = 1.25;
  parameters.emaDecay = 0.0;
  parameters.updatePeriod = 1;

  SharedObjectEdgePenaltyBalanceState state;
  SharedObjectConsensusPenalty information;
  information.rotation = 0.0;
  information.translation = 4.0;
  const Matrix origin = Matrix::Zero(1, 2);
  updateSharedObjectEdgePenaltyBalance(origin, origin, state, parameters,
                                       information, 1.0, 0u, true);

  const Matrix moved = (Matrix(1, 2) << 0.0, 2.0).finished();
  const SharedObjectEdgePenaltyBalanceUpdate update =
      updateSharedObjectEdgePenaltyBalance(moved, moved, state, parameters,
                                           information, 1.0, 1u, true);
  EXPECT_DOUBLE_EQ(update.beta, 0.16);
  EXPECT_TRUE(update.penaltyChanged);
  EXPECT_TRUE(update.penaltyDecreased);
  EXPECT_GT(update.normalizedMidpointMotionResidual,
            parameters.imbalanceRatio *
                update.normalizedPrimalResidual);
}

TEST(testDPGO, EdgePenaltyBalanceDoesNotRescaleThePhysicalMultiplier) {
  SharedObjectDirectEdgeDualState dualState;
  SharedObjectConsensusPenalty initialPenalty;
  initialPenalty.rotation = 0.0;
  initialPenalty.translation = 0.8;
  const Matrix first = (Matrix(1, 2) << 0.0, 2.0).finished();
  const Matrix second = Matrix::Zero(1, 2);
  updateSharedObjectDirectEdgeDualState(first, second, dualState,
                                        initialPenalty);
  updateSharedObjectDirectEdgeDualState(first, second, dualState,
                                        initialPenalty);
  const Matrix multiplier = dualState.physicalMultiplier;

  SharedObjectConsensusPenalty changedPenalty = initialPenalty;
  changedPenalty.translation = 1.0;
  const SharedObjectConsensusUpdate update =
      updateSharedObjectDirectEdgeDualState(first, second, dualState,
                                            changedPenalty, 1.0, 1e-3, 1e-3,
                                            0u, 1.0, 1.0, false);

  EXPECT_EQ(dualState.physicalMultiplier, multiplier);
  EXPECT_LT((update.firstTarget -
             (second - multiplier / changedPenalty.translation))
                .norm(),
            1e-12);
}

TEST(testDPGO, DisabledEdgePenaltyBalancePreservesTheFixedBeta) {
  SharedObjectEdgePenaltyBalanceParameters parameters;
  parameters.enabled = false;
  parameters.initialBeta = 1.0;
  parameters.minimumBeta = 0.05;
  parameters.maximumBeta = 0.5;
  SharedObjectEdgePenaltyBalanceState state;
  const Matrix endpoint = Matrix::Zero(1, 2);

  const SharedObjectEdgePenaltyBalanceUpdate update =
      updateSharedObjectEdgePenaltyBalance(endpoint, endpoint, state,
                                           parameters);

  EXPECT_DOUBLE_EQ(update.beta, 1.0);
  EXPECT_DOUBLE_EQ(state.beta, 1.0);
}

TEST(testDPGO, DirectEdgeDualDefaultsToCanonicalAugmentedLagrangianStep) {
  const Matrix first = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix second = (Matrix(1, 2) << 1.0, 1.0).finished();
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty);
  updateSharedObjectDirectEdgeDualState(first, second, state, penalty);

  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 2.0, 12.0).finished())
                .norm(),
            1e-12);
}

TEST(testDPGO, DirectEdgeDualLineProcessScalesPrimalAndDualUpdates) {
  const Matrix first = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix second = (Matrix(1, 2) << 1.0, 1.0).finished();
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 1.0,
                                        1e-3, 1e-3, 0u, 0.25);
  const SharedObjectConsensusUpdate update =
      updateSharedObjectDirectEdgeDualState(first, second, state, penalty,
                                            1.0, 1e-3, 1e-3, 0u, 0.25);

  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.5, 3.0).finished())
                .norm(),
            1e-12);
  EXPECT_NEAR(update.primalResidual, 0.25 * std::sqrt(10.0), 1e-12);
  EXPECT_NEAR(update.dualResidual, std::sqrt(9.25), 1e-12);
}

TEST(testDPGO, DirectEdgeDualSupportsIndependentChannelSteps) {
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 1.0;
  penalty.translation = 1.0;
  updateSharedObjectDirectEdgeDualState(
      Matrix::Zero(1, 2), Matrix::Zero(1, 2), state, penalty);

  updateSharedObjectDirectEdgeDualStateWithChannelSteps(
      (Matrix(1, 2) << 1.0, 1.0).finished(), Matrix::Zero(1, 2),
      state, penalty, 0.5, 1.5);

  EXPECT_NEAR(state.physicalMultiplier(0, 0), 0.5, 1e-12);
  EXPECT_NEAR(state.physicalMultiplier(0, 1), 1.5, 1e-12);
}

TEST(testDPGO, DirectEdgeDualInertiaAcceptsContractingPoint2Edge) {
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 0.0;
  penalty.translation = 1.0;
  const Matrix second = Matrix::Zero(1, 2);

  const SharedObjectDirectEdgeDualInertialUpdate primed =
      updateSharedObjectDirectEdgeDualStateInertial(
          (Matrix(1, 2) << 0.0, 2.0).finished(), second, state, penalty,
          1.0, 0.5);
  EXPECT_FALSE(primed.attempted);

  const SharedObjectDirectEdgeDualInertialUpdate update =
      updateSharedObjectDirectEdgeDualStateInertial(
          (Matrix(1, 2) << 0.0, 1.0).finished(), second, state, penalty,
          1.0, 0.5);

  EXPECT_TRUE(update.attempted);
  EXPECT_TRUE(update.accepted);
  EXPECT_FALSE(update.restarted);
  EXPECT_LT(update.acceleratedMerit, update.baselineMerit);
  EXPECT_NEAR(update.baselineMerit, 9.0, 1e-12);
  EXPECT_NEAR(update.acceleratedMerit, 3.5, 1e-12);
  EXPECT_NEAR(state.physicalMultiplier(0, 1), 0.5, 1e-12);
  EXPECT_NEAR(update.consensus.primalResidual, 1.0, 1e-12);
  EXPECT_NEAR(update.consensus.firstTarget(0, 1), -0.5, 1e-12);
  EXPECT_NEAR(update.consensus.secondTarget(0, 1), 1.0, 1e-12);
}

TEST(testDPGO, DirectEdgeDualInertiaRestartsOnExpandingPoint2Edge) {
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 0.0;
  penalty.translation = 1.0;
  const Matrix second = Matrix::Zero(1, 2);

  updateSharedObjectDirectEdgeDualStateInertial(
      Matrix::Zero(1, 2), second, state, penalty, 1.0, 0.5);
  const SharedObjectDirectEdgeDualInertialUpdate update =
      updateSharedObjectDirectEdgeDualStateInertial(
          (Matrix(1, 2) << 0.0, 1.0).finished(), second, state, penalty,
          1.0, 0.5);

  EXPECT_TRUE(update.attempted);
  EXPECT_FALSE(update.accepted);
  EXPECT_TRUE(update.restarted);
  EXPECT_LE(update.baselineMerit, update.acceleratedMerit);
  EXPECT_NEAR(state.physicalMultiplier(0, 1), 1.0, 1e-12);
  EXPECT_NEAR(state.previousPhysicalMultiplier(0, 1), 1.0, 1e-12);
  EXPECT_EQ(state.previousFirstEndpoint,
            (Matrix(1, 2) << 0.0, 1.0).finished());
}

TEST(testDPGO, DirectEdgeDualInertiaReportsTheCommittedMultiplierStep) {
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 0.0;
  penalty.translation = 1.0;
  const Matrix second = Matrix::Zero(1, 2);

  updateSharedObjectDirectEdgeDualStateInertial(
      (Matrix(1, 2) << 0.0, 2.0).finished(), second, state, penalty,
      1.0, 0.5);
  updateSharedObjectDirectEdgeDualStateInertial(
      (Matrix(1, 2) << 0.0, 1.0).finished(), second, state, penalty,
      1.0, 0.5);
  const SharedObjectDirectEdgeDualInertialUpdate update =
      updateSharedObjectDirectEdgeDualStateInertial(
          (Matrix(1, 2) << 0.0, 0.4).finished(), second, state, penalty,
          1.0, 0.5);

  ASSERT_TRUE(update.accepted);
  EXPECT_NEAR(state.physicalMultiplier(0, 1), 0.85, 1e-12);
  EXPECT_NEAR(update.consensus.dualResidual, 0.35, 1e-12);
  EXPECT_NEAR(update.consensus.translationDualResidual, 0.35, 1e-12);
}

TEST(testDPGO, DirectEdgeDualAssociationProbabilityRegularizesTheMultiplier) {
  const Matrix first = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix second = (Matrix(1, 2) << 1.0, 1.0).finished();
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 1.0,
                                        1e-3, 1e-3, 0u, 1.0, 0.25);
  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 1.0,
                                        1e-3, 1e-3, 0u, 1.0, 0.25);
  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.5, 3.0).finished())
                .norm(),
            1e-12);

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 1.0,
                                        1e-3, 1e-3, 0u, 1.0, 0.25);
  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.625, 3.75).finished())
                .norm(),
            1e-12);
}

TEST(testDPGO, DirectEdgeDualIgnoresAnUnobservedRotationChannel) {
  Matrix first = Matrix::Zero(2, 3);
  Matrix second = Matrix::Zero(2, 3);
  first.leftCols(2) << 1.0, 0.0,
                       0.0, 1.0;
  second.leftCols(2) << 0.0, -1.0,
                        1.0, 0.0;
  first.col(2) << 4.0, -2.0;
  second.col(2) << 1.0, 2.0;

  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 0.0;
  penalty.translation = 4.0;

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 0.1);
  const SharedObjectConsensusUpdate update =
      updateSharedObjectDirectEdgeDualState(first, second, state, penalty,
                                            0.1);

  EXPECT_DOUBLE_EQ(state.physicalMultiplier.leftCols(2).norm(), 0.0);
  EXPECT_LE((state.physicalMultiplier.col(2) -
             0.4 * (first.col(2) - second.col(2)))
                .norm(),
            1e-12);
  EXPECT_NEAR(update.primalResidual,
              (first.col(2) - second.col(2)).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(update.rotationDualResidual, 0.0);
  EXPECT_DOUBLE_EQ(update.rotationPhysicalMultiplierNorm, 0.0);
}

TEST(testDPGO,
     DirectEdgeDualPreservesMultiplierUntilTheNewCommandPrimalStep) {
  const Matrix first = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix second = (Matrix(1, 2) << 1.0, 1.0).finished();
  SharedObjectDirectEdgeDualState state;
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;

  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 0.1,
                                        1e-3, 1e-3, 0u);
  updateSharedObjectDirectEdgeDualState(first, second, state, penalty, 0.1,
                                        1e-3, 1e-3, 0u);
  ASSERT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.2, 1.2).finished())
                .norm(),
            1e-12);

  const Matrix movedFirst = (Matrix(1, 2) << 3.0, 2.0).finished();
  const SharedObjectConsensusUpdate newCommand =
      updateSharedObjectDirectEdgeDualState(movedFirst, second, state,
                                            penalty, 0.1, 1e-3, 1e-3, 1u);

  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.2, 1.2).finished())
                .norm(),
            1e-12);
  EXPECT_LT((newCommand.firstTarget -
             (Matrix(1, 2) << 0.9, 0.7).finished())
                .norm(),
            1e-12);
  EXPECT_LT((newCommand.secondTarget -
             (Matrix(1, 2) << 3.1, 2.3).finished())
                .norm(),
            1e-12);
  EXPECT_TRUE(std::isinf(newCommand.normalizedDualResidual));

  const SharedObjectConsensusUpdate afterPrimalStep =
      updateSharedObjectDirectEdgeDualState(movedFirst, second, state,
                                            penalty, 0.1, 1e-3, 1e-3, 1u);
  EXPECT_LT((state.physicalMultiplier -
             (Matrix(1, 2) << 0.6, 1.6).finished())
                .norm(),
            1e-12);
  EXPECT_LT((afterPrimalStep.firstTarget -
             (Matrix(1, 2) << 0.7, 0.6).finished())
                .norm(),
            1e-12);
  EXPECT_LT((afterPrimalStep.secondTarget -
             (Matrix(1, 2) << 3.3, 2.4).finished())
                .norm(),
            1e-12);
}

TEST(testDPGO, DirectEdgeDualMultiplierIsGaugeCovariant) {
  const Matrix first =
      (Matrix(2, 3) << 1.0, 0.2, 2.0,
                         -0.1, 0.9, 4.0)
          .finished();
  const Matrix second =
      (Matrix(2, 3) << 0.8, -0.1, 1.0,
                         0.2, 1.1, 1.5)
          .finished();
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 2.0;
  penalty.translation = 4.0;
  SharedObjectDirectEdgeDualState referenceState;
  updateSharedObjectDirectEdgeDualState(first, second, referenceState,
                                        penalty, 0.1, 1e-3, 1e-3, 0u);
  updateSharedObjectDirectEdgeDualState(first, second, referenceState,
                                        penalty, 0.1, 1e-3, 1e-3, 0u);

  SharedObjectDirectEdgeDualState transformedState = referenceState;
  transformedState.penaltyBalance.midpointPrimed = true;
  transformedState.penaltyBalance.previousMidpoint = first;
  transformedState.spectralStepPrimed = true;
  transformedState.previousSpectralScaledMultiplier = first;
  transformedState.previousSpectralWhitenedResidual = second;
  transformedState.inertialHistoryPrimed = true;
  transformedState.previousFirstEndpoint = first;
  transformedState.previousSecondEndpoint = second;
  transformedState.previousPhysicalMultiplier =
      referenceState.physicalMultiplier;
  transformedState.momentumStepHistoryPrimed = true;
  transformedState.previousFirstEndpointStep = first.rightCols(1);
  transformedState.previousSecondEndpointStep = second.rightCols(1);
  const Matrix rotation =
      (Matrix(2, 2) << 0.0, -1.0, 1.0, 0.0).finished();
  const Matrix translation = (Matrix(2, 1) << 3.0, -2.0).finished();
  transportSharedObjectDirectEdgeDualState(transformedState, rotation,
                                           translation);
  EXPECT_LT((transformedState.physicalMultiplier -
             rotation * referenceState.physicalMultiplier)
                .norm(),
            1e-12);
  Matrix expectedMidpoint = rotation * first;
  expectedMidpoint.rightCols(1) += translation;
  EXPECT_LT((transformedState.penaltyBalance.previousMidpoint -
             expectedMidpoint)
                .norm(),
            1e-12);
  EXPECT_TRUE(transformedState.penaltyBalance.midpointPrimed);
  EXPECT_LT((transformedState.previousSpectralScaledMultiplier -
             rotation * first)
                .norm(),
            1e-12);
  EXPECT_LT((transformedState.previousSpectralWhitenedResidual -
             rotation * second)
                .norm(),
            1e-12);
  EXPECT_TRUE(transformedState.spectralStepPrimed);
  Matrix expectedPreviousFirst = rotation * first;
  expectedPreviousFirst.rightCols(1) += translation;
  Matrix expectedPreviousSecond = rotation * second;
  expectedPreviousSecond.rightCols(1) += translation;
  EXPECT_LT((transformedState.previousFirstEndpoint -
             expectedPreviousFirst)
                .norm(),
            1e-12);
  EXPECT_LT((transformedState.previousSecondEndpoint -
             expectedPreviousSecond)
                .norm(),
            1e-12);
  EXPECT_LT((transformedState.previousPhysicalMultiplier -
             rotation * referenceState.physicalMultiplier)
                .norm(),
            1e-12);
  EXPECT_LT((transformedState.previousFirstEndpointStep -
             rotation * first.rightCols(1))
                .norm(),
            1e-12);
  EXPECT_LT((transformedState.previousSecondEndpointStep -
             rotation * second.rightCols(1))
                .norm(),
            1e-12);
  EXPECT_TRUE(transformedState.momentumStepHistoryPrimed);
  EXPECT_TRUE(transformedState.inertialHistoryPrimed);
  EXPECT_TRUE(transformedState.dualCertificatePrimed);

  const SharedObjectConsensusUpdate referenceUpdate =
      updateSharedObjectDirectEdgeDualState(first, second, referenceState,
                                            penalty, 0.1, 1e-3, 1e-3, 0u);
  const SharedObjectConsensusUpdate transformedUpdate =
      updateSharedObjectDirectEdgeDualState(rotation * first,
                                            rotation * second,
                                            transformedState, penalty, 0.1,
                                            1e-3, 1e-3, 0u);
  EXPECT_LT((transformedUpdate.firstTarget -
             rotation * referenceUpdate.firstTarget)
                .norm(),
            1e-12);
  EXPECT_LT((transformedUpdate.secondTarget -
             rotation * referenceUpdate.secondTarget)
                .norm(),
            1e-12);
}

TEST(testDPGO, DirectEdgeDualUsesExactLatentEliminationPenalty) {
  SharedObjectConsensusPenalty splitPenalty;
  splitPenalty.rotation = 8.0;
  splitPenalty.translation = 6.0;

  const SharedObjectConsensusPenalty directPenalty =
      eliminateSharedObjectConsensusLatentPenalty(splitPenalty);

  EXPECT_DOUBLE_EQ(directPenalty.rotation, 4.0);
  EXPECT_DOUBLE_EQ(directPenalty.translation, 3.0);
}

TEST(testDPGO, DirectEdgeDualEliminationPreservesDisabledChannels) {
  SharedObjectConsensusPenalty splitPenalty;
  splitPenalty.rotation = 0.0;
  splitPenalty.translation = 6.0;

  const SharedObjectConsensusPenalty directPenalty =
      eliminateSharedObjectConsensusLatentPenalty(splitPenalty);

  EXPECT_DOUBLE_EQ(directPenalty.rotation, 0.0);
  EXPECT_DOUBLE_EQ(directPenalty.translation, 3.0);
}

TEST(testDPGO, DirectEdgeDualProximalFactorPreservesTheCurrentAlGradient) {
  const Matrix current = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix biasedNeighbor = (Matrix(1, 2) << 0.8, 0.4).finished();
  SharedObjectConsensusPenalty splitPenalty;
  splitPenalty.rotation = 8.0;
  splitPenalty.translation = 6.0;
  const SharedObjectConsensusPenalty directPenalty =
      eliminateSharedObjectConsensusLatentPenalty(splitPenalty);

  const Matrix proximalTarget =
      buildSharedObjectDirectEdgeDualProximalTarget(current, biasedNeighbor);

  EXPECT_LT((proximalTarget -
             (Matrix(1, 2) << 1.4, 2.2).finished())
                .norm(),
            1e-12);
  const Matrix proximalGradient =
      (Matrix(1, 2) <<
           splitPenalty.rotation *
               (current(0, 0) - proximalTarget(0, 0)),
       splitPenalty.translation *
           (current(0, 1) - proximalTarget(0, 1)))
          .finished();
  const Matrix directGradient =
      (Matrix(1, 2) <<
           directPenalty.rotation *
               (current(0, 0) - biasedNeighbor(0, 0)),
       directPenalty.translation *
           (current(0, 1) - biasedNeighbor(0, 1)))
          .finished();
  EXPECT_LT((proximalGradient - directGradient).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(splitPenalty.rotation, 2.0 * directPenalty.rotation);
  EXPECT_DOUBLE_EQ(splitPenalty.translation,
                   2.0 * directPenalty.translation);
}

TEST(testDPGO,
     DirectEdgeDualBlendedCurvaturePreservesTheCurrentAlGradient) {
  const Matrix current = (Matrix(1, 2) << 2.0, 4.0).finished();
  const Matrix biasedNeighbor = (Matrix(1, 2) << 0.8, 0.4).finished();
  constexpr double curvatureScale = 0.75;
  SharedObjectConsensusPenalty splitPenalty;
  splitPenalty.rotation = 8.0;
  splitPenalty.translation = 6.0;
  const SharedObjectConsensusPenalty directPenalty =
      eliminateSharedObjectConsensusLatentPenalty(splitPenalty);

  const Matrix target = buildSharedObjectDirectEdgeDualProximalTarget(
      current, biasedNeighbor, curvatureScale);
  const Matrix blendedGradient =
      (Matrix(1, 2) <<
           curvatureScale * splitPenalty.rotation *
               (current(0, 0) - target(0, 0)),
       curvatureScale * splitPenalty.translation *
           (current(0, 1) - target(0, 1)))
          .finished();
  const Matrix directGradient =
      (Matrix(1, 2) <<
           directPenalty.rotation *
               (current(0, 0) - biasedNeighbor(0, 0)),
       directPenalty.translation *
           (current(0, 1) - biasedNeighbor(0, 1)))
          .finished();

  EXPECT_LT((blendedGradient - directGradient).norm(), 1e-12);
  EXPECT_NEAR(target(0, 0), 1.2, 1e-12);
  EXPECT_NEAR(target(0, 1), 1.6, 1e-12);
}

TEST(testDPGO, DirectEdgeDualSpectralStepUsesWhitenedSecantCurvature) {
  SharedObjectDirectEdgeDualState state;
  state.initialized = true;
  state.physicalMultiplier = Matrix::Zero(1, 1);
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 1.0;
  penalty.translation = 0.0;

  const double fallback = estimateSharedObjectDirectEdgeDualSpectralStep(
      Matrix::Constant(1, 1, 2.0), Matrix::Zero(1, 1), state, penalty,
      1.25, 0.75, 1.5, 0.0);
  EXPECT_DOUBLE_EQ(fallback, 1.25);

  state.physicalMultiplier(0, 0) = 1.0;
  const double clipped = estimateSharedObjectDirectEdgeDualSpectralStep(
      Matrix::Constant(1, 1, 1.5), Matrix::Zero(1, 1), state, penalty,
      1.25, 0.75, 1.5, 0.0);
  EXPECT_NEAR(clipped, 1.5, 1e-12);
}

TEST(testDPGO, DirectEdgeDualSpectralStepRejectsNonconcaveSecant) {
  SharedObjectDirectEdgeDualState state;
  state.initialized = true;
  state.physicalMultiplier = Matrix::Zero(1, 1);
  SharedObjectConsensusPenalty penalty;
  penalty.rotation = 1.0;

  estimateSharedObjectDirectEdgeDualSpectralStep(
      Matrix::Constant(1, 1, 1.0), Matrix::Zero(1, 1), state, penalty,
      1.2, 0.75, 1.5, 0.0);
  state.physicalMultiplier(0, 0) = 1.0;
  const double fallback = estimateSharedObjectDirectEdgeDualSpectralStep(
      Matrix::Constant(1, 1, 2.0), Matrix::Zero(1, 1), state, penalty,
      1.2, 0.75, 1.5, 0.0);
  EXPECT_DOUBLE_EQ(fallback, 1.2);
}

TEST(testDPGO, GlobalDirectEdgeDualSpectralAggregatesAllValidEdges) {
  const std::vector<SharedObjectDirectEdgeDualSpectralContribution>
      contributions{{4.0, 2.0, true}, {5.0, 3.0, true},
                    {100.0, 1.0, false}};
  std::size_t validContributions = 0u;
  const double step = reduceSharedObjectDirectEdgeDualSpectralStep(
      contributions, 1.25, 1.25, 0.75, 2.0, 0.0,
      &validContributions);

  EXPECT_EQ(validContributions, 2u);
  EXPECT_NEAR(step, 9.0 / 5.0, 1e-12);
}

TEST(testDPGO, GlobalDirectEdgeDualSpectralClipsAndAppliesEmaOnce) {
  const std::vector<SharedObjectDirectEdgeDualSpectralContribution>
      contributions{{9.0, 1.0, true}};
  const double step = reduceSharedObjectDirectEdgeDualSpectralStep(
      contributions, 1.25, 1.0, 0.75, 1.5, 0.5);

  EXPECT_NEAR(step, 1.25, 1e-12);
}

TEST(testDPGO, GlobalDirectEdgeDualSpectralRejectsNonconcaveAggregate) {
  const std::vector<SharedObjectDirectEdgeDualSpectralContribution>
      contributions{{1.0, -2.0, true}, {2.0, 0.0, true}};
  std::size_t validContributions = 99u;
  const double step = reduceSharedObjectDirectEdgeDualSpectralStep(
      contributions, 1.2, 1.4, 0.75, 1.5, 0.0,
      &validContributions);

  EXPECT_EQ(validContributions, 0u);
  EXPECT_DOUBLE_EQ(step, 1.2);
}

TEST(testDPGO, DirectEdgeEndpointMomentumRecoversContractiveSecant) {
  const Matrix previous = (Matrix(2, 1) << 1.0, -2.0).finished();
  const Matrix current = 0.2 * previous;

  const SharedObjectDirectEdgeMomentumContribution contribution =
      sharedObjectDirectEdgeEndpointMomentumContribution(previous, current);

  ASSERT_TRUE(contribution.valid);
  EXPECT_NEAR(contribution.numerator / contribution.denominator, 0.25,
              1e-12);
}

TEST(testDPGO, DirectEdgeEndpointMomentumRejectsExpandingSecant) {
  const Matrix previous = (Matrix(2, 1) << 1.0, -2.0).finished();
  const Matrix current = 1.2 * previous;

  const SharedObjectDirectEdgeMomentumContribution contribution =
      sharedObjectDirectEdgeEndpointMomentumContribution(previous, current);

  EXPECT_FALSE(contribution.valid);
}

TEST(testDPGO, OwnerEndpointMomentumAggregatesOnceAndAppliesEma) {
  const std::vector<SharedObjectDirectEdgeMomentumContribution> contributions{
      {1.0, 4.0, true}, {2.0, 4.0, true}, {-1.0, 1.0, false}};
  std::size_t valid = 0u;

  const double momentum = reduceSharedObjectDirectEdgeMomentum(
      contributions, 0.15, 0.10, 0.0, 0.30, 0.5, &valid);

  EXPECT_EQ(valid, 2u);
  EXPECT_NEAR(momentum, 0.20, 1e-12);
}

TEST(testDPGO, DirectEdgeEndpointMomentumHistoryUsesConsecutiveSteps) {
  SharedObjectDirectEdgeDualState state;
  state.inertialHistoryPrimed = true;
  state.previousFirstEndpoint = Matrix::Zero(2, 1);
  state.previousSecondEndpoint = Matrix::Zero(2, 1);
  const Matrix first = (Matrix(2, 1) << 1.0, -2.0).finished();
  const Matrix second = (Matrix(2, 1) << -2.0, 1.0).finished();

  const SharedObjectDirectEdgeMomentumContribution primed =
      updateSharedObjectDirectEdgeEndpointMomentumHistory(first, second,
                                                          state);
  EXPECT_FALSE(primed.valid);
  state.previousFirstEndpoint = first;
  state.previousSecondEndpoint = second;

  const SharedObjectDirectEdgeMomentumContribution contribution =
      updateSharedObjectDirectEdgeEndpointMomentumHistory(
          1.2 * first, 1.2 * second, state);

  ASSERT_TRUE(contribution.valid);
  EXPECT_NEAR(contribution.numerator / contribution.denominator, 0.25,
              1e-12);
}

TEST(testDPGO, DirectEdgeDualRecoversCentralizedWeightedLandmark) {
  const std::array<double, 3> localMeans{0.0, 3.0, 9.0};
  const std::array<double, 3> localInformation{1.0, 2.0, 4.0};
  const std::array<std::array<std::size_t, 2>, 3> endpoints{
      std::array<std::size_t, 2>{0, 1},
      std::array<std::size_t, 2>{1, 2},
      std::array<std::size_t, 2>{2, 0}};
  std::array<double, 3> values = localMeans;
  std::array<SharedObjectDirectEdgeDualState, 3> edgeStates;
  std::array<std::array<double, 2>, 3> edgeTargets{};

  for (unsigned iteration = 0; iteration < 500; ++iteration) {
    for (std::size_t edge = 0; edge < endpoints.size(); ++edge) {
      Matrix first(1, 1);
      Matrix second(1, 1);
      first(0, 0) = values[endpoints[edge][0]];
      second(0, 0) = values[endpoints[edge][1]];
      const SharedObjectConsensusUpdate update =
          updateSharedObjectDirectEdgeDualState(
              first, second, edgeStates[edge], {}, 0.1);
      edgeTargets[edge][0] = update.firstTarget(0, 0);
      edgeTargets[edge][1] = update.secondTarget(0, 0);
    }

    std::array<double, 3> updatedValues{};
    for (std::size_t robot = 0; robot < values.size(); ++robot) {
      double numerator = localInformation[robot] * localMeans[robot];
      double denominator = localInformation[robot];
      for (std::size_t edge = 0; edge < endpoints.size(); ++edge) {
        for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
          if (endpoints[edge][endpoint] == robot) {
            numerator += edgeTargets[edge][endpoint];
            denominator += 1.0;
          }
        }
      }
      updatedValues[robot] = numerator / denominator;
    }
    values = updatedValues;
  }

  const double centralized =
      (localInformation[0] * localMeans[0] +
       localInformation[1] * localMeans[1] +
       localInformation[2] * localMeans[2]) /
      (localInformation[0] + localInformation[1] + localInformation[2]);
  for (const double value : values) {
    EXPECT_NEAR(value, centralized, 1e-8);
  }
}

TEST(testDPGO, PairwiseSharedLatentsRecoverOneThreeRobotLandmark) {
  const std::array<double, 3> localMeans{0.0, 3.0, 9.0};
  const std::array<double, 3> localInformation{1.0, 2.0, 4.0};
  std::array<double, 3> values = localMeans;
  std::array<SharedObjectConsensusState, 3> edgeStates;
  std::array<std::array<double, 2>, 3> edgeTargets{};
  const std::array<std::array<std::size_t, 2>, 3> endpoints{
      std::array<std::size_t, 2>{0, 1},
      std::array<std::size_t, 2>{1, 2},
      std::array<std::size_t, 2>{2, 0}};

  for (unsigned iteration = 0; iteration < 500; ++iteration) {
    for (std::size_t edge = 0; edge < endpoints.size(); ++edge) {
      Matrix first(1, 1);
      Matrix second(1, 1);
      first(0, 0) = values[endpoints[edge][0]];
      second(0, 0) = values[endpoints[edge][1]];
      const SharedObjectConsensusUpdate update =
          updateSharedObjectConsensusState(first, second, edgeStates[edge]);
      edgeTargets[edge][0] = update.firstTarget(0, 0);
      edgeTargets[edge][1] = update.secondTarget(0, 0);
    }

    std::array<double, 3> updatedValues{};
    for (std::size_t robot = 0; robot < values.size(); ++robot) {
      double numerator = localInformation[robot] * localMeans[robot];
      double denominator = localInformation[robot];
      for (std::size_t edge = 0; edge < endpoints.size(); ++edge) {
        for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
          if (endpoints[edge][endpoint] == robot) {
            numerator += edgeTargets[edge][endpoint];
            denominator += 1.0;
          }
        }
      }
      updatedValues[robot] = numerator / denominator;
    }
    values = updatedValues;
  }

  const double centralized =
      (localInformation[0] * localMeans[0] +
       localInformation[1] * localMeans[1] +
       localInformation[2] * localMeans[2]) /
      (localInformation[0] + localInformation[1] + localInformation[2]);
  for (const double value : values) {
    EXPECT_NEAR(value, centralized, 1e-8);
  }
  for (const SharedObjectConsensusState &state : edgeStates) {
    EXPECT_NEAR(state.latent(0, 0), centralized, 1e-8);
  }
}

TEST(testDPGO,
     CoupledSplitLatentOperatorRetainsCrossEdgeComplianceAndMatchesDenseSolve) {
  const std::vector<std::vector<std::size_t>> incidentEdges{
      {0}, {0, 1}, {1}};
  const std::vector<Matrix> localScaledCompliances{
      (Matrix(1, 1) << 0.2).finished(),
      (Matrix(2, 2) << 0.4, 0.25, 0.25, 0.5).finished(),
      (Matrix(1, 1) << 0.3).finished()};
  SharedObjectSplitLatentOperator interfaceOperator(
      2, 1, incidentEdges,
      [&](std::size_t robotId, const Matrix &directions) {
        return directions * localScaledCompliances.at(robotId).transpose();
      });

  Matrix direction(1, 2);
  direction << 2.0, -1.0;
  const Matrix explicitResponseMatrix =
      (Matrix(2, 2) << 0.3, 0.125, 0.125, 0.4).finished();
  const Matrix expectedResponse =
      (explicitResponseMatrix * direction.transpose()).transpose();
  const Matrix response = interfaceOperator.applyResponse(direction);

  EXPECT_NEAR(response(0, 0), expectedResponse(0, 0), 1e-12);
  EXPECT_NEAR(response(0, 1), expectedResponse(0, 1), 1e-12);
  EXPECT_NE(response(0, 0), 0.3 * direction(0, 0));

  Matrix rhs(1, 2);
  rhs << 1.0, -2.0;
  const Matrix system = Matrix::Identity(2, 2) - explicitResponseMatrix;
  const Matrix direct = system.ldlt().solve(rhs.transpose()).transpose();
  Matrix latent = Matrix::Zero(1, 2);
  double previousResidual = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 0; iteration < 100; ++iteration) {
    latent = interfaceOperator.richardsonStep(latent, rhs);
    const double residual =
        (interfaceOperator.applySystem(latent) - rhs).norm();
    EXPECT_LE(residual, previousResidual + 1e-14);
    previousResidual = residual;
  }

  EXPECT_LT(previousResidual, 1e-12);
  EXPECT_LT((latent - direct).norm(), 1e-11);
}

SharedObjectLowerMessage makeSharedObjectLowerMessage(
    const SharedObjectLowerMessageKey &key, std::size_t ownerRobotId,
    std::size_t commandEpoch, std::size_t ownerEndpointSequence,
    std::size_t receiverEndpointSequence, double targetValue) {
  SharedObjectLowerMessage message;
  message.key = key;
  message.ownerRobotId = ownerRobotId;
  message.commandEpoch = commandEpoch;
  message.ownerEndpointSequence = ownerEndpointSequence;
  message.receiverEndpointSequence = receiverEndpointSequence;
  message.target = Matrix::Constant(1, 1, targetValue);
  message.normalizedPrimalResidual = 0.25;
  message.normalizedDualResidual = 0.5;
  return message;
}

TEST(testDPGO, SharedObjectLowerMessageKeyComparesDirectedFields) {
  const SharedObjectLowerMessageKey base{1, 2, 3, 4};
  EXPECT_EQ(base, (SharedObjectLowerMessageKey{1, 2, 3, 4}));
  EXPECT_FALSE(base == (SharedObjectLowerMessageKey{2, 1, 3, 4}));
  EXPECT_FALSE(base == (SharedObjectLowerMessageKey{1, 2, 4, 3}));

  EXPECT_LT((SharedObjectLowerMessageKey{0, 2, 3, 4}), base);
  EXPECT_LT((SharedObjectLowerMessageKey{1, 1, 3, 4}), base);
  EXPECT_LT((SharedObjectLowerMessageKey{1, 2, 2, 4}), base);
  EXPECT_LT((SharedObjectLowerMessageKey{1, 2, 3, 3}), base);
  EXPECT_LT(base, (SharedObjectLowerMessageKey{2, 0, 0, 0}));
  EXPECT_LT(base, (SharedObjectLowerMessageKey{1, 3, 0, 0}));
  EXPECT_LT(base, (SharedObjectLowerMessageKey{1, 2, 4, 0}));
  EXPECT_LT(base, (SharedObjectLowerMessageKey{1, 2, 3, 5}));
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueDeliversReorderedMessagesAndAcknowledges) {
  const SharedObjectLowerMessageKey firstKey{1, 0, 10, 20};
  const SharedObjectLowerMessageKey secondKey{1, 2, 11, 21};
  SharedObjectLowerMessageQueue queue;

  const SharedObjectLowerMessage first = queue.send(
      0, 2, false, makeSharedObjectLowerMessage(firstKey, 0, 3, 7, 4, 1.0));
  const SharedObjectLowerMessage second = queue.send(
      1, 0, false, makeSharedObjectLowerMessage(secondKey, 2, 3, 8, 4, 2.0));

  ASSERT_LT(first.messageSequence, second.messageSequence);
  EXPECT_EQ(queue.outstandingCountBySender(0), 1u);
  EXPECT_EQ(queue.outstandingCountBySender(2), 1u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 2u);
  EXPECT_FALSE(queue.empty());

  const std::vector<SharedObjectLowerMessage> roundOne =
      queue.deliverDueMessages(1, 1);
  ASSERT_EQ(roundOne.size(), 1u);
  EXPECT_EQ(roundOne.front().messageSequence, second.messageSequence);
  EXPECT_EQ(roundOne.front().target(0, 0), 2.0);
  EXPECT_FALSE(queue.acknowledge(0, second.messageSequence));
  EXPECT_TRUE(queue.acknowledge(1, second.messageSequence));
  EXPECT_EQ(queue.outstandingCountBySender(2), 0u);

  const std::vector<SharedObjectLowerMessage> roundTwo =
      queue.deliverDueMessages(1, 2);
  ASSERT_EQ(roundTwo.size(), 1u);
  EXPECT_EQ(roundTwo.front().messageSequence, first.messageSequence);
  EXPECT_EQ(roundTwo.front().target(0, 0), 1.0);
  EXPECT_TRUE(queue.acknowledge(1, first.messageSequence));
  EXPECT_TRUE(queue.empty());
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueDroppedAttemptStaysOutstandingUntilRetryAck) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;

  const SharedObjectLowerMessage dropped = queue.send(
      0, 1, true, makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0));

  EXPECT_EQ(queue.droppedAttempts(), 1u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 0u);
  EXPECT_EQ(queue.outstandingCountBySender(0), 1u);
  EXPECT_TRUE(queue.deliverDueMessages(1, 10).empty());
  EXPECT_FALSE(queue.empty());

  ASSERT_TRUE(queue.retry(dropped.messageSequence, 5, 0));
  const std::vector<SharedObjectLowerMessage> delivered =
      queue.deliverDueMessages(1, 5);
  ASSERT_EQ(delivered.size(), 1u);
  EXPECT_EQ(delivered.front().messageSequence, dropped.messageSequence);
  EXPECT_TRUE(queue.acknowledge(1, dropped.messageSequence));
  EXPECT_TRUE(queue.empty());
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueKeepsDueAttemptPendingUntilLinkActive) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;

  const SharedObjectLowerMessage message = queue.send(
      0, 1, false, makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0));

  EXPECT_TRUE(queue.deliverDueMessages(1, 1, {}).empty());
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);
  EXPECT_EQ(queue.outstandingCountBySender(0), 1u);
  EXPECT_TRUE(queue.retryableMessagesBySender(0).empty());

  const std::vector<SharedObjectLowerMessage> delivered =
      queue.deliverDueMessages(1, 2, {0});
  ASSERT_EQ(delivered.size(), 1u);
  EXPECT_EQ(delivered.front().messageSequence, message.messageSequence);
  EXPECT_TRUE(queue.acknowledge(1, message.messageSequence));
  EXPECT_TRUE(queue.empty());
}

TEST(testDPGO, SharedObjectLowerMessageQueueRejectsNegativeDelayAtomically) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;

  EXPECT_THROW(
      queue.send(0, -1, false,
                 makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0)),
      std::invalid_argument);

  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.hasOutstandingMessages());
  EXPECT_EQ(queue.outstandingCountBySender(0), 0u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 0u);
}

TEST(testDPGO, SharedObjectLowerMessageQueueRejectsOverflowDelayAtomically) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;
  const std::size_t maxRound = std::numeric_limits<std::size_t>::max();

  EXPECT_THROW(
      queue.send(maxRound, 1, false,
                 makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0)),
      std::overflow_error);

  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.hasOutstandingMessages());
  EXPECT_EQ(queue.outstandingCountBySender(0), 0u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 0u);
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueRejectsDuplicateRetryWhileAttemptPending) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;

  const SharedObjectLowerMessage dropped = queue.send(
      0, 1, true, makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0));

  EXPECT_TRUE(queue.retry(dropped.messageSequence, 2, 3));
  EXPECT_FALSE(queue.retry(dropped.messageSequence, 3, 1));
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);

  const std::vector<SharedObjectLowerMessage> delivered =
      queue.deliverDueMessages(1, 5);
  ASSERT_EQ(delivered.size(), 1u);
  EXPECT_EQ(delivered.front().messageSequence, dropped.messageSequence);
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueReportsRetryableMessagesWithoutMutating) {
  SharedObjectLowerMessageQueue queue;
  SharedObjectLowerMessage first;
  first.key.receiverRobotId = 1;
  first.key.senderRobotId = 0;
  first.target = Matrix::Ones(1, 1);
  first.normalizedPrimalResidual = 0.0;
  first.normalizedDualResidual = 0.0;
  const SharedObjectLowerMessage dropped =
      queue.send(0, 0, true, first);

  SharedObjectLowerMessage second = first;
  second.key.receiverRobotId = 2;
  const SharedObjectLowerMessage pending =
      queue.send(0, 3, false, second);

  const std::vector<SharedObjectLowerMessage> retryable =
      queue.retryableMessagesBySender(0);

  ASSERT_EQ(retryable.size(), 1u);
  EXPECT_EQ(retryable.front().messageSequence, dropped.messageSequence);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);
  EXPECT_EQ(queue.outstandingCountBySender(0), 2u);
  EXPECT_EQ(pending.messageSequence, 2u);
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueRetryableQueryExcludesPendingRetry) {
  SharedObjectLowerMessageQueue queue;
  SharedObjectLowerMessage message;
  message.key.receiverRobotId = 1;
  message.key.senderRobotId = 0;
  message.target = Matrix::Ones(1, 1);
  message.normalizedPrimalResidual = 0.0;
  message.normalizedDualResidual = 0.0;
  const SharedObjectLowerMessage dropped =
      queue.send(0, 0, true, message);

  ASSERT_EQ(queue.retryableMessagesBySender(0).size(), 1u);
  ASSERT_TRUE(queue.retry(dropped.messageSequence, 2, 3));

  EXPECT_TRUE(queue.retryableMessagesBySender(0).empty());
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);
  EXPECT_FALSE(queue.retry(dropped.messageSequence, 3, 1));
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueRetryNegativeDelayLeavesStateUnchanged) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;

  const SharedObjectLowerMessage dropped = queue.send(
      0, 1, true, makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0));

  EXPECT_THROW(queue.retry(dropped.messageSequence, 2, -1),
               std::invalid_argument);
  EXPECT_EQ(queue.outstandingCountBySender(0), 1u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 0u);
  EXPECT_EQ(queue.droppedAttempts(), 1u);
  EXPECT_FALSE(queue.empty());
  EXPECT_TRUE(queue.hasOutstandingMessages());

  EXPECT_TRUE(queue.retry(dropped.messageSequence, 3, 0));
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);
}

TEST(testDPGO,
     SharedObjectLowerMessageQueueRetryOverflowDelayLeavesStateUnchanged) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageQueue queue;
  const std::size_t maxRound = std::numeric_limits<std::size_t>::max();

  const SharedObjectLowerMessage dropped = queue.send(
      0, 1, true, makeSharedObjectLowerMessage(key, 0, 2, 5, 6, 3.0));

  EXPECT_THROW(queue.retry(dropped.messageSequence, maxRound, 1),
               std::overflow_error);
  EXPECT_EQ(queue.outstandingCountBySender(0), 1u);
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 0u);
  EXPECT_EQ(queue.droppedAttempts(), 1u);
  EXPECT_FALSE(queue.empty());
  EXPECT_TRUE(queue.hasOutstandingMessages());

  EXPECT_TRUE(queue.retry(dropped.messageSequence, 3, 0));
  EXPECT_EQ(queue.pendingDeliveryAttempts(), 1u);
}

TEST(testDPGO, SharedObjectLowerMessageReceiverRejectsStaleAndInvalidPayloads) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageReceiverState receiver;
  receiver.streamActive = true;
  receiver.expectedCommandEpoch = 4;
  receiver.currentEndpointSequence = 9;

  SharedObjectLowerMessage message =
      makeSharedObjectLowerMessage(key, 0, 4, 5, 9, 1.0);
  message.messageSequence = 2;

  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::Accepted);
  receiver.lastAcceptedMessageSequence = message.messageSequence;

  SharedObjectLowerMessage staleEpoch = message;
  staleEpoch.commandEpoch = 3;
  staleEpoch.messageSequence = 3;
  EXPECT_EQ(classifySharedObjectLowerMessage(staleEpoch, receiver),
            SharedObjectLowerMessageStatus::StaleEpoch);

  SharedObjectLowerMessage staleEndpoint = message;
  staleEndpoint.receiverEndpointSequence = 8;
  staleEndpoint.messageSequence = 3;
  EXPECT_EQ(classifySharedObjectLowerMessage(staleEndpoint, receiver),
            SharedObjectLowerMessageStatus::StaleEndpoint);

  SharedObjectLowerMessage staleSequence = message;
  staleSequence.messageSequence = 2;
  EXPECT_EQ(classifySharedObjectLowerMessage(staleSequence, receiver),
            SharedObjectLowerMessageStatus::StaleSequence);

  SharedObjectLowerMessage invalidPayload = message;
  invalidPayload.messageSequence = 3;
  invalidPayload.target(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_EQ(classifySharedObjectLowerMessage(invalidPayload, receiver),
            SharedObjectLowerMessageStatus::InvalidPayload);
}

TEST(testDPGO, SharedObjectLowerMessageReceiverRejectsInactiveStreamFirst) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageReceiverState receiver;
  receiver.streamActive = false;
  receiver.expectedCommandEpoch = 4;
  receiver.currentEndpointSequence = 9;
  receiver.lastAcceptedMessageSequence = 0;

  SharedObjectLowerMessage message =
      makeSharedObjectLowerMessage(key, 0, 3, 5, 8, 1.0);
  message.messageSequence = 1;

  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::InactiveStream);
  EXPECT_EQ(receiver.lastAcceptedMessageSequence, 0u);
}

TEST(testDPGO, SharedObjectLowerMessageReceiverRejectsMismatchedStreamKey) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  const SharedObjectLowerMessageKey otherSender{1, 2, 10, 20};
  const SharedObjectLowerMessageKey otherObject{1, 0, 10, 21};
  SharedObjectLowerMessageReceiverState receiver;
  receiver.initialized = true;
  receiver.streamActive = true;
  receiver.streamKey = key;
  receiver.expectedCommandEpoch = 4;
  receiver.currentEndpointSequence = 9;

  SharedObjectLowerMessage message =
      makeSharedObjectLowerMessage(otherSender, 2, 4, 5, 9, 1.0);
  message.messageSequence = 1;
  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::KeyMismatch);

  message = makeSharedObjectLowerMessage(otherObject, 0, 4, 5, 9, 1.0);
  message.messageSequence = 1;
  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::KeyMismatch);

  message = makeSharedObjectLowerMessage(key, 0, 4, 5, 9, 1.0);
  message.messageSequence = 1;
  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::Accepted);
}

TEST(testDPGO, SharedObjectLowerMessageReceiverRejectsEmptyTarget) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageReceiverState receiver;
  receiver.streamActive = true;
  receiver.expectedCommandEpoch = 4;
  receiver.currentEndpointSequence = 9;

  SharedObjectLowerMessage message =
      makeSharedObjectLowerMessage(key, 0, 4, 5, 9, 1.0);
  message.messageSequence = 1;
  message.target.resize(0, 0);

  EXPECT_EQ(classifySharedObjectLowerMessage(message, receiver),
            SharedObjectLowerMessageStatus::InvalidPayload);
}

TEST(testDPGO,
     SharedObjectLowerMessageReceiverAcceptsInfinityButRejectsNanAndNegative) {
  const SharedObjectLowerMessageKey key{1, 0, 10, 20};
  SharedObjectLowerMessageReceiverState receiver;
  receiver.streamActive = true;
  receiver.expectedCommandEpoch = 4;
  receiver.currentEndpointSequence = 9;

  SharedObjectLowerMessage infiniteResidual =
      makeSharedObjectLowerMessage(key, 0, 4, 5, 9, 1.0);
  infiniteResidual.messageSequence = 1;
  infiniteResidual.normalizedDualResidual =
      std::numeric_limits<double>::infinity();
  EXPECT_EQ(classifySharedObjectLowerMessage(infiniteResidual, receiver),
            SharedObjectLowerMessageStatus::Accepted);

  SharedObjectLowerMessage nanResidual = infiniteResidual;
  nanResidual.messageSequence = 2;
  nanResidual.normalizedPrimalResidual =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(classifySharedObjectLowerMessage(nanResidual, receiver),
            SharedObjectLowerMessageStatus::InvalidPayload);

  SharedObjectLowerMessage negativeResidual = infiniteResidual;
  negativeResidual.messageSequence = 2;
  negativeResidual.normalizedDualResidual = -1.0;
  EXPECT_EQ(classifySharedObjectLowerMessage(negativeResidual, receiver),
            SharedObjectLowerMessageStatus::InvalidPayload);
}

TEST(testDPGO,
     SharedObjectLowerLocalQuiescenceUsesOnlyLocalCertificateAndOutstanding) {
  LocalSharedObjectLowerQuiescenceState state;
  state.localLowerCertificateConverged = true;
  state.currentIncidentCertificates = 1;
  state.expectedIncidentCertificates = 2;
  state.senderOutstandingMessages = 0;

  EXPECT_FALSE(sharedObjectLowerLocallyQuiescent(state));

  state.currentIncidentCertificates = 2;
  state.senderOutstandingMessages = 1;
  EXPECT_FALSE(sharedObjectLowerLocallyQuiescent(state));

  state.senderOutstandingMessages = 0;
  EXPECT_TRUE(sharedObjectLowerLocallyQuiescent(state));

  state.localLowerCertificateConverged = false;
  EXPECT_FALSE(sharedObjectLowerLocallyQuiescent(state));
}


}  // namespace
}  // namespace DPGO
