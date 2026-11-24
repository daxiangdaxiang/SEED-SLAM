#include <DPGO/ManualObjectDran.h>

#include <DPGO/BilevelAssociation.h>
#include <DPGO/CavityAssociationFeedback.h>
#include <DPGO/GeodesicSE3.h>
#include <DPGO/QuadraticProblem.h>

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

using namespace DPGO;

TEST(testDPGO, ManualObjectDranSummarizesTrackEmpiricalInformation) {
  const TrackEmpiricalInformationEstimate singleton =
      estimateTrackEmpiricalInformation({Vector::Zero(1)}, {1.0});
  const TrackEmpiricalInformationEstimate zeroEnergy =
      estimateTrackEmpiricalInformation(
          std::vector<Vector>(3, Vector::Ones(1)), {1.0, 1.0, 1.0});
  const std::vector<Vector> correlated{
      Vector::Constant(1, -0.2), Vector::Constant(1, -0.1),
      Vector::Constant(1, -0.1), Vector::Constant(1, 0.1),
      Vector::Constant(1, 0.1), Vector::Constant(1, 0.2)};
  const TrackEmpiricalInformationEstimate positiveCorrelation =
      estimateTrackEmpiricalInformation(
          correlated, std::vector<double>(correlated.size(), 1.0));
  const std::vector<Vector> dispersed{
      Vector::Constant(1, -10.0), Vector::Constant(1, 10.0),
      Vector::Constant(1, -10.0), Vector::Constant(1, 10.0)};
  const TrackEmpiricalInformationEstimate overdispersed =
      estimateTrackEmpiricalInformation(
          dispersed, std::vector<double>(dispersed.size(), 1.0));

  const ManualObjectDranTrackEmpiricalInformationSummary summary =
      summarizeManualObjectDranTrackEmpiricalInformation(
          {singleton, zeroEnergy, positiveCorrelation, overdispersed}, true);

  EXPECT_TRUE(summary.enabled);
  EXPECT_EQ(summary.trackCount, 4u);
  EXPECT_EQ(summary.observationCount, 14u);
  EXPECT_EQ(summary.singletonTrackCount, 1u);
  EXPECT_EQ(summary.overdispersedTrackCount, 1u);
  EXPECT_DOUBLE_EQ(summary.singletonTrackFraction, 0.25);
  EXPECT_DOUBLE_EQ(summary.overdispersedTrackFraction, 0.25);
  EXPECT_LE(summary.scaleMin, summary.scaleMedian);
  EXPECT_LE(summary.scaleMedian, summary.scaleP90);
  EXPECT_LE(summary.scaleP90, summary.scaleMax);
  EXPECT_GT(summary.scaleMean, 0.0);
  EXPECT_LE(summary.scaleMean, 1.0);
  EXPECT_GT(summary.effectiveSampleRatioMean, 0.0);
  EXPECT_LE(summary.effectiveSampleRatioMean, 1.0);
  EXPECT_GT(summary.effectiveSampleRatioMedian, 0.0);
  EXPECT_LE(summary.effectiveSampleRatioMedian, 1.0);
}

namespace DPGO {
std::size_t manualObjectDranProjectCompletePoint2ConsensusForTest(
    const std::vector<std::size_t> &objectOffsets, std::size_t numObjects,
    unsigned dimension, double relaxation, std::vector<Matrix> &states,
    std::vector<std::vector<Matrix>> &neighborStateCache);
bool manualObjectDranPoint2ConsensusProjectionActiveForTest(
    bool enabled, std::size_t projectionRounds, unsigned iteration);
bool manualObjectDranCachedSharedObjectLowerCertificateIsCurrentForTest(
    std::size_t acceptedOwnerEndpointSequence,
    std::size_t acceptedReceiverEndpointSequence,
    std::size_t currentReceiverEndpointSequence,
    std::size_t acceptedCommandEpoch, std::size_t expectedCommandEpoch);
bool manualObjectDranSharedObjectLowerDeliveryPumpIsDeliveryOnlyForTest();
bool manualObjectDranSharedObjectLowerOutstandingSenderBlocksQuiescenceForTest();
bool manualObjectDranSharedObjectLowerDeliveryPumpRecountsIdempotentlyForTest();
bool manualObjectDranTransportWaitDoesNotRequireLocalSolveForTest();
bool manualObjectDranProtocolOnlyMissingCertificateStallsWithoutOutstandingForTest();
bool manualObjectDranTerminalLocalBudgetMasksOnlyExhaustedRobotsForTest();
bool manualObjectDranTerminalLatestStateChangesPreserveUnsolvedForTest();
std::size_t manualObjectDranFixedIdentityActiveSetSizeForTest(
    const DPGO::Matrix &weights, double rowCapacity,
    double columnCapacity);
bool manualObjectDranSharedConsensusHypothesisActiveForTest(
    ManualObjectDranSharedLatentMode mode, double associationProbability,
    bool identityActive);
std::vector<Matrix> manualObjectDranRobotGaugeTransformsForTest(
    const std::vector<std::vector<Matrix>> &edgeTransforms,
    bool asynchronousGaussSeidel = false);
std::vector<Matrix> manualObjectDranRobotGaugeTransformsWithRmseForTest(
    const std::vector<std::vector<Matrix>> &edgeTransforms,
    const std::vector<std::vector<double>> &edgeTranslationRmse);
Matrix manualObjectDranSelectedCovarianceForTest(
    const SparseMatrix &quadratic,
    const std::vector<std::size_t> &anchoredColumns,
    const std::vector<std::size_t> &selectedColumns);
std::vector<std::vector<Matrix>>
manualObjectDranPairExcludedAssociationWeightsForTest(
    const std::vector<std::vector<Matrix>> &weights,
    std::size_t firstRobot, std::size_t secondRobot);
Matrix manualObjectDranConditionalTranslationMeanForTest(
    const SparseMatrix &quadratic, const SparseMatrix &linear,
    const Matrix &state, unsigned dimension,
    const std::vector<std::size_t> &anchoredColumns);
double manualObjectDranDescriptorAssociationCostForTest(
    const DPGO::Vector &first, const DPGO::Vector &second, double weight);
Matrix manualObjectDranGraphSignatureAssociationCostsForTest(
    const ObjectPGODirectoryData &dataset, const Matrix &localState,
    const Matrix &targetState, std::size_t robotId,
    std::size_t targetRobotId, unsigned dimension,
    const ManualObjectDranOptions &options);
Matrix manualObjectDranObservationIsolatedUpperStateForTest(
    const ObjectPGODirectoryData &dataset, const Matrix &state,
    std::size_t robotId, unsigned d, const ManualObjectDranOptions &options);
Matrix manualObjectDranObjectCopyValidationPoseForTest(
    const ObjectPGODirectoryData &dataset, const Matrix &state,
    std::size_t robotId, std::size_t objectId, unsigned d);
double manualObjectDranCycleMajorizedCostScaleForTest(
    double blockLipschitz, double stepSize, double temperature);
double manualObjectDranWeightedAlignmentRmseForTest(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &rotation,
    const Vector &translation);
double manualObjectDranMarginalTranslationRmseForTest(
    double fitRmse, double marginalRmse);
double manualObjectDranPredictiveTranslationVarianceForTest(
    double fitRmse, double marginalRmse,
    double synchronizationResidual);
AssociationRotoTranslationEstimate
manualObjectDranRobustSparsePoseAlignmentForTest(
    const std::vector<Matrix> &sourcePoses,
    const std::vector<Matrix> &targetPoses,
    const std::vector<SparsePoseAlignmentWeightEntry> &entries,
    const Matrix &initialRotation, const Vector &initialTranslation);
double manualObjectDranRobustSparsePoseAlignmentRmseForTest(
    const std::vector<Matrix> &sourcePoses,
    const std::vector<Matrix> &targetPoses,
    const std::vector<SparsePoseAlignmentWeightEntry> &entries,
    const Matrix &initialRotation, const Vector &initialTranslation);
Matrix manualObjectDranAcceptCurrentRoundPhysicalCopyMajorizedLocalCandidateForTest(
    const QuadraticProblem &problem, const Matrix &currentRoundAnchor,
    const Matrix &candidate);
std::vector<ObjectCopyConsistencyTerm>
manualObjectDranPhysicalCopyMajorizationTermsForTest(
    const Matrix &association, const Matrix &stableWindows,
    const Matrix &rejectedWindows, double pairValidationTrust,
    double stiffness, const ManualObjectDranOptions &options);
struct ManualObjectDranLowerQuadraticStatsForTest {
  std::size_t activePairs{0};
  double consensusCost{0.0};
  double qSquaredNorm{0.0};
  double gSquaredNorm{0.0};
};
ManualObjectDranLowerQuadraticStatsForTest
manualObjectDranPhysicalCopyLowerQuadraticStatsForTest(
    const Matrix &association, const Matrix &stableWindows,
    const Matrix &rejectedWindows, double pairValidationTrust,
    double stiffness, const ManualObjectDranOptions &options);
}

namespace {

struct SharedObjectMessageTotals {
  std::size_t targetMessages{0};
  std::size_t deliveredMessages{0};
  std::size_t staleCertificates{0};
  std::size_t droppedAttempts{0};
  std::size_t retryAttempts{0};
  std::size_t certificatePayloadBytes{0};
};

SharedObjectMessageTotals sharedObjectMessageTotals(
    const ManualObjectDranRunResult &result) {
  SharedObjectMessageTotals totals;
  for (const ManualObjectDranIterationSummary &summary : result.iterations) {
    totals.targetMessages +=
        summary.lowerSharedObjectConsensusTargetMessages;
    totals.deliveredMessages +=
        summary.lowerSharedObjectConsensusDeliveredMessages;
    totals.staleCertificates +=
        summary.lowerSharedObjectConsensusStaleCertificates;
    totals.droppedAttempts +=
        summary.lowerSharedObjectConsensusDroppedAttempts;
    totals.retryAttempts +=
        summary.lowerSharedObjectConsensusRetryAttempts;
    totals.certificatePayloadBytes +=
        summary.lowerSharedObjectConsensusCertificatePayloadBytes;
  }
  return totals;
}

void expectProperColoring(
    const std::vector<std::vector<std::size_t>> &adjacency,
    const std::vector<std::size_t> &colors) {
  ASSERT_EQ(colors.size(), adjacency.size());
  for (std::size_t robotId = 0; robotId < adjacency.size(); ++robotId) {
    for (const std::size_t neighborId : adjacency[robotId]) {
      ASSERT_LT(neighborId, colors.size());
      EXPECT_NE(colors[robotId], colors[neighborId]);
    }
  }
}

std::filesystem::path makeTinyReciprocalPhysicalCopyDataDir(
    const std::string &name) {
  const std::filesystem::path dataDir =
      std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dataDir);
  std::filesystem::create_directories(dataDir);
  const std::string information =
      "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100";
  {
    std::ofstream out(dataDir / "rob_0.g2o");
    out << "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 0 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 1 " << information << "\n";
  }
  {
    std::ofstream out(dataDir / "rob_1.g2o");
    out << "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 6 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 6 0 0 0 0 0 1 " << information << "\n";
  }
  {
    std::ofstream out(dataDir / "association.csv");
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,0,1\n";
    out << "1,0,0,0,1\n";
  }
  return dataDir;
}

ManualObjectDranOptions tinyReciprocalPhysicalCopyOptions(
    const std::filesystem::path &dataDir) {
  ManualObjectDranOptions options;
  options.numRobots = 2;
  options.numObjects = 1;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = (dataDir / "association.csv").string();
  options.associationRequireObservedObjects = true;
  options.associationResidualUseTransformAlignment = false;
  options.associationGaugeSynchronizationEnabled = false;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::Disabled;
  options.associationLowerPhysicalCopyMajorization = false;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 20;
  options.localTrustRegionMaxInnerIterations = 20;
  return options;
}

std::filesystem::path makeTinyObjectPose3SharedOnlyDataDir(
    const std::string &name) {
  const std::filesystem::path dataDir =
      std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(dataDir);
  std::filesystem::create_directories(dataDir);
  const std::string information =
      "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100";
  {
    std::ofstream out(dataDir / "rob_0.g2o");
    out << "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " << information << "\n";
  }
  {
    std::ofstream out(dataDir / "rob_1.g2o");
    out << "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 3 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 3 0 0 0 0 0 1 " << information << "\n";
  }
  {
    std::ofstream out(dataDir / "association.csv");
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,0,1\n";
    out << "1,0,0,0,1\n";
  }
  return dataDir;
}

ManualObjectDranOptions tinyObjectPose3SharedOnlyOptions(
    const std::filesystem::path &dataDir, unsigned maxIterations) {
  ManualObjectDranOptions options;
  options.numRobots = 2;
  options.numObjects = 1;
  options.maxIterations = maxIterations;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = (dataDir / "association.csv").string();
  options.associationFreezeUpperVariables = true;
  options.associationPolicy = ManualObjectDranAssociationPolicy::Manual;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::DirectEdgeDual;
  options.associationLowerDirectEdgeSynchronousSharedOnly = true;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::CopyBaseline;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;
  return options;
}

std::filesystem::path writeEventSchedule(
    const std::string &name,
    const std::vector<ManualObjectDranEvent> &events) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / name;
  std::ofstream out(path);
  out << "event,robot,logical_time,delay_events\n";
  for (const ManualObjectDranEvent &event : events) {
    out << event.event << "," << event.robot << ","
        << event.logicalTime << "," << event.delayEvents << "\n";
  }
  return path;
}

}  // namespace

TEST(testDPGO, ManualObjectDranEventScheduleStrictLoaderAcceptsValidCsv) {
  const std::filesystem::path path = writeEventSchedule(
      "manual_object_dran_valid_event_schedule.csv",
      {{0, 1, 0.0, 2}, {1, 0, 0.5, 0}, {2, 1, 0.5, 1}});

  const std::vector<ManualObjectDranEvent> events =
      loadManualObjectDranEventScheduleForTest(path.string(), 2, 3);

  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events[0].event, 0u);
  EXPECT_EQ(events[0].robot, 1u);
  EXPECT_DOUBLE_EQ(events[1].logicalTime, 0.5);
  EXPECT_EQ(events[2].delayEvents, 1u);
}

TEST(testDPGO, ManualObjectDranEventScheduleStrictLoaderRejectsBadCsv) {
  auto expectReject = [](const std::string &name, const std::string &body) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << body;
    out.close();
    EXPECT_THROW(loadManualObjectDranEventScheduleForTest(path.string(), 2, 2),
                 std::invalid_argument);
  };

  expectReject("manual_object_dran_event_missing_zero.csv",
               "event,robot,logical_time,delay_events\n1,0,0,0\n2,1,1,0\n");
  expectReject("manual_object_dran_event_duplicate.csv",
               "event,robot,logical_time,delay_events\n0,0,0,0\n0,1,1,0\n");
  expectReject("manual_object_dran_event_nonmonotone.csv",
               "event,robot,logical_time,delay_events\n0,0,1,0\n1,1,0,0\n");
  expectReject("manual_object_dran_event_bad_robot.csv",
               "event,robot,logical_time,delay_events\n0,2,0,0\n1,1,1,0\n");
  expectReject("manual_object_dran_event_negative_time.csv",
               "event,robot,logical_time,delay_events\n0,0,-1,0\n1,1,1,0\n");
  expectReject("manual_object_dran_event_gap.csv",
               "event,robot,logical_time,delay_events\n0,0,0,0\n2,1,1,0\n");
  expectReject("manual_object_dran_event_wrong_count.csv",
               "event,robot,logical_time,delay_events\n0,0,0,0\n");
  expectReject("manual_object_dran_event_bad_header.csv",
               "event,who,logical_time,delay_events\n0,0,0,0\n1,1,1,0\n");
}

TEST(testDPGO, ManualObjectDranEventScheduleOneActiveSenderOnly) {
  const std::filesystem::path dataDir =
      makeTinyObjectPose3SharedOnlyDataDir(
          "manual_object_dran_event_one_active_sender");
  ManualObjectDranOptions options =
      tinyObjectPose3SharedOnlyOptions(dataDir, 2);
  options.eventScheduleFile =
      writeEventSchedule("manual_object_dran_one_active_schedule.csv",
                         {{0, 0, 0.0, 0}, {1, 1, 1.0, 0}})
          .string();

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir.string(), options);

  ASSERT_GE(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].eventRobot, 0u);
  EXPECT_EQ(result.iterations[2].eventRobot, 1u);
  ASSERT_EQ(result.localObjectStateSequences.size(), 2u);
  ASSERT_EQ(result.localObjectStateSequences[0].size(), 1u);
  ASSERT_EQ(result.localObjectStateSequences[1].size(), 1u);
  EXPECT_EQ(result.localObjectStateSequences[0][0], 1u);
  EXPECT_EQ(result.localObjectStateSequences[1][0], 1u);
  EXPECT_GT(result.iterations[1].sharedEndpointStateQueuedMessages, 0u);
  EXPECT_GT(result.iterations[1].lowerSharedObjectConsensusActiveHypotheses,
            0u);
  EXPECT_EQ(result.iterations[1].lowerSharedObjectConsensusTargetMessages, 0u);
  EXPECT_EQ(result.iterations[1].lowerSharedObjectConsensusCertificateMessages,
            0u);
  EXPECT_EQ(result.iterations.back().sharedEndpointStateOutstandingMessages,
            0u);
}

TEST(testDPGO, ManualObjectDranIdentityOracleEventScheduleDrainsLongDelay) {
  const std::filesystem::path dataDir =
      makeTinyObjectPose3SharedOnlyDataDir(
          "manual_object_dran_identity_event_long_delay");
  ManualObjectDranOptions options =
      tinyObjectPose3SharedOnlyOptions(dataDir, 2);
  options.associationMode = ManualObjectDranAssociationMode::IdentityOracle;
  options.associationFile.clear();
  options.eventScheduleFile =
      writeEventSchedule("manual_object_dran_identity_long_delay.csv",
                         {{0, 0, 0.0, 8}, {1, 1, 1.0, 8}})
          .string();

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir.string(), options);

  ASSERT_GT(result.iterations.size(), 3u);
  EXPECT_GT(result.iterations[1].lowerSharedObjectConsensusActiveHypotheses,
            0u);
  EXPECT_TRUE(result.iterations.back().queueDrain);
  EXPECT_EQ(result.iterations.back().sharedEndpointStateOutstandingMessages,
            0u);
  EXPECT_GT(result.iterations.back().sharedEndpointStateDeliveredMessages, 0u);
  EXPECT_EQ(result.iterations.back().lowerSharedObjectConsensusTargetMessages,
            0u);
  EXPECT_EQ(
      result.iterations.back().lowerSharedObjectConsensusCertificateMessages,
      0u);
}

TEST(testDPGO, ManualObjectDranSharedEndpointDelayZeroMatchesDirect) {
  const std::filesystem::path dataDir =
      makeTinyObjectPose3SharedOnlyDataDir(
          "manual_object_dran_delay_zero_matches_direct");
  ManualObjectDranOptions direct =
      tinyObjectPose3SharedOnlyOptions(dataDir, 3);
  direct.localTrustRegionIterations = 0;
  ManualObjectDranOptions queued = direct;
  queued.eventScheduleFile =
      writeEventSchedule("manual_object_dran_delay_zero_equiv_schedule.csv",
                         {{0, 0, 0.0, 0}, {1, 1, 1.0, 0}, {2, 0, 2.0, 0}})
          .string();

  const ManualObjectDranRunResult queuedResult =
      runManualObjectDran(dataDir.string(), queued);
  const ManualObjectDranRunResult replayedQueuedResult =
      runManualObjectDran(dataDir.string(), queued);

  ASSERT_EQ(replayedQueuedResult.finalEstimates.size(),
            queuedResult.finalEstimates.size());
  for (std::size_t robot = 0; robot < replayedQueuedResult.finalEstimates.size();
       ++robot) {
    EXPECT_LE((replayedQueuedResult.finalEstimates[robot] -
               queuedResult.finalEstimates[robot]).norm(),
              1e-8);
  }
  EXPECT_GT(queuedResult.iterations.back().sharedEndpointStateDeliveredMessages,
            0u);
  EXPECT_EQ(
      queuedResult.iterations.back().lowerSharedObjectConsensusPayloadBytes,
      0u);
}

TEST(testDPGO,
     ManualObjectDranSynchronousSharedOnlyPositiveDelayUsesEndpointQueue) {
  const std::filesystem::path dataDir =
      makeTinyObjectPose3SharedOnlyDataDir(
          "manual_object_dran_sync_positive_delay");
  ManualObjectDranOptions options =
      tinyObjectPose3SharedOnlyOptions(dataDir, 3);
  options.associationLowerMessageDelayRounds = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir.string(), options);

  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  EXPECT_GT(summary.sharedEndpointStateDeliveredMessages, 0u);
  EXPECT_EQ(summary.sharedEndpointStateOutstandingMessages, 0u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusTargetMessages, 0u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusCertificateMessages, 0u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusPayloadBytes, 0u);
}

TEST(testDPGO,
     ManualObjectDranReducedLocalSolveSkipsUnusedProblemPreconditioner) {
  ManualObjectDranOptions options;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  EXPECT_FALSE(manualObjectDranLocalSolveNeedsProblemPreconditioner(options,
                                                                    1));

  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  EXPECT_TRUE(manualObjectDranLocalSolveNeedsProblemPreconditioner(options,
                                                                   1));

  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullSolverOuterIterations = 2;
  EXPECT_TRUE(manualObjectDranLocalSolveNeedsProblemPreconditioner(options,
                                                                   2));
  EXPECT_FALSE(manualObjectDranLocalSolveNeedsProblemPreconditioner(options,
                                                                    3));
}

TEST(testDPGO, ManualObjectDranExactPoint2OwnersAreRoundRobin) {
  const std::vector<std::vector<std::size_t>> observers(
      15, std::vector<std::size_t>{0, 1, 2, 3, 4});

  const auto owners = manualObjectDranExactPoint2Owners(5, observers);

  ASSERT_EQ(owners.size(), observers.size());
  for (std::size_t object = 0; object < owners.size(); ++object) {
    EXPECT_EQ(owners[object], object % 5);
  }
}

TEST(testDPGO, ManualObjectDranExactPoint2RequiresEveryRemotePacket) {
  const std::vector<std::vector<std::size_t>> observers{{0, 1, 2}};
  std::vector<ManualObjectDranExactPoint2Statistic> localStatistics;
  std::vector<ManualObjectDranExactPoint2Packet> packets;
  for (std::size_t robot = 0; robot < 3; ++robot) {
    ManualObjectDranExactPoint2Statistic statistic;
    statistic.robot = robot;
    statistic.object = 0;
    statistic.information = 1.0;
    statistic.weightedPosition = Vector::Constant(2, robot + 1.0);
    if (robot == 0) {
      localStatistics.push_back(statistic);
    } else {
      ManualObjectDranExactPoint2Packet packet;
      packet.sender = robot;
      packet.owner = 0;
      packet.object = 0;
      packet.carriesInformation = true;
      packet.information = statistic.information;
      packet.weightedPosition = statistic.weightedPosition;
      packets.push_back(packet);
    }
  }
  packets.pop_back();
  ManualObjectDranExactPoint2InformationCache cache(
      3, std::vector<double>(1, std::numeric_limits<double>::quiet_NaN()));

  const auto update = manualObjectDranAggregateExactPoint2(
      3, observers, localStatistics, packets, cache);

  EXPECT_FALSE(update.valid);
  EXPECT_NE(update.status.find("missing"), std::string::npos);
}

TEST(testDPGO, ManualObjectDranExactPoint2MatchesCentralizedAndCountsBytes) {
  const std::vector<std::vector<std::size_t>> observers(
      15, std::vector<std::size_t>{0, 1, 2, 3, 4});
  const auto owners = manualObjectDranExactPoint2Owners(5, observers);
  std::vector<ManualObjectDranExactPoint2Statistic> localStatistics;
  std::vector<ManualObjectDranExactPoint2Packet> firstPackets;
  for (std::size_t object = 0; object < 15; ++object) {
    for (std::size_t robot = 0; robot < 5; ++robot) {
      ManualObjectDranExactPoint2Statistic statistic;
      statistic.robot = robot;
      statistic.object = object;
      statistic.information = static_cast<double>(robot + 1);
      statistic.weightedPosition.resize(2);
      statistic.weightedPosition <<
          statistic.information * static_cast<double>(object + robot),
          statistic.information *
              (static_cast<double>(object) - static_cast<double>(robot));
      if (robot == owners[object]) {
        localStatistics.push_back(statistic);
      } else {
        ManualObjectDranExactPoint2Packet packet;
        packet.sender = robot;
        packet.owner = owners[object];
        packet.object = object;
        packet.carriesInformation = true;
        packet.information = statistic.information;
        packet.weightedPosition = statistic.weightedPosition;
        firstPackets.push_back(packet);
      }
    }
  }
  ManualObjectDranExactPoint2InformationCache cache(
      5, std::vector<double>(15,
                              std::numeric_limits<double>::quiet_NaN()));

  const auto first = manualObjectDranAggregateExactPoint2(
      5, observers, localStatistics, firstPackets, cache);

  ASSERT_TRUE(first.valid) << first.status;
  EXPECT_EQ(first.statisticMessages, 60u);
  EXPECT_EQ(first.statisticPayloadBytes, 60u * 24u);
  EXPECT_EQ(first.broadcastMessages, 60u);
  EXPECT_EQ(first.broadcastPayloadBytes, 60u * 16u);
  EXPECT_EQ(first.statisticPayloadBytes + first.broadcastPayloadBytes, 2400u);
  ASSERT_EQ(first.points.size(), 15u);
  for (std::size_t object = 0; object < first.points.size(); ++object) {
    const double totalInformation = 15.0;
    Vector expected = Vector::Zero(2);
    for (std::size_t robot = 0; robot < 5; ++robot) {
      const double information = static_cast<double>(robot + 1);
      expected(0) += information * static_cast<double>(object + robot);
      expected(1) += information *
                     (static_cast<double>(object) -
                      static_cast<double>(robot));
    }
    expected /= totalInformation;
    EXPECT_LE((first.points[object] - expected).norm(), 1e-12);
  }

  std::vector<ManualObjectDranExactPoint2Packet> laterPackets = firstPackets;
  for (auto &packet : laterPackets) {
    packet.carriesInformation = false;
    packet.information = 0.0;
  }
  const auto later = manualObjectDranAggregateExactPoint2(
      5, observers, localStatistics, laterPackets, cache);
  ASSERT_TRUE(later.valid) << later.status;
  EXPECT_EQ(later.statisticPayloadBytes + later.broadcastPayloadBytes, 1920u);
  EXPECT_EQ(2400u + 49u * 1920u, 96480u);
}

TEST(testDPGO, ManualObjectDranExactPoint2AccumulatesLocalStatistics) {
  Matrix firstRotation = Matrix::Identity(2, 2);
  Matrix secondRotation(2, 2);
  secondRotation << 0.0, -1.0, 1.0, 0.0;
  Vector firstTranslation(2);
  firstTranslation << 1.0, 2.0;
  Vector secondTranslation(2);
  secondTranslation << -1.0, 1.0;
  Vector firstMeasurement(2);
  firstMeasurement << 2.0, -1.0;
  Vector secondMeasurement(2);
  secondMeasurement << 1.0, 2.0;

  const auto statistic = manualObjectDranExactPoint2LocalStatistic(
      2, 7, {firstRotation, secondRotation},
      {firstTranslation, secondTranslation},
      {firstMeasurement, secondMeasurement}, {2.0, 3.0});

  EXPECT_EQ(statistic.robot, 2u);
  EXPECT_EQ(statistic.object, 7u);
  EXPECT_DOUBLE_EQ(statistic.information, 5.0);
  Vector expected(2);
  expected << -3.0, 8.0;
  EXPECT_LE((statistic.weightedPosition - expected).norm(), 1e-12);
}

TEST(testDPGO, ManualObjectDranConditionalTrajectoryQuadraticMatchesFullCost) {
  constexpr unsigned d = 2;
  constexpr std::size_t trajectoryPoses = 2;
  constexpr Eigen::Index trajectoryColumns =
      trajectoryPoses * static_cast<std::size_t>(d + 1u);
  constexpr Eigen::Index totalColumns = trajectoryColumns + d + 1u;
  Matrix denseSeed(totalColumns, totalColumns);
  for (Eigen::Index row = 0; row < totalColumns; ++row) {
    for (Eigen::Index col = 0; col < totalColumns; ++col) {
      denseSeed(row, col) =
          0.03 * static_cast<double>(1 + row + 2 * col) +
          (row == col ? 1.0 : 0.0);
    }
  }
  const Matrix denseQ = denseSeed.transpose() * denseSeed;
  const SparseMatrix Q = denseQ.sparseView();
  Matrix denseG(3, totalColumns);
  for (Eigen::Index row = 0; row < denseG.rows(); ++row) {
    for (Eigen::Index col = 0; col < denseG.cols(); ++col) {
      denseG(row, col) = 0.02 * static_cast<double>(1 + 3 * row - col);
    }
  }
  const SparseMatrix G = denseG.sparseView();
  Matrix state0(3, totalColumns);
  for (Eigen::Index row = 0; row < state0.rows(); ++row) {
    for (Eigen::Index col = 0; col < state0.cols(); ++col) {
      state0(row, col) = 0.1 * static_cast<double>(1 + row + col);
    }
  }
  Matrix state1 = state0;
  state1.leftCols(trajectoryColumns).array() += 0.07;

  const auto conditional = manualObjectDranConditionTrajectoryQuadratic(
      Q, G, state0, trajectoryPoses, d);

  ASSERT_TRUE(conditional.valid) << conditional.status;
  const auto objective = [](const Matrix &state, const SparseMatrix &quadratic,
                            const SparseMatrix &linear) {
    return 0.5 * ((state * quadratic).cwiseProduct(state)).sum() +
           (state.cwiseProduct(linear)).sum();
  };
  const double fullDifference =
      objective(state1, Q, G) - objective(state0, Q, G);
  const double conditionalDifference =
      objective(state1.leftCols(trajectoryColumns), conditional.Q,
                conditional.G) -
      objective(state0.leftCols(trajectoryColumns), conditional.Q,
                conditional.G);
  EXPECT_NEAR(fullDifference, conditionalDifference, 1e-12);
  EXPECT_LE((conditional.initialState -
             state0.leftCols(trajectoryColumns)).norm(),
            1e-12);
}

TEST(testDPGO, ManualObjectDranPoint2SchurMatchesDirectElimination) {
  Matrix seed(5, 5);
  seed << 1.0, 0.2, -0.1, 0.3, 0.0,
          0.1, 1.1, 0.2, -0.2, 0.1,
          0.0, -0.1, 0.9, 0.1, 0.3,
          0.2, 0.0, 0.1, 1.2, -0.2,
          -0.1, 0.3, 0.0, 0.2, 1.0;
  const Matrix hessian =
      seed.transpose() * seed + 2.0 * Matrix::Identity(5, 5);
  Vector gradient(5);
  gradient << 0.7, -1.1, 0.4, 1.3, -0.8;

  const ManualObjectDranPoint2SchurModel model =
      manualObjectDranPoint2SchurFromDenseForTest(hessian, gradient, 3);

  ASSERT_TRUE(model.valid) << model.status;
  const Matrix hxx = hessian.topLeftCorner(3, 3);
  const Matrix hxp = hessian.topRightCorner(3, 2);
  const Matrix hpx = hessian.bottomLeftCorner(2, 3);
  const Matrix hpp = hessian.bottomRightCorner(2, 2);
  const Vector gx = gradient.head(3);
  const Vector gp = gradient.tail(2);
  const Matrix expectedHessian = hpp - hpx * hxx.ldlt().solve(hxp);
  const Vector expectedGradient = gp - hpx * hxx.ldlt().solve(gx);
  EXPECT_LT((model.hessian - expectedHessian).norm(), 1e-10);
  EXPECT_LT((model.gradient - expectedGradient).norm(), 1e-10);
  EXPECT_LT((model.hessian - model.hessian.transpose()).norm(), 1e-12);

  const ManualObjectDranPoint2SchurSolve solve =
      manualObjectDranSolvePoint2SchurForTest({model});
  ASSERT_TRUE(solve.valid) << solve.status;
  EXPECT_DOUBLE_EQ(solve.spectralShift, 0.0);
  const Vector privateStep =
      manualObjectDranRecoverPrivatePoint2StepForTest(
          hessian, gradient, 3, solve.pointStep);
  Vector recovered(5);
  recovered << privateStep, solve.pointStep;
  const Vector direct = hessian.ldlt().solve(-gradient);
  EXPECT_LT((recovered - direct).norm(), 1e-9);
  EXPECT_LT(solve.relativeResidual, 1e-10);
}

TEST(testDPGO, ManualObjectDranPoint2SchurAggregatesLocalModels) {
  ManualObjectDranPoint2SchurModel first;
  first.valid = true;
  first.status = "ok";
  first.hessian = (Matrix(2, 2) << 3.0, 0.4, 0.4, 2.0).finished();
  first.gradient = (Vector(2) << 1.0, -0.5).finished();
  ManualObjectDranPoint2SchurModel second;
  second.valid = true;
  second.status = "ok";
  second.hessian = (Matrix(2, 2) << 1.5, -0.2, -0.2, 2.5).finished();
  second.gradient = (Vector(2) << -0.3, 0.8).finished();

  const ManualObjectDranPoint2SchurSolve solve =
      manualObjectDranSolvePoint2SchurForTest({first, second});

  ASSERT_TRUE(solve.valid) << solve.status;
  const Matrix hessian = first.hessian + second.hessian;
  const Vector gradient = first.gradient + second.gradient;
  const Vector expected = hessian.ldlt().solve(-gradient);
  EXPECT_LT((solve.pointStep - expected).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(solve.spectralShift, 0.0);
  EXPECT_LT(solve.relativeResidual, 1e-12);
}

TEST(testDPGO, ManualObjectDranCoupledPoint2SchurCorrectionSchedule) {
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      false, 3u, 2u, 3u));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      true, 3u, 2u, 2u));
  EXPECT_TRUE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      true, 3u, 2u, 3u));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      true, 3u, 2u, 4u));
  EXPECT_TRUE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      true, 3u, 2u, 5u));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
      true, 3u, 0u, 5u));
}

TEST(testDPGO, ManualObjectDranCoupledPoint2SchurCorrectionSolverRoute) {
  EXPECT_FALSE(
      manualObjectDranCoupledPoint2SchurUsesMatrixFreeOwnerPcgForTest(
          false, true));
  EXPECT_FALSE(
      manualObjectDranCoupledPoint2SchurUsesMatrixFreeOwnerPcgForTest(
          true, false));
  EXPECT_TRUE(
      manualObjectDranCoupledPoint2SchurUsesMatrixFreeOwnerPcgForTest(
          true, true));
}

TEST(testDPGO, ManualObjectDranCoupledPoint2SchurInitializesOwnerLedger) {
  EXPECT_FALSE(manualObjectDranPoint2OwnerLedgerRequiredForTest(false,
                                                                false));
  EXPECT_TRUE(manualObjectDranPoint2OwnerLedgerRequiredForTest(true,
                                                               false));
  EXPECT_TRUE(manualObjectDranPoint2OwnerLedgerRequiredForTest(false,
                                                               true));
}

TEST(testDPGO, ManualObjectDranCoupledPoint2SchurCorrectionMeritGuard) {
  EXPECT_TRUE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, 9.0));
  EXPECT_TRUE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, 10.0));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, 10.001));
  EXPECT_TRUE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, 10.0 + 5e-8, 1e-8));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, 10.0 + 2e-7, 1e-8));
  EXPECT_FALSE(manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
      10.0, std::numeric_limits<double>::quiet_NaN()));
}

TEST(testDPGO, ManualObjectDranIntegratedPoint2AmmExtrapolatesMixedState) {
  const unsigned d = 2;
  Matrix previous = Matrix::Zero(d, 2 * (d + 1));
  Matrix current = Matrix::Zero(d, 2 * (d + 1));
  previous.block(0, 0, d, d).setIdentity();
  current.block(0, 0, d, d) << 0.0, -1.0, 1.0, 0.0;
  previous.block(0, d + 1, d, d) << 0.0, 1.0, -1.0, 0.0;
  current.block(0, d + 1, d, d) << -1.0, 0.0, 0.0, -1.0;
  previous.col(d) << 1.0, 2.0;
  current.col(d) << 3.0, 6.0;
  previous.col((d + 1) + d) << 4.0, 8.0;
  current.col((d + 1) + d) << 10.0, 14.0;

  const Matrix extrapolated =
      manualObjectDranExtrapolateIntegratedPoint2StateForTest(
          current, previous, 1u, d, 0.5);

  const Matrix trajectoryRotation = extrapolated.block(0, 0, d, d);
  EXPECT_LE((trajectoryRotation.transpose() * trajectoryRotation -
             Matrix::Identity(d, d)).norm(),
            1e-12);
  EXPECT_EQ(extrapolated.block(0, d + 1, d, d),
            current.block(0, d + 1, d, d));
  EXPECT_EQ(extrapolated.col(d), (Vector(2) << 4.0, 8.0).finished());
  EXPECT_EQ(extrapolated.col((d + 1) + d),
            (Vector(2) << 13.0, 17.0).finished());
}

TEST(testDPGO, ManualObjectDranIntegratedPoint2AmmUsesGuardedPortfolio) {
  EXPECT_TRUE(manualObjectDranIntegratedPoint2AmmSelectsAcceleratedForTest(
      10.0, 9.0, 8.0));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2AmmSelectsAcceleratedForTest(
      10.0, 8.0, 9.0));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2AmmSelectsAcceleratedForTest(
      10.0, 11.0, 10.5));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2AmmSelectsAcceleratedForTest(
      10.0, 9.0, std::numeric_limits<double>::quiet_NaN()));
}

TEST(testDPGO, ManualObjectDranIntegratedPoint2AmmUsesCommonEdgeAcceptance) {
  EXPECT_TRUE(
      manualObjectDranIntegratedPoint2AmmCommonAcceptanceForTest(
          10.0, 9.0, 8.0, 4u, 5.0, 4.0));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2AmmCommonAcceptanceForTest(
          10.0, 9.0, 8.0, 4u, 5.0, 6.0));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2AmmCommonAcceptanceForTest(
          10.0, 9.0, 8.0, 0u, 0.0, 0.0));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2AmmCommonAcceptanceForTest(
          10.0, 8.0, 9.0, 4u, 5.0, 4.0));
  EXPECT_TRUE(manualObjectDranIntegratedPoint2EdgeMeritAcceptsForTest(
      4u, 5.0, 4.0));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2EdgeMeritAcceptsForTest(
      4u, 5.0, 6.0));
}

TEST(testDPGO,
     ManualObjectDranIntegratedPoint2AmmUsesPerObjectFactorAcceptance) {
  EXPECT_TRUE(
      manualObjectDranIntegratedPoint2ObjectFactorMeritAcceptsForTest(
          {5.0, 3.0, 0.0}, {4.0, 3.0, 0.0}));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2ObjectFactorMeritAcceptsForTest(
          {5.0, 3.0, 0.0}, {4.0, 3.1, 0.0}));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2ObjectFactorMeritAcceptsForTest(
          {5.0, 3.0}, {4.0}));
  EXPECT_FALSE(
      manualObjectDranIntegratedPoint2ObjectFactorMeritAcceptsForTest(
          {5.0}, {std::numeric_limits<double>::quiet_NaN()}));
}

TEST(testDPGO,
     ManualObjectDranIntegratedPoint2AmmUsesCandidateCopyAcceptance) {
  EXPECT_TRUE(manualObjectDranIntegratedPoint2CopyMeritAcceptsForTest(
      2.0, 1.5));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2CopyMeritAcceptsForTest(
      2.0, 2.1));
  EXPECT_FALSE(manualObjectDranIntegratedPoint2CopyMeritAcceptsForTest(
      2.0, std::numeric_limits<double>::infinity()));
}

TEST(testDPGO,
     ManualObjectDranIntegratedPoint2UsesObjectWiseEdgeMomentum) {
  EXPECT_DOUBLE_EQ(
      manualObjectDranIntegratedPoint2EdgeMomentumForTest(
          0.2, {0.1, 0.4}, true, 1u),
      0.4);
  EXPECT_DOUBLE_EQ(
      manualObjectDranIntegratedPoint2EdgeMomentumForTest(
          0.2, {0.1, 0.4}, false, 1u),
      0.2);
  EXPECT_DOUBLE_EQ(
      manualObjectDranIntegratedPoint2EdgeMomentumForTest(
          0.2, {0.1, 0.4}, true, 2u),
      0.2);
}

TEST(testDPGO, ManualObjectDranIntegratedPoint2DefaultsToSingleLocalSolve) {
  const ManualObjectDranOptions options;
  EXPECT_FALSE(options.integratedPoint2LocalAmm);
  EXPECT_TRUE(options.integratedPoint2CommonAcceptance);
  EXPECT_DOUBLE_EQ(options.integratedPoint2AmmMomentum, 0.10);
  EXPECT_FALSE(options.integratedPoint2AdaptiveMomentum);
  EXPECT_DOUBLE_EQ(options.integratedPoint2AdaptiveMomentumMin, 0.0);
  EXPECT_DOUBLE_EQ(options.integratedPoint2AdaptiveMomentumMax, 0.30);
  EXPECT_DOUBLE_EQ(options.integratedPoint2AdaptiveMomentumEmaDecay, 0.5);
  EXPECT_FALSE(options.integratedPoint2ObjectWiseAdaptiveMomentum);
}

TEST(testDPGO, ManualObjectDranIntegratedPoint2UsesOneOwnerSpectralStep) {
  const double step =
      manualObjectDranIntegratedPoint2OwnerSpectralStepForTest(
          {{4.0, 2.0}, {5.0, 3.0}}, 1.0, 1.0, 0.75, 2.0, 0.0);

  EXPECT_NEAR(step, 9.0 / 5.0, 1e-12);
}

TEST(testDPGO, ManualObjectDranOwnerSpectralStateSurvivesRefresh) {
  std::vector<bool> primed{true, false};
  std::vector<double> steps{1.4, 0.9};

  manualObjectDranResizeOwnerSpectralStateForTest(primed, steps, 2u);
  EXPECT_EQ(primed, (std::vector<bool>{true, false}));
  EXPECT_EQ(steps, (std::vector<double>{1.4, 0.9}));

  manualObjectDranResizeOwnerSpectralStateForTest(primed, steps, 3u);
  EXPECT_EQ(primed, (std::vector<bool>{false, false, false}));
  EXPECT_EQ(steps, (std::vector<double>{1.0, 1.0, 1.0}));
}

TEST(testDPGO, ManualObjectDranPoint2MatrixFreeProductMatchesDenseSchur) {
  Matrix seed(7, 7);
  seed << 1.0, 0.2, 0.0, -0.1, 0.3, 0.0, 0.2,
          0.1, 1.1, -0.2, 0.0, 0.1, 0.2, -0.1,
          0.0, 0.3, 0.9, 0.1, -0.2, 0.1, 0.0,
          0.2, 0.0, 0.1, 1.2, 0.0, -0.1, 0.2,
          -0.1, 0.2, 0.0, 0.1, 1.0, 0.3, 0.0,
          0.0, -0.1, 0.2, 0.0, 0.1, 0.8, 0.2,
          0.1, 0.0, -0.1, 0.2, 0.0, 0.1, 1.1;
  const Matrix hessian =
      seed.transpose() * seed + 2.0 * Matrix::Identity(7, 7);
  const Vector value = (Vector(4) << 0.4, -0.7, 1.2, 0.3).finished();
  const Matrix hxx = hessian.topLeftCorner(3, 3);
  const Matrix hxp = hessian.topRightCorner(3, 4);
  const Matrix hpx = hessian.bottomLeftCorner(4, 3);
  const Matrix hpp = hessian.bottomRightCorner(4, 4);
  const Vector expected =
      (hpp - hpx * hxx.ldlt().solve(hxp)) * value;

  const ManualObjectDranPoint2SchurProduct product =
      manualObjectDranPoint2SchurProductFromDenseForTest(
          hessian, 3, value);

  ASSERT_TRUE(product.valid) << product.status;
  EXPECT_LT((product.product - expected).norm(), 1e-11);
  EXPECT_EQ(product.responseSolves, 1u);
  EXPECT_LT(product.responseRelativeResidual, 1e-12);
}

TEST(testDPGO, ManualObjectDranSchurBlockDiagonalIncludesPrivateResponse) {
  Matrix seed(7, 7);
  seed << 1.0, 0.2, 0.0, -0.1, 0.3, 0.0, 0.2,
          0.1, 1.1, -0.2, 0.0, 0.1, 0.2, -0.1,
          0.0, 0.3, 0.9, 0.1, -0.2, 0.1, 0.0,
          0.2, 0.0, 0.1, 1.2, 0.0, -0.1, 0.2,
          -0.1, 0.2, 0.0, 0.1, 1.0, 0.3, 0.0,
          0.0, -0.1, 0.2, 0.0, 0.1, 0.8, 0.2,
          0.1, 0.0, -0.1, 0.2, 0.0, 0.1, 1.1;
  const Matrix hessian =
      seed.transpose() * seed + 2.0 * Matrix::Identity(7, 7);
  const Matrix hxx = hessian.topLeftCorner(3, 3);
  const Matrix hxp = hessian.topRightCorner(3, 4);
  const Matrix hpp = hessian.bottomRightCorner(4, 4);
  const Matrix schur = hpp - hxp.transpose() * hxx.ldlt().solve(hxp);
  constexpr double damping = 0.05;
  Matrix expected = Matrix::Zero(4, 4);
  expected.topLeftCorner(2, 2) = schur.topLeftCorner(2, 2);
  expected.bottomRightCorner(2, 2) = schur.bottomRightCorner(2, 2);
  expected.diagonal().array() += damping;

  const Matrix actual =
      manualObjectDranSchurBlockDiagonalFromDenseForTest(
          hxx, hxp, hpp, 2u, damping);

  EXPECT_TRUE(actual.isApprox(expected, 1e-12));
  EXPECT_LT(actual.topRightCorner(2, 2).norm(), 1e-15);
  EXPECT_LT(actual.bottomLeftCorner(2, 2).norm(), 1e-15);
}

TEST(testDPGO, ManualObjectDranPrivateBlockJacobiSchurDiagonalIsLocal) {
  Matrix hxx = Matrix::Zero(4, 4);
  hxx.topLeftCorner(2, 2) =
      (Matrix(2, 2) << 4.0, 0.5, 0.5, 3.0).finished();
  hxx.bottomRightCorner(2, 2) =
      (Matrix(2, 2) << 2.5, -0.2, -0.2, 3.5).finished();
  hxx.topRightCorner(2, 2) =
      (Matrix(2, 2) << 0.3, 0.1, -0.2, 0.4).finished();
  hxx.bottomLeftCorner(2, 2) = hxx.topRightCorner(2, 2).transpose();
  Matrix hxp(4, 4);
  hxp << 0.5, 0.1, 0.0, 0.0,
         -0.2, 0.4, 0.0, 0.0,
          0.0, 0.0, 0.3, -0.1,
          0.0, 0.0, 0.2, 0.6;
  Matrix hpp = 3.0 * Matrix::Identity(4, 4);
  hpp(0, 2) = hpp(2, 0) = 0.25;
  constexpr double damping = 0.03;
  Matrix expected = Matrix::Zero(4, 4);
  const Matrix firstResponse =
      hxx.topLeftCorner(2, 2).ldlt().solve(hxp.topRows(2));
  const Matrix secondResponse =
      hxx.bottomRightCorner(2, 2).ldlt().solve(hxp.bottomRows(2));
  const Matrix approximateSchur =
      hpp - hxp.topRows(2).transpose() * firstResponse -
      hxp.bottomRows(2).transpose() * secondResponse;
  expected.topLeftCorner(2, 2) = approximateSchur.topLeftCorner(2, 2);
  expected.bottomRightCorner(2, 2) =
      approximateSchur.bottomRightCorner(2, 2);
  expected.diagonal().array() += damping;

  const Matrix actual =
      manualObjectDranPrivateBlockJacobiSchurDiagonalFromDenseForTest(
          hxx, hxp, hpp, 2u, 2u, damping);

  EXPECT_TRUE(actual.isApprox(expected, 1e-12));
  EXPECT_LT(actual.topRightCorner(2, 2).norm(), 1e-15);
}

TEST(testDPGO, ManualObjectDranBlockTridiagonalSolveMatchesDenseLdlt) {
  constexpr std::size_t blockSize = 2u;
  Matrix hessian = Matrix::Zero(6, 6);
  hessian.diagonal() << 4.0, 5.0, 6.0, 7.0, 5.0, 4.5;
  const Matrix firstCoupling =
      (Matrix(2, 2) << 0.3, -0.1, 0.2, 0.25).finished();
  const Matrix secondCoupling =
      (Matrix(2, 2) << -0.2, 0.15, 0.1, 0.35).finished();
  hessian.block<2, 2>(2, 0) = firstCoupling;
  hessian.block<2, 2>(0, 2) = firstCoupling.transpose();
  hessian.block<2, 2>(4, 2) = secondCoupling;
  hessian.block<2, 2>(2, 4) = secondCoupling.transpose();
  Matrix rhs(6, 3);
  rhs << 1.0, -0.5, 0.2,
         0.3, 0.7, -0.1,
         -0.4, 0.2, 0.8,
         0.6, -0.9, 0.5,
         0.1, 0.4, -0.7,
         -0.2, 0.3, 0.9;

  const auto actual = manualObjectDranSolveBlockTridiagonalForTest(
      hessian, rhs, blockSize);
  const Matrix expected = hessian.ldlt().solve(rhs);

  ASSERT_TRUE(actual.valid) << actual.status;
  ASSERT_EQ(actual.solution.rows(), expected.rows());
  ASSERT_EQ(actual.solution.cols(), expected.cols());
  EXPECT_LT((actual.solution - expected).norm() /
                std::max(1.0, expected.norm()),
            1e-10);
}

TEST(testDPGO, ManualObjectDranBlockTridiagonalSolveRejectsOffBandBlock) {
  Matrix hessian = 4.0 * Matrix::Identity(6, 6);
  hessian.block<2, 2>(0, 4).setConstant(0.1);
  hessian.block<2, 2>(4, 0).setConstant(0.1);

  const auto actual = manualObjectDranSolveBlockTridiagonalForTest(
      hessian, Matrix::Ones(6, 1), 2u);

  EXPECT_FALSE(actual.valid);
  EXPECT_EQ(actual.status, "off_band_block");
}

TEST(testDPGO, ManualObjectDranBlockTridiagonalBlocksMatchDenseLdlt) {
  const std::vector<Matrix> diagonalBlocks{
      (Matrix(2, 2) << 4.0, 0.2, 0.2, 5.0).finished(),
      (Matrix(2, 2) << 6.0, -0.1, -0.1, 7.0).finished(),
      (Matrix(2, 2) << 5.0, 0.3, 0.3, 4.5).finished(),
  };
  const std::vector<Matrix> lowerBlocks{
      (Matrix(2, 2) << 0.3, -0.1, 0.2, 0.25).finished(),
      (Matrix(2, 2) << -0.2, 0.15, 0.1, 0.35).finished(),
  };
  Matrix hessian = Matrix::Zero(6, 6);
  for (std::size_t block = 0; block < diagonalBlocks.size(); ++block) {
    hessian.block<2, 2>(2 * block, 2 * block) = diagonalBlocks[block];
  }
  for (std::size_t block = 0; block < lowerBlocks.size(); ++block) {
    hessian.block<2, 2>(2 * (block + 1u), 2 * block) = lowerBlocks[block];
    hessian.block<2, 2>(2 * block, 2 * (block + 1u)) =
        lowerBlocks[block].transpose();
  }
  Matrix rhs = Matrix::Random(6, 3);

  const auto actual = manualObjectDranSolveBlockTridiagonalBlocksForTest(
      diagonalBlocks, lowerBlocks, rhs);

  ASSERT_TRUE(actual.valid) << actual.status;
  const Matrix expected = hessian.ldlt().solve(rhs);
  EXPECT_LT((actual.solution - expected).norm() /
                std::max(1.0, expected.norm()),
            1e-10);
}

TEST(testDPGO, ManualObjectDranBlockTridiagonalBlocksRejectInvalidCount) {
  const std::vector<Matrix> diagonalBlocks{
      Matrix::Identity(2, 2), Matrix::Identity(2, 2)};
  const std::vector<Matrix> lowerBlocks;

  const auto actual = manualObjectDranSolveBlockTridiagonalBlocksForTest(
      diagonalBlocks, lowerBlocks, Matrix::Ones(4, 1));

  EXPECT_FALSE(actual.valid);
  EXPECT_EQ(actual.status, "invalid_block_count");
}

TEST(testDPGO, ManualObjectDranSparseCouplingAggregatesAndMatchesDense) {
  constexpr std::size_t blockSize = 2u;
  std::vector<ManualObjectDranSparseCouplingEntry> entries{
      {0u, 0u, (Matrix(2, 2) << 1.0, 0.2, -0.1, 0.5).finished()},
      {0u, 0u, (Matrix(2, 2) << 0.3, -0.4, 0.6, 0.1).finished()},
      {1u, 1u, (Matrix(2, 2) << -0.2, 0.7, 0.4, 0.8).finished()},
      {2u, 0u, (Matrix(2, 2) << 0.9, -0.3, 0.2, 0.6).finished()},
  };
  Matrix dense = Matrix::Zero(6, 4);
  for (const auto &entry : entries) {
    dense.block<2, 2>(2 * entry.privateBlock, 2 * entry.objectBlock) +=
        entry.value;
  }
  Matrix objectValues(4, 3);
  objectValues << 0.2, -0.4, 0.8,
                  0.5, 0.1, -0.2,
                  -0.3, 0.7, 0.6,
                  0.9, -0.5, 0.4;
  Matrix privateValues(6, 3);
  privateValues << 0.1, 0.2, -0.3,
                   0.4, -0.5, 0.6,
                   -0.7, 0.8, 0.9,
                   0.3, -0.2, 0.5,
                   0.6, 0.4, -0.1,
                   -0.8, 0.7, 0.2;

  const auto actual = manualObjectDranSparseCouplingOperationsForTest(
      3u, 2u, blockSize, entries, objectValues, privateValues, 0u);

  ASSERT_TRUE(actual.valid) << actual.status;
  EXPECT_EQ(actual.aggregateBlocks, 3u);
  EXPECT_TRUE(actual.forwardProduct.isApprox(dense * objectValues, 1e-12));
  EXPECT_TRUE(actual.transposeProduct.isApprox(
      dense.transpose() * privateValues, 1e-12));
  EXPECT_TRUE(actual.objectColumns.isApprox(dense.leftCols(2), 1e-12));
}

TEST(testDPGO, ManualObjectDranSparseCouplingRejectsOutOfRangeEntry) {
  const std::vector<ManualObjectDranSparseCouplingEntry> entries{
      {2u, 0u, Matrix::Identity(2, 2)},
  };

  const auto actual = manualObjectDranSparseCouplingOperationsForTest(
      2u, 1u, 2u, entries, Matrix::Ones(2, 1), Matrix::Ones(4, 1), 0u);

  EXPECT_FALSE(actual.valid);
  EXPECT_EQ(actual.status, "coupling_index_out_of_range");
}

TEST(testDPGO, ObjectPose3PcgTerminalPolishOnlyRunsAfterFinalActivation) {
  const auto mode =
      ManualObjectDranObjectInterfaceMode::MatrixFreeObjectPose3Pcg;
  EXPECT_FALSE(manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
      mode, false, 19u, 19u, 1u));
  EXPECT_FALSE(manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
      mode, true, 18u, 19u, 1u));
  EXPECT_FALSE(manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
      mode, true, 19u, 19u, 0u));
  EXPECT_FALSE(manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
      ManualObjectDranObjectInterfaceMode::CopyBaseline, true, 19u, 19u,
      1u));
  EXPECT_TRUE(manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
      mode, true, 19u, 19u, 1u));
}

TEST(testDPGO, ObjectPose3PcgCanReplaceOnlyTheActivatedFinalRound) {
  const auto mode =
      ManualObjectDranObjectInterfaceMode::MatrixFreeObjectPose3Pcg;
  EXPECT_FALSE(manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
      mode, false, 20u, 20u, 1u, true));
  EXPECT_FALSE(manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
      mode, true, 19u, 20u, 1u, true));
  EXPECT_FALSE(manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
      mode, true, 20u, 20u, 0u, true));
  EXPECT_FALSE(manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
      mode, true, 20u, 20u, 1u, false));
  EXPECT_TRUE(manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
      mode, true, 20u, 20u, 1u, true));
}

TEST(testDPGO, ManualObjectDranPoint2OwnerPcgMatchesDenseFirstStepAndLedger) {
  ManualObjectDranPoint2SchurModel first;
  first.valid = true;
  first.status = "ok";
  first.hessian =
      (Matrix(4, 4) << 4.0, 0.2, 0.4, 0.0,
                        0.2, 3.0, 0.1, 0.3,
                        0.4, 0.1, 2.5, 0.2,
                        0.0, 0.3, 0.2, 2.0)
          .finished();
  first.gradient = (Vector(4) << 1.0, -0.5, 0.2, -0.8).finished();
  ManualObjectDranPoint2SchurModel second;
  second.valid = true;
  second.status = "ok";
  second.hessian =
      (Matrix(4, 4) << 1.5, 0.1, -0.2, 0.1,
                        0.1, 2.0, 0.0, -0.1,
                        -0.2, 0.0, 1.8, 0.2,
                        0.1, -0.1, 0.2, 1.2)
          .finished();
  second.gradient = (Vector(4) << -0.3, 0.4, -0.1, 0.2).finished();

  const Matrix hessian = first.hessian + second.hessian;
  const Vector rhs = -(first.gradient + second.gradient);
  Matrix preconditioner = Matrix::Zero(4, 4);
  preconditioner.topLeftCorner(2, 2) = hessian.topLeftCorner(2, 2);
  preconditioner.bottomRightCorner(2, 2) =
      hessian.bottomRightCorner(2, 2);
  const Vector z = preconditioner.ldlt().solve(rhs);
  const double alpha = rhs.dot(z) / z.dot(hessian * z);
  const Vector expected = alpha * z;

  const ManualObjectDranPoint2OwnerPcgSolve solve =
      manualObjectDranSolvePoint2OwnerPcgForTest(
          {first, second}, {{0u, 1u}, {0u, 1u}}, {0u, 1u},
          Vector::Zero(4), 1u);

  ASSERT_TRUE(solve.valid) << solve.status;
  EXPECT_EQ(solve.iterations, 1u);
  EXPECT_LT((solve.pointStep - expected).norm(), 1e-12);
  EXPECT_EQ(solve.gradientPayloadBytes, 32u);
  EXPECT_EQ(solve.directionPayloadBytes, 32u);
  EXPECT_EQ(solve.productPayloadBytes, 32u);
  EXPECT_EQ(solve.updatePayloadBytes, 0u);
  EXPECT_EQ(solve.scalarPayloadBytes, 32u);
  EXPECT_EQ(solve.totalPayloadBytes, 128u);
  EXPECT_EQ(solve.gradientMessages, 2u);
  EXPECT_EQ(solve.directionMessages, 2u);
  EXPECT_EQ(solve.productMessages, 2u);
  EXPECT_EQ(solve.scalarMessages, 4u);
  EXPECT_EQ(solve.updateMessages, 0u);
  EXPECT_EQ(solve.totalMessages, 10u);

  const ManualObjectDranPoint2OwnerPcgSolve warmSolve =
      manualObjectDranSolvePoint2OwnerPcgForTest(
          {first, second}, {{0u, 1u}, {0u, 1u}}, {0u, 1u},
          (Vector(4) << 0.1, -0.2, 0.3, -0.4).finished(), 1u);
  ASSERT_TRUE(warmSolve.valid) << warmSolve.status;
  EXPECT_EQ(warmSolve.gradientPayloadBytes, 32u);
  EXPECT_EQ(warmSolve.directionPayloadBytes, 32u);
  EXPECT_EQ(warmSolve.productPayloadBytes, 64u);
  EXPECT_EQ(warmSolve.updatePayloadBytes, 0u);
  EXPECT_EQ(warmSolve.scalarPayloadBytes, 32u);
  EXPECT_EQ(warmSolve.totalPayloadBytes, 160u);
  EXPECT_EQ(warmSolve.gradientMessages, 2u);
  EXPECT_EQ(warmSolve.directionMessages, 2u);
  EXPECT_EQ(warmSolve.productMessages, 4u);
  EXPECT_EQ(warmSolve.scalarMessages, 4u);
  EXPECT_EQ(warmSolve.updateMessages, 0u);
  EXPECT_EQ(warmSolve.totalMessages, 12u);

  const Vector warmStart =
      (Vector(4) << 0.1, -0.2, 0.3, -0.4).finished();
  const double expectedWarmScale = std::clamp(
      rhs.dot(warmStart) / warmStart.dot(hessian * warmStart), 0.0, 1.0);
  const ManualObjectDranPoint2OwnerPcgSolve optimizedWarmSolve =
      manualObjectDranSolvePoint2OwnerPcgForTest(
          {first, second}, {{0u, 1u}, {0u, 1u}}, {0u, 1u}, warmStart,
          1u, 1e-10, true);
  ASSERT_TRUE(optimizedWarmSolve.valid) << optimizedWarmSolve.status;
  EXPECT_NEAR(optimizedWarmSolve.warmStartScale, expectedWarmScale, 1e-12);
  EXPECT_EQ(optimizedWarmSolve.totalPayloadBytes, 160u);
}

TEST(testDPGO, ManualObjectDranRoutedOwnerPcgMatchesDirectNumerically) {
  std::vector<ManualObjectDranPoint2SchurModel> models(3);
  for (std::size_t robot = 0; robot < models.size(); ++robot) {
    models[robot].valid = true;
    models[robot].status = "ok";
    models[robot].hessian = Matrix::Zero(2, 2);
    models[robot].hessian(0, 0) = 2.0 + robot;
    models[robot].hessian(1, 1) = 3.0 + 0.5 * robot;
    models[robot].hessian(0, 1) = models[robot].hessian(1, 0) = 0.1;
    models[robot].gradient =
        (Vector(2) << 0.25 * (robot + 1), -0.2 * (robot + 1)).finished();
  }
  const std::vector<std::vector<std::size_t>> observers{
      {0u, 2u}, {0u, 2u}};
  const std::vector<std::size_t> owners{0u, 0u};
  const Vector warmStart = Vector::Zero(2);
  const auto direct = manualObjectDranSolveInterfaceOwnerPcgForTest(
      models, observers, owners, 1u, warmStart, 3u);
  const auto routed = manualObjectDranSolveInterfaceOwnerPcgRoutedForTest(
      models, observers, owners, 1u, warmStart, 3u,
      {{0u, 1u, 2u}, {1u, 0u, 1u}, {2u, 1u, 0u}});

  ASSERT_TRUE(direct.valid) << direct.status;
  ASSERT_TRUE(routed.valid) << routed.status;
  EXPECT_LT((direct.pointStep - routed.pointStep).norm(), 1e-12);
  EXPECT_NEAR(direct.relativeResidual, routed.relativeResidual, 1e-15);
  EXPECT_EQ(routed.gradientPayloadBytes, 2u * direct.gradientPayloadBytes);
  EXPECT_EQ(routed.directionPayloadBytes,
            2u * direct.directionPayloadBytes);
  EXPECT_EQ(routed.productPayloadBytes, 2u * direct.productPayloadBytes);
  EXPECT_EQ(routed.scalarPayloadBytes, direct.scalarPayloadBytes);
  EXPECT_EQ(routed.gradientMessages, 2u * direct.gradientMessages);
  EXPECT_EQ(routed.directionMessages, 2u * direct.directionMessages);
  EXPECT_EQ(routed.productMessages, 2u * direct.productMessages);
  EXPECT_EQ(routed.scalarMessages, direct.scalarMessages);
}

TEST(testDPGO, ManualObjectDranRoutedOwnerPcgRejectsDisconnectedObserver) {
  ManualObjectDranPoint2SchurModel model;
  model.valid = true;
  model.status = "ok";
  model.hessian = Matrix::Identity(1, 1);
  model.gradient = Vector::Ones(1);
  const auto solve = manualObjectDranSolveInterfaceOwnerPcgRoutedForTest(
      {model, model}, {{0u, 1u}}, {0u}, 1u, Vector::Zero(1), 1u,
      {{0u, std::numeric_limits<std::size_t>::max()},
       {std::numeric_limits<std::size_t>::max(), 0u}});

  EXPECT_FALSE(solve.valid);
  EXPECT_EQ(solve.status, "Point2 owner-PCG observer is unreachable");
}

TEST(testDPGO, ManualObjectDranLocalIterationScheduleUsesBoundedWarmup) {
  ManualObjectDranOptions options;
  options.localTrustRegionIterations = 3;
  options.localInitialTrustRegionIterations = 5;
  options.localInitialTrustRegionRounds = 3;

  EXPECT_EQ(manualObjectDranLocalIterationsForOuterRound(options, 0), 3u);
  EXPECT_EQ(manualObjectDranLocalIterationsForOuterRound(options, 1), 5u);
  EXPECT_EQ(manualObjectDranLocalIterationsForOuterRound(options, 3), 5u);
  EXPECT_EQ(manualObjectDranLocalIterationsForOuterRound(options, 4), 3u);

  options.localInitialTrustRegionIterations = 0;
  options.localInitialTrustRegionRounds = 0;
  EXPECT_EQ(manualObjectDranLocalIterationsForOuterRound(options, 1), 3u);

  options.localInitialTrustRegionIterations = 5;
  EXPECT_THROW(manualObjectDranLocalIterationsForOuterRound(options, 1),
               std::invalid_argument);
}

TEST(testDPGO, ManualObjectDranRobotGraphColoringColorsFourCycleWithTwoColors) {
  const std::vector<std::vector<std::size_t>> adjacency{
      {1, 3},
      {0, 2},
      {1, 3},
      {0, 2},
  };

  const std::vector<std::size_t> colors =
      colorManualObjectDranRobotGraph(adjacency);

  EXPECT_EQ(colors, (std::vector<std::size_t>{0, 1, 0, 1}));
  EXPECT_EQ(*std::max_element(colors.begin(), colors.end()) + 1u, 2u);
  expectProperColoring(adjacency, colors);
}

TEST(testDPGO, ManualObjectDranRobotGraphColoringColorsTriangleWithThreeColors) {
  const std::vector<std::vector<std::size_t>> adjacency{
      {1, 2},
      {0, 2},
      {0, 1},
  };

  const std::vector<std::size_t> colors =
      colorManualObjectDranRobotGraph(adjacency);

  EXPECT_EQ(colors, (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_EQ(*std::max_element(colors.begin(), colors.end()) + 1u, 3u);
  expectProperColoring(adjacency, colors);
}

TEST(testDPGO, ManualObjectDranRobotGraphColoringSymmetrizesOneSidedEdges) {
  const std::vector<std::vector<std::size_t>> adjacency{
      {1},
      {},
      {1},
  };

  const std::vector<std::size_t> colors =
      colorManualObjectDranRobotGraph(adjacency);

  EXPECT_EQ(colors, (std::vector<std::size_t>{0, 1, 0}));
  EXPECT_NE(colors[0], colors[1]);
  EXPECT_NE(colors[2], colors[1]);
}

TEST(testDPGO, ManualObjectDranRobotGraphColoringRejectsMalformedAdjacency) {
  EXPECT_THROW(colorManualObjectDranRobotGraph({{0}}), std::invalid_argument);
  EXPECT_THROW(colorManualObjectDranRobotGraph({{2}, {}}),
               std::invalid_argument);
}

TEST(testDPGO, ManualObjectDranUnavailableDescriptorAddsNoCost) {
  const Vector unavailable = Vector::Zero(4);
  EXPECT_DOUBLE_EQ(manualObjectDranDescriptorAssociationCostForTest(
                       unavailable, unavailable, 2.0),
                   0.0);

  Vector first = Vector::Zero(4);
  Vector second = Vector::Zero(4);
  first(0) = 1.0;
  second(1) = 1.0;
  EXPECT_DOUBLE_EQ(manualObjectDranDescriptorAssociationCostForTest(
                       first, second, 2.0),
                   2.0);
}

TEST(testDPGO,
     ManualObjectDranGraphSignatureSupportIsRotoTranslationInvariant) {
  constexpr unsigned d = 2;
  constexpr std::size_t numObjects = 5;
  ObjectPGODirectoryData dataset;
  dataset.dimension = d;
  dataset.numRobots = 2;
  dataset.numObjects = numObjects;
  dataset.robots.resize(2);

  Matrix local = Matrix::Zero(d, numObjects * (d + 1));
  Matrix target = Matrix::Zero(d, numObjects * (d + 1));
  const std::vector<Vector> points{
      (Vector(2) << 0.0, 0.0).finished(),
      (Vector(2) << 1.1, 0.2).finished(),
      (Vector(2) << 0.3, 2.4).finished(),
      (Vector(2) << 3.7, 1.3).finished(),
      (Vector(2) << 2.2, 4.8).finished()};
  const std::vector<std::size_t> permutation{3, 0, 4, 1, 2};
  const double angle = 0.71;
  Matrix rotation(2, 2);
  rotation << std::cos(angle), -std::sin(angle),
      std::sin(angle), std::cos(angle);
  const Vector translation = (Vector(2) << 8.0, -3.5).finished();
  for (std::size_t object = 0; object < numObjects; ++object) {
    const int localCol = static_cast<int>(object * (d + 1));
    const int targetCol =
        static_cast<int>(permutation[object] * (d + 1));
    local.block(0, localCol, d, d).setIdentity();
    local.col(localCol + d) = points[object];
    target.block(0, targetCol, d, d) = rotation;
    target.col(targetCol + d) = rotation * points[object] + translation;
  }

  ManualObjectDranOptions options;
  options.associationRequireObservedObjects = false;
  options.associationGeometryScale = 1.0;
  const Matrix costs =
      manualObjectDranGraphSignatureAssociationCostsForTest(
          dataset, local, target, 0, 1, d, options);

  ASSERT_EQ(costs.rows(), static_cast<int>(numObjects));
  ASSERT_EQ(costs.cols(), static_cast<int>(numObjects));
  for (std::size_t object = 0; object < numObjects; ++object) {
    Eigen::Index bestColumn = -1;
    costs.row(static_cast<int>(object)).minCoeff(&bestColumn);
    EXPECT_EQ(bestColumn,
              static_cast<Eigen::Index>(permutation[object]));
    EXPECT_NEAR(costs(static_cast<int>(object), bestColumn), 0.0, 1e-12);
  }
}

TEST(testDPGO,
     ManualObjectDranObservationIsolatedUpperStateIgnoresCopyDeformation) {
  ObjectPGODirectoryData dataset;
  dataset.dimension = 2;
  dataset.numRobots = 1;
  dataset.numObjects = 1;
  dataset.robots.resize(1);
  ObjectPGORobotData &robot = dataset.robots[0];
  robot.robotId = 0;
  robot.trajectoryVertexIds = {0};

  ObjectObservationMeasurement observation;
  observation.robotId = 0;
  observation.trajectoryVertexId = 0;
  observation.trajectoryLocalIndex = 0;
  observation.objectGlobalVertexId = 0;
  observation.measurement = RelativeSEMeasurement(
      0, 0, 0, 0, Matrix::Identity(2, 2),
      (Vector(2) << 2.0, 3.0).finished(), 1.0, 1.0);
  robot.objectObservationMeasurements.push_back(observation);

  Matrix state = Matrix::Zero(2, 6);
  state.block(0, 0, 2, 2) = Matrix::Identity(2, 2);
  state.block(0, 3, 2, 2) = Matrix::Identity(2, 2);
  state.col(2) = (Vector(2) << 1.0, -1.0).finished();
  state.col(5) = (Vector(2) << 99.0, -40.0).finished();

  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  const Matrix isolated = manualObjectDranObservationIsolatedUpperStateForTest(
      dataset, state, 0, 2, options);

  EXPECT_TRUE(isolated.leftCols(3).isApprox(state.leftCols(3), 1e-12));
  EXPECT_TRUE(isolated.block(0, 3, 2, 2).isApprox(Matrix::Identity(2, 2),
                                                  1e-12));
  EXPECT_TRUE(isolated.col(5).isApprox(
      (Vector(2) << 3.0, 2.0).finished(), 1e-12));
  const Matrix copyValidationPose =
      manualObjectDranObjectCopyValidationPoseForTest(
          dataset, state, 0, 0, 2);
  ASSERT_EQ(copyValidationPose.rows(), 2);
  ASSERT_EQ(copyValidationPose.cols(), 3);
  EXPECT_TRUE(copyValidationPose.col(2).isApprox(
      (Vector(2) << 3.0, 2.0).finished(), 1e-12));

  options.associationPolicy = ManualObjectDranAssociationPolicy::Manual;
  const Matrix manualState =
      manualObjectDranObservationIsolatedUpperStateForTest(dataset, state, 0,
                                                            2, options);
  EXPECT_TRUE(manualState.isApprox(state, 1e-12));
}

TEST(testDPGO, ManualObjectDranAlignmentRmseUsesPointRowsAsObjectIndices) {
  Matrix sourcePoints = Matrix::Zero(4, 2);
  Matrix targetPoints = Matrix::Zero(4, 2);
  sourcePoints.row(0) << 0.0, 0.0;
  sourcePoints.row(1) << 1.0, 0.0;
  sourcePoints.row(2) << 2.0, 0.0;
  sourcePoints.row(3) << 3.0, 0.0;
  targetPoints = sourcePoints;
  targetPoints.row(3) << 7.0, 0.0;
  const Matrix assignment = Matrix::Identity(4, 4);

  const double rmse = manualObjectDranWeightedAlignmentRmseForTest(
      sourcePoints, targetPoints, assignment, Matrix::Identity(2, 2),
      Vector::Zero(2));

  EXPECT_NEAR(rmse, 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(
      manualObjectDranMarginalTranslationRmseForTest(0.2, 1.5), 1.5);
  EXPECT_NEAR(
      manualObjectDranPredictiveTranslationVarianceForTest(0.2, 1.5, 0.3),
      2.25, 1e-12);
}

TEST(testDPGO,
     ManualObjectDranRobustPoseAlignmentContinuouslySuppressesAliasResidual) {
  std::vector<Matrix> targetPoses(3, Matrix::Zero(2, 3));
  targetPoses[0].leftCols(2) = Matrix::Identity(2, 2);
  targetPoses[1].leftCols(2) = Matrix::Identity(2, 2);
  targetPoses[2].leftCols(2) = Matrix::Identity(2, 2);
  targetPoses[0].col(2) = (Vector(2) << 0.0, 0.0).finished();
  targetPoses[1].col(2) = (Vector(2) << 1.0, 0.0).finished();
  targetPoses[2].col(2) = (Vector(2) << 0.0, 1.0).finished();
  const Vector expectedTranslation =
      (Vector(2) << 5.0, 2.0).finished();
  std::vector<Matrix> sourcePoses = targetPoses;
  for (Matrix &pose : sourcePoses) {
    pose.col(2) += expectedTranslation;
  }
  const std::vector<SparsePoseAlignmentWeightEntry> entries{
      {0, 0, 1.0, 0.0, 1.0},
      {1, 1, 1.0, 0.0, 1.0},
      {2, 2, 1.0, 0.0, 1.0},
      {0, 2, 1.0, 0.0, 1.0},
  };

  const AssociationRotoTranslationEstimate estimate =
      manualObjectDranRobustSparsePoseAlignmentForTest(
          sourcePoses, targetPoses, entries, Matrix::Identity(2, 2),
          expectedTranslation);

  EXPECT_LT((estimate.rotation - Matrix::Identity(2, 2)).norm(), 1e-8);
  EXPECT_LT((estimate.translation - expectedTranslation).norm(), 1e-8);
  EXPECT_LT(manualObjectDranRobustSparsePoseAlignmentRmseForTest(
                sourcePoses, targetPoses, entries,
                Matrix::Identity(2, 2), expectedTranslation),
            1e-8);
}

TEST(testDPGO,
     ManualObjectDranAssociationAwareTedCciUsesPermutedSoftAssociation) {
  const std::filesystem::path dataDir =
      std::filesystem::temp_directory_path() /
      "manual_object_dran_association_aware_ted_cci";
  std::filesystem::remove_all(dataDir);
  std::filesystem::create_directories(dataDir);
  const std::string information =
      "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100";
  {
    std::ofstream out(dataDir / "rob_0.g2o");
    out << "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 2 2 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 3 4 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " << information << "\n";
    out << "EDGE_SE3:QUAT 0 2 2 0 0 0 0 0 1 " << information << "\n";
    out << "EDGE_SE3:QUAT 1 3 3 0 0 0 0 0 1 " << information << "\n";
  }
  {
    std::ofstream out(dataDir / "rob_1.g2o");
    out << "VERTEX_SE3:QUAT 0 10 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 1 11 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 2 14 0 0 0 0 0 1\n";
    out << "VERTEX_SE3:QUAT 3 12 0 0 0 0 0 1\n";
    out << "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " << information << "\n";
    out << "EDGE_SE3:QUAT 0 2 4 0 0 0 0 0 1 " << information << "\n";
    out << "EDGE_SE3:QUAT 1 3 1 0 0 0 0 0 1 " << information << "\n";
  }
  const std::filesystem::path associationFile = dataDir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,1,1\n";
    out << "0,1,1,0,1\n";
    out << "1,0,0,1,1\n";
    out << "1,0,1,0,1\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = 2;
  options.numObjects = 2;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationRequireObservedObjects = true;
  options.associationResidualUseTransformAlignment = false;
  options.associationGaugeSynchronizationEnabled = false;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::Disabled;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult baseline =
      runManualObjectDran(dataDir.string(), options);
  options.associationAwareTedCciInitialization = true;
  const ManualObjectDranRunResult initialized =
      runManualObjectDran(dataDir.string(), options);
  options.associationAwareTedCciInitialization = false;
  options.maxIterations = 1;
  const ManualObjectDranRunResult jacobi =
      runManualObjectDran(dataDir.string(), options);
  options.associationLowerPhysicalCopyMajorization = true;
  const ManualObjectDranRunResult majorized =
      runManualObjectDran(dataDir.string(), options);
  std::filesystem::remove_all(dataDir);

  ASSERT_EQ(baseline.finalEstimates.size(), 2u);
  ASSERT_EQ(initialized.finalEstimates.size(), 2u);
  const Vector baselineFirst = baseline.finalEstimates[0].block(0, 11, 3, 1);
  const Vector baselineSecond = baseline.finalEstimates[1].block(0, 15, 3, 1);
  const Vector initializedFirst =
      initialized.finalEstimates[0].block(0, 11, 3, 1);
  const Vector initializedSecond =
      initialized.finalEstimates[1].block(0, 15, 3, 1);
  EXPECT_GT((baselineFirst - baselineSecond).norm(), 5.0);
  EXPECT_LT((initializedFirst - initializedSecond).norm(), 1e-6);
  EXPECT_TRUE(initialized.associationAwareTedCciInitializationApplied);
  EXPECT_GT(initialized.associationAwareTedCciActiveVertices, 0u);
  EXPECT_EQ(initialized.associationAwareTedCciCrossMeasurements, 2u);
  const Vector jacobiFirst = jacobi.finalEstimates[0].block(0, 11, 3, 1);
  const Vector jacobiSecond = jacobi.finalEstimates[1].block(0, 15, 3, 1);
  const Vector majorizedFirst =
      majorized.finalEstimates[0].block(0, 11, 3, 1);
  const Vector majorizedSecond =
      majorized.finalEstimates[1].block(0, 15, 3, 1);
  EXPECT_LT((majorizedFirst - majorizedSecond).norm(),
            (jacobiFirst - jacobiSecond).norm());
}

TEST(testDPGO, ManualObjectDranSharedEdgeJointDbaV2PreservesJointInvariants) {
  ManualObjectDranDirectedAssociationWeights weights(
      3, std::vector<Matrix>(3));
  weights[0][1] = (Matrix(2, 3) << 0.7, 0.2, 0.0, 0.1, 0.6, 0.0).finished();
  weights[1][0] = weights[0][1].transpose();
  weights[0][2] = (Matrix(2, 2) << 0.1, 0.2, 0.3, 0.4).finished();
  const Matrix unrelatedBefore = weights[0][2];

  const double inf = std::numeric_limits<double>::infinity();
  ManualObjectDranDirectedAssociationWeights objectives(
      3, std::vector<Matrix>(3));
  objectives[0][1] = (Matrix(2, 3) << 0.1, 0.9, inf, 0.8, 0.2, inf).finished();
  objectives[1][0] = (Matrix(3, 2) << 0.3, 0.6, 0.7, 0.4, inf, inf).finished();

  std::vector<std::vector<bool>> edges(3, std::vector<bool>(3, false));
  edges[0][1] = true;
  edges[1][0] = true;
  PartialSinkhornOptions options;
  options.iterations = 40;
  options.rowCapacity = 1.0;
  options.columnCapacity = 1.0;
  const ManualObjectDranDirectedAssociationWeights previousWeights = weights;

  const ManualObjectDranSharedEdgeSyncDiagnostics diagnostics =
      synchronizeManualObjectDranSharedEdgeJointWeights(
          weights, previousWeights, objectives, edges, true, options);

  EXPECT_EQ(diagnostics.synchronizedEdgeCount, 1u);
  EXPECT_GT(diagnostics.preProjectionDisagreementNorm, 0.0);
  EXPECT_GT(diagnostics.projectionDisplacementNorm, 0.0);
  EXPECT_LT((weights[0][1] - weights[1][0].transpose()).norm(), 1e-12);
  EXPECT_LE(weights[0][1](0, 2), 1e-12);
  EXPECT_LE(weights[0][1](1, 2), 1e-12);
  EXPECT_LE(weights[0][1].rowwise().sum().maxCoeff(), 1.0 + 1e-12);
  EXPECT_LE(weights[0][1].colwise().sum().maxCoeff(), 1.0 + 1e-12);
  EXPECT_EQ(weights[0][2], unrelatedBefore);

  ManualObjectDranDirectedAssociationWeights swappedWeights(
      2, std::vector<Matrix>(2));
  swappedWeights[0][1] = (Matrix(3, 2) << 0.7, 0.1, 0.2, 0.6, 0.0, 0.0).finished();
  swappedWeights[1][0] = swappedWeights[0][1].transpose();
  ManualObjectDranDirectedAssociationWeights swappedObjectives(
      2, std::vector<Matrix>(2));
  swappedObjectives[0][1] = objectives[1][0];
  swappedObjectives[1][0] = objectives[0][1];
  std::vector<std::vector<bool>> swappedEdges(2, std::vector<bool>(2, false));
  swappedEdges[0][1] = true;
  swappedEdges[1][0] = true;
  const ManualObjectDranDirectedAssociationWeights swappedPreviousWeights =
      swappedWeights;
  synchronizeManualObjectDranSharedEdgeJointWeights(
      swappedWeights, swappedPreviousWeights, swappedObjectives, swappedEdges,
      true, options);
  EXPECT_LT((weights[0][1] - swappedWeights[0][1].transpose()).norm(),
            1e-12);
}

TEST(testDPGO,
     ManualObjectDranFeedbackUotUsesOnlyEndpointsAndFreshSurrogateSolve) {
  const double inf = std::numeric_limits<double>::infinity();
  ManualObjectDranDirectedAssociationWeights baseWeights(
      2, std::vector<Matrix>(2));
  baseWeights[0][1] =
      (Matrix(2, 2) << 0.45, 0.05, 0.05, 0.45).finished();
  baseWeights[1][0] = baseWeights[0][1].transpose();
  ManualObjectDranDirectedAssociationWeights baseObjectives(
      2, std::vector<Matrix>(2));
  baseObjectives[0][1] =
      (Matrix(2, 2) << 0.1, 1.5, 1.4, 0.2).finished();
  baseObjectives[1][0] =
      (Matrix(2, 2) << 0.2, 1.6, 1.3, 0.3).finished();
  std::vector<std::vector<bool>> baseEdges(
      2, std::vector<bool>(2, false));
  baseEdges[0][1] = true;
  baseEdges[1][0] = true;
  std::vector<double> baseReadiness{0.6, 0.8};
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> baseState(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  PartialSinkhornOptions options;
  options.iterations = 80;
  options.temperature = 0.2;
  options.unmatchedCost = 1.0;

  ManualObjectDranDirectedAssociationWeights twoRobotWeights = baseWeights;
  const ManualObjectDranSharedEdgeSyncDiagnostics first =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          twoRobotWeights, baseObjectives, baseEdges, baseReadiness, baseState,
          options);
  const Matrix firstShared = twoRobotWeights[0][1];
  const ManualObjectDranSharedEdgeSyncDiagnostics second =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          twoRobotWeights, baseObjectives, baseEdges, baseReadiness, baseState,
          options);

  EXPECT_EQ(first.synchronizedEdgeCount, 1u);
  EXPECT_EQ(first.sparseResponseCandidateCount, first.candidateCount);
  EXPECT_EQ(second.sparseResponseCandidateCount, second.candidateCount);
  EXPECT_LT((twoRobotWeights[0][1] -
             twoRobotWeights[1][0].transpose())
                .norm(),
            1e-12);
  // The first shared response initializes the command. Subsequent responses
  // use endpoint-local lower readiness as the fixed-point step.
  EXPECT_DOUBLE_EQ(first.mannStepMean, 1.0);
  EXPECT_NEAR(second.mannStepMean, std::sqrt(0.6 * 0.8), 1e-12);
  EXPECT_DOUBLE_EQ(baseState[0][1].lastMannStep, second.mannStepMean);
  EXPECT_DOUBLE_EQ(baseState[1][0].lastMannStep, second.mannStepMean);
  EXPECT_LT(second.upperFixedPointResidualNorm,
            first.upperFixedPointResidualNorm);
  EXPECT_GT(twoRobotWeights[0][1].sum(), 0.5 * firstShared.sum());

  ManualObjectDranDirectedAssociationWeights threeRobotWeights(
      3, std::vector<Matrix>(3));
  threeRobotWeights[0][1] = baseWeights[0][1];
  threeRobotWeights[1][0] = baseWeights[1][0];
  threeRobotWeights[0][2] =
      (Matrix(2, 2) << 0.2, 0.1, 0.0, 0.3).finished();
  const Matrix unrelatedBefore = threeRobotWeights[0][2];
  ManualObjectDranDirectedAssociationWeights threeRobotObjectives(
      3, std::vector<Matrix>(3));
  threeRobotObjectives[0][1] = baseObjectives[0][1];
  threeRobotObjectives[1][0] = baseObjectives[1][0];
  threeRobotObjectives[0][2] = Matrix::Constant(2, 2, inf);
  std::vector<std::vector<bool>> threeRobotEdges(
      3, std::vector<bool>(3, false));
  threeRobotEdges[0][1] = true;
  threeRobotEdges[1][0] = true;
  std::vector<double> threeRobotReadiness{0.6, 0.8, 0.01};
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> threeRobotState(
      3, std::vector<ManualObjectDranEdgeFeedbackState>(3));
  synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
      threeRobotWeights, threeRobotObjectives, threeRobotEdges,
      threeRobotReadiness, threeRobotState, options);
  EXPECT_LT((threeRobotWeights[0][1] - firstShared).norm(), 1e-12);
  EXPECT_EQ(threeRobotWeights[0][2], unrelatedBefore);
}

TEST(testDPGO,
     ManualObjectDranSparseFeedbackUotMatchesDenseEndpointObjectives) {
  const double inf = std::numeric_limits<double>::infinity();
  ManualObjectDranDirectedAssociationWeights denseWeights(
      2, std::vector<Matrix>(2));
  denseWeights[0][1] =
      (Matrix(2, 3) << 0.4, 0.1, 0.0, 0.0, 0.3, 0.2).finished();
  denseWeights[1][0] = denseWeights[0][1].transpose();
  ManualObjectDranDirectedAssociationWeights sparseWeights = denseWeights;

  ManualObjectDranDirectedAssociationWeights denseObjectives(
      2, std::vector<Matrix>(2));
  denseObjectives[0][1] =
      (Matrix(2, 3) << 0.1, 0.8, inf, inf, 0.2, 0.7).finished();
  denseObjectives[1][0] =
      (Matrix(3, 2) << 0.3, inf, 0.9, 0.4, inf, 0.5).finished();

  ManualObjectDranDirectedSparseAssociationObjectives sparseObjectives(
      2, std::vector<ManualObjectDranSparseAssociationObjective>(2));
  for (std::size_t i = 0; i < denseObjectives.size(); ++i) {
    for (std::size_t j = 0; j < denseObjectives[i].size(); ++j) {
      const Matrix &dense = denseObjectives[i][j];
      ManualObjectDranSparseAssociationObjective &sparse =
          sparseObjectives[i][j];
      sparse.rows = dense.rows();
      sparse.cols = dense.cols();
      for (int row = 0; row < dense.rows(); ++row) {
        for (int col = 0; col < dense.cols(); ++col) {
          if (std::isfinite(dense(row, col))) {
            sparse.entries.push_back({row, col, dense(row, col)});
          }
        }
      }
    }
  }

  std::vector<std::vector<bool>> edges(2, std::vector<bool>(2, false));
  edges[0][1] = true;
  edges[1][0] = true;
  const std::vector<double> readiness{0.7, 0.9};
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> denseState(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> sparseState(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  PartialSinkhornOptions options;
  options.iterations = 80;
  options.temperature = 0.25;
  options.unmatchedCost = 1.0;

  const ManualObjectDranSharedEdgeSyncDiagnostics denseDiagnostics =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          denseWeights, denseObjectives, edges, readiness, denseState,
          options);
  const ManualObjectDranSharedEdgeSyncDiagnostics sparseDiagnostics =
      synchronizeManualObjectDranDecentralizedFeedbackUotSparseWeights(
          sparseWeights, sparseObjectives, edges, readiness, sparseState,
          options);

  EXPECT_LT((denseWeights[0][1] - sparseWeights[0][1]).norm(), 1e-12);
  EXPECT_EQ(sparseDiagnostics.candidateCount,
            denseDiagnostics.candidateCount);
  EXPECT_EQ(sparseDiagnostics.sparseResponseCandidateCount,
            denseDiagnostics.sparseResponseCandidateCount);
  EXPECT_NEAR(sparseDiagnostics.preProjectionDisagreementNorm,
              denseDiagnostics.preProjectionDisagreementNorm, 1e-12);
  EXPECT_NEAR(sparseDiagnostics.upperFixedPointResidualNorm,
              denseDiagnostics.upperFixedPointResidualNorm, 1e-12);
}

TEST(testDPGO,
     ManualObjectDranSparseFeedbackUotEvaluatesEachUndirectedEdgeOnce) {
  constexpr std::size_t numRobots = 3;
  ManualObjectDranDirectedAssociationWeights weights(
      numRobots, std::vector<Matrix>(numRobots));
  ManualObjectDranDirectedSparseAssociationObjectives objectives(
      numRobots,
      std::vector<ManualObjectDranSparseAssociationObjective>(numRobots));
  std::vector<std::vector<bool>> edges(
      numRobots, std::vector<bool>(numRobots, false));
  for (std::size_t i = 0; i < numRobots; ++i) {
    for (std::size_t j = i + 1; j < numRobots; ++j) {
      weights[i][j] = Matrix::Constant(1, 1, 0.4);
      weights[j][i] = weights[i][j].transpose();
      objectives[i][j].rows = 1;
      objectives[i][j].cols = 1;
      objectives[i][j].entries.push_back({0, 0, 0.1 + 0.01 * i});
      objectives[j][i].rows = 1;
      objectives[j][i].cols = 1;
      objectives[j][i].entries.push_back({0, 0, 0.2 + 0.01 * j});
      edges[i][j] = true;
      edges[j][i] = true;
    }
  }
  const std::vector<double> readiness(numRobots, 0.8);
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> feedbackStates(
      numRobots,
      std::vector<ManualObjectDranEdgeFeedbackState>(numRobots));
  PartialSinkhornOptions options;
  options.iterations = 20;
  options.temperature = 0.25;
  options.unmatchedCost = 1.0;

  const ManualObjectDranSharedEdgeSyncDiagnostics diagnostics =
      synchronizeManualObjectDranDecentralizedFeedbackUotSparseWeights(
          weights, objectives, edges, readiness, feedbackStates, options);

  EXPECT_EQ(diagnostics.edgeResponseTaskCount, 3u);
  EXPECT_EQ(diagnostics.synchronizedEdgeCount, 3u);
  for (std::size_t i = 0; i < numRobots; ++i) {
    for (std::size_t j = i + 1; j < numRobots; ++j) {
      EXPECT_LT((weights[i][j] - weights[j][i].transpose()).norm(), 1e-15);
    }
  }
}

TEST(testDPGO,
     ManualObjectDranSparseFeedbackUotReportsNonGroundTruthResponseTrace) {
  ManualObjectDranDirectedAssociationWeights weights(
      2, std::vector<Matrix>(2));
  weights[0][1] = (Matrix(1, 2) << 0.4, 0.0).finished();
  weights[1][0] = weights[0][1].transpose();
  ManualObjectDranDirectedSparseAssociationObjectives objectives(
      2, std::vector<ManualObjectDranSparseAssociationObjective>(2));
  objectives[0][1].rows = 1;
  objectives[0][1].cols = 2;
  objectives[0][1].entries = {{0, 0, 0.0}, {0, 1, 3.0}};
  objectives[1][0].rows = 2;
  objectives[1][0].cols = 1;
  objectives[1][0].entries = {{0, 0, 0.0}, {1, 0, 3.0}};
  std::vector<std::vector<bool>> edges(2, std::vector<bool>(2, false));
  edges[0][1] = true;
  edges[1][0] = true;
  const std::vector<double> readiness{1.0, 1.0};
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> feedbackStates(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  PartialSinkhornOptions options;
  options.iterations = 30;
  options.temperature = 1.0;
  options.unmatchedCost = 2.0;
  std::vector<ManualObjectDranAssociationUpperResponseDiagnostic> trace;

  synchronizeManualObjectDranDecentralizedFeedbackUotSparseWeights(
      weights, objectives, edges, readiness, feedbackStates, options, &trace);

  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].firstRobot, 0u);
  EXPECT_EQ(trace[0].secondRobot, 1u);
  EXPECT_EQ(trace[0].row, 0u);
  EXPECT_EQ(trace[0].col, 0u);
  EXPECT_DOUBLE_EQ(trace[0].currentCost, 0.0);
  EXPECT_DOUBLE_EQ(trace[0].rawScore, 2.4);
  EXPECT_DOUBLE_EQ(trace[0].responseWeight, 1.0);
  EXPECT_DOUBLE_EQ(trace[0].proposalWeight, 1.0);
  EXPECT_EQ(trace[1].col, 1u);
  EXPECT_DOUBLE_EQ(trace[1].currentCost, 3.0);
  EXPECT_DOUBLE_EQ(trace[1].rawScore, 0.0);
  EXPECT_DOUBLE_EQ(trace[1].responseWeight, 0.0);
  EXPECT_DOUBLE_EQ(trace[1].proposalWeight, 0.0);

  objectives[0][1].entries[0].cost = 100.0;
  objectives[1][0].entries[0].cost = 100.0;
  trace.clear();
  synchronizeManualObjectDranDecentralizedFeedbackUotSparseWeights(
      weights, objectives, edges, readiness, feedbackStates, options, &trace);

  ASSERT_EQ(trace.size(), 2u);
  EXPECT_DOUBLE_EQ(trace[0].fixedCost, 0.0);
  EXPECT_DOUBLE_EQ(trace[0].currentCost, 100.0);
  EXPECT_NEAR(trace[0].feedbackCost, 200.0 / 102.0, 1e-12);
  EXPECT_LT(trace[0].feedbackCost, options.unmatchedCost);
  EXPECT_DOUBLE_EQ(trace[0].proposalWeight, 1.0);
}

TEST(testDPGO,
     ManualObjectDranFeedbackUotUsesTrueCappedSimplexProjection) {
  ManualObjectDranDirectedAssociationWeights weights(
      2, std::vector<Matrix>(2));
  weights[0][1] = Matrix::Zero(1, 2);
  weights[1][0] = Matrix::Zero(2, 1);
  ManualObjectDranDirectedAssociationWeights objectives(
      2, std::vector<Matrix>(2));
  objectives[0][1] = (Matrix(1, 2) << 0.0, 1.5).finished();
  objectives[1][0] = objectives[0][1].transpose();
  std::vector<std::vector<bool>> edges(
      2, std::vector<bool>(2, false));
  edges[0][1] = true;
  edges[1][0] = true;
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> feedbackStates(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  PartialSinkhornOptions options;
  options.iterations = 30;
  options.stepSize = 1.0;
  options.temperature = 1.0;
  options.unmatchedCost = 2.0;

  synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
      weights, objectives, edges, std::vector<double>{0.0, 0.0},
      feedbackStates, options);

  EXPECT_NEAR(weights[0][1](0, 0), 1.0, 1e-12);
  EXPECT_NEAR(weights[0][1](0, 1), 0.0, 1e-12);
  EXPECT_LT((weights[0][1] - weights[1][0].transpose()).norm(), 1e-12);
}

TEST(testDPGO,
     ManualObjectDranCycleMajorizerCancelsTransportStepScale) {
  const double lipschitz = 3.0;
  const double stepSize = 0.8;
  const double temperature = 0.5;
  const double costScale = manualObjectDranCycleMajorizedCostScaleForTest(
      lipschitz, stepSize, temperature);

  EXPECT_NEAR(costScale, temperature / (stepSize * lipschitz), 1e-12);
  EXPECT_NEAR(stepSize * costScale / temperature, 1.0 / lipschitz,
              1e-12);
  EXPECT_DOUBLE_EQ(manualObjectDranCycleMajorizedCostScaleForTest(
                       0.0, stepSize, temperature),
                   0.0);
}

TEST(testDPGO, ManualObjectDranRunResultDefaultsToCommonGaugeEstimates) {
  const ManualObjectDranRunResult result;

  EXPECT_FALSE(result.finalEstimatesAreRobotLocal);
  EXPECT_TRUE(result.finalRobotGauges.empty());
}

TEST(testDPGO, ManualObjectDranFeedbackUotPolicyUsesFixedLocalScalesAndLastIterate) {
  ManualObjectDranLocalLowerFeedbackState feedback;
  const double firstReadiness =
      updateManualObjectDranLocalLowerFeedback(10.0, 4.0, feedback);
  const double stationarityScale = feedback.stationarityScale;
  const double latentScale = feedback.robustLatentScale;
  EXPECT_DOUBLE_EQ(firstReadiness, 0.0);
  const double secondReadiness =
      updateManualObjectDranLocalLowerFeedback(5.0, 2.0, feedback);
  EXPECT_GT(secondReadiness, firstReadiness);
  EXPECT_NEAR(secondReadiness, 1.0 / (1.0 + std::sqrt(0.5)), 1e-12);
  EXPECT_DOUBLE_EQ(feedback.stationarityScale, stationarityScale);
  EXPECT_DOUBLE_EQ(feedback.robustLatentScale, latentScale);
  const double currentCommandWorseReadiness =
      updateManualObjectDranLocalLowerFeedback(20.0, 2.0, feedback);
  EXPECT_LT(currentCommandWorseReadiness, secondReadiness);
  EXPECT_NEAR(currentCommandWorseReadiness,
              1.0 / (1.0 + std::hypot(2.0, 0.5)), 1e-12);
  const double invalidCurrentCommandReadiness =
      updateManualObjectDranLocalLowerFeedback(
          std::numeric_limits<double>::infinity(), 2.0, feedback);
  EXPECT_DOUBLE_EQ(invalidCurrentCommandReadiness, 0.0);

  ManualObjectDranOptions requested;
  requested.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  requested.associationLowerValidationSelectBestMeasurementState = true;
  requested.associationStateValidationTrajectoryGuardEnabled = true;
  requested.associationPairSupportLatentContinuation = true;
  requested.associationPairSupportGatePower = 7.0;
  requested.associationReciprocalFreshSupportBlend = 1.0;
  requested.associationEdgeLatentConsensusWeight = 1.0;
  requested.associationTemporalBlend = 0.5;
  requested.associationTemporalKlWeight = 0.5;
  requested.associationCycleObjectiveWeight = 1.0;
  requested.associationTransformDimension = 2;
  requested.associationTransformPlaneAxes = "yz";
  requested.associationTransformMinInliers = 3;
  requested.associationTransformMaxResidual = 0.75;
  requested.associationTransformMinScore = 0.4;
  requested.associationTransformWeightScale = 1.25;
  requested.associationTransformMaxSeedCandidates = 64;
  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_EQ(effective.associationPolicy,
            ManualObjectDranAssociationPolicy::
                DecentralizedFeedbackUotDbaV1);
  EXPECT_EQ(effective.associationLowerSharedLatentMode,
            ManualObjectDranSharedLatentMode::DirectEdgeDual);
  EXPECT_TRUE(effective.associationSharedEdgeVariableEnabled);
  EXPECT_FALSE(effective.associationLowerValidationSelectBestMeasurementState);
  EXPECT_FALSE(effective.associationStateValidationTrajectoryGuardEnabled);
  EXPECT_FALSE(effective.associationPairSupportLatentContinuation);
  EXPECT_DOUBLE_EQ(effective.associationPairSupportGatePower, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationReciprocalFreshSupportBlend, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationTemporalBlend, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationTemporalKlWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationCycleObjectiveWeight, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationRelationalGeometryWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationRelationalGeometryDistanceScale, 0.0);
  EXPECT_TRUE(effective.associationRelationalGeometryCenterRows);
  EXPECT_DOUBLE_EQ(effective.associationStateValidationHypergradientWeight,
                   0.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityUpperCostWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityMin, 0.05);
  EXPECT_DOUBLE_EQ(
      effective.associationLowerObjectCopyRobustResidualScale, -1.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerObjectCopyRobustMinWeight, 0.0);
  EXPECT_FALSE(effective.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_FALSE(effective.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_FALSE(effective.reducedRotationGradientBoundaryCandidate);
  EXPECT_FALSE(effective.associationTransformUpdateInLowerFixedPoint);
  EXPECT_FALSE(effective.associationTransformUseInitialObjectPoses);
  EXPECT_TRUE(effective.associationResidualTransformBootstrap);
  EXPECT_EQ(effective.associationTransformDimension, 2u);
  EXPECT_EQ(effective.associationTransformPlaneAxes, "yz");
  EXPECT_EQ(effective.associationTransformMinInliers, 3u);
  EXPECT_DOUBLE_EQ(effective.associationTransformMaxResidual, 0.75);
  EXPECT_DOUBLE_EQ(effective.associationTransformMinScore, 0.4);
  EXPECT_DOUBLE_EQ(effective.associationTransformWeightScale, 1.25);
  EXPECT_EQ(effective.associationTransformMaxSeedCandidates, 64u);
  EXPECT_TRUE(effective.associationGaugeSynchronizationApplyToStates);
  EXPECT_FALSE(
      effective.associationGaugeSynchronizationApplyOnceToInitialStates);
  EXPECT_DOUBLE_EQ(
      effective.associationLowerObjectCopySupportNormalizationScale, 1.0);

  requested.associationRelationalGeometryWeight = 0.4;
  requested.associationRelationalGeometryDistanceScale = 0.0;
  const ManualObjectDranOptions relational =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_DOUBLE_EQ(relational.associationRelationalGeometryWeight, 0.4);
  EXPECT_DOUBLE_EQ(relational.associationRelationalGeometryDistanceScale, 0.0);
  EXPECT_TRUE(relational.associationRelationalGeometryCenterRows);

  requested.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::ExactFixed;
  const ManualObjectDranOptions exactFixed =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_EQ(exactFixed.associationLowerSharedLatentMode,
            ManualObjectDranSharedLatentMode::ExactFixed);
  EXPECT_FALSE(exactFixed.associationTransformUpdateInLowerFixedPoint);
  EXPECT_FALSE(exactFixed.associationLowerSharedLatentFusion);
  EXPECT_FALSE(exactFixed.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_FALSE(exactFixed.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_DOUBLE_EQ(exactFixed.associationLowerObjectCopyRobustResidualScale,
                   0.0);

  requested.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::ExactWeighted;
  const ManualObjectDranOptions exactWeighted =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_EQ(exactWeighted.associationLowerSharedLatentMode,
            ManualObjectDranSharedLatentMode::ExactWeighted);
  EXPECT_FALSE(exactWeighted.associationLowerSharedLatentFusion);
  EXPECT_FALSE(
      exactWeighted.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_FALSE(exactWeighted.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_DOUBLE_EQ(
      exactWeighted.associationLowerObjectCopyRobustResidualScale, 0.0);
  EXPECT_TRUE(exactWeighted.associationGaugeSynchronizationApplyToStates);
  EXPECT_FALSE(
      exactWeighted.associationGaugeSynchronizationApplyOnceToInitialStates);

  requested.associationGaugeCovariantPreserveLocalStates = true;
  const ManualObjectDranOptions exactWeightedLocalView =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_TRUE(
      exactWeightedLocalView.associationGaugeCovariantPreserveLocalStates);
  EXPECT_FALSE(
      exactWeightedLocalView.associationGaugeSynchronizationApplyToStates);
  EXPECT_FALSE(exactWeightedLocalView
                   .associationGaugeSynchronizationApplyOnceToInitialStates);
  requested.associationGaugeCovariantPreserveLocalStates = false;

  requested.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::KktActiveHybrid;
  const ManualObjectDranOptions kktHybrid =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_EQ(kktHybrid.associationLowerSharedLatentMode,
            ManualObjectDranSharedLatentMode::KktActiveHybrid);
  EXPECT_FALSE(kktHybrid.associationLowerSharedLatentFusion);
  EXPECT_TRUE(
      kktHybrid.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_TRUE(kktHybrid.associationLowerObjectCopyNormalizeBySupport);

  requested.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::Disabled;
  const ManualObjectDranOptions physical =
      applyManualObjectDranAssociationPolicy(requested);
  EXPECT_TRUE(physical.associationGaugeSynchronizationEnabled);
  EXPECT_TRUE(physical.associationGaugeSynchronizationApplyToStates);
  EXPECT_FALSE(physical.associationLowerObjectCopyUseTransformAlignment);
  EXPECT_DOUBLE_EQ(physical.associationLowerObjectCopyRobustResidualScale,
                   -1.0);
  EXPECT_FALSE(physical.associationLowerObjectCopyNormalizeBySupport);

  ManualObjectDranOptions frozenFixedRequest;
  frozenFixedRequest.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  frozenFixedRequest.associationMode =
      ManualObjectDranAssociationMode::FixedFileSoft;
  frozenFixedRequest.associationFreezeUpperVariables = true;
  const ManualObjectDranOptions frozenFixed =
      applyManualObjectDranAssociationPolicy(frozenFixedRequest);
  EXPECT_EQ(frozenFixed.associationMode,
            ManualObjectDranAssociationMode::FixedFileSoft);
  EXPECT_FALSE(frozenFixed.associationResidualTransformBootstrap);

  ManualObjectDranOptions externalFrozenRequest;
  externalFrozenRequest.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  externalFrozenRequest.associationFile = "external_association.csv";
  externalFrozenRequest.associationFreezeUpperVariables = true;
  const ManualObjectDranOptions externalFrozen =
      applyManualObjectDranAssociationPolicy(externalFrozenRequest);
  EXPECT_EQ(externalFrozen.associationMode,
            ManualObjectDranAssociationMode::FixedFileSoft);
  EXPECT_FALSE(externalFrozen.associationResidualTransformBootstrap);

}

TEST(testDPGO,
     ManualObjectDranFeedbackUotBoundsCurrentCommandInnovation) {
  ManualObjectDranDirectedAssociationWeights weights(
      2, std::vector<Matrix>(2));
  weights[0][1] = Matrix::Zero(1, 1);
  weights[1][0] = Matrix::Zero(1, 1);
  ManualObjectDranDirectedAssociationWeights objectives(
      2, std::vector<Matrix>(2));
  objectives[0][1] = Matrix::Zero(1, 1);
  objectives[1][0] = Matrix::Zero(1, 1);
  std::vector<std::vector<bool>> edges(
      2, std::vector<bool>(2, false));
  edges[0][1] = true;
  edges[1][0] = true;
  std::vector<std::vector<ManualObjectDranEdgeFeedbackState>> feedbackStates(
      2, std::vector<ManualObjectDranEdgeFeedbackState>(2));
  PartialSinkhornOptions options;
  options.iterations = 20;
  options.stepSize = 1.0;
  options.temperature = 1.0;
  options.unmatchedCost = 1.0;

  const ManualObjectDranSharedEdgeSyncDiagnostics initialized =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          weights, objectives, edges, std::vector<double>{0.0, 0.0},
          feedbackStates, options);
  ASSERT_NEAR(weights[0][1](0, 0), 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(initialized.mannStepMean, 1.0);
  const std::size_t initializedEpoch = feedbackStates[0][1].commandEpoch;

  objectives[0][1](0, 0) = 2.0;
  objectives[1][0](0, 0) = 2.0;
  const ManualObjectDranSharedEdgeSyncDiagnostics quarterStep =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          weights, objectives, edges, std::vector<double>{0.25, 0.25},
          feedbackStates, options);
  EXPECT_NEAR(weights[0][1](0, 0), 1.0, 1e-12);
  EXPECT_NEAR(quarterStep.mannStepMean, 0.25, 1e-12);
  EXPECT_EQ(feedbackStates[0][1].commandEpoch, initializedEpoch);
  const std::size_t steppedEpoch = feedbackStates[0][1].commandEpoch;

  const Matrix heldCommand = weights[0][1];
  const ManualObjectDranSharedEdgeSyncDiagnostics held =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          weights, objectives, edges, std::vector<double>{0.0, 0.0},
          feedbackStates, options);
  EXPECT_NEAR((weights[0][1] - heldCommand).norm(), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(held.mannStepMean, 0.0);
  EXPECT_EQ(feedbackStates[0][1].commandEpoch, steppedEpoch);

  objectives[0][1](0, 0) = 1.25;
  objectives[1][0](0, 0) = 1.25;
  const ManualObjectDranSharedEdgeSyncDiagnostics proximalStep =
      synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
          weights, objectives, edges, std::vector<double>{1.0, 1.0},
          feedbackStates, options);
  EXPECT_NEAR(weights[0][1](0, 0), 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(proximalStep.mannStepMean, 1.0);
}

TEST(testDPGO, ManualObjectDranLocalLowerFeedbackRejectsNonfiniteResiduals) {
  ManualObjectDranLocalLowerFeedbackState stationarityFeedback;
  const double stationarityBaseline =
      updateManualObjectDranLocalLowerFeedback(
          10.0, 4.0, stationarityFeedback);
  const double stationarityAfterNonfinite =
      updateManualObjectDranLocalLowerFeedback(
          std::numeric_limits<double>::infinity(), 2.0,
          stationarityFeedback);
  EXPECT_DOUBLE_EQ(stationarityBaseline, 0.0);
  EXPECT_DOUBLE_EQ(stationarityAfterNonfinite, 0.0);
  EXPECT_FALSE(std::isfinite(stationarityFeedback.stationarityResidual));
  EXPECT_TRUE(std::isfinite(
      stationarityFeedback.robustLatentFixedPointResidual));

  ManualObjectDranLocalLowerFeedbackState latentFeedback;
  const double latentBaseline =
      updateManualObjectDranLocalLowerFeedback(10.0, 4.0, latentFeedback);
  const double latentAfterNonfinite =
      updateManualObjectDranLocalLowerFeedback(
          5.0, std::numeric_limits<double>::quiet_NaN(), latentFeedback);
  EXPECT_DOUBLE_EQ(latentBaseline, 0.0);
  EXPECT_DOUBLE_EQ(latentAfterNonfinite, 0.0);
  EXPECT_TRUE(std::isfinite(latentFeedback.stationarityResidual));
  EXPECT_FALSE(std::isfinite(
      latentFeedback.robustLatentFixedPointResidual));
}

TEST(testDPGO, ManualObjectDranLocalLowerFeedbackAllowsFiniteZeroLatentResidual) {
  ManualObjectDranLocalLowerFeedbackState feedback;
  const double firstReadiness =
      updateManualObjectDranLocalLowerFeedback(10.0, 4.0, feedback);
  const double secondReadiness =
      updateManualObjectDranLocalLowerFeedback(5.0, 0.0, feedback);
  EXPECT_GT(secondReadiness, firstReadiness);
  EXPECT_DOUBLE_EQ(firstReadiness, 0.0);
  EXPECT_NEAR(secondReadiness, 2.0 / 3.0, 1e-12);
  EXPECT_DOUBLE_EQ(feedback.robustLatentFixedPointResidual, 0.0);
  EXPECT_TRUE(std::isfinite(feedback.robustLatentFixedPointResidual));
}

TEST(testDPGO, ManualObjectDranChangedUpperCommandResetsLowerScales) {
  ManualObjectDranLocalLowerFeedbackState feedback;
  EXPECT_DOUBLE_EQ(
      updateManualObjectDranLocalLowerFeedback(100.0, 10.0, feedback), 0.0);
  EXPECT_NEAR(
      updateManualObjectDranLocalLowerFeedback(50.0, 5.0, feedback),
      1.0 / (1.0 + std::sqrt(0.5)), 1e-12);

  invalidateManualObjectDranLocalLowerFeedbackCommand(feedback);

  EXPECT_FALSE(feedback.stationarityScaleInitialized);
  EXPECT_FALSE(feedback.robustLatentScaleInitialized);
  EXPECT_DOUBLE_EQ(feedback.readiness, 0.0);
  EXPECT_DOUBLE_EQ(
      updateManualObjectDranLocalLowerFeedback(40.0, 4.0, feedback), 0.0);
  EXPECT_DOUBLE_EQ(feedback.stationarityScale, 40.0);
  EXPECT_DOUBLE_EQ(feedback.robustLatentScale, 4.0);
  EXPECT_NEAR(
      updateManualObjectDranLocalLowerFeedback(20.0, 2.0, feedback),
      1.0 / (1.0 + std::sqrt(0.5)), 1e-12);
}

TEST(testDPGO, ManualObjectDranFixedOracleInitializesIdentityActiveSet) {
  const DPGO::Matrix weights =
      (DPGO::Matrix(2, 3) << 1.0, 0.999, 0.0,
       0.0, 1.0, 0.25)
          .finished();

  EXPECT_EQ(manualObjectDranFixedIdentityActiveSetSizeForTest(
                weights, 1.0, 1.0),
            2u);
  EXPECT_EQ(manualObjectDranFixedIdentityActiveSetSizeForTest(
                0.5 * weights, 0.5, 0.75),
            2u);
}

TEST(testDPGO,
     ManualObjectDranDirectEdgeDualContinuouslyActivatesFractionalIdentity) {
  EXPECT_TRUE(manualObjectDranSharedConsensusHypothesisActiveForTest(
      ManualObjectDranSharedLatentMode::DirectEdgeDual, 0.37, false));
  EXPECT_FALSE(manualObjectDranSharedConsensusHypothesisActiveForTest(
      ManualObjectDranSharedLatentMode::DirectEdgeDual, 0.0, true));

  // The historical KKT hybrid remains a strict active-set ablation.
  EXPECT_FALSE(manualObjectDranSharedConsensusHypothesisActiveForTest(
      ManualObjectDranSharedLatentMode::KktActiveHybrid, 0.37, false));
  EXPECT_TRUE(manualObjectDranSharedConsensusHypothesisActiveForTest(
      ManualObjectDranSharedLatentMode::KktActiveHybrid, 1.0, true));
}

TEST(testDPGO,
     ManualObjectDranDirectEdgeDualContinuationUsesDualFixedPointResidual) {
  EXPECT_DOUBLE_EQ(
      sharedObjectLowerIncidentFixedPointResidual(
          ManualObjectDranSharedLatentMode::DirectEdgeDual, 9.0, 0.25),
      0.25);
  EXPECT_DOUBLE_EQ(
      sharedObjectLowerIncidentFixedPointResidual(
          ManualObjectDranSharedLatentMode::ExactFixed, 9.0, 0.25),
      9.0);
}

TEST(testDPGO, ManualObjectDranAssociationCostAccountsForInformationVolume) {
  EXPECT_NEAR(objectAssociationInformationVolumeCorrection(0.5, 1.0),
              std::log(2.0), 1e-12);
  EXPECT_NEAR(objectAssociationInformationVolumeCorrection(2.0, 1.0),
              -std::log(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(objectAssociationInformationVolumeCorrection(1.0, 1.0),
                   0.0);
  EXPECT_DOUBLE_EQ(objectAssociationInformationVolumeCorrection(0.0, 1.0),
                   0.0);
}

TEST(testDPGO, ManualObjectDranRobotGaugeSynchronizationConvergesOnNoisyCycle) {
  std::vector<std::vector<Matrix>> edges(3, std::vector<Matrix>(3));
  const auto translation = [](double x) {
    Matrix transform = Matrix::Zero(3, 4);
    transform.leftCols(3).setIdentity();
    transform(0, 3) = x;
    return transform;
  };
  edges[0][1] = translation(10.0);
  edges[1][0] = translation(-10.0);
  edges[0][2] = translation(20.0);
  edges[2][0] = translation(-20.0);
  edges[1][2] = translation(4.0);
  edges[2][1] = translation(-4.0);

  const std::vector<Matrix> gauges =
      manualObjectDranRobotGaugeTransformsForTest(edges);
  const std::vector<Matrix> asynchronousGauges =
      manualObjectDranRobotGaugeTransformsForTest(edges, true);

  ASSERT_EQ(gauges.size(), 3u);
  ASSERT_EQ(gauges[1].rows(), 3);
  ASSERT_EQ(gauges[2].rows(), 3);
  EXPECT_NEAR(gauges[1](0, 3), 12.0, 1e-6);
  EXPECT_NEAR(gauges[2](0, 3), 18.0, 1e-6);
  ASSERT_EQ(asynchronousGauges.size(), gauges.size());
  for (std::size_t robotId = 0; robotId < gauges.size(); ++robotId) {
    EXPECT_LT((asynchronousGauges[robotId] - gauges[robotId]).norm(), 1e-6);
  }
}

TEST(testDPGO,
     ManualObjectDranRobotGaugeSynchronizationUsesEdgePrecisionAndCycleRobustness) {
  std::vector<std::vector<Matrix>> edges(3, std::vector<Matrix>(3));
  std::vector<std::vector<double>> rmse(
      3, std::vector<double>(3, std::numeric_limits<double>::infinity()));
  const auto translation = [](double x) {
    Matrix transform = Matrix::Zero(3, 4);
    transform.leftCols(3).setIdentity();
    transform(0, 3) = x;
    return transform;
  };
  edges[0][1] = translation(10.0);
  edges[1][0] = translation(-10.0);
  rmse[0][1] = rmse[1][0] = 0.2;
  edges[0][2] = translation(20.0);
  edges[2][0] = translation(-20.0);
  rmse[0][2] = rmse[2][0] = 0.5;
  edges[1][2] = translation(4.0);
  edges[2][1] = translation(-4.0);
  rmse[1][2] = rmse[2][1] = 0.1;

  const std::vector<Matrix> gauges =
      manualObjectDranRobotGaugeTransformsWithRmseForTest(edges, rmse);

  ASSERT_EQ(gauges.size(), 3u);
  // The precise 0->1->2 path predicts 10 and 14, whereas the less precise
  // inconsistent direct edge predicts 20 for robot 2.
  EXPECT_NEAR(gauges[1](0, 3), 10.0, 0.02);
  EXPECT_NEAR(gauges[2](0, 3), 14.0, 0.02);
}

TEST(testDPGO, ManualObjectDranSharedEdgePolicyIsIdempotentAndSeparate) {
  ManualObjectDranOptions finalOptions;
  finalOptions.associationPolicy =
      ManualObjectDranAssociationPolicy::FinalDbaP4203;
  const ManualObjectDranOptions frozen =
      applyManualObjectDranAssociationPolicy(finalOptions);
  EXPECT_FALSE(frozen.associationSharedEdgeVariableEnabled);

  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SharedEdgeDbaV1;
  const ManualObjectDranOptions once =
      applyManualObjectDranAssociationPolicy(options);
  const ManualObjectDranOptions twice =
      applyManualObjectDranAssociationPolicy(once);

  EXPECT_TRUE(once.associationSharedEdgeVariableEnabled);
  EXPECT_DOUBLE_EQ(once.associationEdgeLatentConsensusWeight, 0.0);
  EXPECT_DOUBLE_EQ(once.associationReciprocalFreshSupportBlend, 0.0);
  EXPECT_FALSE(once.associationCycleConsistencyEnabled);
  EXPECT_DOUBLE_EQ(once.associationCycleObjectiveWeight, 0.0);
  EXPECT_DOUBLE_EQ(once.associationStateValidationHypergradientWeight,
                   frozen.associationStateValidationHypergradientWeight);
  EXPECT_DOUBLE_EQ(once.associationStateValidationMeritTrustScale,
                   frozen.associationStateValidationMeritTrustScale);
  EXPECT_EQ(once.associationStateValidationMeritLowerIterations,
            frozen.associationStateValidationMeritLowerIterations);
  EXPECT_EQ(once.associationStateValidationUseEdgeTransformConfidence,
            frozen.associationStateValidationUseEdgeTransformConfidence);
  EXPECT_DOUBLE_EQ(
      once.associationStateValidationTransformConfidenceDistanceScale,
      frozen.associationStateValidationTransformConfidenceDistanceScale);
  EXPECT_EQ(once.associationStateValidationMeritUseMeasurementCost,
            frozen.associationStateValidationMeritUseMeasurementCost);
  EXPECT_EQ(once.associationStateValidationMeritGateFullUpdate,
            frozen.associationStateValidationMeritGateFullUpdate);
  EXPECT_EQ(once.associationStateValidationTrajectoryGuardEnabled,
            frozen.associationStateValidationTrajectoryGuardEnabled);
  EXPECT_DOUBLE_EQ(once.associationPairSupportGatePower,
                   frozen.associationPairSupportGatePower);
  EXPECT_DOUBLE_EQ(once.associationPairSupportGateMinEffectiveSupport,
                   frozen.associationPairSupportGateMinEffectiveSupport);
  EXPECT_EQ(once.associationPairSupportLatentContinuation,
            frozen.associationPairSupportLatentContinuation);
  EXPECT_EQ(once.associationObjectReliabilityUseSharedState,
            frozen.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(once.associationObjectReliabilityUpperCostWeight,
                   frozen.associationObjectReliabilityUpperCostWeight);
  EXPECT_DOUBLE_EQ(once.associationObjectReliabilityResidualScale,
                   frozen.associationObjectReliabilityResidualScale);
  EXPECT_DOUBLE_EQ(once.associationObjectReliabilityMin,
                   frozen.associationObjectReliabilityMin);
  EXPECT_EQ(once.associationLowerObjectCopyObservationPredictiveGate,
            frozen.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_EQ(once.associationPostUpdateLowerFixedPointRounds,
            frozen.associationPostUpdateLowerFixedPointRounds);
  EXPECT_EQ(once.associationPostUpdateLocalRefine,
            frozen.associationPostUpdateLocalRefine);
  EXPECT_EQ(once.associationPostUpdateLocalRefineIterations,
            frozen.associationPostUpdateLocalRefineIterations);
  EXPECT_EQ(once.associationPostUpdateLocalRefineExchange,
            frozen.associationPostUpdateLocalRefineExchange);
  EXPECT_EQ(twice.associationPolicy, once.associationPolicy);
  EXPECT_TRUE(twice.associationSharedEdgeVariableEnabled);
  EXPECT_DOUBLE_EQ(twice.associationEdgeLatentConsensusWeight,
                   once.associationEdgeLatentConsensusWeight);
  EXPECT_DOUBLE_EQ(twice.associationReciprocalFreshSupportBlend,
                   once.associationReciprocalFreshSupportBlend);

  options.associationFinalAblation = ManualObjectDranFinalAblation::EdgeConsensus;
  EXPECT_THROW(applyManualObjectDranAssociationPolicy(options),
               std::invalid_argument);
}

TEST(testDPGO, ManualObjectDranDefaultsCoupledTranslationTrustRegion) {
  ManualObjectDranOptions options;
  EXPECT_TRUE(options.coupledTranslationTrustRegion);
  EXPECT_EQ(options.objectInitialization,
            ManualObjectDranObjectInitialization::CentralizedChordal);
  EXPECT_EQ(options.objectInterfaceMode,
            ManualObjectDranObjectInterfaceMode::CopyBaseline);
  EXPECT_EQ(options.objectInterfaceResponseStrategy,
            ManualObjectDranObjectInterfaceResponseStrategy::Serial);
  EXPECT_EQ(options.associationMode,
            ManualObjectDranAssociationMode::IdentityOracle);
  EXPECT_EQ(options.associationPolicy,
            ManualObjectDranAssociationPolicy::Manual);
  EXPECT_EQ(options.associationFinalAblation,
            ManualObjectDranFinalAblation::None);
  EXPECT_EQ(options.diagnosticProfile,
            ManualObjectDranDiagnosticProfile::Auto);
  EXPECT_EQ(options.associationUpdatePeriod, 1u);
  EXPECT_EQ(options.associationMaxCandidatesPerEdge, 0u);
  EXPECT_EQ(options.associationSinkhornMaxNonzeros, 0u);
  EXPECT_DOUBLE_EQ(options.associationRowCapacity, 1.0);
  EXPECT_DOUBLE_EQ(options.associationColumnCapacity, 1.0);
  EXPECT_FALSE(options.associationRequireObservedObjects);
  EXPECT_DOUBLE_EQ(options.associationTemporalBlend, 0.0);
  EXPECT_FALSE(options.associationRenormalizeSparseWeights);
  EXPECT_FALSE(options.associationResidualWarmStartUseGeometry);
  EXPECT_FALSE(options.associationResidualUseTransformAlignment);
  EXPECT_FALSE(options.associationResidualTransformUpdateFromWeights);
  EXPECT_TRUE(options.associationLowerObjectCopyUseTransformAlignment);
  EXPECT_TRUE(options.associationLowerObjectCopyDirectEnabled);
  EXPECT_FALSE(options.associationLowerSharedLatentFusion);
  EXPECT_EQ(options.associationLowerSharedLatentMode,
            ManualObjectDranSharedLatentMode::Auto);
  EXPECT_DOUBLE_EQ(options.associationLowerObjectCopyRobustResidualScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationLowerObjectCopyRobustMinWeight, 0.0);
  EXPECT_FALSE(options.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_FALSE(options.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_FALSE(options.associationLowerObjectCopyUseObservationImpliedTarget);
  EXPECT_DOUBLE_EQ(options.associationLowerTrajectoryTranslationProxWeight, 0.0);
  EXPECT_FALSE(
      options.associationLowerTrajectoryObjectCopyFeedbackMassConserving);
  EXPECT_FALSE(options.associationResidualTransformBootstrap);
  EXPECT_FALSE(options.associationResidualTrustPreviousSupport);
  EXPECT_FALSE(options.associationSparseUpperSupport);
  EXPECT_FALSE(options.associationPairSupportLatentContinuation);
  EXPECT_DOUBLE_EQ(options.associationReferenceTrustWeight, 0.0);
  EXPECT_FALSE(options.associationReferenceTrustScaleWithUpperCost);
  EXPECT_DOUBLE_EQ(options.associationReferenceTrustMinimumCostRmsScale, 1.0);
  EXPECT_DOUBLE_EQ(options.associationReferenceMassTrustWeight, 0.0);
  EXPECT_FALSE(options.associationReferenceMassTrustScaleWithUpperCost);
  EXPECT_DOUBLE_EQ(options.associationReferenceMassTrustMinimumCostRmsScale,
                   1.0);
  EXPECT_DOUBLE_EQ(options.associationRelationalGeometryWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRelationalGeometryDistanceScale, 0.0);
  EXPECT_FALSE(options.associationRelationalGeometryCenterRows);
  EXPECT_DOUBLE_EQ(options.associationRotoTranslationAlignmentWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRotoTranslationAlignmentDistanceScale,
                   0.0);
  EXPECT_EQ(options.associationRotoTranslationHypothesisTopK, 0u);
  EXPECT_EQ(options.associationRotoTranslationHypothesisAngleBins, 36u);
  EXPECT_DOUBLE_EQ(options.associationRotoTranslationHypothesisSoftminTemperature,
                   1.0);
  EXPECT_DOUBLE_EQ(
      options.associationUpperTrackingLowerValidationMeritTrustScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationStateValidationHypergradientWeight, 0.0);
  EXPECT_FALSE(options.associationStateValidationMeritUseMeasurementCost);
  EXPECT_FALSE(options.associationStateValidationMeritGateFullUpdate);
  EXPECT_TRUE(options.associationStateValidationTrajectoryGuardEnabled);
  EXPECT_FALSE(options.associationLowerValidationSelectBestMeasurementState);
  EXPECT_FALSE(options.associationLowerValidationMeasurementIncumbentProx);
  EXPECT_FALSE(options.associationSaveIterationEstimateSnapshots);
  EXPECT_TRUE(options.recordObjectInterfaceDiagnostics);
  EXPECT_TRUE(options.recordDetailedAssociationDiagnostics);
  EXPECT_TRUE(options.recordPredictiveMeasurementCost);
  EXPECT_EQ(options.associationHeldoutObjectObservationModulo, 0u);
  EXPECT_EQ(options.associationHeldoutObjectObservationRemainder, 0u);
  EXPECT_FALSE(options.associationPairValidationTrustEnabled);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationWeight, 0.0);
  EXPECT_TRUE(options.associationLowerValidationCenterRows);
  EXPECT_TRUE(options.associationLowerValidationCenterColumns);
  EXPECT_TRUE(options.associationLowerValidationNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationTargetRms, 1.0);
  EXPECT_TRUE(options.associationLowerValidationScaleWithUpperCost);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationMinimumCostRmsScale, 1.0);
  EXPECT_DOUBLE_EQ(
      options.associationLowerValidationStationarityTrustMaxGradientNorm, 0.0);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationMeritTrustScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationPairwiseWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationPairwiseDistanceScale,
                   0.0);
  EXPECT_DOUBLE_EQ(options.associationLowerValidationReferenceTrustWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationImplicitValidationAdjointWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationImplicitValidationAdjointDamping, 1.0);
  EXPECT_DOUBLE_EQ(options.associationImplicitValidationAdjointMaxEffectiveRatio,
                   0.0);
  EXPECT_DOUBLE_EQ(options.associationImplicitValidationAdjointTargetRms, 1.0);
  EXPECT_DOUBLE_EQ(
      options.associationImplicitValidationAdjointTargetRmsUnmatchedFraction,
      0.0);
  EXPECT_DOUBLE_EQ(
      options.associationImplicitValidationAdjointMeritTrustScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationValidationRiskWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationValidationRiskTargetRms, 1.0);
  EXPECT_DOUBLE_EQ(
      options.associationValidationRiskTargetRmsUnmatchedFraction, 0.0);
  EXPECT_FALSE(options.associationImplicitValidationAdjointUseCoupledSolve);
  EXPECT_EQ(options.associationImplicitValidationAdjointMaxActiveObjects, 0u);
  EXPECT_DOUBLE_EQ(
      options.associationImplicitValidationAdjointObservationRhsWeight, 0.0);
  EXPECT_FALSE(
      options.associationImplicitValidationAdjointProjectAgainstBackend);
  EXPECT_FALSE(
      options.associationImplicitValidationAdjointProjectAgainstUpperPrefix);
  EXPECT_DOUBLE_EQ(options.associationCoveragePriorWeight, 0.0);
  EXPECT_FALSE(options.associationCoveragePriorMatchRowColumn);
  EXPECT_FALSE(options.associationCoveragePriorNormalizeByReferenceMass);
  EXPECT_FALSE(options.associationBackendScaleNegativeDeltaByTransformConfidence);
  EXPECT_DOUBLE_EQ(options.associationSignedFreshSupportBlend, 0.0);
  EXPECT_DOUBLE_EQ(options.associationSignedFreshSupportSigmoidScale, 1.0);
  EXPECT_DOUBLE_EQ(
      options.associationSignedFreshSupportStateValidationMeritTrustScale,
      0.0);
  EXPECT_FALSE(options.associationNormalizeUpperCosts);
  EXPECT_DOUBLE_EQ(options.associationUpperCostNormalizationTargetRms, 1.0);
  EXPECT_EQ(options.associationTransformMinInliers, 3u);
  EXPECT_DOUBLE_EQ(options.associationTransformMaxResidual, 1.0);
  EXPECT_DOUBLE_EQ(options.associationTransformMinScore, 0.0);
  EXPECT_DOUBLE_EQ(options.associationTransformWeightScale, 1.0);
  EXPECT_EQ(options.associationTransformMaxSeedCandidates, 80u);
  EXPECT_EQ(options.associationTransformDimension, 0u);
  EXPECT_EQ(options.associationTransformPlaneAxes, "xy");
  EXPECT_FALSE(options.associationCycleConsistencyEnabled);
  EXPECT_FALSE(options.associationTransformUseInitialObjectPoses);
  EXPECT_DOUBLE_EQ(options.associationCycleConsistencyMinDirectWeight, 1e-12);
  EXPECT_DOUBLE_EQ(options.associationCycleConsistencyMinComposedWeight, 1e-12);
  EXPECT_EQ(options.associationCycleConsistencyMinTwoHopSupport, 1u);
  EXPECT_TRUE(options.associationCycleConsistencyKeepWithoutEvidence);
  EXPECT_DOUBLE_EQ(options.associationCycleObjectiveWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationCycleObjectiveMinEdgeMass, 1e-12);
  EXPECT_FALSE(options.associationAdaptiveUpperEnabled);
  EXPECT_DOUBLE_EQ(options.associationAdaptiveUpperInitialStep, 1.0);
  EXPECT_DOUBLE_EQ(options.associationAdaptiveUpperEpsilon, 1e-12);
  EXPECT_DOUBLE_EQ(options.associationAdaptiveUpperMaxStep, 1.0);
  EXPECT_DOUBLE_EQ(options.associationTemporalKlWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationTemporalKlEpsilon, 1e-12);
  EXPECT_DOUBLE_EQ(options.associationEntropyRegularizationWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendFeedbackGradientScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendFeedbackMinTrust, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendResponseGain, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendResponseDamping, 1.0);
  EXPECT_DOUBLE_EQ(options.associationBackendResponseMaxReductionFraction,
                   0.5);
  EXPECT_DOUBLE_EQ(options.associationBackendAdjointGain, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendAdjointDamping, 1.0);
  EXPECT_DOUBLE_EQ(options.associationBackendAdjointMaxReductionFraction,
                   0.5);
  EXPECT_DOUBLE_EQ(options.associationBackendAdjointMaxIncreaseFraction, 0.5);
  EXPECT_DOUBLE_EQ(options.associationBackendCoupledAdjointGain, 0.0);
  EXPECT_DOUBLE_EQ(options.associationBackendCoupledAdjointDamping, 1.0);
  EXPECT_DOUBLE_EQ(options.associationBackendCoupledAdjointMaxReductionFraction,
                   0.5);
  EXPECT_DOUBLE_EQ(options.associationBackendCoupledAdjointMaxIncreaseFraction,
                   0.5);
  EXPECT_DOUBLE_EQ(options.associationBackendAdjointReductionMetadataMargin,
                   -1.0);
  EXPECT_TRUE(options.associationBackendAdjointAllowCostReduction);
  EXPECT_EQ(options.associationBackendSnapshotPeriod, 1u);
  EXPECT_DOUBLE_EQ(options.associationBackendMaxEffectiveRatio, 0.0);
  EXPECT_FALSE(options.associationBackendNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(options.associationBackendTargetRms, 1.0);
  EXPECT_DOUBLE_EQ(options.associationBackendTargetRmsUnmatchedFraction, 0.0);
  EXPECT_FALSE(options.associationBackendCapPositiveDeltaAtUnmatched);
  EXPECT_TRUE(options.associationBackendCenterRows);
  EXPECT_TRUE(options.associationBackendCenterColumns);
  EXPECT_DOUBLE_EQ(options.associationBackendStationarityTrustMaxGradientNorm,
                   0.0);
  EXPECT_EQ(options.associationBackendMaxActiveFactorsPerEdge, 0u);
  EXPECT_DOUBLE_EQ(options.associationSupportSharpnessWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationSupportSharpnessTargetRatio, 0.0);
  EXPECT_DOUBLE_EQ(options.associationSupportSharpnessValidationBlend, 0.0);
  EXPECT_DOUBLE_EQ(options.associationSupportSharpnessCoverageBlend, 0.0);
  EXPECT_DOUBLE_EQ(options.associationStructuralExpansionWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationStructuralExpansionDistanceScale, 0.0);
  EXPECT_TRUE(options.associationStructuralExpansionUseProposalGate);
  EXPECT_FALSE(options.associationPairSupportLatentContinuation);
  EXPECT_DOUBLE_EQ(options.associationReferenceSupportSharpeningWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationReferenceSupportSharpeningTarget, 1.0);
  EXPECT_FALSE(options.associationPostUpdateLocalRefine);
  EXPECT_FALSE(options.associationLocalCavityFeedbackSweepEnabled);
  EXPECT_FALSE(options.associationComponentResponseGateEnabled);
  EXPECT_FALSE(options.associationNetworkPartitionGateEnabled);
  EXPECT_FALSE(
      options.associationNetworkPartitionMulticolorFeedbackEnabled);
  EXPECT_EQ(options.associationPostUpdateLocalRefineIterations, 1u);
  EXPECT_TRUE(options.associationPostUpdateLocalRefineExchange);
  EXPECT_EQ(options.associationTwoTimeScaleLowerInnerRounds, 1u);
  EXPECT_EQ(options.associationTwoTimeScaleLowerInnerIterations, 0u);
  EXPECT_TRUE(options.associationTwoTimeScaleLowerInnerExchange);
  EXPECT_TRUE(options.associationTwoTimeScaleFreezeUpperInLowerInner);
  EXPECT_DOUBLE_EQ(options.associationMinConsensusWeight, 0.0);
  EXPECT_DOUBLE_EQ(options.associationMinConsensusMargin, 0.0);
  EXPECT_DOUBLE_EQ(options.associationConsensusWeightPower, 1.0);
  EXPECT_EQ(options.associationActiveMinStableUpdates, 0u);
  EXPECT_EQ(options.associationActivationTrustRampUpdates, 0u);
  EXPECT_DOUBLE_EQ(options.associationTentativeWeightScale, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRejectResidualThreshold, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRejectResidualBestMargin, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRejectResidualQuantile, 0.0);
  EXPECT_DOUBLE_EQ(options.associationRejectResidualQuantileScale, 1.0);
  EXPECT_EQ(options.associationRejectMinBadUpdates, 0u);
}

TEST(testDPGO, ManualObjectDranRunsWithDirectObjectCopyDisabled) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = true;
  options.associationResidualTrustPreviousSupport = false;
  options.associationResidualAllowSupportExpansion = true;
  options.associationUnmatchedCost = 2.0;
  options.associationLowerObjectCopyDirectEnabled = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_TRUE(std::isfinite(result.iterations.front().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.front().consensusCost));
  EXPECT_TRUE(std::isfinite(result.iterations.front().gradient));
}

TEST(testDPGO, ManualObjectDranTerminalLowerEpochKeepsUpperCommandFixed) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult control =
      runManualObjectDran(dataDir, options);
  options.associationTerminalLowerRounds = 2;
  const ManualObjectDranRunResult terminal =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(control.iterations.size(), 1u);
  ASSERT_EQ(terminal.iterations.size(), 2u);
  EXPECT_EQ(terminal.iterations.back().iter, 3u);
  EXPECT_DOUBLE_EQ(
      terminal.iterations.back()
          .postUpdateLowerMaxOuterContinuationResidual,
      0.0);
  ASSERT_EQ(control.finalAssociationWeights.size(),
            terminal.finalAssociationWeights.size());
  for (std::size_t receiver = 0;
       receiver < control.finalAssociationWeights.size(); ++receiver) {
    ASSERT_EQ(control.finalAssociationWeights[receiver].size(),
              terminal.finalAssociationWeights[receiver].size());
    for (std::size_t sender = 0;
         sender < control.finalAssociationWeights[receiver].size(); ++sender) {
      EXPECT_EQ(control.finalAssociationWeights[receiver][sender],
                terminal.finalAssociationWeights[receiver][sender]);
    }
  }
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerZeroDelayMessagesAreAuditable) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = dataDir + "/gt_associations.csv";
  options.associationFreezeUpperVariables = true;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_GE(result.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  const SharedObjectMessageTotals totals = sharedObjectMessageTotals(result);
  EXPECT_GT(totals.deliveredMessages, 0u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusCurrentIncidentCertificates,
            summary.lowerSharedObjectConsensusExpectedIncidentCertificates);
  EXPECT_EQ(summary.lowerSharedObjectConsensusOutstandingMessages, 0u);
  EXPECT_EQ(result.selectedIterationCount, result.evaluatedIterationCount);
  EXPECT_FALSE(result.lowerValidationSelectedBestMeasurementState);
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerDelayDoesNotFalseQuiesce) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = dataDir + "/gt_associations.csv";
  options.associationFreezeUpperVariables = true;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationLowerMessageDelayRounds = 1;
  options.associationTerminalLowerRounds = 4;
  options.associationLowerValidationSelectBestMeasurementState = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  const SharedObjectMessageTotals totals = sharedObjectMessageTotals(result);
  EXPECT_GT(totals.deliveredMessages, 0u);
  EXPECT_LT(totals.staleCertificates, totals.deliveredMessages);
  EXPECT_EQ(summary.lowerSharedObjectConsensusCurrentIncidentCertificates,
            summary.lowerSharedObjectConsensusExpectedIncidentCertificates);
  EXPECT_EQ(summary.lowerSharedObjectConsensusOutstandingMessages, 0u);
  EXPECT_EQ(result.selectedIterationCount, result.evaluatedIterationCount);
  EXPECT_FALSE(result.lowerValidationSelectedBestMeasurementState);
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerDropFirstAttemptRecoversByRetry) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = dataDir + "/gt_associations.csv";
  options.associationFreezeUpperVariables = true;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationLowerMessageDelayRounds = 1;
  options.associationLowerDropFirstAttempt = true;
  options.associationTerminalLowerRounds = 4;
  options.associationLowerValidationSelectBestMeasurementState = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  const SharedObjectMessageTotals totals = sharedObjectMessageTotals(result);
  EXPECT_GT(totals.droppedAttempts, 0u);
  EXPECT_GT(totals.retryAttempts, 0u);
  EXPECT_GE(totals.certificatePayloadBytes,
            totals.retryAttempts *
                (2u * sizeof(double) + 3u * sizeof(std::uint64_t)));
  EXPECT_EQ(summary.lowerSharedObjectConsensusCurrentIncidentCertificates,
            summary.lowerSharedObjectConsensusExpectedIncidentCertificates);
  EXPECT_EQ(summary.lowerSharedObjectConsensusOutstandingMessages, 0u);
  EXPECT_LT(summary.postUpdateLowerConvergedRobots, options.numRobots);
  EXPECT_EQ(result.selectedIterationCount, result.evaluatedIterationCount);
  EXPECT_FALSE(result.lowerValidationSelectedBestMeasurementState);
  EXPECT_LE(totals.droppedAttempts, totals.targetMessages);
  EXPECT_LE(totals.retryAttempts, totals.droppedAttempts);
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerTimeVaryingSparseTopologyDeliversEventually) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  const std::filesystem::path topologyPath =
      std::filesystem::temp_directory_path() /
      "manual_object_dran_lower_sparse_topology_test.csv";
  {
    std::ofstream out(topologyPath);
    out << "round,src,dst,weight\n";
    out << "0,0,1,1.0\n";
    out << "0,1,2,1.0\n";
    out << "0,2,3,1.0\n";
    out << "1,0,3,1.0\n";
    out << "1,2,3,1.0\n";
    out << "1,1,2,1.0\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topologyFile = topologyPath.string();
  options.topologyWeightMode = "unit";
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = dataDir + "/gt_associations.csv";
  options.associationFreezeUpperVariables = true;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationLowerMessageDelayRounds = 1;
  options.associationTerminalLowerRounds = 20;
  options.associationLowerValidationSelectBestMeasurementState = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  const SharedObjectMessageTotals totals = sharedObjectMessageTotals(result);
  EXPECT_GT(totals.deliveredMessages, 0u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusCurrentIncidentCertificates,
            summary.lowerSharedObjectConsensusExpectedIncidentCertificates);
  EXPECT_LT(summary.lowerSharedObjectConsensusExpectedIncidentCertificates,
            3372u);
  EXPECT_EQ(summary.lowerSharedObjectConsensusOutstandingMessages, 0u);
  EXPECT_GT(summary.iter, summary.postUpdateLowerRoundsExecuted);
  EXPECT_LT(summary.measurementCost, 1000000.0);
  EXPECT_TRUE(std::isfinite(summary.consensusCost));
  EXPECT_LT(summary.postUpdateLowerConvergedRobots, options.numRobots);
  EXPECT_EQ(result.selectedIterationCount, result.evaluatedIterationCount);
  EXPECT_FALSE(result.lowerValidationSelectedBestMeasurementState);
}

TEST(testDPGO,
     ManualObjectDranCachedSharedObjectLowerCertificateIsReceiverLocal) {
  EXPECT_TRUE(manualObjectDranCachedSharedObjectLowerCertificateIsCurrentForTest(
      7, 3, 3, 2, 2));

  EXPECT_TRUE(manualObjectDranCachedSharedObjectLowerCertificateIsCurrentForTest(
      8, 3, 3, 2, 2));

  EXPECT_FALSE(manualObjectDranCachedSharedObjectLowerCertificateIsCurrentForTest(
      7, 3, 4, 2, 2));
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerDeliveryPumpIsDeliveryOnly) {
  EXPECT_TRUE(
      manualObjectDranSharedObjectLowerDeliveryPumpIsDeliveryOnlyForTest());
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerOutstandingSenderBlocksQuiescence) {
  EXPECT_TRUE(
      manualObjectDranSharedObjectLowerOutstandingSenderBlocksQuiescenceForTest());
}

TEST(testDPGO,
     ManualObjectDranSharedObjectLowerDeliveryPumpRecountsIdempotently) {
  EXPECT_TRUE(
      manualObjectDranSharedObjectLowerDeliveryPumpRecountsIdempotentlyForTest());
}

TEST(testDPGO,
     ManualObjectDranTransportWaitDoesNotRequireLocalSolve) {
  EXPECT_TRUE(manualObjectDranTransportWaitDoesNotRequireLocalSolveForTest());
}

TEST(testDPGO,
     ManualObjectDranProtocolOnlyMissingCertificateStallsWithoutOutstanding) {
  EXPECT_TRUE(
      manualObjectDranProtocolOnlyMissingCertificateStallsWithoutOutstandingForTest());
}

TEST(testDPGO,
     ManualObjectDranTerminalLocalBudgetMasksOnlyExhaustedRobots) {
  EXPECT_TRUE(
      manualObjectDranTerminalLocalBudgetMasksOnlyExhaustedRobotsForTest());
}

TEST(testDPGO,
     ManualObjectDranTerminalLatestStateChangesPreserveUnsolved) {
  EXPECT_TRUE(
      manualObjectDranTerminalLatestStateChangesPreserveUnsolvedForTest());
}

TEST(testDPGO, ManualObjectDranCanSaveIterationEstimateSnapshots) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = true;
  options.associationResidualTrustPreviousSupport = false;
  options.associationResidualAllowSupportExpansion = true;
  options.associationUnmatchedCost = 2.0;
  options.associationSaveIterationEstimateSnapshots = true;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(result.iterationEstimateSnapshots.size(), result.iterations.size());
  ASSERT_EQ(result.iterationEstimateSnapshots.front().size(), options.numRobots);
  ASSERT_EQ(result.iterationEstimateSnapshots.back().size(), options.numRobots);
}

TEST(testDPGO, ManualObjectDranReportsHeldoutObjectObservationCost) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = true;
  options.associationResidualTrustPreviousSupport = false;
  options.associationResidualAllowSupportExpansion = true;
  options.associationUnmatchedCost = 2.0;
  options.associationHeldoutObjectObservationModulo = 2;
  options.associationHeldoutObjectObservationRemainder = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_TRUE(std::isfinite(result.iterations.front().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.front().heldoutMeasurementCost));
  EXPECT_GT(result.iterations.front().heldoutMeasurementCost, 0.0);
}

TEST(testDPGO, ManualObjectDranReportsPredictiveObjectObservationCost) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = true;
  options.associationResidualTrustPreviousSupport = false;
  options.associationResidualAllowSupportExpansion = true;
  options.associationUnmatchedCost = 2.0;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_TRUE(
      std::isfinite(result.iterations.front().predictiveMeasurementCost));
  EXPECT_GE(result.iterations.front().predictiveMeasurementCost, 0.0);
}

TEST(testDPGO, ManualObjectDranTwoTimeScaleBilevelPolicySetsContinuousDefaults) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::TwoTimeScaleBilevelV1;
  options.associationBackendFeedbackGradientScale = 1.0;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(effective.associationPolicy,
            ManualObjectDranAssociationPolicy::TwoTimeScaleBilevelV1);
  EXPECT_EQ(effective.associationMode,
            ManualObjectDranAssociationMode::ResidualSoft);
  EXPECT_TRUE(effective.associationAdaptiveUpperEnabled);
  EXPECT_EQ(effective.associationUpdatePeriod, 2u);
  EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
  EXPECT_TRUE(effective.associationResidualUseTransformAlignment);
  EXPECT_TRUE(effective.associationLowerObjectCopyUseTransformAlignment);
  EXPECT_TRUE(effective.associationResidualTransformBootstrap);
  EXPECT_TRUE(effective.associationResidualTrustPreviousSupport);
  EXPECT_EQ(effective.associationSinkhornMaxNonzeros, 0u);
  EXPECT_FALSE(effective.associationNormalizeUpperCosts);
  EXPECT_DOUBLE_EQ(effective.associationReferenceMassTrustWeight, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationConsensusWeightPower, 3.0);
  EXPECT_GT(effective.associationTemporalKlWeight, 0.0);
  EXPECT_EQ(effective.associationBackendSnapshotPeriod, 2u);
  EXPECT_GT(effective.associationBackendMaxEffectiveRatio, 0.0);
  EXPECT_EQ(effective.associationTwoTimeScaleLowerInnerRounds, 3u);
  EXPECT_EQ(effective.associationTwoTimeScaleLowerInnerIterations, 1u);
  EXPECT_TRUE(effective.associationTwoTimeScaleLowerInnerExchange);
  EXPECT_TRUE(effective.associationTwoTimeScaleFreezeUpperInLowerInner);

  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationUpdatePeriod = 5;
  options.associationReferenceMassTrustWeight = 2.5;
  options.associationConsensusWeightPower = 2.0;
  options.associationTemporalKlWeight = 0.25;
  options.associationBackendSnapshotPeriod = 7;
  options.associationBackendMaxEffectiveRatio = 0.004;
  options.associationBackendNormalizeActiveRms = true;
  options.associationBackendTargetRms = 0.75;
  options.associationBackendTargetRmsUnmatchedFraction = 0.25;
  options.associationBackendCapPositiveDeltaAtUnmatched = true;
  options.associationBackendCenterRows = false;
  options.associationBackendCenterColumns = false;
  options.associationPostUpdateLocalRefineExchange = false;
  options.associationTwoTimeScaleLowerInnerRounds = 4;
  options.associationTwoTimeScaleLowerInnerIterations = 2;
  options.associationTwoTimeScaleLowerInnerExchange = false;
  options.associationTwoTimeScaleFreezeUpperInLowerInner = false;
  options.associationLowerObjectCopyUseTransformAlignment = false;
  const ManualObjectDranOptions explicitEffective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(explicitEffective.associationMode,
            ManualObjectDranAssociationMode::FixedFileSoft);
  EXPECT_EQ(explicitEffective.associationUpdatePeriod, 5u);
  EXPECT_DOUBLE_EQ(explicitEffective.associationReferenceMassTrustWeight, 2.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationConsensusWeightPower, 2.0);
  EXPECT_DOUBLE_EQ(explicitEffective.associationTemporalKlWeight, 0.25);
  EXPECT_EQ(explicitEffective.associationBackendSnapshotPeriod, 7u);
  EXPECT_DOUBLE_EQ(explicitEffective.associationBackendMaxEffectiveRatio,
                   0.004);
  EXPECT_TRUE(explicitEffective.associationBackendNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(explicitEffective.associationBackendTargetRms, 0.75);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationBackendTargetRmsUnmatchedFraction, 0.25);
  EXPECT_TRUE(explicitEffective.associationBackendCapPositiveDeltaAtUnmatched);
  EXPECT_FALSE(explicitEffective.associationBackendCenterRows);
  EXPECT_FALSE(explicitEffective.associationBackendCenterColumns);
  EXPECT_FALSE(explicitEffective.associationPostUpdateLocalRefineExchange);
  EXPECT_EQ(explicitEffective.associationTwoTimeScaleLowerInnerRounds, 4u);
  EXPECT_EQ(explicitEffective.associationTwoTimeScaleLowerInnerIterations, 2u);
  EXPECT_FALSE(explicitEffective.associationTwoTimeScaleLowerInnerExchange);
  EXPECT_FALSE(explicitEffective.associationTwoTimeScaleFreezeUpperInLowerInner);
  EXPECT_FALSE(
      explicitEffective.associationLowerObjectCopyUseTransformAlignment);
}

TEST(testDPGO,
     ManualObjectDranSingleLoopBilevelPolicySetsBestDiagnosticDefaults) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopBilevelV2;
  options.associationBackendAdjointGain = 1.0;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(effective.associationPolicy,
            ManualObjectDranAssociationPolicy::SingleLoopBilevelV2);
  EXPECT_EQ(effective.associationMode,
            ManualObjectDranAssociationMode::ResidualSoft);
  EXPECT_FALSE(effective.associationAdaptiveUpperEnabled);
  EXPECT_EQ(effective.associationUpdatePeriod, 1u);
  EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
  EXPECT_TRUE(effective.associationResidualUseTransformAlignment);
  EXPECT_TRUE(effective.associationResidualTransformBootstrap);
  EXPECT_TRUE(effective.associationResidualTrustPreviousSupport);
  EXPECT_EQ(effective.associationSinkhornMaxNonzeros, 0u);
  EXPECT_FALSE(effective.associationNormalizeUpperCosts);
  EXPECT_DOUBLE_EQ(effective.associationReferenceMassTrustWeight, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationConsensusWeightPower, 3.0);
  EXPECT_DOUBLE_EQ(effective.associationTemporalKlWeight, 0.0);
  EXPECT_EQ(effective.associationBackendSnapshotPeriod, 1u);
  EXPECT_DOUBLE_EQ(effective.associationBackendMaxEffectiveRatio, 0.01);

  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationUpdatePeriod = 4;
  options.associationReferenceMassTrustWeight = 2.5;
  options.associationConsensusWeightPower = 2.0;
  options.associationTemporalKlWeight = 0.25;
  options.associationBackendSnapshotPeriod = 7;
  options.associationBackendMaxEffectiveRatio = 0.004;
  options.associationBackendNormalizeActiveRms = true;
  options.associationBackendTargetRms = 0.75;
  options.associationBackendTargetRmsUnmatchedFraction = 0.25;
  options.associationBackendCapPositiveDeltaAtUnmatched = true;
  options.associationBackendCenterRows = false;
  options.associationBackendCenterColumns = false;
  options.associationPostUpdateLocalRefineExchange = false;
  const ManualObjectDranOptions explicitEffective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(explicitEffective.associationMode,
            ManualObjectDranAssociationMode::FixedFileSoft);
  EXPECT_EQ(explicitEffective.associationUpdatePeriod, 4u);
  EXPECT_DOUBLE_EQ(explicitEffective.associationReferenceMassTrustWeight, 2.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationConsensusWeightPower, 2.0);
  EXPECT_DOUBLE_EQ(explicitEffective.associationTemporalKlWeight, 0.25);
  EXPECT_EQ(explicitEffective.associationBackendSnapshotPeriod, 7u);
  EXPECT_DOUBLE_EQ(explicitEffective.associationBackendMaxEffectiveRatio,
                   0.004);
  EXPECT_TRUE(explicitEffective.associationBackendNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(explicitEffective.associationBackendTargetRms, 0.75);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationBackendTargetRmsUnmatchedFraction, 0.25);
  EXPECT_TRUE(explicitEffective.associationBackendCapPositiveDeltaAtUnmatched);
  EXPECT_FALSE(explicitEffective.associationBackendCenterRows);
  EXPECT_FALSE(explicitEffective.associationBackendCenterColumns);
  EXPECT_FALSE(explicitEffective.associationPostUpdateLocalRefineExchange);
}

TEST(testDPGO,
     ManualObjectDranSingleLoopRotoBilevelPolicyAddsContinuousGeometryDefaults) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(effective.associationPolicy,
            ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3);
  EXPECT_EQ(effective.associationMode,
            ManualObjectDranAssociationMode::ResidualSoft);
  EXPECT_EQ(effective.associationUpdatePeriod, 1u);
  EXPECT_FALSE(effective.associationAdaptiveUpperEnabled);
  EXPECT_DOUBLE_EQ(effective.associationReferenceMassTrustWeight, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationConsensusWeightPower, 3.0);
  EXPECT_DOUBLE_EQ(effective.associationTemporalKlWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationRotoTranslationAlignmentWeight, 0.2);
  EXPECT_EQ(effective.associationRotoTranslationHypothesisTopK, 3u);
  EXPECT_EQ(effective.associationRotoTranslationHypothesisAngleBins, 36u);
  EXPECT_DOUBLE_EQ(
      effective.associationRotoTranslationHypothesisSoftminTemperature, 0.1);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationWeight, 0.0);
  EXPECT_TRUE(effective.associationLowerValidationCenterRows);
  EXPECT_TRUE(effective.associationLowerValidationCenterColumns);
  EXPECT_TRUE(effective.associationLowerValidationNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationTargetRms, 1.0);
  EXPECT_TRUE(effective.associationLowerValidationScaleWithUpperCost);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationMinimumCostRmsScale, 1.0);
  EXPECT_DOUBLE_EQ(
      effective.associationLowerValidationStationarityTrustMaxGradientNorm,
      0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationMeritTrustScale, 0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationUpperTrackingLowerValidationMeritTrustScale, 0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationStateValidationHypergradientWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationStateValidationMeritTrustScale, 0.0);
  EXPECT_EQ(effective.associationStateValidationMeritLowerIterations, 1u);
  EXPECT_FALSE(effective.associationStateValidationUseEdgeTransformConfidence);
  EXPECT_DOUBLE_EQ(
      effective.associationStateValidationTransformConfidenceDistanceScale,
      2.0);
  EXPECT_DOUBLE_EQ(
      effective.associationSignedFreshSupportStateValidationMeritTrustScale,
      0.0);
  EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight, 0.0);
  EXPECT_FALSE(effective.associationEdgeLatentConsensusUseTransformConfidence);
  EXPECT_DOUBLE_EQ(
      effective.associationEdgeLatentConsensusConfidenceDistanceScale, 2.0);
  EXPECT_DOUBLE_EQ(effective.associationReciprocalFreshSupportBlend, 0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationReciprocalFreshSupportUniquenessScale, 0.0);
  EXPECT_FALSE(
      effective.associationReciprocalFreshSupportUseTransformConfidence);
  EXPECT_DOUBLE_EQ(
      effective.associationReciprocalFreshSupportConfidenceDistanceScale, 2.0);
  EXPECT_FALSE(effective.associationTransformUpdateInLowerFixedPoint);
  EXPECT_FALSE(effective.associationTransformLowerFixedPointUseReliability);
  EXPECT_FALSE(
      effective.associationTransformLowerFixedPointUseReliabilityTrust);
  EXPECT_FALSE(effective.associationTransformUseProjectedWeights);
  EXPECT_FALSE(effective.associationResidualTransformUpdateFromWeights);
  EXPECT_FALSE(effective.associationGaugeSynchronizationEnabled);
  EXPECT_FALSE(effective.associationGaugeSynchronizationApplyToStates);
  EXPECT_FALSE(effective.associationGaugeSynchronizationApplyOnceToInitialStates);
  EXPECT_FALSE(effective.associationLowerTrajectoryObjectCopyFeedback);
  EXPECT_FALSE(
      effective.associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust);
  EXPECT_FALSE(
      effective.associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass);
  EXPECT_FALSE(
      effective.associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor);
  EXPECT_DOUBLE_EQ(effective.associationLowerObjectCopyRobustResidualScale,
                   0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerObjectCopyRobustMinWeight, 0.0);
  EXPECT_FALSE(effective.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_FALSE(effective.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_DOUBLE_EQ(effective.associationLowerTrajectoryTranslationProxWeight,
                   0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerProjectionPower, 1.0);
  EXPECT_FALSE(effective.associationLowerProjectionHard);
  EXPECT_DOUBLE_EQ(effective.associationLowerHighConfidenceAnchorGain, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerContinuousSupportGatePower, 0.0);
  EXPECT_FALSE(effective.associationLowerUseCurrentResidualReliability);
  EXPECT_FALSE(
      effective.associationLowerUseBoundedCurrentResidualReliability);
  EXPECT_FALSE(
      effective.associationLowerUseSupportConfidenceReliability);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityResidualScale, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityMin, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityUpperCostWeight, 0.0);
  EXPECT_FALSE(effective.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(effective.associationObjectObservationRobustResidualScale,
                   0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationObjectObservationRobustMinInformationScale, 0.0);
  EXPECT_FALSE(effective.associationObjectObservationRobustNormalizeInformation);
  EXPECT_DOUBLE_EQ(
      effective.associationObjectObservationUncertaintyCountPrior, 0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationObjectObservationUncertaintyDistanceScale, 0.0);
  EXPECT_DOUBLE_EQ(
      effective.associationObjectObservationUncertaintyMinReliability, 0.0);
  EXPECT_FALSE(
      effective.associationObjectObservationUncertaintyNormalizeInformation);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationPairwiseWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationPairwiseDistanceScale,
                   0.0);
  EXPECT_DOUBLE_EQ(effective.associationLowerValidationReferenceTrustWeight,
                   0.0);
  EXPECT_DOUBLE_EQ(effective.associationCoveragePriorWeight, 0.0);
  EXPECT_FALSE(effective.associationCoveragePriorMatchRowColumn);
  EXPECT_FALSE(effective.associationCoveragePriorNormalizeByReferenceMass);

  options.associationRotoTranslationAlignmentWeight = 0.35;
  options.associationRotoTranslationHypothesisTopK = 5;
  options.associationRotoTranslationHypothesisAngleBins = 72;
  options.associationRotoTranslationHypothesisSoftminTemperature = 0.25;
  options.associationLowerValidationWeight = 0.12;
  options.associationLowerValidationCenterRows = false;
  options.associationLowerValidationCenterColumns = false;
  options.associationLowerValidationNormalizeActiveRms = false;
  options.associationLowerValidationTargetRms = 0.2;
  options.associationLowerValidationScaleWithUpperCost = false;
  options.associationLowerValidationMinimumCostRmsScale = 3.0;
  options.associationLowerValidationStationarityTrustMaxGradientNorm = 4.0;
  options.associationLowerValidationMeritTrustScale = 5.0;
  options.associationUpperTrackingLowerValidationMeritTrustScale = 0.8;
  options.associationStateValidationHypergradientWeight = 0.9;
  options.associationStateValidationMeritTrustScale = 0.7;
  options.associationStateValidationMeritLowerIterations = 2;
  options.associationStateValidationUseEdgeTransformConfidence = true;
  options.associationStateValidationTransformConfidenceDistanceScale = 3.25;
  options.associationStateValidationMeritUseMeasurementCost = true;
  options.associationStateValidationMeritGateFullUpdate = true;
  options.associationStateValidationTrajectoryGuardEnabled = false;
  options.associationLowerValidationSelectBestMeasurementState = true;
  options.associationLowerValidationMeasurementIncumbentProx = true;
  options.associationPairValidationTrustEnabled = true;
  options.associationBackendScaleNegativeDeltaByTransformConfidence = true;
  options.associationSignedFreshSupportBlend = 0.75;
  options.associationSignedFreshSupportSigmoidScale = 2.0;
  options.associationSignedFreshSupportStateValidationMeritTrustScale = 1.5;
  options.associationEdgeLatentConsensusWeight = 0.25;
  options.associationEdgeLatentConsensusUseTransformConfidence = true;
  options.associationEdgeLatentConsensusConfidenceDistanceScale = 3.5;
  options.associationReciprocalFreshSupportBlend = 0.6;
  options.associationReciprocalFreshSupportUniquenessScale = 1.25;
  options.associationReciprocalFreshSupportUseTransformConfidence = true;
  options.associationReciprocalFreshSupportConfidenceDistanceScale = 4.5;
  options.associationTransformUpdateInLowerFixedPoint = true;
  options.associationTransformLowerFixedPointUseReliability = true;
  options.associationTransformLowerFixedPointUseReliabilityTrust = true;
  options.associationTransformUseProjectedWeights = true;
  options.associationTransformUseInitialObjectPoses = true;
  options.associationResidualTransformUpdateFromWeights = true;
  options.associationGaugeSynchronizationEnabled = true;
  options.associationGaugeSynchronizationApplyToStates = true;
  options.associationGaugeSynchronizationApplyOnceToInitialStates = true;
  options.associationLowerTrajectoryObjectCopyFeedback = true;
  options.associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust = true;
  options.associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass =
      true;
  options.associationLowerTrajectoryObjectCopyFeedbackMassConserving = true;
  options.associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor = true;
  options.associationLowerSharedLatentFusion = true;
  options.associationLowerObjectCopyRobustResidualScale = 0.25;
  options.associationLowerObjectCopyRobustMinWeight = 0.1;
  options.associationLowerObjectCopyObservationPredictiveGate = true;
  options.associationLowerObjectCopyNormalizeBySupport = true;
  options.associationLowerObjectCopyUseObservationImpliedTarget = true;
  options.associationLowerTrajectoryTranslationProxWeight = 1.0;
  options.associationLowerProjectionPower = 8.0;
  options.associationLowerProjectionHard = true;
  options.associationLowerHighConfidenceAnchorGain = 1.5;
  options.associationLowerContinuousSupportGatePower = 2.0;
  options.associationLowerUseCurrentResidualReliability = true;
  options.associationLowerUseBoundedCurrentResidualReliability = true;
  options.associationLowerUseSupportConfidenceReliability = true;
  options.associationObjectReliabilityResidualScale = 2.5;
  options.associationObjectReliabilityMin = 0.2;
  options.associationObjectReliabilityUpperCostWeight = 0.4;
  options.associationObjectReliabilityUseSharedState = true;
  options.associationObjectObservationRobustResidualScale = 3.0;
  options.associationObjectObservationRobustMinInformationScale = 0.15;
  options.associationObjectObservationRobustNormalizeInformation = true;
  options.associationObjectObservationUncertaintyCountPrior = 3.0;
  options.associationObjectObservationUncertaintyDistanceScale = 30.0;
  options.associationObjectObservationUncertaintyMinReliability = 0.1;
  options.associationObjectObservationUncertaintyNormalizeInformation = true;
  options.associationLowerValidationPairwiseWeight = 0.7;
  options.associationLowerValidationPairwiseDistanceScale = 2.0;
  options.associationLowerValidationReferenceTrustWeight = 0.45;
  options.associationImplicitValidationAdjointMeritTrustScale = 0.6;
  options.associationCoveragePriorWeight = 0.3;
  options.associationCoveragePriorMatchRowColumn = true;
  options.associationCoveragePriorNormalizeByReferenceMass = true;
  options.associationSupportSharpnessWeight = 0.4;
  options.associationSupportSharpnessTargetRatio = 3e-8;
  options.associationSupportSharpnessValidationBlend = 0.75;
  options.associationSupportSharpnessCoverageBlend = 0.5;
  options.associationStructuralExpansionWeight = 0.08;
  options.associationStructuralExpansionDistanceScale = 2.5;
  options.associationStructuralExpansionUseProposalGate = false;
  options.associationPairSupportLatentContinuation = true;
  options.associationReferenceSupportSharpeningWeight = 0.12;
  options.associationReferenceSupportSharpeningTarget = 1.0;
  options.associationPostUpdateLocalRefine = true;
  options.associationPostUpdateLocalRefineIterations = 2;
  options.associationTwoTimeScaleLowerInnerRounds = 5;
  options.associationTwoTimeScaleLowerInnerIterations = 3;
  options.associationTwoTimeScaleLowerInnerExchange = false;
  options.associationTwoTimeScaleFreezeUpperInLowerInner = false;
  options.associationSinkhornMaxNonzeros = 64;
  const ManualObjectDranOptions explicitEffective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_DOUBLE_EQ(explicitEffective.associationRotoTranslationAlignmentWeight,
                   0.35);
  EXPECT_EQ(explicitEffective.associationRotoTranslationHypothesisTopK, 5u);
  EXPECT_EQ(explicitEffective.associationRotoTranslationHypothesisAngleBins,
            72u);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationRotoTranslationHypothesisSoftminTemperature,
      0.25);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerValidationWeight, 0.12);
  EXPECT_FALSE(explicitEffective.associationLowerValidationCenterRows);
  EXPECT_FALSE(explicitEffective.associationLowerValidationCenterColumns);
  EXPECT_FALSE(explicitEffective.associationLowerValidationNormalizeActiveRms);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerValidationTargetRms, 0.2);
  EXPECT_FALSE(explicitEffective.associationLowerValidationScaleWithUpperCost);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerValidationMinimumCostRmsScale,
                   3.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationLowerValidationStationarityTrustMaxGradientNorm,
      4.0);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerValidationMeritTrustScale,
                   5.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationUpperTrackingLowerValidationMeritTrustScale,
      0.8);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationStateValidationHypergradientWeight, 0.9);
  EXPECT_DOUBLE_EQ(explicitEffective.associationStateValidationMeritTrustScale,
                   0.7);
  EXPECT_EQ(explicitEffective.associationStateValidationMeritLowerIterations,
            2u);
  EXPECT_TRUE(
      explicitEffective.associationStateValidationUseEdgeTransformConfidence);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationStateValidationTransformConfidenceDistanceScale,
      3.25);
  EXPECT_TRUE(
      explicitEffective.associationStateValidationMeritUseMeasurementCost);
  EXPECT_TRUE(explicitEffective.associationStateValidationMeritGateFullUpdate);
  EXPECT_FALSE(
      explicitEffective.associationStateValidationTrajectoryGuardEnabled);
  EXPECT_TRUE(
      explicitEffective.associationLowerValidationSelectBestMeasurementState);
  EXPECT_TRUE(explicitEffective.associationLowerValidationMeasurementIncumbentProx);
  EXPECT_TRUE(explicitEffective.associationPairValidationTrustEnabled);
  EXPECT_TRUE(
      explicitEffective.associationBackendScaleNegativeDeltaByTransformConfidence);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSignedFreshSupportBlend, 0.75);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSignedFreshSupportSigmoidScale,
                   2.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationSignedFreshSupportStateValidationMeritTrustScale,
      1.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationEdgeLatentConsensusWeight,
                   0.25);
  EXPECT_TRUE(
      explicitEffective.associationEdgeLatentConsensusUseTransformConfidence);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationEdgeLatentConsensusConfidenceDistanceScale,
      3.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationReciprocalFreshSupportBlend,
                   0.6);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationReciprocalFreshSupportUniquenessScale,
      1.25);
  EXPECT_TRUE(
      explicitEffective.associationReciprocalFreshSupportUseTransformConfidence);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationReciprocalFreshSupportConfidenceDistanceScale,
      4.5);
  EXPECT_TRUE(explicitEffective.associationTransformUpdateInLowerFixedPoint);
  EXPECT_TRUE(explicitEffective.associationTransformLowerFixedPointUseReliability);
  EXPECT_TRUE(
      explicitEffective.associationTransformLowerFixedPointUseReliabilityTrust);
  EXPECT_TRUE(explicitEffective.associationTransformUseProjectedWeights);
  EXPECT_TRUE(explicitEffective.associationTransformUseInitialObjectPoses);
  EXPECT_TRUE(explicitEffective.associationResidualTransformUpdateFromWeights);
  EXPECT_TRUE(explicitEffective.associationGaugeSynchronizationEnabled);
  EXPECT_TRUE(explicitEffective.associationGaugeSynchronizationApplyToStates);
  EXPECT_TRUE(explicitEffective.associationGaugeSynchronizationApplyOnceToInitialStates);
  EXPECT_TRUE(explicitEffective.associationLowerTrajectoryObjectCopyFeedback);
  EXPECT_TRUE(
      explicitEffective.associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust);
  EXPECT_TRUE(
      explicitEffective
          .associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass);
  EXPECT_TRUE(
      explicitEffective
          .associationLowerTrajectoryObjectCopyFeedbackMassConserving);
  EXPECT_TRUE(
      explicitEffective.associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor);
  EXPECT_TRUE(explicitEffective.associationLowerSharedLatentFusion);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationLowerObjectCopyRobustResidualScale, 0.25);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerObjectCopyRobustMinWeight,
                   0.1);
  EXPECT_TRUE(
      explicitEffective.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_TRUE(
      explicitEffective.associationLowerObjectCopyUseObservationImpliedTarget);
  EXPECT_TRUE(explicitEffective.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationLowerTrajectoryTranslationProxWeight, 1.0);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerProjectionPower, 8.0);
  EXPECT_TRUE(explicitEffective.associationLowerProjectionHard);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerHighConfidenceAnchorGain,
                   1.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerContinuousSupportGatePower,
                   2.0);
  EXPECT_TRUE(explicitEffective.associationLowerUseCurrentResidualReliability);
  EXPECT_TRUE(
      explicitEffective.associationLowerUseBoundedCurrentResidualReliability);
  EXPECT_TRUE(
      explicitEffective.associationLowerUseSupportConfidenceReliability);
  EXPECT_DOUBLE_EQ(explicitEffective.associationObjectReliabilityResidualScale,
                   2.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationObjectReliabilityMin, 0.2);
  EXPECT_DOUBLE_EQ(explicitEffective.associationObjectReliabilityUpperCostWeight,
                   0.4);
  EXPECT_TRUE(explicitEffective.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationObjectObservationRobustResidualScale, 3.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationObjectObservationRobustMinInformationScale,
      0.15);
  EXPECT_TRUE(
      explicitEffective.associationObjectObservationRobustNormalizeInformation);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationObjectObservationUncertaintyCountPrior, 3.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationObjectObservationUncertaintyDistanceScale,
      30.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationObjectObservationUncertaintyMinReliability,
      0.1);
  EXPECT_TRUE(
      explicitEffective.associationObjectObservationUncertaintyNormalizeInformation);
  EXPECT_DOUBLE_EQ(explicitEffective.associationLowerValidationPairwiseWeight,
                   0.7);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationLowerValidationPairwiseDistanceScale, 2.0);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationLowerValidationReferenceTrustWeight, 0.45);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationImplicitValidationAdjointMeritTrustScale,
      0.6);
  EXPECT_DOUBLE_EQ(explicitEffective.associationCoveragePriorWeight, 0.3);
  EXPECT_TRUE(explicitEffective.associationCoveragePriorMatchRowColumn);
  EXPECT_TRUE(explicitEffective.associationCoveragePriorNormalizeByReferenceMass);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSupportSharpnessWeight, 0.4);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSupportSharpnessTargetRatio,
                   3e-8);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSupportSharpnessValidationBlend,
                   0.75);
  EXPECT_DOUBLE_EQ(explicitEffective.associationSupportSharpnessCoverageBlend,
                   0.5);
  EXPECT_DOUBLE_EQ(explicitEffective.associationStructuralExpansionWeight,
                   0.08);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationStructuralExpansionDistanceScale, 2.5);
  EXPECT_FALSE(explicitEffective.associationStructuralExpansionUseProposalGate);
  EXPECT_TRUE(explicitEffective.associationPairSupportLatentContinuation);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationReferenceSupportSharpeningWeight, 0.12);
  EXPECT_DOUBLE_EQ(
      explicitEffective.associationReferenceSupportSharpeningTarget, 1.0);
  EXPECT_TRUE(explicitEffective.associationPostUpdateLocalRefine);
  EXPECT_EQ(explicitEffective.associationPostUpdateLocalRefineIterations, 2u);
  EXPECT_EQ(explicitEffective.associationTwoTimeScaleLowerInnerRounds, 5u);
  EXPECT_EQ(explicitEffective.associationTwoTimeScaleLowerInnerIterations, 3u);
  EXPECT_FALSE(explicitEffective.associationTwoTimeScaleLowerInnerExchange);
  EXPECT_FALSE(explicitEffective.associationTwoTimeScaleFreezeUpperInLowerInner);
  EXPECT_EQ(explicitEffective.associationSinkhornMaxNonzeros, 64u);
}

TEST(testDPGO, ManualObjectDranFinalDbaPresetKeepsOnlyPaperPathMechanisms) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::FinalDbaP4203;
  options.associationMode = ManualObjectDranAssociationMode::IdentityOracle;
  options.associationStateValidationMeritUseMeasurementCost = true;
  options.associationStateValidationMeritGateFullUpdate = true;
  options.associationStateValidationHypergradientWeight = 42.0;
  options.associationEdgeLatentConsensusWeight = 42.0;
  options.associationLowerValidationMeasurementIncumbentProx = true;
  options.associationPairValidationTrustEnabled = true;
  options.associationLowerTrajectoryObjectCopyFeedback = true;
  options.associationLowerObjectCopyUseObservationImpliedTarget = true;
  options.associationLowerObjectCopyNormalizeBySupport = true;
  options.associationSupportSharpnessWeight = 0.5;
  options.associationImplicitValidationAdjointWeight = 0.25;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(effective.associationPolicy,
            ManualObjectDranAssociationPolicy::FinalDbaP4203);
  EXPECT_EQ(effective.associationMode,
            ManualObjectDranAssociationMode::ResidualSoft);
  EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
  EXPECT_TRUE(effective.associationResidualUseTransformAlignment);
  EXPECT_FALSE(effective.associationResidualTransformBootstrap);
  EXPECT_TRUE(effective.associationResidualAllowSupportExpansion);
  EXPECT_FALSE(effective.associationResidualTrustPreviousSupport);
  EXPECT_TRUE(effective.associationSparseUpperSupport);
  EXPECT_EQ(effective.associationMaxCandidatesPerEdge, 20u);
  EXPECT_TRUE(effective.associationResidualTransformUpdateFromWeights);
  EXPECT_TRUE(effective.associationTransformUseInitialObjectPoses);
  EXPECT_DOUBLE_EQ(effective.associationRotoTranslationAlignmentWeight, 0.0);
  EXPECT_EQ(effective.associationRotoTranslationHypothesisTopK, 0u);
  EXPECT_DOUBLE_EQ(effective.associationReferenceMassTrustWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationConsensusWeightPower, 1.0);
  EXPECT_TRUE(effective.associationGaugeSynchronizationEnabled);
  EXPECT_TRUE(effective.associationGaugeSynchronizationApplyOnceToInitialStates);
  EXPECT_TRUE(effective.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityUpperCostWeight, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityResidualScale, 2.0);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityMin, 0.05);
  EXPECT_DOUBLE_EQ(effective.associationPairSupportGatePower, 2.0);
  EXPECT_DOUBLE_EQ(effective.associationPairSupportGateMinEffectiveSupport, 2.0);
  EXPECT_TRUE(effective.associationPairSupportLatentContinuation);
  EXPECT_EQ(effective.associationPostUpdateLowerFixedPointRounds, 5u);
  EXPECT_TRUE(effective.associationPostUpdateLocalRefine);
  EXPECT_EQ(effective.associationPostUpdateLocalRefineIterations, 1u);
  EXPECT_TRUE(effective.associationPostUpdateLocalRefineExchange);
  EXPECT_DOUBLE_EQ(effective.associationStateValidationHypergradientWeight,
                   0.1);
  EXPECT_DOUBLE_EQ(effective.associationStateValidationMeritTrustScale, 0.0);
  EXPECT_TRUE(effective.associationStateValidationUseEdgeTransformConfidence);
  EXPECT_DOUBLE_EQ(
      effective.associationStateValidationTransformConfidenceDistanceScale,
      2.0);
  EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight, 0.05);
  EXPECT_TRUE(effective.associationEdgeLatentConsensusUseTransformConfidence);
  EXPECT_DOUBLE_EQ(
      effective.associationEdgeLatentConsensusConfidenceDistanceScale, 2.0);
  EXPECT_TRUE(effective.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_TRUE(effective.associationLowerValidationSelectBestMeasurementState);
  EXPECT_DOUBLE_EQ(effective.associationSignedFreshSupportBlend, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationSignedFreshSupportSigmoidScale, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationReciprocalFreshSupportBlend, 1.0);
  EXPECT_FALSE(effective.associationStateValidationTrajectoryGuardEnabled);
  EXPECT_DOUBLE_EQ(effective.associationBackendFeedbackGradientScale, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationBackendResponseGain, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationBackendAdjointGain, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationBackendCoupledAdjointGain, 0.0);
  EXPECT_EQ(effective.diagnosticProfile,
            ManualObjectDranDiagnosticProfile::Lightweight);
  EXPECT_FALSE(effective.recordObjectInterfaceDiagnostics);
  EXPECT_FALSE(effective.recordDetailedAssociationDiagnostics);
  EXPECT_FALSE(effective.recordPredictiveMeasurementCost);

  EXPECT_FALSE(effective.associationStateValidationMeritUseMeasurementCost);
  EXPECT_FALSE(effective.associationStateValidationMeritGateFullUpdate);
  EXPECT_FALSE(effective.associationLowerValidationMeasurementIncumbentProx);
  EXPECT_FALSE(effective.associationPairValidationTrustEnabled);
  EXPECT_FALSE(effective.associationLowerTrajectoryObjectCopyFeedback);
  EXPECT_FALSE(effective.associationLowerObjectCopyUseObservationImpliedTarget);
  EXPECT_FALSE(effective.associationLowerObjectCopyNormalizeBySupport);
  EXPECT_DOUBLE_EQ(effective.associationSupportSharpnessWeight, 0.0);
  EXPECT_DOUBLE_EQ(effective.associationImplicitValidationAdjointWeight, 0.0);
}

TEST(testDPGO,
     ManualObjectDranScaleInvariantRelationalGeometryKeepsBroadSupport) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationRelationalGeometryNormalizeBySquaredMass = true;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_TRUE(
      effective.associationRelationalGeometryNormalizeBySquaredMass);
  EXPECT_EQ(effective.associationMaxCandidatesPerEdge, 0u);
}

TEST(testDPGO, ManualObjectDranRelationalNullEvidenceKeepsBroadSupport) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationRelationalNullEvidence = true;
  options.associationRelationalNullEvidenceOptions.calibration =
      AssociationRelationalNullCalibration::AdaptiveEmpiricalRank;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_TRUE(effective.associationRelationalNullEvidence);
  EXPECT_EQ(effective.associationRelationalNullEvidenceOptions.calibration,
            AssociationRelationalNullCalibration::AdaptiveEmpiricalRank);
  EXPECT_EQ(effective.associationMaxCandidatesPerEdge, 0u);
}

TEST(testDPGO, ManualObjectDranRelationalNullEvidenceDefaultsStayFixed) {
  const ManualObjectDranOptions options;

  EXPECT_EQ(options.associationRelationalNullEvidenceOptions.calibration,
            AssociationRelationalNullCalibration::FixedStudentT);
  EXPECT_DOUBLE_EQ(
      options.associationRelationalNullEvidenceOptions.fixedAnchorThreshold,
      0.5);
  EXPECT_EQ(
      options.associationRelationalNullEvidenceOptions.fixedMinimumAnchors,
      8u);
  EXPECT_DOUBLE_EQ(
      options.associationRelationalNullEvidenceOptions
          .fixedStudentTDegreesOfFreedom,
      3.0);
}

TEST(testDPGO, ManualObjectDranFinalDbaAblationNoneKeepsFrozenSettings) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::FinalDbaP4203;
  options.associationFinalAblation = ManualObjectDranFinalAblation::None;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_DOUBLE_EQ(effective.associationStateValidationHypergradientWeight,
                   0.1);
  EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight, 0.05);
  EXPECT_DOUBLE_EQ(effective.associationReciprocalFreshSupportBlend, 1.0);
  EXPECT_DOUBLE_EQ(effective.associationPairSupportGatePower, 2.0);
  EXPECT_TRUE(effective.associationPairSupportLatentContinuation);
  EXPECT_TRUE(effective.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityUpperCostWeight, 1.0);
  EXPECT_TRUE(effective.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
  EXPECT_TRUE(effective.associationGaugeSynchronizationEnabled);
  EXPECT_EQ(effective.associationPostUpdateLowerFixedPointRounds, 5u);
}

TEST(testDPGO,
     ManualObjectDranFinalDbaFactoryAppliesDiagnosticsAfterSelectedDeletion) {
  const std::vector<ManualObjectDranFinalAblation> ablations = {
      ManualObjectDranFinalAblation::None,
      ManualObjectDranFinalAblation::EdgeConsensus,
  };
  for (const ManualObjectDranFinalAblation ablation : ablations) {
    ManualObjectDranOptions options;
    options.associationFinalAblation = ablation;

    const ManualObjectDranOptions effective =
        makeFinalDbaObjectAssociationOptions(options);

    EXPECT_EQ(effective.diagnosticProfile,
              ManualObjectDranDiagnosticProfile::Lightweight);
    EXPECT_FALSE(effective.recordObjectInterfaceDiagnostics);
    EXPECT_FALSE(effective.recordDetailedAssociationDiagnostics);
    EXPECT_FALSE(effective.recordPredictiveMeasurementCost);
    EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight,
                     ablation == ManualObjectDranFinalAblation::EdgeConsensus
                         ? 0.0
                         : 0.05);
    EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
    EXPECT_TRUE(effective.associationGaugeSynchronizationEnabled);
  }
}

TEST(testDPGO, ManualObjectDranFinalDbaFactoryIsIdempotentForAllAblations) {
  const std::vector<ManualObjectDranFinalAblation> ablations = {
      ManualObjectDranFinalAblation::None,
      ManualObjectDranFinalAblation::StateValidation,
      ManualObjectDranFinalAblation::EdgeConsensus,
      ManualObjectDranFinalAblation::ReciprocalSupport,
      ManualObjectDranFinalAblation::PairSupportGate,
      ManualObjectDranFinalAblation::PairSupportContinuation,
      ManualObjectDranFinalAblation::SharedReliability,
      ManualObjectDranFinalAblation::ObservationPredictiveGate,
  };
  for (const ManualObjectDranFinalAblation ablation : ablations) {
    ManualObjectDranOptions options;
    options.associationFinalAblation = ablation;

    const ManualObjectDranOptions once =
        makeFinalDbaObjectAssociationOptions(options);
    const ManualObjectDranOptions twice =
        makeFinalDbaObjectAssociationOptions(once);

    EXPECT_EQ(twice.associationFinalAblation, once.associationFinalAblation);
    EXPECT_EQ(twice.associationPolicy, once.associationPolicy);
    EXPECT_EQ(twice.associationMode, once.associationMode);
    EXPECT_EQ(twice.diagnosticProfile, once.diagnosticProfile);
    EXPECT_EQ(twice.recordObjectInterfaceDiagnostics,
              once.recordObjectInterfaceDiagnostics);
    EXPECT_EQ(twice.recordDetailedAssociationDiagnostics,
              once.recordDetailedAssociationDiagnostics);
    EXPECT_EQ(twice.recordPredictiveMeasurementCost,
              once.recordPredictiveMeasurementCost);
    EXPECT_DOUBLE_EQ(twice.associationStateValidationHypergradientWeight,
                     once.associationStateValidationHypergradientWeight);
    EXPECT_DOUBLE_EQ(twice.associationEdgeLatentConsensusWeight,
                     once.associationEdgeLatentConsensusWeight);
    EXPECT_DOUBLE_EQ(twice.associationReciprocalFreshSupportBlend,
                     once.associationReciprocalFreshSupportBlend);
    EXPECT_DOUBLE_EQ(twice.associationPairSupportGatePower,
                     once.associationPairSupportGatePower);
    EXPECT_EQ(twice.associationPairSupportLatentContinuation,
              once.associationPairSupportLatentContinuation);
    EXPECT_EQ(twice.associationObjectReliabilityUseSharedState,
              once.associationObjectReliabilityUseSharedState);
    EXPECT_DOUBLE_EQ(twice.associationObjectReliabilityUpperCostWeight,
                     once.associationObjectReliabilityUpperCostWeight);
    EXPECT_EQ(twice.associationLowerObjectCopyObservationPredictiveGate,
              once.associationLowerObjectCopyObservationPredictiveGate);
    EXPECT_TRUE(twice.associationResidualWarmStartUseGeometry);
    EXPECT_TRUE(twice.associationResidualUseTransformAlignment);
    EXPECT_TRUE(twice.associationSparseUpperSupport);
    EXPECT_EQ(twice.associationMaxCandidatesPerEdge, 20u);
    EXPECT_TRUE(twice.associationGaugeSynchronizationEnabled);
    EXPECT_TRUE(twice.associationPostUpdateLocalRefine);
    EXPECT_EQ(twice.associationPostUpdateLowerFixedPointRounds, 5u);
  }
}

TEST(testDPGO, ManualObjectDranFinalDbaAblationsDeleteOnlyTheirTarget) {
  struct AblationCase {
    ManualObjectDranFinalAblation ablation;
  };
  const std::vector<AblationCase> cases = {
      {ManualObjectDranFinalAblation::StateValidation},
      {ManualObjectDranFinalAblation::EdgeConsensus},
      {ManualObjectDranFinalAblation::ReciprocalSupport},
      {ManualObjectDranFinalAblation::PairSupportGate},
      {ManualObjectDranFinalAblation::PairSupportContinuation},
      {ManualObjectDranFinalAblation::SharedReliability},
      {ManualObjectDranFinalAblation::ObservationPredictiveGate},
  };

  for (const AblationCase &testCase : cases) {
    ManualObjectDranOptions options;
    options.associationPolicy =
        ManualObjectDranAssociationPolicy::FinalDbaP4203;
    options.associationFinalAblation = testCase.ablation;
    const ManualObjectDranOptions effective =
        applyManualObjectDranAssociationPolicy(options);

    const double expectedStateValidation =
        testCase.ablation == ManualObjectDranFinalAblation::StateValidation
            ? 0.0
            : 0.1;
    const double expectedEdgeConsensus =
        testCase.ablation == ManualObjectDranFinalAblation::EdgeConsensus
            ? 0.0
            : 0.05;
    const double expectedReciprocalSupport =
        testCase.ablation == ManualObjectDranFinalAblation::ReciprocalSupport
            ? 0.0
            : 1.0;
    const double expectedPairSupportGate =
        testCase.ablation == ManualObjectDranFinalAblation::PairSupportGate
            ? 0.0
            : 2.0;
    const bool expectedContinuation =
        testCase.ablation !=
        ManualObjectDranFinalAblation::PairSupportContinuation;
    const bool expectedSharedReliability =
        testCase.ablation != ManualObjectDranFinalAblation::SharedReliability;
    const double expectedSharedReliabilityUpperCost =
        expectedSharedReliability ? 1.0 : 0.0;
    const bool expectedObservationPredictiveGate =
        testCase.ablation !=
        ManualObjectDranFinalAblation::ObservationPredictiveGate;

    EXPECT_DOUBLE_EQ(effective.associationStateValidationHypergradientWeight,
                     expectedStateValidation);
    EXPECT_DOUBLE_EQ(effective.associationEdgeLatentConsensusWeight,
                     expectedEdgeConsensus);
    EXPECT_DOUBLE_EQ(effective.associationReciprocalFreshSupportBlend,
                     expectedReciprocalSupport);
    EXPECT_DOUBLE_EQ(effective.associationPairSupportGatePower,
                     expectedPairSupportGate);
    EXPECT_EQ(effective.associationPairSupportLatentContinuation,
              expectedContinuation);
    EXPECT_EQ(effective.associationObjectReliabilityUseSharedState,
              expectedSharedReliability);
    EXPECT_DOUBLE_EQ(effective.associationObjectReliabilityUpperCostWeight,
                     expectedSharedReliabilityUpperCost);
    EXPECT_EQ(effective.associationLowerObjectCopyObservationPredictiveGate,
              expectedObservationPredictiveGate);

    EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
    EXPECT_TRUE(effective.associationResidualUseTransformAlignment);
    EXPECT_TRUE(effective.associationGaugeSynchronizationEnabled);
    EXPECT_TRUE(effective.associationPostUpdateLocalRefine);
    EXPECT_EQ(effective.associationPostUpdateLowerFixedPointRounds, 5u);
    EXPECT_EQ(effective.associationMaxCandidatesPerEdge, 20u);
  }
}

TEST(testDPGO, ManualObjectDranFinalDbaAblationsRequireFinalPolicy) {
  ManualObjectDranOptions options;
  options.associationPolicy = ManualObjectDranAssociationPolicy::Manual;
  options.associationFinalAblation =
      ManualObjectDranFinalAblation::EdgeConsensus;

  EXPECT_THROW(applyManualObjectDranAssociationPolicy(options),
               std::invalid_argument);
}

TEST(testDPGO, ManualObjectDranFinalDbaAblationApplicationIsIdempotent) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::FinalDbaP4203;
  options.associationFinalAblation =
      ManualObjectDranFinalAblation::SharedReliability;

  const ManualObjectDranOptions once =
      applyManualObjectDranAssociationPolicy(options);
  const ManualObjectDranOptions twice =
      applyManualObjectDranAssociationPolicy(once);

  EXPECT_EQ(twice.associationFinalAblation, once.associationFinalAblation);
  EXPECT_EQ(twice.diagnosticProfile, once.diagnosticProfile);
  EXPECT_DOUBLE_EQ(twice.associationStateValidationHypergradientWeight,
                   once.associationStateValidationHypergradientWeight);
  EXPECT_DOUBLE_EQ(twice.associationEdgeLatentConsensusWeight,
                   once.associationEdgeLatentConsensusWeight);
  EXPECT_DOUBLE_EQ(twice.associationReciprocalFreshSupportBlend,
                   once.associationReciprocalFreshSupportBlend);
  EXPECT_DOUBLE_EQ(twice.associationPairSupportGatePower,
                   once.associationPairSupportGatePower);
  EXPECT_EQ(twice.associationPairSupportLatentContinuation,
            once.associationPairSupportLatentContinuation);
  EXPECT_EQ(twice.associationObjectReliabilityUseSharedState,
            once.associationObjectReliabilityUseSharedState);
  EXPECT_DOUBLE_EQ(twice.associationObjectReliabilityUpperCostWeight,
                   once.associationObjectReliabilityUpperCostWeight);
  EXPECT_EQ(twice.associationLowerObjectCopyObservationPredictiveGate,
            once.associationLowerObjectCopyObservationPredictiveGate);
  EXPECT_TRUE(twice.associationGaugeSynchronizationEnabled);
  EXPECT_EQ(twice.associationPostUpdateLowerFixedPointRounds, 5u);
}

TEST(testDPGO, ManualObjectDranStateWindowsSkipInactiveMechanisms) {
  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;

  EXPECT_FALSE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranStateWindowsKeepActivationWindows) {
  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationActiveMinStableUpdates = 1;

  EXPECT_TRUE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranStateWindowsSkipFixedFileActivationOnly) {
  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationActiveMinStableUpdates = 1;

  EXPECT_FALSE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranStateWindowsKeepFixedFileRejectionWindows) {
  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationRejectMinBadUpdates = 2;

  options.associationRejectResidualThreshold = 0.5;
  EXPECT_TRUE(associationStateWindowsCanAffectConsensus(options));

  options.associationRejectResidualThreshold = 0.0;
  options.associationRejectResidualBestMargin = 0.25;
  EXPECT_TRUE(associationStateWindowsCanAffectConsensus(options));

  options.associationRejectResidualBestMargin = 0.0;
  options.associationRejectResidualQuantile = 0.9;
  EXPECT_TRUE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranStateWindowsSkipTransformSoftRejection) {
  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::TransformSoft;
  options.associationRejectMinBadUpdates = 2;
  options.associationRejectResidualThreshold = 0.5;

  EXPECT_FALSE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranStateWindowsIgnoreIncompleteRejectionConfig) {
  ManualObjectDranOptions options;
  options.associationRejectMinBadUpdates = 2;

  EXPECT_FALSE(associationStateWindowsCanAffectConsensus(options));
}

TEST(testDPGO, ManualObjectDranEstimatesAssociationTermReserveCapacity) {
  ManualObjectDranOptions options;
  EXPECT_EQ(estimatedObjectAssociationTermReserveCapacity(0u, options), 0u);
  EXPECT_EQ(estimatedObjectAssociationTermReserveCapacity(12u, options),
            144u);

  options.associationMaxCandidatesPerEdge = 3u;
  EXPECT_EQ(estimatedObjectAssociationTermReserveCapacity(12u, options),
            36u);

  options.associationSinkhornMaxNonzeros = 17u;
  EXPECT_EQ(estimatedObjectAssociationTermReserveCapacity(12u, options),
            17u);

  options.associationMaxCandidatesPerEdge = 20u;
  options.associationSinkhornMaxNonzeros = 0u;
  EXPECT_EQ(estimatedObjectAssociationTermReserveCapacity(12u, options),
            144u);
}

TEST(testDPGO, ManualObjectDranAppendAssociationTermReserveKeepsExistingTerms) {
  ManualObjectDranOptions options;
  options.associationMaxCandidatesPerEdge = 4u;
  std::vector<ObjectCopyConsistencyTerm> terms;
  terms.reserve(1u);
  ObjectCopyConsistencyTerm existing;
  existing.localObjectId = 7u;
  existing.targetObjectId = 8u;
  terms.push_back(existing);

  reserveAdditionalObjectAssociationTerms(terms, 10u, options);

  ASSERT_EQ(terms.size(), 1u);
  EXPECT_EQ(terms.front().localObjectId, 7u);
  EXPECT_EQ(terms.front().targetObjectId, 8u);
  EXPECT_GE(terms.capacity(), 41u);
  EXPECT_LT(terms.capacity(), 101u);
}

TEST(testDPGO,
     ManualObjectDranPhysicalCopyMajorizationFactorUsesSymmetricInformation) {
  ObjectCopyConsistencyTerm forward;
  forward.associationProbability = 0.37;
  forward.weight = 2.5;
  forward.reliability = 0.42;
  forward.localPrecisionScale = 3.0;
  forward.targetPrecisionScale = 4.0;
  forward.localInformation = ObjectCopyChannelInformation{9.0, 25.0};
  forward.targetInformation = ObjectCopyChannelInformation{36.0, 100.0};
  forward.robustResidualScale = 7.0;
  forward.robustMinWeight = 0.2;
  forward.robustLatentTarget = 0.3;
  forward.robustLatentValue = 0.4;

  ObjectCopyConsistencyTerm reverse = forward;
  std::swap(reverse.localInformation, reverse.targetInformation);
  configurePhysicalObjectCopyMajorizationFactor(forward);
  configurePhysicalObjectCopyMajorizationFactor(reverse);

  EXPECT_DOUBLE_EQ(forward.associationProbability, 0.37);
  EXPECT_DOUBLE_EQ(reverse.associationProbability, 0.37);
  EXPECT_DOUBLE_EQ(forward.weight, 2.5);
  EXPECT_DOUBLE_EQ(reverse.weight, 2.5);
  EXPECT_DOUBLE_EQ(forward.reliability, 0.42);
  EXPECT_DOUBLE_EQ(reverse.reliability, 0.42);
  EXPECT_DOUBLE_EQ(forward.localPrecisionScale, 1.0);
  EXPECT_DOUBLE_EQ(forward.targetPrecisionScale, 1.0);
  EXPECT_TRUE(forward.useChannelInformation);
  EXPECT_TRUE(reverse.useChannelInformation);
  EXPECT_DOUBLE_EQ(forward.localInformation.rotation,
                   reverse.localInformation.rotation);
  EXPECT_DOUBLE_EQ(forward.localInformation.translation,
                   reverse.localInformation.translation);
  EXPECT_DOUBLE_EQ(forward.targetInformation.rotation,
                   reverse.targetInformation.rotation);
  EXPECT_DOUBLE_EQ(forward.targetInformation.translation,
                   reverse.targetInformation.translation);
  EXPECT_DOUBLE_EQ(forward.localInformation.rotation,
                   forward.targetInformation.rotation);
  EXPECT_DOUBLE_EQ(forward.localInformation.translation,
                   forward.targetInformation.translation);
  EXPECT_DOUBLE_EQ(forward.robustResidualScale, 0.0);
  EXPECT_DOUBLE_EQ(forward.robustMinWeight, 0.0);
  EXPECT_DOUBLE_EQ(forward.robustLatentTarget, 1.0);
  EXPECT_DOUBLE_EQ(forward.robustLatentValue, 1.0);
}

TEST(testDPGO,
     ManualObjectDranCurrentRoundPhysicalCopyMmAcceptanceRejectsAscentOnly) {
  QuadraticProblem problem(1u, 2u, 2u);
  SparseMatrix Q(3, 3);
  SparseMatrix G(2, 3);
  G.coeffRef(0, 2) = -1.0;
  problem.setQ(Q);
  problem.setG(G);

  const Matrix anchor = Matrix::Zero(2, 3);
  Matrix descent = anchor;
  descent(0, 2) = 1.0;
  Matrix equal = anchor;
  Matrix ascent = anchor;
  ascent(0, 2) = -1.0;
  Matrix nonfinite = descent;
  nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();

  EXPECT_EQ(
      manualObjectDranAcceptCurrentRoundPhysicalCopyMajorizedLocalCandidateForTest(
          problem, anchor, descent),
      descent);
  EXPECT_EQ(
      manualObjectDranAcceptCurrentRoundPhysicalCopyMajorizedLocalCandidateForTest(
          problem, anchor, equal),
      equal);
  EXPECT_EQ(
      manualObjectDranAcceptCurrentRoundPhysicalCopyMajorizedLocalCandidateForTest(
          problem, anchor, ascent),
      anchor);
  EXPECT_EQ(
      manualObjectDranAcceptCurrentRoundPhysicalCopyMajorizedLocalCandidateForTest(
          problem, anchor, nonfinite),
      anchor);
}

TEST(testDPGO,
     ManualObjectDranPhysicalCopyMajorizationIgnoresLegacyScalarGates) {
  Matrix association(2, 2);
  association << 0.4, 0.2, 0.1, 0.3;
  const Matrix stableWindows = Matrix::Zero(2, 2);
  const Matrix rejectedWindows = Matrix::Ones(2, 2);
  const double stiffness = 7.0;

  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::Disabled;
  options.associationLowerPhysicalCopyMajorization = true;
  options.associationConsensusWeightPower = 2.0;
  options.associationActiveMinStableUpdates = 5u;
  options.associationActivationTrustRampUpdates = 10u;
  options.associationTentativeWeightScale = 0.0;
  options.associationLowerHighConfidenceAnchorGain = 100.0;
  options.associationLowerContinuousSupportGatePower = 3.0;
  options.associationLowerUseSupportConfidenceReliability = true;
  options.associationPairValidationTrustEnabled = true;
  options.associationObjectReliabilityResidualScale = 0.25;
  options.associationObjectObservationUncertaintyCountPrior = 10.0;
  options.associationObjectObservationUncertaintyDistanceScale = 10.0;
  options.associationObjectObservationUncertaintyMinReliability = 0.01;

  const std::vector<ObjectCopyConsistencyTerm> terms =
      manualObjectDranPhysicalCopyMajorizationTermsForTest(
          association, stableWindows, rejectedWindows, 0.01, stiffness,
          options);

  ASSERT_EQ(terms.size(), 4u);
  for (const ObjectCopyConsistencyTerm &term : terms) {
    const double p =
        association(static_cast<int>(term.localObjectId),
                    static_cast<int>(term.targetObjectId));
    EXPECT_DOUBLE_EQ(term.weight, stiffness * p * p);
    EXPECT_DOUBLE_EQ(term.reliability, 1.0);
    EXPECT_TRUE(term.useChannelInformation);
  }
}

TEST(testDPGO,
     ManualObjectDranColoredPhysicalCopyUsesSymmetricFactorsWithoutJacobiProx) {
  Matrix association(2, 2);
  association << 0.4, 0.2, 0.1, 0.3;
  const Matrix stableWindows = Matrix::Zero(2, 2);
  const Matrix rejectedWindows = Matrix::Ones(2, 2);
  const double stiffness = 7.0;

  ManualObjectDranOptions options;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::Disabled;
  options.associationConsensusWeightPower = 2.0;
  options.associationActiveMinStableUpdates = 5u;
  options.associationActivationTrustRampUpdates = 10u;
  options.associationTentativeWeightScale = 0.0;
  options.associationLowerHighConfidenceAnchorGain = 100.0;
  options.associationLowerContinuousSupportGatePower = 3.0;
  options.associationLowerUseSupportConfidenceReliability = true;
  options.associationPairValidationTrustEnabled = true;
  options.associationObjectReliabilityResidualScale = 0.25;
  options.associationObjectObservationUncertaintyCountPrior = 10.0;
  options.associationObjectObservationUncertaintyDistanceScale = 10.0;
  options.associationObjectObservationUncertaintyMinReliability = 0.01;

  options.associationLowerPhysicalCopyColoredBcd = true;
  const std::vector<ObjectCopyConsistencyTerm> coloredTerms =
      manualObjectDranPhysicalCopyMajorizationTermsForTest(
          association, stableWindows, rejectedWindows, 0.01, stiffness,
          options);
  const ManualObjectDranLowerQuadraticStatsForTest colored =
      manualObjectDranPhysicalCopyLowerQuadraticStatsForTest(
          association, stableWindows, rejectedWindows, 0.01, stiffness,
          options);

  options.associationLowerPhysicalCopyMajorization = true;
  const ManualObjectDranLowerQuadraticStatsForTest bothFlags =
      manualObjectDranPhysicalCopyLowerQuadraticStatsForTest(
          association, stableWindows, rejectedWindows, 0.01, stiffness,
          options);

  options.associationLowerPhysicalCopyColoredBcd = false;
  const ManualObjectDranLowerQuadraticStatsForTest jacobi =
      manualObjectDranPhysicalCopyLowerQuadraticStatsForTest(
          association, stableWindows, rejectedWindows, 0.01, stiffness,
          options);

  ASSERT_EQ(coloredTerms.size(), 4u);
  for (const ObjectCopyConsistencyTerm &term : coloredTerms) {
    const double p =
        association(static_cast<int>(term.localObjectId),
                    static_cast<int>(term.targetObjectId));
    EXPECT_DOUBLE_EQ(term.weight, stiffness * p * p);
    EXPECT_DOUBLE_EQ(term.reliability, 1.0);
    EXPECT_TRUE(term.useChannelInformation);
  }
  EXPECT_EQ(colored.activePairs, 4u);
  EXPECT_GT(colored.consensusCost, 0.0);
  EXPECT_DOUBLE_EQ(colored.consensusCost, bothFlags.consensusCost);
  EXPECT_DOUBLE_EQ(colored.qSquaredNorm, bothFlags.qSquaredNorm);
  EXPECT_DOUBLE_EQ(colored.gSquaredNorm, bothFlags.gSquaredNorm);
  EXPECT_GT(jacobi.qSquaredNorm, colored.qSquaredNorm);
  EXPECT_NE(jacobi.gSquaredNorm, colored.gSquaredNorm);
}

TEST(testDPGO,
     ManualObjectDranColoredPhysicalCopySweepUsesFreshPriorColorCopies) {
  const std::filesystem::path dataDir =
      makeTinyReciprocalPhysicalCopyDataDir(
          "manual_object_dran_colored_physical_copy_sweep");

  ManualObjectDranOptions options =
      tinyReciprocalPhysicalCopyOptions(dataDir);
  options.associationLowerPhysicalCopyColoredBcd = true;

  ManualObjectDranOptions initialOptions = options;
  initialOptions.maxIterations = 0;
  const ManualObjectDranRunResult initial =
      runManualObjectDran(dataDir.string(), initialOptions);
  const ManualObjectDranRunResult colored =
      runManualObjectDran(dataDir.string(), options);
  ManualObjectDranOptions simultaneousOptions = options;
  simultaneousOptions.associationLowerPhysicalCopyColoredBcd = false;
  const ManualObjectDranRunResult simultaneous =
      runManualObjectDran(dataDir.string(), simultaneousOptions);
  std::filesystem::remove_all(dataDir);

  ASSERT_EQ(initial.iterations.size(), 1u);
  ASSERT_EQ(colored.iterations.size(), 2u);
  ASSERT_EQ(simultaneous.iterations.size(), 2u);
  ASSERT_EQ(colored.finalEstimates.size(), 2u);
  ASSERT_EQ(simultaneous.finalEstimates.size(), 2u);

  const double initialObjective =
      initial.iterations.back().measurementCost +
      0.5 * initial.iterations.back().consensusCost;
  const double coloredObjective =
      colored.iterations.back().measurementCost +
      0.5 * colored.iterations.back().consensusCost;
  ASSERT_TRUE(std::isfinite(initialObjective));
  ASSERT_TRUE(std::isfinite(coloredObjective));
  EXPECT_LE(coloredObjective, initialObjective + 1e-8);

  const Vector coloredObject0 = colored.finalEstimates[0].block(0, 7, 3, 1);
  const Vector coloredObject1 = colored.finalEstimates[1].block(0, 7, 3, 1);
  const Vector simultaneousObject0 =
      simultaneous.finalEstimates[0].block(0, 7, 3, 1);
  const Vector simultaneousObject1 =
      simultaneous.finalEstimates[1].block(0, 7, 3, 1);
  EXPECT_GT((coloredObject1 - simultaneousObject1).norm(), 1e-4);
  EXPECT_LT((coloredObject1 - coloredObject0).norm(),
            (simultaneousObject1 - simultaneousObject0).norm());
}

TEST(testDPGO,
     ManualObjectDranPhysicalCopyFalseFlagMatchesDefaultTinyRunExactly) {
  const std::filesystem::path dataDir =
      makeTinyReciprocalPhysicalCopyDataDir(
          "manual_object_dran_physical_copy_false_flag_default");
  const ManualObjectDranOptions defaultOptions =
      tinyReciprocalPhysicalCopyOptions(dataDir);
  ManualObjectDranOptions explicitFalseOptions = defaultOptions;
  explicitFalseOptions.associationLowerPhysicalCopyColoredBcd = false;

  const ManualObjectDranRunResult defaultRun =
      runManualObjectDran(dataDir.string(), defaultOptions);
  const ManualObjectDranRunResult explicitFalseRun =
      runManualObjectDran(dataDir.string(), explicitFalseOptions);
  std::filesystem::remove_all(dataDir);

  ASSERT_EQ(defaultRun.finalEstimates.size(),
            explicitFalseRun.finalEstimates.size());
  for (std::size_t robotId = 0; robotId < defaultRun.finalEstimates.size();
       ++robotId) {
    EXPECT_EQ(defaultRun.finalEstimates[robotId],
              explicitFalseRun.finalEstimates[robotId]);
  }
  ASSERT_EQ(defaultRun.iterations.size(), explicitFalseRun.iterations.size());
  EXPECT_DOUBLE_EQ(defaultRun.iterations.back().measurementCost,
                   explicitFalseRun.iterations.back().measurementCost);
  EXPECT_DOUBLE_EQ(defaultRun.iterations.back().consensusCost,
                   explicitFalseRun.iterations.back().consensusCost);
}

TEST(testDPGO, ManualObjectDranBuildsObjectObservationCounts) {
  ObjectPGORobotData robot;
  robot.trajectoryVertexIds = {0, 1};
  const auto makeObservation = [](std::size_t objectLocalIndex) {
    ObjectObservationMeasurement obs;
    obs.objectLocalIndex = objectLocalIndex;
    obs.measurement.r1 = 0;
    obs.measurement.r2 = 0;
    obs.measurement.p1 = 0;
    obs.measurement.p2 = 0;
    obs.measurement.R = Matrix::Identity(3, 3);
    obs.measurement.t = Vector::Zero(3);
    obs.measurement.kappa = 1.0;
    obs.measurement.tau = 1.0;
    obs.measurement.isKnownInlier = false;
    obs.measurement.weight = 1.0;
    return obs;
  };
  robot.objectObservationMeasurements = {
      makeObservation(2), makeObservation(2), makeObservation(4)};

  const std::vector<std::size_t> counts =
      objectObservationCounts(robot, 6u);

  ASSERT_EQ(counts.size(), 6u);
  EXPECT_EQ(counts[0], 0u);
  EXPECT_EQ(counts[2], 2u);
  EXPECT_EQ(counts[4], 1u);
}

TEST(testDPGO, ManualObjectDranExtractsActiveAssociationEntries) {
  Matrix weights = Matrix::Zero(3, 4);
  weights(0, 1) = 0.2;
  weights(1, 0) = std::numeric_limits<double>::quiet_NaN();
  weights(1, 3) = 1e-14;
  weights(2, 2) = 0.7;

  const std::vector<ManualObjectDranAssociationEntry> entries =
      activeAssociationEntries(weights, 1e-12);

  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].row, 0u);
  EXPECT_EQ(entries[0].col, 1u);
  EXPECT_DOUBLE_EQ(entries[0].weight, 0.2);
  EXPECT_EQ(entries[1].row, 2u);
  EXPECT_EQ(entries[1].col, 2u);
  EXPECT_DOUBLE_EQ(entries[1].weight, 0.7);
}

TEST(testDPGO, ManualObjectDranExtractsFiniteAssociationCandidateEdges) {
  Matrix supportCosts = Matrix::Constant(
      3, 4, std::numeric_limits<double>::infinity());
  supportCosts(0, 2) = 1.25;
  supportCosts(1, 0) = 0.5;
  supportCosts(2, 3) = 2.0;
  supportCosts(2, 1) = std::numeric_limits<double>::quiet_NaN();

  const std::vector<ManualObjectDranAssociationCandidateEdge> edges =
      finiteAssociationCandidateEdges(supportCosts);

  ASSERT_EQ(edges.size(), 3u);
  EXPECT_EQ(edges[0].row, 0u);
  EXPECT_EQ(edges[0].col, 2u);
  EXPECT_DOUBLE_EQ(edges[0].cost, 1.25);
  EXPECT_EQ(edges[1].row, 1u);
  EXPECT_EQ(edges[1].col, 0u);
  EXPECT_DOUBLE_EQ(edges[1].cost, 0.5);
  EXPECT_EQ(edges[2].row, 2u);
  EXPECT_EQ(edges[2].col, 3u);
  EXPECT_DOUBLE_EQ(edges[2].cost, 2.0);
}

TEST(testDPGO, ManualObjectDranSemanticMismatchLowerBoundPrunesOnlyImpossible) {
  ManualObjectDranOptions options;
  options.associationSemanticWeight = 1.25;
  options.associationTransformMinScore = 0.35;

  EXPECT_DOUBLE_EQ(semanticAssociationMismatchLowerBoundCost(3u, 1.0, 3u, 1.0,
                                                             options),
                   0.0);
  EXPECT_FALSE(semanticAssociationMismatchCannotPassMinScore(
      3u, 1.0, 3u, 1.0, options));

  EXPECT_DOUBLE_EQ(semanticAssociationMismatchLowerBoundCost(3u, 1.0, 4u, 1.0,
                                                             options),
                   1.25);
  EXPECT_TRUE(semanticAssociationMismatchCannotPassMinScore(
      3u, 1.0, 4u, 1.0, options));

  EXPECT_DOUBLE_EQ(semanticAssociationMismatchLowerBoundCost(3u, 0.1, 4u, 0.1,
                                                             options),
                   0.125);
  EXPECT_FALSE(semanticAssociationMismatchCannotPassMinScore(
      3u, 0.1, 4u, 0.1, options));

  options.associationSemanticWeight = 0.0;
  EXPECT_FALSE(semanticAssociationMismatchCannotPassMinScore(
      3u, 1.0, 4u, 1.0, options));
  options.associationSemanticWeight = 1.25;
  options.associationTransformMinScore = 0.0;
  EXPECT_FALSE(semanticAssociationMismatchCannotPassMinScore(
      3u, 1.0, 4u, 1.0, options));
}

TEST(testDPGO, ManualObjectDranComputesAssociationSupportCoverage) {
  Matrix supportCosts = Matrix::Constant(
      2, 3, std::numeric_limits<double>::infinity());
  supportCosts(0, 0) = 1.0;
  supportCosts(1, 2) = 0.5;
  Matrix assignment = Matrix::Zero(2, 3);
  assignment(0, 0) = 0.25;
  assignment(0, 1) = 0.25;
  assignment(1, 2) = 0.5;

  const ManualObjectDranAssociationSupportStats stats =
      associationFiniteSupportStats(supportCosts, assignment);

  EXPECT_EQ(stats.supportEntries, 2u);
  EXPECT_EQ(stats.denseEntries, 6u);
  EXPECT_DOUBLE_EQ(stats.assignmentMass, 1.0);
  EXPECT_DOUBLE_EQ(stats.coveredAssignmentMass, 0.75);
  EXPECT_DOUBLE_EQ(stats.density, 2.0 / 6.0);
  EXPECT_DOUBLE_EQ(stats.massCoverage, 0.75);
}

TEST(testDPGO, ManualObjectDranRecordsRuntimeProfileBreakdown) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationTransformMinInliers = 100000;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_GE(result.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &summary = result.iterations.back();
  EXPECT_GE(summary.profileRunSetupSeconds, 0.0);
  EXPECT_GE(summary.profileDataLoadSeconds, 0.0);
  EXPECT_GE(summary.profileGraphInitializationSeconds, 0.0);
  EXPECT_GE(summary.profileInitialAssociationSeconds, 0.0);
  EXPECT_GE(summary.profileInitialAssociationBootstrapSeconds, 0.0);
  EXPECT_GE(summary.profileInitialAssociationStateWindowSeconds, 0.0);
  EXPECT_GE(summary.profileInitialGaugeReliabilitySeconds, 0.0);
  EXPECT_GE(summary.profileLowerInnerRoundsSeconds, 0.0);
  EXPECT_GE(summary.profileLowerFactorAssemblySeconds, 0.0);
  EXPECT_GE(summary.profileLowerSolveSeconds, 0.0);
  EXPECT_GE(summary.profileLowerFeedbackCertificateSeconds, 0.0);
  EXPECT_GE(summary.postUpdateLowerRoundsExecuted, 0u);
  EXPECT_GE(summary.postUpdateLowerRobotSolves, 0u);
  EXPECT_GE(summary.postUpdateLowerConvergedRobots, 0u);
  EXPECT_GE(summary.postUpdateLowerMaxNormalizedResidual, 0.0);
  EXPECT_GE(summary.postUpdateLowerMaxStationarityResidual, 0.0);
  EXPECT_GE(summary.postUpdateLowerMaxOuterContinuationResidual, 0.0);
  EXPECT_GE(summary.postUpdateLowerMaxIncidentPrimalResidual, 0.0);
  EXPECT_GE(summary.postUpdateLowerMaxIncidentDualResidual, 0.0);
  EXPECT_GE(summary.lowerSharedObjectConsensusNormalizedPrimalResidual, 0.0);
  EXPECT_GE(summary.lowerSharedObjectConsensusNormalizedDualResidual, 0.0);
  EXPECT_GE(summary.lowerSharedObjectConsensusCertificateMessages, 0u);
  EXPECT_GE(summary.lowerSharedObjectConsensusStaleCertificates, 0u);
  EXPECT_GE(summary.lowerSharedObjectConsensusCertificatePayloadBytes, 0u);
  EXPECT_GE(summary.profileAssociationUpdateSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationUnprofiledSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationWeightUpdateSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationWeightUpdateUnprofiledSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationLocalCostSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationStateValidationPrecomputeSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationStateValidationCostSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationEdgeLatentConsensusSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationReferenceMassTrustSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationTemporalStatsSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationSupportStatsSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationTransformRefreshSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationCycleConsistencySeconds, 0.0);
  EXPECT_GE(summary.profileAssociationStateWindowSeconds, 0.0);
  EXPECT_GE(summary.profileObjectInterfaceResponseSeconds, 0.0);
  EXPECT_GE(summary.profilePairValidationSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationRotoTranslationSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationProjectionSeconds, 0.0);
  EXPECT_GE(summary.profileAssociationSupportPostprocessSeconds, 0.0);
  EXPECT_LE(summary.profileAssociationSupportEntries,
            summary.profileAssociationDenseEntries);
  EXPECT_GE(summary.profileAssociationSupportDensity, 0.0);
  EXPECT_LE(summary.profileAssociationSupportDensity, 1.0);
  EXPECT_GE(summary.profileAssociationSupportMassCoverage, 0.0);
  EXPECT_LE(summary.profileAssociationSupportMassCoverage, 1.0);
  EXPECT_GE(summary.profileAssociationPreCostSupportSeconds, 0.0);
  EXPECT_LE(summary.profileAssociationPreCostSupportEntries,
            summary.profileAssociationPreCostDenseEntries);
  EXPECT_GE(summary.profileAssociationPreCostSupportDensity, 0.0);
  EXPECT_LE(summary.profileAssociationPreCostSupportDensity, 1.0);
  EXPECT_GE(summary.profileAssociationPreCostSupportMassCoverage, 0.0);
  EXPECT_LE(summary.profileAssociationPreCostSupportMassCoverage, 1.0);
  EXPECT_GE(summary.profileObjectExchangeSeconds, 0.0);
  EXPECT_GE(summary.profileGaugeReliabilityUpdateSeconds, 0.0);
  EXPECT_GE(summary.profilePostUpdateRefineSeconds, 0.0);
  EXPECT_GE(summary.profileSummaryFactorAssemblySeconds, 0.0);
  EXPECT_GE(summary.profileSummaryAssociationDiagnosticsSeconds, 0.0);
  EXPECT_GT(summary.profileLowerFactorAssemblySeconds +
                summary.profileLowerSolveSeconds +
                summary.profileAssociationLocalCostSeconds +
                summary.profileAssociationProjectionSeconds +
                summary.profileSummaryFactorAssemblySeconds +
                summary.profileSummaryAssociationDiagnosticsSeconds,
            0.0);
}

TEST(testDPGO, ManualObjectDranExtractsObjectBlockStiffnesses) {
  const unsigned d = 3;
  const std::size_t trajectoryPoses = 2;
  const std::size_t numObjects = 3;
  const unsigned blockWidth = d + 1;
  SparseMatrix Q(static_cast<int>((trajectoryPoses + numObjects) * blockWidth),
                 static_cast<int>((trajectoryPoses + numObjects) * blockWidth));
  std::vector<Eigen::Triplet<double>> triplets;
  for (std::size_t objectId = 0; objectId < numObjects; ++objectId) {
    const std::size_t block = trajectoryPoses + objectId;
    for (unsigned c = 0; c < blockWidth; ++c) {
      const double value =
          (objectId == 1 && c == 2) ? -5.0 : static_cast<double>(objectId + c + 1);
      const int col = static_cast<int>(block * blockWidth + c);
      triplets.emplace_back(col, col, value);
    }
  }
  Q.setFromTriplets(triplets.begin(), triplets.end());

  const std::vector<double> stiffnesses =
      objectBlockDiagonalStiffnesses(Q, trajectoryPoses, numObjects, d);

  ASSERT_EQ(stiffnesses.size(), numObjects);
  EXPECT_DOUBLE_EQ(stiffnesses[0], 10.0);
  EXPECT_DOUBLE_EQ(stiffnesses[1], 15.0);
  EXPECT_DOUBLE_EQ(stiffnesses[2], 18.0);

  const std::vector<ObjectCopyChannelInformation> information =
      objectBlockDiagonalChannelInformation(Q, trajectoryPoses, numObjects, d);

  ASSERT_EQ(information.size(), numObjects);
  EXPECT_DOUBLE_EQ(information[0].rotation, 2.0);
  EXPECT_DOUBLE_EQ(information[0].translation, 4.0);
  EXPECT_DOUBLE_EQ(information[1].rotation, 10.0 / 3.0);
  EXPECT_DOUBLE_EQ(information[1].translation, 5.0);
  EXPECT_DOUBLE_EQ(information[2].rotation, 4.0);
  EXPECT_DOUBLE_EQ(information[2].translation, 6.0);
}

TEST(testDPGO, ManualObjectDranFinalDbaAllowsExplicitFullDiagnostics) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::FinalDbaP4203;
  options.diagnosticProfile = ManualObjectDranDiagnosticProfile::Full;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_EQ(effective.diagnosticProfile, ManualObjectDranDiagnosticProfile::Full);
  EXPECT_TRUE(effective.recordObjectInterfaceDiagnostics);
  EXPECT_TRUE(effective.recordDetailedAssociationDiagnostics);
  EXPECT_TRUE(effective.recordPredictiveMeasurementCost);
}

TEST(testDPGO,
     ManualObjectDranLowerValidationReferenceTrustProducesFeedbackWithoutBaseWeight) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = false;
  options.associationResidualTrustPreviousSupport = true;
  options.associationRotoTranslationAlignmentWeight = 0.2;
  options.associationRotoTranslationHypothesisTopK = 3;
  options.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  options.associationLowerValidationWeight = 0.0;
  options.associationLowerValidationReferenceTrustWeight = 0.2;
  options.associationLowerValidationScaleWithUpperCost = false;
  options.associationTransformMinInliers = 100000;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationActualLowerValidationAlpha,
                   0.0);
  EXPECT_GT(result.iterations.front().associationLowerValidationGradientNorm,
            0.0);
}

TEST(testDPGO,
     ManualObjectDranSingleLoopRotoPolicyCanAllowSupportExpansion) {
  ManualObjectDranOptions options;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  options.associationResidualAllowSupportExpansion = true;

  const ManualObjectDranOptions effective =
      applyManualObjectDranAssociationPolicy(options);

  EXPECT_TRUE(effective.associationResidualWarmStartUseGeometry);
  EXPECT_TRUE(effective.associationResidualUseTransformAlignment);
  EXPECT_TRUE(effective.associationResidualTransformBootstrap);
  EXPECT_FALSE(effective.associationResidualTrustPreviousSupport);
}

TEST(testDPGO,
     ManualObjectDranLowerValidationUsesSoftRotoTransformWhenHardTransformFails) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = false;
  options.associationResidualTrustPreviousSupport = true;
  options.associationRotoTranslationAlignmentWeight = 0.2;
  options.associationRotoTranslationHypothesisTopK = 3;
  options.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  options.associationLowerValidationWeight = 0.1;
  options.associationLowerValidationScaleWithUpperCost = false;
  options.associationTransformMinInliers = 100000;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationActualLowerValidationAlpha,
                   0.1);
  EXPECT_GT(result.iterations.front().associationLowerValidationGradientNorm,
            0.0);
}

TEST(testDPGO,
     ManualObjectDranImplicitValidationAdjointProducesSeparateFeedback) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationResidualTransformBootstrap = false;
  options.associationResidualTrustPreviousSupport = true;
  options.associationRotoTranslationAlignmentWeight = 0.2;
  options.associationRotoTranslationHypothesisTopK = 3;
  options.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  options.associationLowerValidationWeight = 0.0;
  options.associationLowerValidationNormalizeActiveRms = false;
  options.associationLowerValidationCenterRows = false;
  options.associationLowerValidationCenterColumns = false;
  options.associationImplicitValidationAdjointWeight = 0.1;
  options.associationImplicitValidationAdjointDamping = 1.0;
  options.associationTransformMinInliers = 100000;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  const ManualObjectDranIterationSummary &summary = result.iterations.front();
  EXPECT_DOUBLE_EQ(summary.associationActualLowerValidationAlpha, 0.0);
  EXPECT_DOUBLE_EQ(summary.associationActualImplicitValidationAdjointAlpha,
                   0.1);
  EXPECT_GT(summary.associationImplicitValidationAdjointGradientNorm, 0.0);
  EXPECT_GT(summary.associationImplicitValidationAdjointRhsGradientNorm, 0.0);
}

TEST(testDPGO,
     ManualObjectDranLowerValidationStationarityGateDownweightsUnsettledBackend) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 0;
  baseOptions.topology = ManualObjectDranTopology::Ring;
  baseOptions.ringHops = 1;
  baseOptions.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  baseOptions.associationResidualWarmStartUseGeometry = true;
  baseOptions.associationResidualUseTransformAlignment = true;
  baseOptions.associationResidualTransformBootstrap = false;
  baseOptions.associationResidualTrustPreviousSupport = true;
  baseOptions.associationRotoTranslationAlignmentWeight = 0.2;
  baseOptions.associationRotoTranslationHypothesisTopK = 3;
  baseOptions.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  baseOptions.associationLowerValidationWeight = 0.1;
  baseOptions.associationLowerValidationScaleWithUpperCost = false;
  baseOptions.associationTransformMinInliers = 100000;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult ungated =
      runManualObjectDran(dataDir, baseOptions);
  ManualObjectDranOptions gatedOptions = baseOptions;
  gatedOptions.associationLowerValidationStationarityTrustMaxGradientNorm =
      1e-12;
  const ManualObjectDranRunResult gated =
      runManualObjectDran(dataDir, gatedOptions);

  ASSERT_EQ(ungated.iterations.size(), 1u);
  ASSERT_EQ(gated.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(ungated.iterations.front().associationActualLowerValidationAlpha,
                   0.1);
  EXPECT_GT(ungated.iterations.front().associationLowerValidationGradientNorm,
            0.0);
  EXPECT_GT(gated.iterations.front().associationBackendStationarityGradientNorm,
            0.0);
  EXPECT_LT(gated.iterations.front().associationActualLowerValidationAlpha,
            ungated.iterations.front().associationActualLowerValidationAlpha);
  EXPECT_LT(gated.iterations.front().associationLowerValidationGradientNorm,
            ungated.iterations.front().associationLowerValidationGradientNorm);
}

TEST(testDPGO,
     ManualObjectDranReferenceSupportSharpeningPreservesBootstrapWhenUnsettled) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 1;
  baseOptions.topology = ManualObjectDranTopology::Ring;
  baseOptions.ringHops = 1;
  baseOptions.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  baseOptions.associationResidualWarmStartUseGeometry = true;
  baseOptions.associationResidualUseTransformAlignment = true;
  baseOptions.associationResidualTransformBootstrap = false;
  baseOptions.associationResidualTrustPreviousSupport = true;
  baseOptions.associationRotoTranslationAlignmentWeight = 0.2;
  baseOptions.associationRotoTranslationHypothesisTopK = 3;
  baseOptions.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  baseOptions.associationLowerValidationWeight = 0.1;
  baseOptions.associationLowerValidationScaleWithUpperCost = false;
  baseOptions.associationReferenceSupportSharpeningWeight = 0.12;
  baseOptions.associationReferenceSupportSharpeningTarget = 1.0;
  baseOptions.associationTransformMinInliers = 100000;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult ungated =
      runManualObjectDran(dataDir, baseOptions);
  ManualObjectDranOptions gatedOptions = baseOptions;
  gatedOptions.associationLowerValidationStationarityTrustMaxGradientNorm =
      1e-12;
  const ManualObjectDranRunResult gated =
      runManualObjectDran(dataDir, gatedOptions);

  ASSERT_EQ(ungated.iterations.size(), 2u);
  ASSERT_EQ(gated.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &ungatedSummary =
      ungated.iterations.back();
  const ManualObjectDranIterationSummary &gatedSummary =
      gated.iterations.back();
  EXPECT_NEAR(ungatedSummary.associationActualReferenceSupportSharpeningAlpha,
              0.12, 1e-12);
  EXPECT_GT(ungatedSummary.associationReferenceSupportSharpeningGradientNorm,
            0.0);
  EXPECT_GT(gatedSummary.associationBackendStationarityGradientNorm, 0.0);
  EXPECT_NEAR(gatedSummary.associationActualReferenceSupportSharpeningAlpha,
              ungatedSummary.associationActualReferenceSupportSharpeningAlpha,
              1e-12);
  EXPECT_GT(gatedSummary.associationReferenceSupportSharpeningGradientNorm,
            0.0);
}

TEST(testDPGO,
     ManualObjectDranPostAssociationRefineReducesUpdatedWeightGradient) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 1;
  baseOptions.topology = ManualObjectDranTopology::Ring;
  baseOptions.ringHops = 1;
  baseOptions.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  baseOptions.associationResidualWarmStartUseGeometry = true;
  baseOptions.associationResidualUseTransformAlignment = true;
  baseOptions.associationResidualTransformBootstrap = true;
  baseOptions.associationResidualAllowSupportExpansion = true;
  baseOptions.associationMaxCandidatesPerEdge = 8;
  baseOptions.associationRotoTranslationAlignmentWeight = 0.2;
  baseOptions.associationRotoTranslationHypothesisTopK = 3;
  baseOptions.associationRotoTranslationHypothesisSoftminTemperature = 0.1;
  baseOptions.associationLowerValidationWeight = 0.01;
  baseOptions.associationLowerValidationScaleWithUpperCost = true;
  baseOptions.associationReferenceTrustWeight = 0.2;
  baseOptions.associationReferenceSupportSharpeningWeight = 0.12;
  baseOptions.associationReferenceSupportSharpeningTarget = 1.0;
  baseOptions.associationTransformMinInliers = 100000;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult baseline =
      runManualObjectDran(dataDir, baseOptions);
  ManualObjectDranOptions refinedOptions = baseOptions;
  refinedOptions.associationPostUpdateLocalRefine = true;
  refinedOptions.associationPostUpdateLocalRefineIterations = 1;
  const ManualObjectDranRunResult refined =
      runManualObjectDran(dataDir, refinedOptions);

  ASSERT_EQ(baseline.iterations.size(), 2u);
  ASSERT_EQ(refined.iterations.size(), 2u);
  EXPECT_LT(refined.iterations.back().gradient,
            baseline.iterations.back().gradient);
}

TEST(testDPGO,
     ManualObjectDranFixedFileSoftTransformAlignmentReducesGtOracleGradient) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  const std::string associationFile = dataDir + "/gt_associations.csv";
  if (!std::filesystem::exists(dataDir) ||
      !std::filesystem::exists(associationFile)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO oracle data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 4;
  baseOptions.numObjects = 386;
  baseOptions.maxIterations = 0;
  baseOptions.topology = ManualObjectDranTopology::Complete;
  baseOptions.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  baseOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  baseOptions.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  baseOptions.associationFile = associationFile;
  baseOptions.associationRequireObservedObjects = true;
  baseOptions.associationTransformDimension = 0;
  baseOptions.associationTransformMinInliers = 8;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult unaligned =
      runManualObjectDran(dataDir, baseOptions);
  ManualObjectDranOptions alignedOptions = baseOptions;
  alignedOptions.associationResidualUseTransformAlignment = true;
  const ManualObjectDranRunResult aligned =
      runManualObjectDran(dataDir, alignedOptions);

  ASSERT_EQ(unaligned.iterations.size(), 1u);
  ASSERT_EQ(aligned.iterations.size(), 1u);
  EXPECT_LT(aligned.iterations.front().gradient,
            0.1 * unaligned.iterations.front().gradient);
  EXPECT_LT(aligned.iterations.front().consensusCost,
            0.1 * unaligned.iterations.front().consensusCost);
}

TEST(testDPGO,
     ManualObjectDranInitialObjectTransformSourceChangesFixedAlignment) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  const std::string associationFile = dataDir + "/gt_associations.csv";
  if (!std::filesystem::exists(dataDir) ||
      !std::filesystem::exists(associationFile)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO oracle data";
  }

  ManualObjectDranOptions stateOptions;
  stateOptions.numRobots = 4;
  stateOptions.numObjects = 386;
  stateOptions.maxIterations = 0;
  stateOptions.topology = ManualObjectDranTopology::Complete;
  stateOptions.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  stateOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  stateOptions.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  stateOptions.associationFile = associationFile;
  stateOptions.associationRequireObservedObjects = true;
  stateOptions.associationResidualUseTransformAlignment = true;
  stateOptions.associationTransformDimension = 0;
  stateOptions.associationTransformMinInliers = 8;
  stateOptions.localTrustRegionIterations = 1;
  stateOptions.localTrustRegionMaxInnerIterations = 2;

  ManualObjectDranOptions initialOptions = stateOptions;
  initialOptions.associationTransformUseInitialObjectPoses = true;

  const ManualObjectDranRunResult stateSource =
      runManualObjectDran(dataDir, stateOptions);
  const ManualObjectDranRunResult initialSource =
      runManualObjectDran(dataDir, initialOptions);

  ASSERT_EQ(stateSource.iterations.size(), 1u);
  ASSERT_EQ(initialSource.iterations.size(), 1u);
  EXPECT_TRUE(std::isfinite(initialSource.iterations.front().consensusCost));
  EXPECT_GT(std::abs(initialSource.iterations.front().consensusCost -
                     stateSource.iterations.front().consensusCost),
            1e-9);
}

TEST(testDPGO,
     ManualObjectDranInitialObjectTransformSourceStaysGaugeConsistentAfterOneStep) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  const std::string associationFile = dataDir + "/gt_associations.csv";
  if (!std::filesystem::exists(dataDir) ||
      !std::filesystem::exists(associationFile)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO oracle data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile;
  options.associationRequireObservedObjects = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationTransformUseInitialObjectPoses = true;
  options.associationTransformDimension = 0;
  options.associationTransformMinInliers = 8;
  options.associationGaugeSynchronizationEnabled = true;
  options.associationGaugeSynchronizationApplyOnceToInitialStates = true;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result = runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const double initialConsensus = result.iterations.front().consensusCost;
  const double afterOneStepConsensus = result.iterations.back().consensusCost;
  ASSERT_TRUE(std::isfinite(initialConsensus));
  ASSERT_TRUE(std::isfinite(afterOneStepConsensus));
  ASSERT_GT(initialConsensus, 0.0);
  EXPECT_LT(afterOneStepConsensus, 10.0 * initialConsensus);
}

TEST(testDPGO, ManualObjectDranUsesTopologyFileSchedule) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  const std::filesystem::path topologyPath =
      std::filesystem::temp_directory_path() /
      "manual_object_dran_topology_file_test.csv";
  {
    std::ofstream out(topologyPath);
    out << "round,src,dst,weight\n";
    out << "0,0,1,0.25\n";
    out << "1,1,2,0.25\n";
    out << "1,2,3,0.25\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.topologyFile = topologyPath.string();
  options.topologyWeightMode = "unit";
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);
  std::filesystem::remove(topologyPath);

  ASSERT_EQ(result.iterations.size(), 3u);
  ASSERT_GT(result.numObjects, 0u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs,
            2u * result.numObjects);
  EXPECT_EQ(result.iterations[1].activeConsensusPairs,
            2u * result.numObjects);
  EXPECT_EQ(result.iterations.back().activeConsensusPairs,
            4u * result.numObjects);
}

TEST(testDPGO, ManualObjectDranExposesFinalAssociationWeights) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationSemanticWeight = 1.0;
  options.associationSinkhornIterations = 5;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.finalAssociationWeights.size(), result.numRobots);
  ASSERT_GT(result.numObjects, 0u);
  ASSERT_GE(result.numRobots, 2u);
  ASSERT_EQ(result.finalAssociationWeights[0].size(), result.numRobots);
  ASSERT_EQ(result.finalAssociationWeights[0][1].rows(),
            static_cast<int>(result.numObjects));
  ASSERT_EQ(result.finalAssociationWeights[0][1].cols(),
            static_cast<int>(result.numObjects));
  EXPECT_GE(result.finalAssociationWeights[0][1].sum(), 0.0);
}

TEST(testDPGO,
     ManualObjectDranAssociationMetricsDoNotFallbackWhenGtPairIsKnownEmpty) {
  Matrix association = Matrix::Zero(3, 3);
  association(0, 0) = 0.8;
  association(1, 2) = 0.1;
  Matrix emptyGroundTruth = Matrix::Zero(3, 3);

  const ManualObjectDranAssociationMetricSummary withKnownEmptyGt =
      evaluateManualObjectDranAssociationMatrixMetrics(
          association, emptyGroundTruth, true);

  EXPECT_EQ(withKnownEmptyGt.candidateCount, 2u);
  EXPECT_DOUBLE_EQ(withKnownEmptyGt.correctMass, 0.0);
  EXPECT_DOUBLE_EQ(withKnownEmptyGt.falseMergeMass, 0.9);
  EXPECT_DOUBLE_EQ(withKnownEmptyGt.recallDenominator, 0.0);
  EXPECT_DOUBLE_EQ(withKnownEmptyGt.recall, 0.0);
  EXPECT_DOUBLE_EQ(withKnownEmptyGt.precision, 0.0);

  const ManualObjectDranAssociationMetricSummary withoutGt =
      evaluateManualObjectDranAssociationMatrixMetrics(
          association, emptyGroundTruth, false);

  EXPECT_EQ(withoutGt.candidateCount, 2u);
  EXPECT_DOUBLE_EQ(withoutGt.correctMass, 0.8);
  EXPECT_DOUBLE_EQ(withoutGt.falseMergeMass, 0.1);
  EXPECT_DOUBLE_EQ(withoutGt.recallDenominator, 3.0);
  EXPECT_GT(withoutGt.precision, 0.0);
}

TEST(testDPGO, ManualObjectDranRejectsMalformedTopologyFiles) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  auto expectTopologyThrows = [&](const std::string &name,
                                  const std::string &contents) {
    const std::filesystem::path topologyPath =
        std::filesystem::temp_directory_path() / name;
    {
      std::ofstream out(topologyPath);
      out << contents;
    }
    ManualObjectDranOptions options;
    options.numRobots = 21;
    options.numObjects = 0;
    options.maxIterations = 0;
    options.topologyFile = topologyPath.string();
    options.topologyWeightMode = "matrix";
    EXPECT_THROW(runManualObjectDran(dataDir, options), std::runtime_error);
    std::filesystem::remove(topologyPath);
  };

  expectTopologyThrows("manual_object_dran_topology_missing_round.csv",
                       "round,src,dst,weight\n1,0,1,0.25\n");
  expectTopologyThrows("manual_object_dran_topology_bad_weight.csv",
                       "round,src,dst,weight\n0,0,1,nan\n");
}

TEST(testDPGO, ManualObjectDranRunsDroneObjectCopiesWithReducedSolver) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.beta = 1.0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 5;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.numRobots, 21u);
  ASSERT_GT(result.numObjects, 0u);
  EXPECT_EQ(result.knownObjectCopies, result.numRobots * result.numObjects);
  EXPECT_GT(result.observedObjectCopies, 0u);
  EXPECT_GT(result.relayObjectCopies, 0u);
  EXPECT_EQ(result.relayOnlyObjectCopies, result.relayObjectCopies);
  EXPECT_LT(result.relayObjectCopies,
            result.numRobots * result.numObjects);
  EXPECT_EQ(result.initializedFromNeighborObjectCopies, 0u);
  EXPECT_EQ(result.staleObjectCopies, 0u);
  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(result.objectInterfaceDiagnostics.size(),
            result.iterations.size() * result.numRobots * result.numObjects);
  bool sawObservedObject = false;
  bool sawRelayObject = false;
  bool sawSelectedForSend = false;
  for (const ManualObjectDranObjectInterfaceDiagnostic &row :
       result.objectInterfaceDiagnostics) {
    EXPECT_LT(row.robotId, result.numRobots);
    EXPECT_LT(row.objectId, result.numObjects);
    EXPECT_EQ(row.relayOnly, !row.observed);
    EXPECT_TRUE(std::isfinite(row.residualNorm));
    EXPECT_TRUE(std::isfinite(row.gradientNorm));
    EXPECT_TRUE(std::isfinite(row.stiffness));
    EXPECT_TRUE(std::isfinite(row.neighborDisagreement));
    sawObservedObject = sawObservedObject || row.observed;
    sawRelayObject = sawRelayObject || row.relayOnly;
    sawSelectedForSend = sawSelectedForSend || row.selectedForSend;
  }
  EXPECT_TRUE(sawObservedObject);
  EXPECT_TRUE(sawRelayObject);
  EXPECT_TRUE(sawSelectedForSend);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 2520u);
  EXPECT_EQ(result.iterations.front().knownObjectCopies,
            result.knownObjectCopies);
  EXPECT_EQ(result.iterations.front().relayObjectCopies,
            result.relayObjectCopies);
  EXPECT_EQ(result.iterations.front().observedObjectCopies,
            result.observedObjectCopies);
  EXPECT_EQ(result.iterations.front().relayOnlyObjectCopies,
            result.relayOnlyObjectCopies);
  EXPECT_EQ(result.iterations.front().initializedFromNeighborObjectCopies,
            result.initializedFromNeighborObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCopies,
            result.staleObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCommTriggers, 0u);
  EXPECT_GT(result.iterations.front().commPoseCount, 0u);
  EXPECT_TRUE(std::isfinite(result.iterations.back().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().consensusCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().gradient));
}

TEST(testDPGO, ManualObjectDranCanSkipObjectInterfaceDiagnostics) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.recordObjectInterfaceDiagnostics = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.objectInterfaceDiagnostics.size(), 0u);
  EXPECT_GT(result.numObjects, 0u);
}

TEST(testDPGO, ManualObjectDranCanSkipDetailedAssociationDiagnostics) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.recordDetailedAssociationDiagnostics = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  const ManualObjectDranIterationSummary &summary = result.iterations.front();
  EXPECT_EQ(summary.highConfidenceAssociationCandidateCount, 0u);
  EXPECT_EQ(summary.activatedAssociationCandidateCount, 0u);
  EXPECT_EQ(summary.activeAssociationCandidateCount, 0u);
  EXPECT_EQ(summary.associationRejectedCandidateCount, 0u);
}

TEST(testDPGO, ManualObjectDranCanSkipPredictiveMeasurementCost) {
  const std::string dataDir =
      "association_result/object_results/kitti360_4r_public_default/object_pgo";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing KITTI-360 object-PGO data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 4;
  options.numObjects = 386;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::Triggered;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.recordPredictiveMeasurementCost = false;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(result.iterations.front().predictiveMeasurementCost, 0.0);
}

TEST(testDPGO, ManualObjectDranNeighborAverageInitializationReportsRelays) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  ASSERT_EQ(result.objectInterfaceDiagnostics.size(),
            result.numRobots * result.numObjects);
  EXPECT_EQ(result.knownObjectCopies, result.numRobots * result.numObjects);
  EXPECT_GT(result.relayOnlyObjectCopies, 0u);
  EXPECT_EQ(result.relayObjectCopies, result.relayOnlyObjectCopies);
  EXPECT_GT(result.initializedFromNeighborObjectCopies, 0u);
  EXPECT_LE(result.initializedFromNeighborObjectCopies,
            result.relayOnlyObjectCopies);
  EXPECT_LE(result.staleObjectCopies, result.relayOnlyObjectCopies);
  EXPECT_EQ(result.iterations.front().initializedFromNeighborObjectCopies,
            result.initializedFromNeighborObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCopies,
            result.staleObjectCopies);
  EXPECT_LT(result.iterations.front().measurementCost, 10.0);
  EXPECT_LT(result.iterations.front().consensusCost, 1000.0);
}

TEST(testDPGO, ManualObjectDranInitializationConsensusDoesNotAffectCentralized) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 0;
  baseOptions.objectInitialization =
      ManualObjectDranObjectInitialization::CentralizedChordal;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions consensusOptions = baseOptions;
  consensusOptions.objectInitializationConsensusRounds = 5;
  consensusOptions.objectInitializationObservedAnchorWeight = 2.0;
  consensusOptions.objectInitializationRelayAnchorWeight = 1.0;

  const ManualObjectDranRunResult base =
      runManualObjectDran(dataDir, baseOptions);
  const ManualObjectDranRunResult consensus =
      runManualObjectDran(dataDir, consensusOptions);

  ASSERT_EQ(base.iterations.size(), 1u);
  ASSERT_EQ(consensus.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(consensus.iterations.front().measurementCost,
                   base.iterations.front().measurementCost);
  EXPECT_DOUBLE_EQ(consensus.iterations.front().consensusCost,
                   base.iterations.front().consensusCost);
  EXPECT_EQ(consensus.initializedFromNeighborObjectCopies,
            base.initializedFromNeighborObjectCopies);
  EXPECT_EQ(consensus.staleObjectCopies, base.staleObjectCopies);
}

TEST(testDPGO, ManualObjectDranInitializationConsensusReducesDisagreement) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 0;
  baseOptions.topology = ManualObjectDranTopology::Ring;
  baseOptions.ringHops = 1;
  baseOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions consensusOptions = baseOptions;
  consensusOptions.objectInitializationConsensusRounds = 5;
  consensusOptions.objectInitializationObservedAnchorWeight = 1.0;

  const ManualObjectDranRunResult base =
      runManualObjectDran(dataDir, baseOptions);
  const ManualObjectDranRunResult consensus =
      runManualObjectDran(dataDir, consensusOptions);

  ASSERT_EQ(base.iterations.size(), 1u);
  ASSERT_EQ(consensus.iterations.size(), 1u);
  EXPECT_LT(consensus.iterations.front().consensusCost,
            base.iterations.front().consensusCost);
  EXPECT_LT(consensus.iterations.front().measurementCost, 10.0);
}

TEST(testDPGO, ManualObjectDranInformationAnchorChangesInitializationConsensus) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions constantOptions;
  constantOptions.numRobots = 21;
  constantOptions.numObjects = 0;
  constantOptions.maxIterations = 0;
  constantOptions.topology = ManualObjectDranTopology::Ring;
  constantOptions.ringHops = 1;
  constantOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  constantOptions.objectInitializationConsensusRounds = 5;
  constantOptions.objectInitializationObservedAnchorWeight = 1.0;
  constantOptions.localTrustRegionIterations = 1;
  constantOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions informationOptions = constantOptions;
  informationOptions.objectInitializationAnchorMode =
      ManualObjectDranObjectInitializationAnchorMode::Information;

  const ManualObjectDranRunResult constant =
      runManualObjectDran(dataDir, constantOptions);
  const ManualObjectDranRunResult information =
      runManualObjectDran(dataDir, informationOptions);

  ASSERT_EQ(constant.iterations.size(), 1u);
  ASSERT_EQ(information.iterations.size(), 1u);
  const double costDelta =
      std::abs(information.iterations.front().measurementCost -
               constant.iterations.front().measurementCost) +
      std::abs(information.iterations.front().consensusCost -
               constant.iterations.front().consensusCost);
  EXPECT_GT(costDelta, 1e-9);
  EXPECT_LT(information.iterations.front().measurementCost, 10.0);
  EXPECT_TRUE(std::isfinite(information.iterations.front().consensusCost));
}

TEST(testDPGO, ManualObjectDranRunsFullThenReducedSchedule) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  options.fullSolverOuterIterations = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_TRUE(std::isfinite(result.iterations.back().measurementCost));
}

TEST(testDPGO, ManualObjectDranReducedSchurResponseModeReportsWork) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceSchurDamping = 0.01;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 3;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_EQ(summary.iter, 1u);
  EXPECT_GT(summary.activeRobots, 0u);
  EXPECT_GT(summary.candidateObjectBlocks, 0u);
  EXPECT_EQ(summary.candidateObjectBlocks,
            summary.acceptedObjectBlocks + summary.rejectedObjectBlocks);
  EXPECT_TRUE(std::isfinite(summary.modelCostBefore));
  EXPECT_TRUE(std::isfinite(summary.modelCostAfter));
  EXPECT_TRUE(std::isfinite(summary.acceptedModelDecrease));
  EXPECT_TRUE(std::isfinite(summary.stepNorm));
}

TEST(testDPGO, ManualObjectDranReducedSchurBatchResponseReportsBatches) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 3;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_GT(summary.activeRobots, 0u);
  EXPECT_EQ(summary.batchResponseRobots, summary.activeRobots);
  EXPECT_EQ(summary.batchResponseRobots,
            summary.acceptedRobotBatches + summary.rejectedRobotBatches);
  EXPECT_GT(summary.candidateObjectBlocks, 0u);
  EXPECT_EQ(summary.candidateObjectBlocks,
            summary.acceptedObjectBlocks + summary.rejectedObjectBlocks);
}

TEST(testDPGO, ManualObjectDranReducedSchurResponseHonorsPeriod) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 3;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponsePeriod = 2;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 3u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[0].iter, 1u);
  EXPECT_FALSE(result.objectInterfaceResponseSummaries[0].scheduled);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[0].activeRobots, 0u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[1].iter, 2u);
  EXPECT_TRUE(result.objectInterfaceResponseSummaries[1].scheduled);
  EXPECT_GT(result.objectInterfaceResponseSummaries[1].activeRobots, 0u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[2].iter, 3u);
  EXPECT_FALSE(result.objectInterfaceResponseSummaries[2].scheduled);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[2].activeRobots, 0u);
}

TEST(testDPGO, ManualObjectDranAdaptiveResponseSkipsLowPredictedDecrease) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::PredictedDecrease;
  options.objectInterfaceMinPredictedDecrease = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_TRUE(summary.scheduled);
  EXPECT_EQ(summary.activeRobots, 0u);
  EXPECT_GT(summary.triggerCandidateRobots, 0u);
  EXPECT_EQ(summary.triggeredRobots, 0u);
  EXPECT_EQ(summary.triggerCandidateRobots, summary.skippedTriggerRobots);
  EXPECT_GT(summary.predictedModelDecrease, 0.0);
  EXPECT_TRUE(std::isfinite(summary.predictedModelDecrease));
  EXPECT_EQ(summary.candidateObjectBlocks, 0u);
}

TEST(testDPGO, ManualObjectDranLocalInnovationTriggerSkipsBeforeSchurModel) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::LocalInnovation;
  options.objectInterfaceMinInnovationScore = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_TRUE(summary.scheduled);
  EXPECT_EQ(summary.activeRobots, 0u);
  EXPECT_GT(summary.triggerCandidateRobots, 0u);
  EXPECT_EQ(summary.triggeredRobots, 0u);
  EXPECT_EQ(summary.triggerCandidateRobots, summary.skippedTriggerRobots);
  EXPECT_GT(summary.interfaceInnovationScore, 0.0);
  EXPECT_EQ(summary.predictedModelDecrease, 0.0);
  EXPECT_EQ(summary.candidateObjectBlocks, 0u);
}

TEST(testDPGO, ManualObjectDranAccumulatedInnovationCarriesSkippedSignal) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::AccumulatedInnovation;
  options.objectInterfaceMinInnovationScore = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 2u);
  const ManualObjectDranObjectInterfaceResponseSummary &first =
      result.objectInterfaceResponseSummaries[0];
  const ManualObjectDranObjectInterfaceResponseSummary &second =
      result.objectInterfaceResponseSummaries[1];
  EXPECT_TRUE(first.scheduled);
  EXPECT_TRUE(second.scheduled);
  EXPECT_EQ(first.activeRobots, 0u);
  EXPECT_EQ(second.activeRobots, 0u);
  EXPECT_EQ(first.triggerCandidateRobots, first.skippedTriggerRobots);
  EXPECT_EQ(second.triggerCandidateRobots, second.skippedTriggerRobots);
  EXPECT_GT(first.accumulatedInnovationScore, 0.0);
  EXPECT_GT(second.accumulatedInnovationScore,
            first.accumulatedInnovationScore);
  EXPECT_EQ(first.predictedModelDecrease, 0.0);
  EXPECT_EQ(second.predictedModelDecrease, 0.0);
}

TEST(testDPGO, ManualObjectDranPairCavityCovarianceMatchesDenseInverse) {
  Matrix dense(3, 3);
  dense << 4.0, 0.5, -0.2,
           0.5, 3.0, 0.4,
          -0.2, 0.4, 2.0;
  const SparseMatrix quadratic = dense.sparseView();

  const Matrix covariance = manualObjectDranSelectedCovarianceForTest(
      quadratic, {}, {0u, 2u});
  const Matrix expectedFull = dense.inverse();
  Matrix expected(2, 2);
  expected << expectedFull(0, 0), expectedFull(0, 2),
              expectedFull(2, 0), expectedFull(2, 2);

  EXPECT_LE((covariance - expected).norm(), 1e-12);
  EXPECT_LE((covariance - covariance.transpose()).norm(), 1e-15);
}

TEST(testDPGO, ManualObjectDranPairCavityCovarianceRejectsSingularSystem) {
  Matrix dense = Matrix::Zero(2, 2);
  dense(0, 0) = 1.0;
  const SparseMatrix quadratic = dense.sparseView();

  EXPECT_THROW(manualObjectDranSelectedCovarianceForTest(
                   quadratic, {}, {0u}),
               std::runtime_error);
}

TEST(testDPGO, ManualObjectDranPairCavityExclusionZerosOnlyTargetPair) {
  std::vector<std::vector<Matrix>> weights(
      3, std::vector<Matrix>(3, Matrix::Ones(2, 2)));
  weights[0][2] *= 2.0;
  weights[2][0] *= 3.0;

  const std::vector<std::vector<Matrix>> excluded =
      manualObjectDranPairExcludedAssociationWeightsForTest(
          weights, 2u, 0u);

  EXPECT_EQ(excluded[0][2].norm(), 0.0);
  EXPECT_EQ(excluded[2][0].norm(), 0.0);
  EXPECT_EQ((excluded[0][1] - weights[0][1]).norm(), 0.0);
  EXPECT_GT(weights[0][2].norm(), 0.0);
  EXPECT_THROW(manualObjectDranPairExcludedAssociationWeightsForTest(
                   weights, 1u, 1u),
               std::invalid_argument);
  EXPECT_THROW(manualObjectDranPairExcludedAssociationWeightsForTest(
                   weights, 0u, 3u),
               std::invalid_argument);
}

TEST(testDPGO, ManualObjectDranPairCavityConditionalMeanSolvesFreeTranslations) {
  const unsigned d = 1u;
  Matrix denseQ = Matrix::Zero(4, 4);
  denseQ.diagonal() << 2.0, 4.0, 3.0, 3.0;
  denseQ(1, 3) = 1.0;
  denseQ(3, 1) = 1.0;
  denseQ(0, 3) = 0.5;
  denseQ(3, 0) = 0.5;
  SparseMatrix quadratic = denseQ.sparseView();
  Matrix denseLinear = Matrix::Zero(1, 4);
  denseLinear(0, 3) = -2.0;
  SparseMatrix linear = denseLinear.sparseView();
  Matrix state(1, 4);
  state << 1.5, 2.0, -0.5, 7.0;

  const Matrix recovered =
      manualObjectDranConditionalTranslationMeanForTest(
          quadratic, linear, state, d, {1u});

  EXPECT_EQ(recovered(0, 1), state(0, 1));
  EXPECT_EQ(recovered(0, 0), state(0, 0));
  EXPECT_EQ(recovered(0, 2), state(0, 2));
  const Matrix gradient = recovered * denseQ + denseLinear;
  EXPECT_NEAR(gradient(0, 3), 0.0, 1e-12);
  EXPECT_NE(recovered(0, 3), state(0, 3));
}

TEST(testDPGO, CavityAssociationProfilesPairGaugeWithoutTuningWeights) {
  Matrix first(3, 2);
  first << 0.0, 0.0,
           1.0, 0.0,
           0.0, 2.0;
  Matrix second(3, 2);
  second << 2.0, 1.0,
            2.0, 2.0,
            0.0, 1.0;
  Matrix firstCovariance = Vector(3).setConstant(0.0).asDiagonal();
  firstCovariance.diagonal() << 0.2, 0.3, 0.4;
  Matrix secondCovariance = Vector(3).setConstant(0.0).asDiagonal();
  secondCovariance.diagonal() << 0.5, 0.6, 0.7;

  const CavityPairBlockEvidence evidence = profileCavityPairBlockEvidence(
      first, second, firstCovariance, secondCovariance);

  ASSERT_TRUE(evidence.valid);
  EXPECT_EQ(evidence.gaugeRank, 3u);
  EXPECT_EQ(evidence.numMatches, 3u);
  EXPECT_NEAR(evidence.profileChiSquared, 0.0, 1e-12);
  EXPECT_NEAR(evidence.covarianceLogDet, -0.7334505595844678, 1e-12);
  EXPECT_NEAR(evidence.gaugeInformationLogDet, 3.7088780797980947, 1e-12);
  EXPECT_NEAR(evidence.score, 4.244529359720831, 1e-12);
}

TEST(testDPGO, CavityAssociationDiagonalProfileMatchesDenseReference) {
  Matrix first(6, 2);
  first << 0.0, 0.0,
           1.2, -0.1,
           -0.4, 2.0,
           2.1, 1.7,
           -1.0, -0.5,
           0.8, 3.2;
  Matrix second(6, 2);
  second << 1.7, -0.8,
            2.1, 0.4,
            -0.2, 0.9,
            1.1, 2.8,
            1.2, -1.9,
            -0.3, 3.0;
  Vector firstVariance(6);
  firstVariance << 0.2, 0.5, 0.3, 0.9, 0.4, 0.7;
  Vector secondVariance(6);
  secondVariance << 0.8, 0.1, 0.6, 0.2, 0.5, 0.3;

  const CavityPairBlockEvidence dense = profileCavityPairBlockEvidence(
      first, second, firstVariance.asDiagonal(),
      secondVariance.asDiagonal());
  const CavityPairBlockEvidence diagonal =
      profileDiagonalCavityPairBlockEvidence(
          first, second, firstVariance, secondVariance);

  ASSERT_TRUE(dense.valid);
  ASSERT_TRUE(diagonal.valid);
  EXPECT_EQ(diagonal.gaugeRank, dense.gaugeRank);
  EXPECT_EQ(diagonal.numMatches, dense.numMatches);
  EXPECT_NEAR(diagonal.profileChiSquared, dense.profileChiSquared, 1e-12);
  EXPECT_NEAR(diagonal.covarianceLogDet, dense.covarianceLogDet, 1e-12);
  EXPECT_NEAR(diagonal.gaugeInformationLogDet,
              dense.gaugeInformationLogDet, 1e-12);
  EXPECT_NEAR(diagonal.score, dense.score, 1e-12);
}

TEST(testDPGO, CavityAssociationDiagonalProfileMatchesDenseAcrossScales) {
  constexpr Eigen::Index numMatches = 37;
  for (const double scale : {0.01, 1.0, 100.0}) {
    Matrix first(numMatches, 2);
    Matrix second(numMatches, 2);
    Vector firstVariance(numMatches);
    Vector secondVariance(numMatches);
    for (Eigen::Index match = 0; match < numMatches; ++match) {
      const double index = static_cast<double>(match);
      second(match, 0) = scale * (0.23 * index + std::sin(0.7 * index));
      second(match, 1) = scale * (std::cos(0.31 * index) - 0.11 * index);
      first(match, 0) =
          0.91 * second(match, 0) - 0.37 * second(match, 1) +
          scale * (1.3 + 0.002 * index * index);
      first(match, 1) =
          0.37 * second(match, 0) + 0.91 * second(match, 1) +
          scale * (-0.8 + 0.003 * std::sin(index));
      firstVariance(match) = 0.05 + 0.013 * static_cast<double>(match % 9);
      secondVariance(match) = 0.08 + 0.017 * static_cast<double>(match % 7);
    }

    const CavityPairBlockEvidence dense = profileCavityPairBlockEvidence(
        first, second, firstVariance.asDiagonal(),
        secondVariance.asDiagonal());
    const CavityPairBlockEvidence diagonal =
        profileDiagonalCavityPairBlockEvidence(
            first, second, firstVariance, secondVariance);

    ASSERT_TRUE(dense.valid);
    ASSERT_TRUE(diagonal.valid);
    const auto tolerance = [](double reference) {
      return 1e-10 * std::max(1.0, std::abs(reference));
    };
    EXPECT_NEAR(diagonal.profileChiSquared, dense.profileChiSquared,
                tolerance(dense.profileChiSquared));
    EXPECT_NEAR(diagonal.covarianceLogDet, dense.covarianceLogDet,
                tolerance(dense.covarianceLogDet));
    EXPECT_NEAR(diagonal.gaugeInformationLogDet,
                dense.gaugeInformationLogDet,
                tolerance(dense.gaugeInformationLogDet));
    EXPECT_NEAR(diagonal.score, dense.score, tolerance(dense.score));
  }
}

TEST(testDPGO, CavityAssociationPairScoreIsEndpointInvariantWithNoisyPoints) {
  Matrix first(5, 2);
  first << -320.0, 20.0,
           -310.0, 20.0,
           -315.0, 30.0,
           -300.0, 15.0,
           -330.0, 10.0;
  const double angle = 0.1;
  Matrix rotation(2, 2);
  rotation << std::cos(angle), -std::sin(angle),
              std::sin(angle), std::cos(angle);
  Matrix second =
      (rotation.transpose() *
       (first.rowwise() - Eigen::RowVector2d(50.0, -20.0)).transpose())
          .transpose();
  second.row(0) += Eigen::RowVector2d(0.2, -0.1);
  Vector firstVariance(5);
  firstVariance << 1e5, 2e5, 1.5e5, 2.5e5, 1.2e5;
  const Vector secondVariance = 2.0 * firstVariance;

  const CavityPairBlockEvidence forward = profileCavityPairBlockEvidence(
      first, second, firstVariance.asDiagonal(),
      secondVariance.asDiagonal());
  const CavityPairBlockEvidence reverse = profileCavityPairBlockEvidence(
      second, first, secondVariance.asDiagonal(),
      firstVariance.asDiagonal());
  const CavityPairBlockEvidence diagonalForward =
      profileDiagonalCavityPairBlockEvidence(
          first, second, firstVariance, secondVariance);
  const CavityPairBlockEvidence diagonalReverse =
      profileDiagonalCavityPairBlockEvidence(
          second, first, secondVariance, firstVariance);

  ASSERT_TRUE(forward.valid);
  ASSERT_TRUE(reverse.valid);
  ASSERT_TRUE(diagonalForward.valid);
  ASSERT_TRUE(diagonalReverse.valid);
  EXPECT_NEAR(forward.score, reverse.score, 1e-10);
  EXPECT_NEAR(diagonalForward.score, diagonalReverse.score, 1e-10);
}

TEST(testDPGO, CavityAssociationAssignmentCoversFixedCardinalityExactly) {
  Matrix costs(4, 3);
  costs << 9.0, 2.0, 7.0,
           6.0, 4.0, 3.0,
           5.0, 8.0, 1.0,
           7.0, 6.0, 9.0;

  const CavityComponentAssignment result = solveCavityComponentAssignment(
      {10u, 11u, 12u, 13u}, {20u, 21u, 22u}, costs);

  EXPECT_DOUBLE_EQ(result.totalCost, 9.0);
  const std::vector<std::pair<std::size_t, std::size_t>> expected{
      {10u, 21u}, {11u, 20u}, {12u, 22u}};
  EXPECT_EQ(result.assignments, expected);
}

TEST(testDPGO, CavityAssociationReassignsIdentityWithStrictPairCertificate) {
  CavityRobotAssociationState first;
  first.objectIds = {0u, 1u, 2u, 3u, 4u};
  first.categories = {0u, 1u, 2u, 3u, 3u};
  first.points.resize(5, 2);
  first.points << 0.0, 0.0,
                  1.0, 0.0,
                  0.0, 2.0,
                  2.0, 2.0,
                 10.0, 10.0;
  first.covarianceDiagonal = Vector::Constant(5, 0.2);

  CavityRobotAssociationState second;
  second.objectIds = {10u, 11u, 12u, 13u};
  second.categories = {0u, 1u, 2u, 3u};
  second.points.resize(4, 2);
  second.points << 2.0, 1.0,
                   2.0, 2.0,
                   0.0, 1.0,
                   0.0, 3.0;
  second.covarianceDiagonal = Vector::Constant(4, 0.5);

  const std::vector<std::pair<std::size_t, std::size_t>> active{
      {0u, 10u}, {1u, 11u}, {2u, 12u}, {4u, 13u}};
  const CavityPairReassignment result =
      reassignCavityPairIdentities(first, second, active);

  ASSERT_TRUE(result.accepted);
  EXPECT_LT(result.scoreAfterCandidate, result.scoreBefore);
  EXPECT_EQ(result.assignmentsBefore, active);
  const std::vector<std::pair<std::size_t, std::size_t>> expected{
      {0u, 10u}, {1u, 11u}, {2u, 12u}, {3u, 13u}};
  EXPECT_EQ(result.assignmentsAfter, expected);
  EXPECT_EQ(result.changedEdgeCount, 1u);
  EXPECT_EQ(result.configuredScoreWeights, 0u);
  EXPECT_EQ(result.configuredAcceptanceThresholds, 0u);
  EXPECT_EQ(result.payloadBytes, 9u * 3u * sizeof(double));
}

TEST(testDPGO, CavityAssociationRejectsPairScoreTieWithoutMarginParameter) {
  CavityRobotAssociationState first;
  first.objectIds = {0u, 1u, 2u};
  first.categories = {0u, 1u, 2u};
  first.points = Matrix::Zero(3, 2);
  first.points << 0.0, 0.0,
                  1.0, 0.0,
                  0.0, 1.0;
  first.covarianceDiagonal = Vector::Ones(3);
  CavityRobotAssociationState second = first;
  second.objectIds = {10u, 11u, 12u};

  const std::vector<std::pair<std::size_t, std::size_t>> active{
      {0u, 10u}, {1u, 11u}, {2u, 12u}};
  const CavityPairReassignment result =
      reassignCavityPairIdentities(first, second, active);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.assignmentsAfter, active);
  EXPECT_DOUBLE_EQ(result.scoreAfterCandidate, result.scoreBefore);
}

TEST(testDPGO, CavityAssociationExtractsMutualPositiveMaximaWithoutThreshold) {
  Matrix weights = Matrix::Zero(3, 3);
  weights << 0.9, 0.2, 0.0,
             0.1, 0.8, 0.0,
             0.0, 0.0, 1e-20;

  const std::vector<std::pair<std::size_t, std::size_t>> active =
      extractCavityActiveAssignments(weights);

  const std::vector<std::pair<std::size_t, std::size_t>> expected{
      {0u, 0u}, {1u, 1u}, {2u, 2u}};
  EXPECT_EQ(active, expected);
}

TEST(testDPGO, NetworkEditZoneSeparatesChangedComponentsAndOmitsUnchanged) {
  using Node = ManualObjectDranObjectCopyNode;
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{0u, 0u}, {1u, 0u}},
      {{2u, 0u}, {3u, 0u}},
      {{4u, 0u}, {5u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 0u}, {1u, 1u}},
      {{2u, 1u}, {3u, 0u}},
      {{4u, 0u}, {5u, 0u}}};

  const std::vector<ManualObjectDranNetworkEditZone> zones =
      manualObjectDranNetworkEditZonesForTest(incumbent, candidate);

  ASSERT_EQ(zones.size(), 2u);
  EXPECT_EQ(zones[0].id, 0u);
  EXPECT_EQ(zones[0].nodes,
            (std::vector<Node>{{0u, 0u}, {1u, 0u}, {1u, 1u}}));
  EXPECT_EQ(zones[0].removedEdges,
            (std::vector<Edge>{{{0u, 0u}, {1u, 0u}}}));
  EXPECT_EQ(zones[0].addedEdges,
            (std::vector<Edge>{{{0u, 0u}, {1u, 1u}}}));
  EXPECT_EQ(zones[1].id, 1u);
  EXPECT_EQ(zones[1].nodes,
            (std::vector<Node>{{2u, 0u}, {2u, 1u}, {3u, 0u}}));
  EXPECT_EQ(zones[1].removedEdges,
            (std::vector<Edge>{{{2u, 0u}, {3u, 0u}}}));
  EXPECT_EQ(zones[1].addedEdges,
            (std::vector<Edge>{{{2u, 1u}, {3u, 0u}}}));
}

TEST(testDPGO, NetworkEditZoneMergesPairChangesTouchingOneIdentity) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {2u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 1u}, {1u, 0u}}, {{1u, 0u}, {2u, 1u}}};

  const std::vector<ManualObjectDranNetworkEditZone> zones =
      manualObjectDranNetworkEditZonesForTest(incumbent, candidate);

  ASSERT_EQ(zones.size(), 1u);
  EXPECT_EQ(zones[0].nodes.size(), 5u);
  EXPECT_EQ(zones[0].removedEdges.size(), 2u);
  EXPECT_EQ(zones[0].addedEdges.size(), 2u);
}

TEST(testDPGO, NetworkPartitionRejectsRepeatedRobotInsideOneComponent) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> feasible{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {2u, 0u}}};
  const std::vector<Edge> infeasible{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {0u, 1u}}};

  EXPECT_TRUE(manualObjectDranNetworkPartitionFeasibleForTest(feasible));
  EXPECT_FALSE(manualObjectDranNetworkPartitionFeasibleForTest(infeasible));
}

TEST(testDPGO, NetworkEditZoneIsIndependentOfInputEdgeOrder) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{2u, 0u}, {3u, 0u}}, {{1u, 0u}, {0u, 0u}}};
  const std::vector<Edge> candidate{
      {{3u, 0u}, {2u, 1u}}, {{1u, 1u}, {0u, 0u}}};
  std::vector<Edge> reversedIncumbent(incumbent.rbegin(), incumbent.rend());
  std::vector<Edge> reversedCandidate(candidate.rbegin(), candidate.rend());

  const auto forward = manualObjectDranNetworkEditZonesForTest(
      incumbent, candidate);
  const auto reverse = manualObjectDranNetworkEditZonesForTest(
      reversedIncumbent, reversedCandidate);

  ASSERT_EQ(forward.size(), reverse.size());
  for (std::size_t index = 0; index < forward.size(); ++index) {
    EXPECT_EQ(forward[index].id, reverse[index].id);
    EXPECT_EQ(forward[index].nodes, reverse[index].nodes);
    EXPECT_EQ(forward[index].removedEdges, reverse[index].removedEdges);
    EXPECT_EQ(forward[index].addedEdges, reverse[index].addedEdges);
  }
}

TEST(testDPGO, DistributedNetworkEditZonesMatchCentralizedAtQueueQuiescence) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {2u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 0u}, {1u, 1u}}, {{1u, 1u}, {2u, 0u}}};
  const std::vector<std::pair<std::size_t, std::size_t>> topology{
      {0u, 1u}, {1u, 2u}};

  const ManualObjectDranDistributedEditZoneResult distributed =
      manualObjectDranDistributedNetworkEditZonesForTest(
          incumbent, candidate, 3u, topology);
  const auto centralized = manualObjectDranNetworkEditZonesForTest(
      incumbent, candidate);
  const ManualObjectDranDistributedEditZoneResult completeTopology =
      manualObjectDranDistributedNetworkEditZonesForTest(
          std::vector<Edge>(incumbent.rbegin(), incumbent.rend()),
          std::vector<Edge>(candidate.rbegin(), candidate.rend()), 3u,
          {{0u, 1u}, {0u, 2u}, {1u, 2u}});

  ASSERT_TRUE(distributed.complete);
  EXPECT_TRUE(distributed.queueQuiescent);
  EXPECT_GT(distributed.labelMessageCount, 0u);
  EXPECT_EQ(distributed.labelPayloadBytes,
            2u * sizeof(std::size_t) * distributed.labelMessageCount);
  ASSERT_EQ(distributed.zones.size(), centralized.size());
  ASSERT_TRUE(completeTopology.complete);
  ASSERT_EQ(completeTopology.zones.size(), centralized.size());
  for (std::size_t index = 0; index < centralized.size(); ++index) {
    EXPECT_EQ(distributed.zones[index].id, centralized[index].id);
    EXPECT_EQ(distributed.zones[index].nodes, centralized[index].nodes);
    EXPECT_EQ(distributed.zones[index].removedEdges,
              centralized[index].removedEdges);
    EXPECT_EQ(distributed.zones[index].addedEdges,
              centralized[index].addedEdges);
    EXPECT_EQ(completeTopology.zones[index].nodes,
              centralized[index].nodes);
    EXPECT_EQ(completeTopology.zones[index].removedEdges,
              centralized[index].removedEdges);
    EXPECT_EQ(completeTopology.zones[index].addedEdges,
              centralized[index].addedEdges);
  }
}

TEST(testDPGO, DistributedNetworkEditZonesReportDisconnectedSchedule) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{{{0u, 0u}, {2u, 0u}}};
  const std::vector<Edge> candidate{{{0u, 0u}, {2u, 1u}}};

  const ManualObjectDranDistributedEditZoneResult result =
      manualObjectDranDistributedNetworkEditZonesForTest(
          incumbent, candidate, 3u, {{0u, 1u}});

  EXPECT_TRUE(result.queueQuiescent);
  EXPECT_FALSE(result.complete);
  EXPECT_EQ(result.status, "disconnected_component_topology");
}

TEST(testDPGO, DistributedNetworkFeasibilityReducesRobotConflicts) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<std::pair<std::size_t, std::size_t>> topology{
      {0u, 1u}, {1u, 2u}};
  const std::vector<Edge> feasible{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {2u, 0u}}};
  const std::vector<Edge> infeasible{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {0u, 1u}}};

  const ManualObjectDranDistributedFeasibilityResult accepted =
      manualObjectDranDistributedNetworkPartitionFeasibleForTest(
          feasible, 3u, topology);
  const ManualObjectDranDistributedFeasibilityResult rejected =
      manualObjectDranDistributedNetworkPartitionFeasibleForTest(
          infeasible, 3u, topology);

  EXPECT_TRUE(accepted.complete);
  EXPECT_TRUE(accepted.feasible);
  EXPECT_GT(accepted.reductionMessageCount, 0u);
  EXPECT_EQ(accepted.reductionPayloadBytes,
            accepted.reductionMessageCount * sizeof(std::uint8_t));
  EXPECT_TRUE(rejected.complete);
  EXPECT_FALSE(rejected.feasible);
}

TEST(testDPGO, DistributedSchurPacketsReduceExactlyAlongComponentTree) {
  ManualObjectDranTranslationSchurPacket first;
  first.robot = 0u;
  first.hessian = Matrix::Identity(2, 2);
  first.gradient = Matrix::Constant(2, 2, 1.0);
  first.positions = Matrix::Constant(2, 2, 2.0);
  first.information = Vector::Constant(2, 3.0);
  first.payloadBytes = 8u;
  ManualObjectDranTranslationSchurPacket second = first;
  second.robot = 1u;
  second.hessian *= 2.0;
  second.gradient *= 2.0;
  second.positions *= 2.0;
  second.information *= 2.0;
  second.payloadBytes = 16u;
  ManualObjectDranTranslationSchurPacket third = first;
  third.robot = 2u;
  third.hessian *= 3.0;
  third.gradient *= 3.0;
  third.positions *= 3.0;
  third.information *= 3.0;
  third.payloadBytes = 24u;

  const ManualObjectDranTranslationSchurReduction reduced =
      manualObjectDranReduceTranslationSchurPacketsForTest(
          3u, {{0u, 1u}, {1u, 2u}}, {0u, 1u, 2u}, 0u,
          {third, first, second});
  const ManualObjectDranTranslationSchurReduction rerooted =
      manualObjectDranReduceTranslationSchurPacketsForTest(
          3u, {{0u, 1u}, {1u, 2u}, {0u, 2u}}, {2u, 0u, 1u}, 2u,
          {second, third, first});

  ASSERT_TRUE(reduced.valid);
  EXPECT_EQ(reduced.rootRobot, 0u);
  EXPECT_LE((reduced.hessian - 6.0 * Matrix::Identity(2, 2)).norm(),
            1e-12);
  EXPECT_LE((reduced.gradient - Matrix::Constant(2, 2, 6.0)).norm(),
            1e-12);
  EXPECT_LE((reduced.positions - Matrix::Constant(2, 2, 12.0)).norm(),
            1e-12);
  EXPECT_LE((reduced.information - Vector::Constant(2, 18.0)).norm(),
            1e-12);
  EXPECT_EQ(reduced.reductionMessageCount, 2u);
  EXPECT_EQ(reduced.reductionPayloadBytes, 64u);
  EXPECT_EQ(reduced.broadcastMessageCount, 2u);
  EXPECT_EQ(reduced.broadcastPayloadBytes,
            2u * sizeof(std::uint8_t));
  ASSERT_TRUE(rerooted.valid);
  EXPECT_EQ(rerooted.rootRobot, 2u);
  EXPECT_LE((rerooted.hessian - reduced.hessian).norm(), 1e-12);
  EXPECT_LE((rerooted.gradient - reduced.gradient).norm(), 1e-12);
  EXPECT_LE((rerooted.positions - reduced.positions).norm(), 1e-12);
  EXPECT_LE((rerooted.information - reduced.information).norm(), 1e-12);
}

TEST(testDPGO, DistributedSchurReductionReportsDisconnectedParticipants) {
  ManualObjectDranTranslationSchurPacket first;
  first.robot = 0u;
  first.hessian = Matrix::Identity(1, 1);
  first.gradient = Matrix::Zero(1, 2);
  first.positions = Matrix::Zero(1, 2);
  first.information = Vector::Ones(1);
  first.payloadBytes = 8u;
  ManualObjectDranTranslationSchurPacket second = first;
  second.robot = 2u;

  const ManualObjectDranTranslationSchurReduction reduced =
      manualObjectDranReduceTranslationSchurPacketsForTest(
          3u, {{0u, 1u}}, {0u, 2u}, 0u, {first, second});

  EXPECT_FALSE(reduced.valid);
  EXPECT_EQ(reduced.status, "disconnected_component_topology");
}

TEST(testDPGO, NetworkEditZoneColorSharesColorOnlyAcrossRobotDisjointZones) {
  using Node = ManualObjectDranObjectCopyNode;
  std::vector<ManualObjectDranNetworkEditZone> zones(3);
  zones[0].id = 0u;
  zones[0].nodes = std::vector<Node>{{0u, 0u}, {1u, 0u}};
  zones[1].id = 1u;
  zones[1].nodes = std::vector<Node>{{2u, 0u}, {3u, 0u}};
  zones[2].id = 2u;
  zones[2].nodes = std::vector<Node>{{0u, 1u}, {4u, 0u}};

  const std::vector<std::size_t> colors =
      manualObjectDranNetworkEditZoneColorsForTest(zones);

  ASSERT_EQ(colors.size(), zones.size());
  EXPECT_EQ(colors[0], 0u);
  EXPECT_EQ(colors[1], 0u);
  EXPECT_EQ(colors[2], 1u);
}

TEST(testDPGO, NetworkEditZoneColorUsesRobotConflictNotNodeConnectivity) {
  using Node = ManualObjectDranObjectCopyNode;
  std::vector<ManualObjectDranNetworkEditZone> zones(2);
  zones[0].id = 4u;
  zones[0].nodes = std::vector<Node>{{0u, 10u}, {1u, 3u}};
  zones[1].id = 9u;
  zones[1].nodes = std::vector<Node>{{0u, 11u}, {2u, 8u}};

  const std::vector<std::size_t> colors =
      manualObjectDranNetworkEditZoneColorsForTest(zones);

  ASSERT_EQ(colors.size(), 2u);
  EXPECT_NE(colors[0], colors[1]);
}

TEST(testDPGO, NetworkEditZoneColorUsesThreeColorsForConflictTriangle) {
  using Node = ManualObjectDranObjectCopyNode;
  std::vector<ManualObjectDranNetworkEditZone> zones(3);
  zones[0].id = 0u;
  zones[0].nodes = std::vector<Node>{{0u, 0u}, {1u, 0u}};
  zones[1].id = 1u;
  zones[1].nodes = std::vector<Node>{{1u, 1u}, {2u, 0u}};
  zones[2].id = 2u;
  zones[2].nodes = std::vector<Node>{{0u, 1u}, {2u, 1u}};

  const std::vector<std::size_t> colors =
      manualObjectDranNetworkEditZoneColorsForTest(zones);

  EXPECT_EQ(colors, (std::vector<std::size_t>{0u, 1u, 2u}));
}

TEST(testDPGO, NetworkEditZoneColorIsInvariantToZoneInputOrder) {
  using Node = ManualObjectDranObjectCopyNode;
  std::vector<ManualObjectDranNetworkEditZone> ordered(3);
  ordered[0].id = 0u;
  ordered[0].nodes = std::vector<Node>{{0u, 0u}, {1u, 0u}};
  ordered[1].id = 1u;
  ordered[1].nodes = std::vector<Node>{{2u, 0u}, {3u, 0u}};
  ordered[2].id = 2u;
  ordered[2].nodes = std::vector<Node>{{0u, 1u}, {4u, 0u}};
  const std::vector<ManualObjectDranNetworkEditZone> permuted{
      ordered[2], ordered[0], ordered[1]};

  const std::vector<std::size_t> orderedColors =
      manualObjectDranNetworkEditZoneColorsForTest(ordered);
  const std::vector<std::size_t> permutedColors =
      manualObjectDranNetworkEditZoneColorsForTest(permuted);

  std::map<std::size_t, std::size_t> byId;
  for (std::size_t index = 0; index < ordered.size(); ++index) {
    byId[ordered[index].id] = orderedColors[index];
  }
  for (std::size_t index = 0; index < permuted.size(); ++index) {
    EXPECT_EQ(permutedColors[index], byId.at(permuted[index].id));
  }
}

TEST(testDPGO, NetworkPartitionPlanCommitsOnlyBeneficialEditZone) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{0u, 0u}, {1u, 0u}}, {{2u, 0u}, {3u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 0u}, {1u, 1u}}, {{2u, 1u}, {3u, 0u}}};
  ManualObjectDranTranslationResponse beneficial;
  beneficial.valid = true;
  beneficial.predictedPhysicalDelta = -2.0;
  beneficial.numericalErrorBound = 1e-12;
  ManualObjectDranTranslationResponse harmful = beneficial;
  harmful.predictedPhysicalDelta = 3.0;

  const ManualObjectDranNetworkPartitionPlan plan =
      manualObjectDranPlanNetworkPartitionForTest(
          incumbent, candidate, {beneficial, harmful});

  ASSERT_EQ(plan.zoneSummaries.size(), 2u);
  EXPECT_TRUE(plan.zoneSummaries[0].committed);
  EXPECT_FALSE(plan.zoneSummaries[1].committed);
  EXPECT_EQ(plan.committedEdges,
            (std::vector<Edge>{{{0u, 0u}, {1u, 1u}},
                               {{2u, 0u}, {3u, 0u}}}));
}

TEST(testDPGO, NetworkPartitionPlanRejectsInfeasibleZoneBeforeResponse) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{{{0u, 0u}, {1u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 0u}, {1u, 0u}}, {{1u, 0u}, {0u, 1u}}};
  ManualObjectDranTranslationResponse misleadingResponse;
  misleadingResponse.valid = true;
  misleadingResponse.predictedPhysicalDelta = -100.0;

  const ManualObjectDranNetworkPartitionPlan plan =
      manualObjectDranPlanNetworkPartitionForTest(
          incumbent, candidate, {misleadingResponse});

  ASSERT_EQ(plan.zoneSummaries.size(), 1u);
  EXPECT_TRUE(plan.zoneSummaries[0].valid);
  EXPECT_FALSE(plan.zoneSummaries[0].partitionFeasible);
  EXPECT_FALSE(plan.zoneSummaries[0].responseAccepted);
  EXPECT_FALSE(plan.zoneSummaries[0].committed);
  EXPECT_EQ(plan.zoneSummaries[0].status, "partition_infeasible");
  EXPECT_EQ(plan.committedEdges, incumbent);
}

TEST(testDPGO, NetworkPartitionPlanPreservesIncumbentWhenAllZonesReject) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{0u, 0u}, {1u, 0u}}, {{2u, 0u}, {3u, 0u}}};
  const std::vector<Edge> candidate{
      {{0u, 0u}, {1u, 1u}}, {{2u, 1u}, {3u, 0u}}};
  ManualObjectDranTranslationResponse reject;
  reject.valid = true;
  reject.predictedPhysicalDelta = 1.0;

  const ManualObjectDranNetworkPartitionPlan plan =
      manualObjectDranPlanNetworkPartitionForTest(
          incumbent, candidate, {reject, reject});

  EXPECT_EQ(plan.committedEdges, incumbent);
  ASSERT_EQ(plan.zoneSummaries.size(), 2u);
  EXPECT_FALSE(plan.zoneSummaries[0].committed);
  EXPECT_FALSE(plan.zoneSummaries[1].committed);
}

TEST(testDPGO, NetworkPartitionPlanIsIndependentOfEdgeInsertionOrder) {
  using Edge = ManualObjectDranObjectCopyEdge;
  const std::vector<Edge> incumbent{
      {{2u, 0u}, {3u, 0u}}, {{0u, 0u}, {1u, 0u}}};
  const std::vector<Edge> candidate{
      {{2u, 1u}, {3u, 0u}}, {{0u, 0u}, {1u, 1u}}};
  std::vector<Edge> reverseIncumbent(incumbent.rbegin(), incumbent.rend());
  std::vector<Edge> reverseCandidate(candidate.rbegin(), candidate.rend());
  ManualObjectDranTranslationResponse accept;
  accept.valid = true;
  accept.predictedPhysicalDelta = -1.0;

  const ManualObjectDranNetworkPartitionPlan first =
      manualObjectDranPlanNetworkPartitionForTest(
          incumbent, candidate, {accept, accept});
  const ManualObjectDranNetworkPartitionPlan second =
      manualObjectDranPlanNetworkPartitionForTest(
          reverseIncumbent, reverseCandidate, {accept, accept});

  EXPECT_EQ(first.committedEdges, second.committedEdges);
  ASSERT_EQ(first.zoneSummaries.size(), second.zoneSummaries.size());
  for (std::size_t index = 0; index < first.zoneSummaries.size(); ++index) {
    EXPECT_EQ(first.zoneSummaries[index].zoneId,
              second.zoneSummaries[index].zoneId);
    EXPECT_EQ(first.zoneSummaries[index].status,
              second.zoneSummaries[index].status);
    EXPECT_EQ(first.zoneSummaries[index].committed,
              second.zoneSummaries[index].committed);
  }
}

TEST(testDPGO, ComponentTranslationResponseSeparatesWrongAndRepairEdges) {
  const Matrix physicalHessian = Matrix::Identity(3, 3);
  const Matrix physicalGradient = Matrix::Zero(3, 2);
  Matrix positions(3, 2);
  positions << 0.0, 0.0,
               1.0, 0.0,
              10.0, 0.0;
  const std::vector<ManualObjectDranTranslationAssociationEdge> current{
      {0u, 1u, 1.0}};
  const std::vector<ManualObjectDranTranslationAssociationEdge> wrong{
      {0u, 2u, 1.0}};

  const ManualObjectDranTranslationResponse forward =
      manualObjectDranTranslationResponseForTest(
          physicalHessian, physicalGradient, positions, current, wrong);
  const ManualObjectDranTranslationResponse repair =
      manualObjectDranTranslationResponseForTest(
          physicalHessian, physicalGradient, positions, wrong, current);

  ASSERT_TRUE(forward.valid);
  ASSERT_TRUE(repair.valid);
  EXPECT_NEAR(forward.currentPhysicalChange, 1.0 / 9.0, 1e-12);
  EXPECT_NEAR(forward.candidatePhysicalChange, 100.0 / 9.0, 1e-12);
  EXPECT_NEAR(forward.predictedPhysicalDelta, 11.0, 1e-12);
  EXPECT_NEAR(repair.predictedPhysicalDelta, -11.0, 1e-12);
  EXPECT_LT(forward.numericalErrorBound, 1e-12);
  EXPECT_LT(repair.numericalErrorBound, 1e-12);
  EXPECT_EQ(forward.configuredDampingParameters, 0u);
  EXPECT_EQ(forward.configuredAcceptanceThresholds, 0u);
}

TEST(testDPGO, NetworkEditZoneResponseSeparatesMixedSignsHiddenByBatch) {
  const Matrix physicalHessian = Matrix::Identity(6, 6);
  const Matrix physicalGradient = Matrix::Zero(6, 2);
  Matrix positions(6, 2);
  positions << 0.0, 0.0,
               1.0, 0.0,
              20.0, 0.0,
               0.0, 0.0,
               1.0, 0.0,
              10.0, 0.0;
  const std::vector<ManualObjectDranTranslationAssociationEdge> current{
      {0u, 2u, 1.0}, {3u, 4u, 1.0}};
  const std::vector<ManualObjectDranTranslationAssociationEdge> candidate{
      {0u, 1u, 1.0}, {3u, 5u, 1.0}};

  const ManualObjectDranTranslationResponse batch =
      manualObjectDranTranslationResponseForTest(
          physicalHessian, physicalGradient, positions, current, candidate);
  const ManualObjectDranTranslationResponse beneficial =
      manualObjectDranTranslationResponseForTest(
          physicalHessian, physicalGradient, positions,
          {{0u, 2u, 1.0}}, {{0u, 1u, 1.0}});
  const ManualObjectDranTranslationResponse harmful =
      manualObjectDranTranslationResponseForTest(
          physicalHessian, physicalGradient, positions,
          {{3u, 4u, 1.0}}, {{3u, 5u, 1.0}});

  ASSERT_TRUE(batch.valid);
  ASSERT_TRUE(beneficial.valid);
  ASSERT_TRUE(harmful.valid);
  EXPECT_LT(batch.predictedPhysicalDelta, 0.0);
  EXPECT_LT(beneficial.predictedPhysicalDelta, 0.0);
  EXPECT_GT(harmful.predictedPhysicalDelta, 0.0);
  EXPECT_NEAR(batch.predictedPhysicalDelta,
              beneficial.predictedPhysicalDelta +
                  harmful.predictedPhysicalDelta,
              1e-12);
}

TEST(testDPGO, ComponentTranslationQuadraticUsesPhysicalMeasurementFactors) {
  Matrix state = Matrix::Zero(2, 6);
  state.middleCols(0, 2) = Matrix::Identity(2, 2);
  state.col(2) << 1.0, 2.0;
  state.middleCols(3, 2) = Matrix::Identity(2, 2);
  state.col(5) << 4.0, 6.0;
  RelativeSEMeasurement measurement;
  measurement.r1 = 0u;
  measurement.r2 = 0u;
  measurement.p1 = 0u;
  measurement.p2 = 1u;
  measurement.R = Matrix::Identity(2, 2);
  measurement.t = Vector(2);
  measurement.t << 2.0, 1.0;
  measurement.kappa = 1.0;
  measurement.tau = 3.0;
  measurement.isKnownInlier = true;
  measurement.weight = 0.5;

  const ManualObjectDranTranslationSchurModel model =
      manualObjectDranTranslationQuadraticForTest(
          {measurement}, state, 2u, 2u);

  Matrix expectedHessian(2, 2);
  expectedHessian << 3.0, -3.0, -3.0, 3.0;
  Matrix expectedGradient(2, 2);
  expectedGradient << -3.0, -9.0, 3.0, 9.0;
  ASSERT_TRUE(model.valid);
  EXPECT_LT((model.hessian - expectedHessian).norm(), 1e-12);
  EXPECT_LT((model.gradient - expectedGradient).norm(), 1e-12);
}

TEST(testDPGO, ComponentTranslationSchurMatchesDirectBoundaryResponse) {
  Matrix a(5, 5);
  a << 2.0, -0.4, 0.1, 0.0, 0.2,
      -0.4, 1.8, -0.2, 0.3, 0.0,
       0.1, -0.2, 1.6, -0.1, 0.2,
       0.0, 0.3, -0.1, 2.2, -0.3,
       0.2, 0.0, 0.2, -0.3, 1.9;
  const Matrix hessian = a.transpose() * a + Matrix::Identity(5, 5);
  Matrix gradient(5, 2);
  gradient << 0.2, -0.1,
             -0.4, 0.3,
              0.1, 0.5,
             -0.2, -0.3,
              0.6, 0.2;
  const std::vector<std::size_t> boundary{1u, 4u};

  const ManualObjectDranTranslationSchurModel reduced =
      manualObjectDranTranslationSchurForTest(
          hessian, gradient, boundary);
  const Matrix boundaryStep =
      -reduced.hessian.ldlt().solve(reduced.gradient);
  const Matrix fullStep = -hessian.ldlt().solve(gradient);

  ASSERT_TRUE(reduced.valid);
  EXPECT_LT((boundaryStep.row(0) - fullStep.row(1)).norm(), 1e-10);
  EXPECT_LT((boundaryStep.row(1) - fullStep.row(4)).norm(), 1e-10);
  EXPECT_EQ(reduced.configuredAnchors, 0u);
  EXPECT_EQ(reduced.configuredDampingParameters, 0u);
}

TEST(testDPGO, CompletePoint2ConsensusProjectionUsesReceivedEndpointsOnly) {
  constexpr unsigned d = 3u;
  constexpr std::size_t blockWidth = d + 1u;
  const std::vector<std::size_t> objectOffsets{1u, 1u, 1u};
  std::vector<Matrix> states(3, Matrix::Zero(d, 3u * blockWidth));
  for (std::size_t robot = 0; robot < states.size(); ++robot) {
    for (std::size_t pose = 0; pose < 3u; ++pose) {
      states[robot].block(0, pose * blockWidth, d, d) =
          Matrix::Identity(d, d);
    }
    states[robot].col(d) << 100.0 + robot, 200.0 + robot, 300.0 + robot;
    states[robot].col((objectOffsets[robot] + 0u) * blockWidth + d)
        << 3.0 * robot, 6.0 * robot, 10.0 + robot;
    states[robot].col((objectOffsets[robot] + 1u) * blockWidth + d)
        << 2.0 + robot, -4.0 + 2.0 * robot, 20.0 + robot;
  }
  const std::vector<Matrix> before = states;
  std::vector<std::vector<Matrix>> cache(
      states.size(), std::vector<Matrix>(states.size()));
  for (std::size_t receiver = 0; receiver < states.size(); ++receiver) {
    for (std::size_t sender = 0; sender < states.size(); ++sender) {
      cache[receiver][sender] = states[sender];
    }
  }

  const std::size_t projected =
      manualObjectDranProjectCompletePoint2ConsensusForTest(
          objectOffsets, 2u, d, 1.0, states, cache);

  EXPECT_EQ(projected, 6u);
  for (std::size_t robot = 0; robot < states.size(); ++robot) {
    EXPECT_TRUE(states[robot].block(0, 0, d, blockWidth).isApprox(
        before[robot].block(0, 0, d, blockWidth), 0.0));
    const std::size_t firstCol = objectOffsets[robot] * blockWidth + d;
    const std::size_t secondCol = (objectOffsets[robot] + 1u) * blockWidth + d;
    EXPECT_TRUE(states[robot].col(firstCol).head(2).isApprox(
        (Vector(2) << 3.0, 6.0).finished(), 1e-15));
    EXPECT_TRUE(states[robot].col(secondCol).head(2).isApprox(
        (Vector(2) << 3.0, -2.0).finished(), 1e-15));
    EXPECT_DOUBLE_EQ(states[robot](2, firstCol), before[robot](2, firstCol));
    EXPECT_DOUBLE_EQ(states[robot](2, secondCol), before[robot](2, secondCol));
    for (std::size_t sender = 0; sender < states.size(); ++sender) {
      const std::size_t senderFirst = objectOffsets[sender] * blockWidth + d;
      const std::size_t senderSecond =
          (objectOffsets[sender] + 1u) * blockWidth + d;
      EXPECT_TRUE(cache[robot][sender].col(senderFirst).head(2).isApprox(
          (Vector(2) << 3.0, 6.0).finished(), 1e-15));
      EXPECT_TRUE(cache[robot][sender].col(senderSecond).head(2).isApprox(
          (Vector(2) << 3.0, -2.0).finished(), 1e-15));
      EXPECT_DOUBLE_EQ(cache[robot][sender](2, senderFirst),
                       before[sender](2, senderFirst));
      EXPECT_DOUBLE_EQ(cache[robot][sender](2, senderSecond),
                       before[sender](2, senderSecond));
    }
  }
}

TEST(testDPGO, CompletePoint2ConsensusProjectionSupportsRelaxation) {
  constexpr unsigned d = 3u;
  constexpr std::size_t blockWidth = d + 1u;
  const std::vector<std::size_t> objectOffsets{1u, 1u};
  std::vector<Matrix> states(2, Matrix::Zero(d, 2u * blockWidth));
  for (Matrix &state : states) {
    state.block(0, 0, d, d) = Matrix::Identity(d, d);
    state.block(0, blockWidth, d, d) = Matrix::Identity(d, d);
  }
  states[0].col(blockWidth + d) << 0.0, 2.0, 10.0;
  states[1].col(blockWidth + d) << 4.0, 6.0, 20.0;
  std::vector<std::vector<Matrix>> cache(
      states.size(), std::vector<Matrix>(states.size()));
  cache[0][1] = states[1];
  cache[1][0] = states[0];

  manualObjectDranProjectCompletePoint2ConsensusForTest(
      objectOffsets, 1u, d, 0.25, states, cache);

  EXPECT_TRUE(states[0].col(blockWidth + d).head(2).isApprox(
      (Vector(2) << 0.5, 2.5).finished(), 1e-15));
  EXPECT_TRUE(states[1].col(blockWidth + d).head(2).isApprox(
      (Vector(2) << 3.5, 5.5).finished(), 1e-15));
  EXPECT_DOUBLE_EQ(states[0](2, blockWidth + d), 10.0);
  EXPECT_DOUBLE_EQ(states[1](2, blockWidth + d), 20.0);
}

TEST(testDPGO, CompleteObjectPoseMeanUsesAllReceivedPhysicalCopies) {
  constexpr unsigned d = 3u;
  constexpr std::size_t blockWidth = d + 1u;
  const std::vector<std::size_t> objectOffsets{1u, 1u};
  std::vector<Matrix> states(2, Matrix::Zero(d, 2u * blockWidth));
  const double angle = 30.0 * M_PI / 180.0;
  Matrix positive = Matrix::Identity(d, d);
  positive(0, 0) = std::cos(angle);
  positive(0, 1) = -std::sin(angle);
  positive(1, 0) = std::sin(angle);
  positive(1, 1) = std::cos(angle);
  Matrix negative = positive.transpose();
  for (Matrix &state : states) {
    state.block(0, 0, d, d) = Matrix::Identity(d, d);
  }
  states[0].block(0, blockWidth, d, d) = positive;
  states[1].block(0, blockWidth, d, d) = negative;
  states[0].col(blockWidth + d) << 0.0, 2.0, 4.0;
  states[1].col(blockWidth + d) << 4.0, 6.0, 8.0;
  std::vector<std::vector<Matrix>> cache(
      2u, std::vector<Matrix>(2u));
  cache[0][1] = states[1];
  cache[1][0] = states[0];
  const auto cacheBefore = cache;

  const auto means = manualObjectDranCompleteObjectPoseMeansForTest(
      objectOffsets, 1u, d, states, cache);

  ASSERT_EQ(means.size(), 2u);
  for (const auto &receiverMeans : means) {
    ASSERT_EQ(receiverMeans.size(), 1u);
    EXPECT_TRUE(receiverMeans[0].leftCols(d).isApprox(
        Matrix::Identity(d, d), 1e-12));
    EXPECT_TRUE(receiverMeans[0].rightCols(1).isApprox(
        (Vector(3) << 2.0, 4.0, 6.0).finished(), 1e-12));
  }
  EXPECT_TRUE(cache[0][1].isApprox(cacheBefore[0][1], 0.0));
  EXPECT_TRUE(cache[1][0].isApprox(cacheBefore[1][0], 0.0));
}

TEST(testDPGO, CompleteObjectPoseGaugeRecoversCommonLeftAction) {
  constexpr unsigned d = 3u;
  constexpr std::size_t blockWidth = d + 1u;
  const std::size_t objectOffset = 1u;
  Matrix state = Matrix::Zero(d, 3u * blockWidth);
  for (std::size_t block = 0; block < 3u; ++block) {
    state.block(0, block * blockWidth, d, d) = Matrix::Identity(d, d);
  }
  state.col(blockWidth + d) << 1.0, 2.0, 3.0;
  state.col(2u * blockWidth + d) << -2.0, 1.0, 4.0;

  const double angle = 20.0 * std::acos(-1.0) / 180.0;
  Matrix expected = Matrix::Zero(d, blockWidth);
  expected.leftCols(d) = Matrix::Identity(d, d);
  expected(0, 0) = std::cos(angle);
  expected(0, 1) = -std::sin(angle);
  expected(1, 0) = std::sin(angle);
  expected(1, 1) = std::cos(angle);
  expected.col(d) << 0.5, -1.0, 2.0;

  std::vector<Matrix> means(2u, Matrix::Zero(d, blockWidth));
  for (std::size_t object = 0; object < means.size(); ++object) {
    const Matrix source = state.block(
        0, (objectOffset + object) * blockWidth, d, blockWidth);
    means[object].leftCols(d).noalias() =
        expected.leftCols(d) * source.leftCols(d);
    means[object].col(d).noalias() =
        expected.leftCols(d) * source.col(d) + expected.col(d);
  }

  const Matrix estimated = manualObjectDranCompleteObjectPoseGaugeForTest(
      state, objectOffset, means, d);

  EXPECT_TRUE(estimated.isApprox(expected, 1e-12));
}

TEST(testDPGO, ObjectPose3InterfaceOwnerPcgMatchesDenseSolve) {
  constexpr std::size_t blockDimension = 6u;
  constexpr std::size_t numObjects = 2u;
  constexpr Eigen::Index size = blockDimension * numObjects;
  std::vector<ManualObjectDranPoint2SchurModel> models(2u);
  for (std::size_t robot = 0; robot < models.size(); ++robot) {
    Matrix seed = Matrix::Zero(size, size);
    for (Eigen::Index row = 0; row < size; ++row) {
      for (Eigen::Index col = 0; col < size; ++col) {
        seed(row, col) =
            0.01 * static_cast<double>((row + 1) * (col + 2) + robot);
      }
    }
    models[robot].hessian =
        seed.transpose() * seed +
        (2.0 + static_cast<double>(robot)) * Matrix::Identity(size, size);
    models[robot].gradient =
        Vector::LinSpaced(size, -0.4 + 0.1 * robot, 0.7 + 0.1 * robot);
    models[robot].valid = true;
  }
  const std::vector<std::vector<std::size_t>> observers{{0u, 1u}, {0u, 1u}};
  const std::vector<std::size_t> owners{0u, 1u};
  const Vector warmStart = Vector::Zero(size);

  const auto solve = manualObjectDranSolveInterfaceOwnerPcgForTest(
      models, observers, owners, blockDimension, warmStart, size, 1e-12,
      false);
  Matrix hessian = models[0].hessian + models[1].hessian;
  Vector gradient = models[0].gradient + models[1].gradient;
  const Vector expected = hessian.ldlt().solve(-gradient);

  ASSERT_TRUE(solve.valid) << solve.status;
  EXPECT_TRUE(solve.pointStep.isApprox(expected, 1e-10));
  EXPECT_LE(solve.relativeResidual, 1e-10);
  EXPECT_GT(solve.totalPayloadBytes, 0u);
  EXPECT_EQ(solve.totalMessages,
            2u + solve.iterations * (2u + 2u + 4u));
}

TEST(testDPGO, ChordalPose3FactorLinearizationMatchesFiniteDifferences) {
  Matrix first = Matrix::Zero(3, 4);
  Matrix measurement = Matrix::Zero(3, 4);
  Matrix second = Matrix::Zero(3, 4);
  first.leftCols(3) = so3Exp((Eigen::Vector3d() << 0.1, -0.2, 0.05).finished());
  measurement.leftCols(3) =
      so3Exp((Eigen::Vector3d() << -0.03, 0.08, 0.12).finished());
  second.leftCols(3) =
      so3Exp((Eigen::Vector3d() << 0.15, -0.06, 0.2).finished());
  first.col(3) << 0.4, -0.7, 1.2;
  measurement.col(3) << 1.0, 0.3, -0.2;
  second.col(3) << 1.5, -0.1, 0.8;
  const auto linearization =
      manualObjectDranLinearizeChordalPose3FactorForTest(
          first, measurement, second);
  const double epsilon = 1e-7;
  auto residual = [&](const Matrix &a, const Matrix &b) {
    Vector value(12);
    const Matrix rotation =
        b.leftCols(3) - a.leftCols(3) * measurement.leftCols(3);
    Eigen::Map<Matrix>(value.data(), 3, 3) = rotation;
    value.tail(3) = b.col(3) - a.col(3) -
                    a.leftCols(3) * measurement.col(3);
    return value;
  };
  Matrix numericFirst = Matrix::Zero(12, 6);
  Matrix numericSecond = Matrix::Zero(12, 6);
  for (Eigen::Index column = 0; column < 6; ++column) {
    Matrix firstPlus = first;
    Matrix firstMinus = first;
    Matrix secondPlus = second;
    Matrix secondMinus = second;
    if (column < 3) {
      Eigen::Vector3d delta = Eigen::Vector3d::Zero();
      delta(column) = epsilon;
      firstPlus.leftCols(3) = first.leftCols(3) * so3Exp(delta);
      firstMinus.leftCols(3) = first.leftCols(3) * so3Exp(-delta);
      secondPlus.leftCols(3) = second.leftCols(3) * so3Exp(delta);
      secondMinus.leftCols(3) = second.leftCols(3) * so3Exp(-delta);
    } else {
      firstPlus(column - 3, 3) += epsilon;
      firstMinus(column - 3, 3) -= epsilon;
      secondPlus(column - 3, 3) += epsilon;
      secondMinus(column - 3, 3) -= epsilon;
    }
    numericFirst.col(column) =
        (residual(firstPlus, second) - residual(firstMinus, second)) /
        (2.0 * epsilon);
    numericSecond.col(column) =
        (residual(first, secondPlus) - residual(first, secondMinus)) /
        (2.0 * epsilon);
  }
  Matrix analyticFirst = Matrix::Zero(12, 6);
  Matrix analyticSecond = Matrix::Zero(12, 6);
  analyticFirst.topRows(9) = linearization.firstRotationJacobian;
  analyticFirst.bottomRows(3) = linearization.firstTranslationJacobian;
  analyticSecond.topRows(9) = linearization.secondRotationJacobian;
  analyticSecond.bottomRows(3) = linearization.secondTranslationJacobian;

  EXPECT_TRUE(analyticFirst.isApprox(numericFirst, 1e-7));
  EXPECT_TRUE(analyticSecond.isApprox(numericSecond, 1e-7));
}

TEST(testDPGO, CompletePoint2ConsensusProjectionWarmupStopsDeterministically) {
  EXPECT_FALSE(manualObjectDranPoint2ConsensusProjectionActiveForTest(
      false, 5u, 1u));
  EXPECT_TRUE(manualObjectDranPoint2ConsensusProjectionActiveForTest(
      true, 5u, 1u));
  EXPECT_TRUE(manualObjectDranPoint2ConsensusProjectionActiveForTest(
      true, 5u, 5u));
  EXPECT_FALSE(manualObjectDranPoint2ConsensusProjectionActiveForTest(
      true, 5u, 6u));
  EXPECT_TRUE(manualObjectDranPoint2ConsensusProjectionActiveForTest(
      true, 0u, 100u));
}

TEST(testDPGO, CompletePoint2ConsensusDualResetOccursOnceAtWarmupBoundary) {
  EXPECT_FALSE(manualObjectDranPoint2ConsensusShouldResetDualForTest(
      false, 2u, 2u));
  EXPECT_FALSE(manualObjectDranPoint2ConsensusShouldResetDualForTest(
      true, 0u, 2u));
  EXPECT_FALSE(manualObjectDranPoint2ConsensusShouldResetDualForTest(
      true, 2u, 1u));
  EXPECT_TRUE(manualObjectDranPoint2ConsensusShouldResetDualForTest(
      true, 2u, 2u));
  EXPECT_FALSE(manualObjectDranPoint2ConsensusShouldResetDualForTest(
      true, 2u, 3u));
}

TEST(testDPGO, DirectEdgeDualRelaxationWaitsForPoint2Warmup) {
  EXPECT_TRUE(manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
      false, 2u, 0u));
  EXPECT_FALSE(manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
      true, 2u, 0u));
  EXPECT_FALSE(manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
      true, 2u, 2u));
  EXPECT_TRUE(manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
      true, 2u, 3u));
  EXPECT_FALSE(manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
      true, 0u, 100u));
}
