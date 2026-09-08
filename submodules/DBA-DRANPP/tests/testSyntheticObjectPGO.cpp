#include <DPGO/ManualObjectDran.h>
#include <DPGO/ObjectPGOData.h>
#include <DPGO/SyntheticObjectPGO.h>

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace DPGO;

namespace {

std::filesystem::path syntheticTestDirectory(const std::string &name) {
  return std::filesystem::temp_directory_path() / name;
}

std::size_t countCsvDataRows(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string line;
  std::size_t rows = 0;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      ++rows;
    }
  }
  return rows > 0 ? rows - 1 : 0;
}

std::string readTextFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not read test file: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

std::string readSyntheticGraphs(const std::filesystem::path &directory,
                                std::size_t numRobots) {
  std::ostringstream contents;
  for (std::size_t robotId = 0; robotId < numRobots; ++robotId) {
    contents << readTextFile(directory /
                             ("rob_" + std::to_string(robotId) + ".g2o"));
  }
  return contents.str();
}

std::vector<std::string> splitCsvTestLine(const std::string &line) {
  std::stringstream ss(line);
  std::vector<std::string> tokens;
  std::string token;
  while (std::getline(ss, token, ',')) {
    tokens.push_back(token);
  }
  return tokens;
}

Eigen::Quaterniond firstSe3EdgeQuaternionToObject(
    const std::filesystem::path &path, std::size_t objectVertexId) {
  std::ifstream input(path);
  std::string tag;
  while (input >> tag) {
    if (tag != "EDGE_SE3:QUAT") {
      std::string rest;
      std::getline(input, rest);
      continue;
    }
    std::size_t from = 0;
    std::size_t to = 0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
    input >> from >> to >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    std::string rest;
    std::getline(input, rest);
    if (to == objectVertexId) {
      return Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    }
  }
  throw std::runtime_error("Could not find object EDGE_SE3:QUAT");
}

}  // namespace

TEST(testDPGO, ManualObjectDranDormantDirectEdgeDualRetentionPolicy) {
  EXPECT_FALSE(manualObjectDranShouldPruneInactiveDirectEdgeDualForTest(
      true, false));
  EXPECT_TRUE(manualObjectDranShouldPruneInactiveDirectEdgeDualForTest(
      false, false));
  EXPECT_FALSE(manualObjectDranShouldPruneInactiveDirectEdgeDualForTest(
      false, true));
}

TEST(testDPGO, ManualObjectDranFixedUnionModelRetainsInactiveTargets) {
  EXPECT_FALSE(manualObjectDranShouldPruneInactiveDirectEdgeTargetForTest(
      true, false));
  EXPECT_TRUE(manualObjectDranShouldPruneInactiveDirectEdgeTargetForTest(
      false, false));
  EXPECT_FALSE(manualObjectDranShouldPruneInactiveDirectEdgeTargetForTest(
      false, true));
}

TEST(testDPGO,
     SyntheticObjectPGODirectEdgePoint2ProjectionSendsNoProtocolPayload) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_direct_edge_point2_shared_only");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 3;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.1;
  synthetic.objectInitialDriftPerRobot = 0.1;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t objectId = 0; objectId < synthetic.numObjects;
         ++objectId) {
      out << "0,1," << objectId << "," << objectId << ",1\n";
      out << "1,0," << objectId << "," << objectId << ",1\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::InputState;
  options.objectOrientationObserved = false;
  options.objectCommunicationRepresentation =
      ManualObjectDranObjectCommunicationRepresentation::Point2;
  options.associationPolicy = ManualObjectDranAssociationPolicy::Manual;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationFreezeUpperVariables = true;
  options.associationRequireObservedObjects = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationTransformDimension = 3;
  options.associationGaugeSynchronizationEnabled = true;
  options.associationGaugeSynchronizationApplyToStates = true;
  options.associationGaugeSynchronizationApplyOnceToInitialStates = false;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::DirectEdgeDual;
  options.associationLowerDirectEdgeSynchronousSharedOnly = true;
  options.associationLowerDirectEdgePoint2ConsensusProjection = true;
  options.associationLowerDirectEdgePoint2ConsensusProjectionRelaxation = 0.5;
  options.associationLowerDirectEdgePoint2ConsensusProjectionRounds = 1;
  options.beta = 0.2;
  options.associationLowerDirectEdgeAdaptiveBeta = true;
  options.associationLowerDirectEdgeBetaMin = 0.05;
  options.associationLowerDirectEdgeBetaMax = 0.5;
  options.associationLowerDirectEdgeBetaUpdatePeriod = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const std::size_t directedBlocksPerRound =
      synthetic.numRobots * (synthetic.numRobots - 1) * synthetic.numObjects;
  const double expectedRoundMb =
      static_cast<double>(directedBlocksPerRound * 2u * sizeof(double)) /
      (1024.0 * 1024.0);
  for (const ManualObjectDranIterationSummary &row : result.iterations) {
    EXPECT_EQ(row.commPoseCount, directedBlocksPerRound);
    EXPECT_DOUBLE_EQ(row.iterCommMb, expectedRoundMb);
    EXPECT_EQ(row.lowerSharedObjectConsensusActiveHypotheses,
              directedBlocksPerRound);
    EXPECT_EQ(row.lowerSharedObjectConsensusTargetMessages, 0u);
    EXPECT_EQ(row.lowerSharedObjectConsensusDeliveredMessages, 0u);
    EXPECT_EQ(row.lowerSharedObjectConsensusCertificateMessages, 0u);
    EXPECT_EQ(row.lowerSharedObjectConsensusPayloadBytes, 0u);
    EXPECT_EQ(row.lowerSharedObjectConsensusCertificatePayloadBytes, 0u);
    EXPECT_EQ(row.lowerSharedObjectConsensusAdaptiveBetaEdges,
              directedBlocksPerRound);
    EXPECT_GE(row.lowerSharedObjectConsensusAdaptiveBetaMean, 0.05);
    EXPECT_LE(row.lowerSharedObjectConsensusAdaptiveBetaMean, 0.5);
    EXPECT_TRUE(std::isfinite(row.measurementCost));
  }
  EXPECT_DOUBLE_EQ(result.iterations.back().cumulativeCommMb,
                   2.0 * expectedRoundMb);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOExactPoint2RoundIsMonotoneAndSynchronized) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_exact_point2_owner");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 3;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.1;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t object = 0; object < synthetic.numObjects; ++object) {
      out << "0,1," << object << "," << object << ",1\n";
      out << "1,0," << object << "," << object << ",1\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.communicationPolicy = ManualObjectDranCommunicationPolicy::All;
  options.objectInitialization = ManualObjectDranObjectInitialization::InputState;
  options.objectOrientationObserved = false;
  options.objectCommunicationRepresentation =
      ManualObjectDranObjectCommunicationRepresentation::Point2;
  options.associationPolicy = ManualObjectDranAssociationPolicy::Manual;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationFreezeUpperVariables = true;
  options.associationRequireObservedObjects = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationTransformDimension = 3;
  options.associationGaugeSynchronizationEnabled = true;
  options.associationGaugeSynchronizationApplyToStates = true;
  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::ExactPoint2Owner;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const auto &initial = result.iterations.front();
  const auto &updated = result.iterations.back();
  EXPECT_EQ(initial.exactPoint2IterationPayloadBytes, 0u);
  EXPECT_EQ(updated.exactPoint2StatisticMessages, 4u);
  EXPECT_EQ(updated.exactPoint2BroadcastMessages, 4u);
  EXPECT_EQ(updated.exactPoint2IterationPayloadBytes, 4u * (24u + 16u));
  EXPECT_EQ(updated.exactPoint2CumulativePayloadBytes, 160u);
  EXPECT_EQ(updated.lowerSharedObjectConsensusPayloadBytes, 0u);
  EXPECT_EQ(updated.lowerSharedObjectConsensusCertificatePayloadBytes, 0u);
  EXPECT_LE(updated.measurementCost,
            initial.measurementCost +
                1e-10 * std::max(1.0, std::abs(initial.measurementCost)))
      << " conditional_before="
      << updated.exactPoint2TrajectoryConditionalCostBefore
      << " conditional_after="
      << updated.exactPoint2TrajectoryConditionalCostAfter
      << " physical_after_trajectory="
      << updated.exactPoint2PhysicalCostAfterTrajectory
      << " physical_after_update="
      << updated.exactPoint2PhysicalCostAfterUpdate
      << " accepted=" << updated.exactPoint2TrajectoryAcceptedSolves
      << " rejected=" << updated.exactPoint2TrajectoryRejectedSolves;
  EXPECT_LE(updated.exactPoint2MaxCopySpread, 1e-12);
  EXPECT_EQ(updated.exactPoint2Status, "ok");

  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::ExactPoint2SchurOracle;
  options.maxIterations = 3;
  const ManualObjectDranRunResult oracle =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(oracle.iterations.size(), 4u);
  const auto &oracleInitial = oracle.iterations.front();
  const auto &oracleUpdated = oracle.iterations.back();
  EXPECT_TRUE(oracleUpdated.point2SchurOracleActive);
  EXPECT_EQ(oracleUpdated.exactPoint2TrajectoryAcceptedSolves, 0u);
  EXPECT_EQ(oracleUpdated.exactPoint2TrajectoryRejectedSolves, 0u);
  EXPECT_GT(oracleUpdated.point2SchurOracleResponseSolves, 0u);
  EXPECT_TRUE(std::isfinite(
      oracleUpdated.point2SchurOracleMaxResponseRelativeResidual));
  EXPECT_TRUE(std::isfinite(oracleUpdated.point2SchurOracleSymmetryError));
  EXPECT_TRUE(std::isfinite(oracleUpdated.point2SchurOracleSpectralShift));
  EXPECT_TRUE(
      std::isfinite(oracleUpdated.point2SchurOracleLinearRelativeResidual));
  EXPECT_EQ(oracleUpdated.exactPoint2IterationPayloadBytes, 0u);
  EXPECT_DOUBLE_EQ(oracleUpdated.iterCommMb, 0.0);
  EXPECT_LE(oracleUpdated.measurementCost,
            oracleInitial.measurementCost +
                1e-10 * std::max(1.0, std::abs(oracleInitial.measurementCost)))
      << oracleUpdated.exactPoint2Status;
  for (std::size_t index = 1; index < oracle.iterations.size(); ++index) {
    const double previousCost = oracle.iterations[index - 1].measurementCost;
    const double currentCost = oracle.iterations[index].measurementCost;
    EXPECT_LE(currentCost,
              previousCost +
                  1e-10 * std::max(1.0, std::abs(previousCost)))
        << "iteration " << oracle.iterations[index].iter << " status="
        << oracle.iterations[index].exactPoint2Status;
  }
  EXPECT_LE(oracleUpdated.exactPoint2MaxCopySpread, 1e-12);
  EXPECT_TRUE(oracleUpdated.exactPoint2Status == "ok" ||
              oracleUpdated.exactPoint2Status ==
                  "no_decreasing_schur_step" ||
              oracleUpdated.exactPoint2Status == "global_merit_rejected" ||
              oracleUpdated.exactPoint2Status ==
                  "outer_global_merit_rejected")
      << oracleUpdated.exactPoint2Status;

  options.associationLowerSharedLatentMode =
      ManualObjectDranSharedLatentMode::ExactPoint2OwnerPcg;
  options.exactPoint2OwnerPcgIterations = 1u;
  const ManualObjectDranRunResult ownerPcg =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(ownerPcg.iterations.size(), 4u);
  for (std::size_t index = 1; index < ownerPcg.iterations.size(); ++index) {
    const auto &previous = ownerPcg.iterations[index - 1];
    const auto &current = ownerPcg.iterations[index];
    EXPECT_TRUE(current.point2OwnerPcgActive);
    EXPECT_EQ(current.point2OwnerPcgIterations, 1u);
    EXPECT_GT(current.exactPoint2IterationPayloadBytes, 0u);
    EXPECT_EQ(current.lowerSharedObjectConsensusPayloadBytes, 0u);
    EXPECT_EQ(current.lowerSharedObjectConsensusCertificatePayloadBytes, 0u);
    EXPECT_LE(current.measurementCost,
              previous.measurementCost +
                  1e-10 * std::max(1.0, std::abs(previous.measurementCost)));
    EXPECT_LE(current.exactPoint2MaxCopySpread, 1e-12);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     ManualObjectDranAutomaticTrajectoryTranslationProxUsesActivePairDensity) {
  EXPECT_DOUBLE_EQ(
      automaticLowerTrajectoryTranslationProxWeight(0.25, 10, 5), 0.25);
  EXPECT_DOUBLE_EQ(
      automaticLowerTrajectoryTranslationProxWeight(-1.0, 10, 5), 2.0);
  EXPECT_DOUBLE_EQ(
      automaticLowerTrajectoryTranslationProxWeight(-1.0, 0, 5), 0.0);
  EXPECT_DOUBLE_EQ(
      automaticLowerTrajectoryTranslationProxWeight(-1.0, 10, 0), 0.0);
  EXPECT_DOUBLE_EQ(automaticLowerTrajectoryTranslationProxWeight(
                       std::numeric_limits<double>::quiet_NaN(), 10, 5),
                   0.0);
}

TEST(testDPGO,
     ManualObjectDranAutomaticObjectCopyTargetPrecisionUsesActiveMean) {
  EXPECT_DOUBLE_EQ(automaticObjectCopyTargetPrecisionScale(4.0, 2.0), 2.0);
  EXPECT_DOUBLE_EQ(automaticObjectCopyTargetPrecisionScale(0.5, 2.0), 0.25);
  EXPECT_DOUBLE_EQ(automaticObjectCopyTargetPrecisionScale(0.0, 2.0), 0.0);
  EXPECT_DOUBLE_EQ(automaticObjectCopyTargetPrecisionScale(4.0, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(automaticObjectCopyTargetPrecisionScale(
                       std::numeric_limits<double>::quiet_NaN(), 2.0),
                   1.0);
}

TEST(testDPGO,
     ManualObjectDranStateValidationTrajectoryGuardRejectsIncrease) {
  EXPECT_FALSE(associationStateValidationTrajectoryGuardRejects(1.0, 0.5));
  EXPECT_FALSE(associationStateValidationTrajectoryGuardRejects(1.0, 1.0));
  EXPECT_TRUE(associationStateValidationTrajectoryGuardRejects(1.0, 1.25));
  EXPECT_FALSE(associationStateValidationTrajectoryGuardRejects(
      std::numeric_limits<double>::quiet_NaN(), 1.25));
  EXPECT_FALSE(associationStateValidationTrajectoryGuardRejects(
      1.0, std::numeric_limits<double>::quiet_NaN()));
}

TEST(testDPGO, SyntheticObjectPGOWritesLoadableObjectAwareDirectory) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_loadable");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions options;
  options.numRobots = 2;
  options.numObjects = 3;
  options.posesPerRobot = 4;
  writeSyntheticObjectPGODirectory(dir.string(), options);

  const ObjectPGODirectoryData dataset =
      loadObjectAwareG2ODirectory(dir.string(), options.numRobots, 0);

  EXPECT_EQ(dataset.dimension, 2u);
  EXPECT_EQ(dataset.numRobots, options.numRobots);
  EXPECT_EQ(dataset.numObjects, options.numObjects);
  ASSERT_EQ(dataset.robots.size(), options.numRobots);
  for (const ObjectPGORobotData &robot : dataset.robots) {
    EXPECT_EQ(robot.trajectoryVertexIds.size(), options.posesPerRobot);
    EXPECT_EQ(robot.objectLocalVertexIds.size(), options.numObjects);
    EXPECT_EQ(robot.trajectoryMeasurements.size(), options.posesPerRobot - 1);
    EXPECT_EQ(robot.objectObservationMeasurements.size(), options.numObjects);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOWritesRealistic3DDirectory) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_realistic_3d");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions options;
  options.dimension = 3;
  options.numRobots = 3;
  options.numObjects = 8;
  options.posesPerRobot = 10;
  options.numCategories = 4;
  options.observeAllObjects = false;
  options.detectionRange = 9.0;
  options.horizontalFovDeg = 140.0;
  options.verticalFovDeg = 90.0;
  options.missDetectionRate = 0.15;
  options.observationTranslationNoise = 0.03;
  options.observationRotationNoise = 0.01;
  options.worldLength = 20.0;
  options.worldWidth = 12.0;
  options.worldHeight = 4.0;
  writeSyntheticObjectPGODirectory(dir.string(), options);

  const ObjectPGODirectoryData dataset =
      loadObjectAwareG2ODirectory(dir.string(), options.numRobots,
                                  options.numObjects);

  EXPECT_EQ(dataset.dimension, 3u);
  EXPECT_EQ(dataset.numRobots, options.numRobots);
  EXPECT_EQ(dataset.numObjects, options.numObjects);
  ASSERT_EQ(dataset.robots.size(), options.numRobots);
  std::size_t totalObjectObservations = 0;
  for (const ObjectPGORobotData &robot : dataset.robots) {
    EXPECT_EQ(robot.trajectoryVertexIds.size(), options.posesPerRobot);
    EXPECT_EQ(robot.objectLocalVertexIds.size(), options.numObjects);
    EXPECT_EQ(robot.trajectoryMeasurements.size(), options.posesPerRobot - 1);
    totalObjectObservations += robot.objectObservationMeasurements.size();
  }
  EXPECT_GT(totalObjectObservations, 0u);
  EXPECT_LT(totalObjectObservations,
            options.numRobots * options.numObjects);

  EXPECT_TRUE(std::filesystem::exists(dir / "ground_truth_poses.csv"));
  EXPECT_TRUE(std::filesystem::exists(dir / "ground_truth_objects.csv"));
  EXPECT_TRUE(std::filesystem::exists(dir / "observations.csv"));
  EXPECT_TRUE(std::filesystem::exists(dir / "metadata.csv"));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOWritesHardAssociation3DDirectory) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_hard_assoc_3d");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions options;
  options.dimension = 3;
  options.numRobots = 3;
  options.numObjects = 8;
  options.posesPerRobot = 12;
  options.numCategories = 4;
  options.observeAllObjects = false;
  options.detectionRange = 14.0;
  options.horizontalFovDeg = 180.0;
  options.verticalFovDeg = 120.0;
  options.objectObservationsPerObject = 2;
  options.falsePositiveObjectsPerRobot = 2;
  options.duplicateObjectRatio = 0.25;
  options.permuteLocalObjectIds = true;
  options.observationTranslationNoise = 0.04;
  options.observationRotationNoise = 0.01;
  options.worldLength = 22.0;
  options.worldWidth = 14.0;
  options.worldHeight = 4.0;
  writeSyntheticObjectPGODirectory(dir.string(), options);

  const std::size_t localSlots = syntheticLocalObjectSlotCount(options);
  EXPECT_GT(localSlots, options.numObjects);

  const ObjectPGODirectoryData dataset =
      loadObjectAwareG2ODirectory(dir.string(), options.numRobots,
                                  localSlots);
  EXPECT_EQ(dataset.dimension, 3u);
  EXPECT_EQ(dataset.numObjects, localSlots);
  std::size_t totalObjectObservations = 0;
  for (const ObjectPGORobotData &robot : dataset.robots) {
    EXPECT_EQ(robot.objectLocalVertexIds.size(), localSlots);
    totalObjectObservations += robot.objectObservationMeasurements.size();
  }
  EXPECT_GT(totalObjectObservations, options.numRobots * options.numObjects);

  const std::filesystem::path objectMapPath = dir / "object_id_map.csv";
  const std::filesystem::path gtPath = dir / "gt_associations.csv";
  const std::filesystem::path difficultyPath = dir / "difficulty_report.csv";
  const std::filesystem::path localObjectsPath = dir / "local_objects.csv";
  const std::filesystem::path localMetadataPath =
      dir / "local_object_metadata.csv";
  EXPECT_TRUE(std::filesystem::exists(objectMapPath));
  EXPECT_TRUE(std::filesystem::exists(gtPath));
  EXPECT_TRUE(std::filesystem::exists(difficultyPath));
  EXPECT_TRUE(std::filesystem::exists(localObjectsPath));
  EXPECT_TRUE(std::filesystem::exists(localMetadataPath));
  EXPECT_EQ(countCsvDataRows(localMetadataPath),
            options.numRobots * localSlots);

  std::ifstream mapInput(objectMapPath);
  std::string line;
  bool sawDuplicate = false;
  bool sawFalsePositive = false;
  bool sawPermutedPrimary = false;
  std::getline(mapInput, line);
  while (std::getline(mapInput, line)) {
    const std::vector<std::string> tokens = splitCsvTestLine(line);
    ASSERT_GE(tokens.size(), 5u);
    const std::size_t localSlot =
        static_cast<std::size_t>(std::stoul(tokens[1]));
    const int trueObject = std::stoi(tokens[3]);
    if (tokens[4] == "duplicate") {
      sawDuplicate = true;
    }
    if (tokens[4] == "false_positive") {
      sawFalsePositive = true;
    }
    if (tokens[4] == "primary" && trueObject >= 0 &&
        localSlot != static_cast<std::size_t>(trueObject)) {
      sawPermutedPrimary = true;
    }
  }
  EXPECT_TRUE(sawDuplicate);
  EXPECT_TRUE(sawFalsePositive);
  EXPECT_TRUE(sawPermutedPrimary);

  std::ifstream gtInput(gtPath);
  bool sawOffDiagonalGt = false;
  std::getline(gtInput, line);
  while (std::getline(gtInput, line)) {
    const std::vector<std::string> tokens = splitCsvTestLine(line);
    ASSERT_GE(tokens.size(), 5u);
    if (tokens[2] != tokens[3]) {
      sawOffDiagonalGt = true;
      break;
    }
  }
  EXPECT_TRUE(sawOffDiagonalGt);
  EXPECT_GT(countCsvDataRows(difficultyPath), 5u);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOGeneratesDeterministicObjectMetadata) {
  SyntheticObjectPGOOptions options;
  options.numRobots = 3;
  options.numObjects = 5;
  options.numCategories = 2;
  options.descriptorDim = 6;
  options.semanticAccuracy = 0.75;
  options.descriptorNoise = 0.05;
  options.shapeNoise = 0.02;
  options.repeatedObjectRatio = 0.5;

  const SyntheticObjectMetadataSet first =
      generateSyntheticObjectMetadata(options);
  const SyntheticObjectMetadataSet second =
      generateSyntheticObjectMetadata(options);

  ASSERT_EQ(first.objects.size(), options.numObjects);
  ASSERT_EQ(first.observations.size(), options.numRobots);
  ASSERT_EQ(second.objects.size(), first.objects.size());
  for (std::size_t objectId = 0; objectId < options.numObjects; ++objectId) {
    EXPECT_EQ(first.objects[objectId].category,
              second.objects[objectId].category);
    EXPECT_EQ(first.objects[objectId].shape.rows(), 3);
    EXPECT_EQ(first.objects[objectId].descriptor.rows(),
              static_cast<int>(options.descriptorDim));
    EXPECT_TRUE(first.objects[objectId].shape.isApprox(
        second.objects[objectId].shape, 1e-12));
    EXPECT_TRUE(first.objects[objectId].descriptor.isApprox(
        second.objects[objectId].descriptor, 1e-12));
  }
  for (std::size_t robotId = 0; robotId < options.numRobots; ++robotId) {
    ASSERT_EQ(first.observations[robotId].size(), options.numObjects);
    for (std::size_t objectId = 0; objectId < options.numObjects;
         ++objectId) {
      EXPECT_EQ(first.observations[robotId][objectId].descriptor.rows(),
                static_cast<int>(options.descriptorDim));
      EXPECT_EQ(first.observations[robotId][objectId].symmetry,
                first.objects[objectId].symmetry);
      EXPECT_TRUE(std::isfinite(
          first.observations[robotId][objectId].semanticConfidence));
      EXPECT_TRUE(first.observations[robotId][objectId].shape.allFinite());
      EXPECT_TRUE(
          first.observations[robotId][objectId].descriptor.allFinite());
    }
  }
}

TEST(testDPGO, SyntheticObjectPGOInstanceSeedIsReproducible) {
  const std::filesystem::path seedZeroDir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_seed_zero");
  const std::filesystem::path seedOneFirstDir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_seed_one_first");
  const std::filesystem::path seedOneSecondDir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_seed_one_second");
  std::filesystem::remove_all(seedZeroDir);
  std::filesystem::remove_all(seedOneFirstDir);
  std::filesystem::remove_all(seedOneSecondDir);

  SyntheticObjectPGOOptions options;
  options.dimension = 3;
  options.numRobots = 3;
  options.numObjects = 12;
  options.posesPerRobot = 10;
  options.worldLength = 30.0;
  options.worldWidth = 12.0;
  options.worldHeight = 4.0;
  options.detectionRange = 15.0;
  options.observeAllObjects = false;
  options.missDetectionRate = 0.2;
  options.objectObservationsPerObject = 2;
  options.falsePositiveObjectsPerRobot = 2;
  options.duplicateObjectRatio = 0.2;
  options.permuteLocalObjectIds = true;
  options.semanticAccuracy = 0.8;
  options.descriptorNoise = 0.08;
  options.shapeNoise = 0.04;
  options.observationTranslationNoise = 0.03;
  options.observationRotationNoise = 0.02;
  options.symmetryYawAmbiguityRate = 0.35;

  options.instanceSeed = 0;
  writeSyntheticObjectPGODirectory(seedZeroDir.string(), options);
  options.instanceSeed = 1;
  writeSyntheticObjectPGODirectory(seedOneFirstDir.string(), options);
  writeSyntheticObjectPGODirectory(seedOneSecondDir.string(), options);

  const std::string seedZeroMetadata =
      readTextFile(seedZeroDir / "local_object_metadata.csv");
  const std::string seedOneFirstMetadata =
      readTextFile(seedOneFirstDir / "local_object_metadata.csv");
  const std::string seedOneSecondMetadata =
      readTextFile(seedOneSecondDir / "local_object_metadata.csv");
  const std::string seedZeroObservations =
      readTextFile(seedZeroDir / "observations.csv");
  const std::string seedOneFirstObservations =
      readTextFile(seedOneFirstDir / "observations.csv");
  const std::string seedOneSecondObservations =
      readTextFile(seedOneSecondDir / "observations.csv");
  const std::string seedZeroGraphs =
      readSyntheticGraphs(seedZeroDir, options.numRobots);
  const std::string seedOneFirstGraphs =
      readSyntheticGraphs(seedOneFirstDir, options.numRobots);
  const std::string seedOneSecondGraphs =
      readSyntheticGraphs(seedOneSecondDir, options.numRobots);

  EXPECT_EQ(seedOneFirstMetadata, seedOneSecondMetadata);
  EXPECT_EQ(seedOneFirstObservations, seedOneSecondObservations);
  EXPECT_EQ(seedOneFirstGraphs, seedOneSecondGraphs);
  EXPECT_NE(seedZeroMetadata, seedOneFirstMetadata);
  EXPECT_NE(seedZeroObservations, seedOneFirstObservations);
  EXPECT_NE(seedZeroGraphs, seedOneFirstGraphs);
  EXPECT_NE(readTextFile(seedZeroDir / "difficulty_report.csv")
                .find("instance_seed,0\n"),
            std::string::npos);
  EXPECT_NE(readTextFile(seedOneFirstDir / "difficulty_report.csv")
                .find("instance_seed,1\n"),
            std::string::npos);

  std::filesystem::remove_all(seedZeroDir);
  std::filesystem::remove_all(seedOneFirstDir);
  std::filesystem::remove_all(seedOneSecondDir);
}

TEST(testDPGO, SyntheticObjectPGOWritesLocalObjectSymmetryMetadata) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_local_symmetry");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions options;
  options.dimension = 3;
  options.numRobots = 1;
  options.numObjects = 3;
  options.posesPerRobot = 4;
  options.descriptorDim = 2;
  writeSyntheticObjectPGODirectory(dir.string(), options);

  std::ifstream in(dir / "local_object_metadata.csv");
  ASSERT_TRUE(in.is_open());
  std::string header;
  ASSERT_TRUE(std::getline(in, header));
  EXPECT_NE(header.find("symmetry"), std::string::npos);

  std::vector<std::string> symmetries;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> tokens = splitCsvTestLine(line);
    ASSERT_GE(tokens.size(), 10u);
    symmetries.push_back(tokens[5]);
  }
  ASSERT_EQ(symmetries.size(), 3u);
  EXPECT_EQ(symmetries[0], "identity");
  EXPECT_EQ(symmetries[1], "c2");
  EXPECT_EQ(symmetries[2], "c4");

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOWritesSymmetryAmbiguousObjectObservations) {
  const std::filesystem::path plainDir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_symmetry_plain");
  const std::filesystem::path ambiguousDir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_symmetry_ambiguous");
  std::filesystem::remove_all(plainDir);
  std::filesystem::remove_all(ambiguousDir);

  SyntheticObjectPGOOptions options;
  options.dimension = 3;
  options.numRobots = 1;
  options.numObjects = 3;
  options.posesPerRobot = 4;
  options.observeAllObjects = true;
  options.objectObservationsPerObject = 1;
  options.observationRotationNoise = 0.0;
  options.symmetryYawAmbiguityRate = 0.0;
  writeSyntheticObjectPGODirectory(plainDir.string(), options);

  options.symmetryYawAmbiguityRate = 1.0;
  writeSyntheticObjectPGODirectory(ambiguousDir.string(), options);

  const std::size_t c4ObjectVertexId = options.posesPerRobot + 2;
  const Eigen::Quaterniond plain =
      firstSe3EdgeQuaternionToObject(plainDir / "rob_0.g2o",
                                     c4ObjectVertexId);
  const Eigen::Quaterniond ambiguous =
      firstSe3EdgeQuaternionToObject(ambiguousDir / "rob_0.g2o",
                                     c4ObjectVertexId);
  const double dot =
      std::abs(std::max(-1.0, std::min(1.0, plain.dot(ambiguous))));
  const double angle = 2.0 * std::acos(dot);
  EXPECT_GT(angle, 1.0);

  std::filesystem::remove_all(plainDir);
  std::filesystem::remove_all(ambiguousDir);
}

TEST(testDPGO, SyntheticObjectPGORunsManualObjectDranOracleSmoke) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_manual_dran");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 4;
  synthetic.robotDriftPerRobot = 0.1;
  synthetic.objectInitialDriftPerRobot = 0.1;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.numRobots, synthetic.numRobots);
  EXPECT_EQ(result.numObjects, synthetic.numObjects);
  EXPECT_EQ(result.knownObjectCopies,
            synthetic.numRobots * synthetic.numObjects);
  EXPECT_EQ(result.observedObjectCopies,
            synthetic.numRobots * synthetic.numObjects);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects);
  EXPECT_EQ(result.iterations.front().associationCandidateCount,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRecall, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationF1, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationFalseMergeRate, 0.0);
  EXPECT_TRUE(std::isfinite(result.iterations.back().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().consensusCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().gradient));
  const SyntheticObjectPGOEvaluation evaluation =
      evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);
  EXPECT_EQ(evaluation.trajectoryPoseCount,
            synthetic.numRobots * synthetic.posesPerRobot);
  EXPECT_EQ(evaluation.objectCopyCount,
            synthetic.numRobots * synthetic.numObjects);
  EXPECT_TRUE(std::isfinite(evaluation.trajectoryTranslationRmse));
  EXPECT_TRUE(std::isfinite(evaluation.objectTranslationRmse));
  EXPECT_LT(evaluation.trajectoryTranslationRmse, 1e-8);
  EXPECT_LT(evaluation.objectTranslationRmse, 1e-8);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsManualObjectDranGeometryAssociations) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_association_modes");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.2;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions hardOptions;
  hardOptions.numRobots = synthetic.numRobots;
  hardOptions.numObjects = synthetic.numObjects;
  hardOptions.maxIterations = 0;
  hardOptions.topology = ManualObjectDranTopology::Complete;
  hardOptions.associationMode = ManualObjectDranAssociationMode::GeometryHard;

  const ManualObjectDranRunResult hard =
      runManualObjectDran(dir.string(), hardOptions);
  ASSERT_EQ(hard.iterations.size(), 1u);
  EXPECT_EQ(hard.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects);
  EXPECT_TRUE(std::isfinite(hard.iterations.front().measurementCost));
  EXPECT_TRUE(std::isfinite(hard.iterations.front().consensusCost));

  ManualObjectDranOptions softOptions = hardOptions;
  softOptions.associationMode = ManualObjectDranAssociationMode::GeometrySoft;
  softOptions.associationSinkhornIterations = 30;
  softOptions.associationTemperature = 0.5;

  const ManualObjectDranRunResult soft =
      runManualObjectDran(dir.string(), softOptions);
  ASSERT_EQ(soft.iterations.size(), 1u);
  EXPECT_EQ(soft.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects * synthetic.numObjects);
  EXPECT_TRUE(std::isfinite(soft.iterations.front().measurementCost));
  EXPECT_TRUE(std::isfinite(soft.iterations.front().consensusCost));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOBackendActiveFactorBudgetCapsSoftFactors) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_backend_factor_budget");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 4;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.2;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::GeometrySoft;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;

  const ManualObjectDranRunResult uncapped =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(uncapped.iterations.size(), 1u);
  ASSERT_GT(uncapped.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) * 3u);

  options.associationBackendMaxActiveFactorsPerEdge = 3;
  const ManualObjectDranRunResult capped =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(capped.iterations.size(), 1u);
  EXPECT_EQ(capped.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                options.associationBackendMaxActiveFactorsPerEdge);
  EXPECT_LT(capped.iterations.front().activeConsensusPairs,
            uncapped.iterations.front().activeConsensusPairs);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsSlideGraphAssociationBaseline) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_slide_graph");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 8;
  synthetic.posesPerRobot = 8;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.semanticAccuracy = 0.7;
  synthetic.descriptorNoise = 0.05;
  synthetic.shapeNoise = 0.03;
  synthetic.repeatedObjectRatio = 0.5;
  synthetic.permuteLocalObjectIds = true;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.4;
  synthetic.objectInitialDriftPerRobot = 0.4;
  synthetic.worldLength = 20.0;
  synthetic.worldWidth = 14.0;
  synthetic.worldHeight = 5.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = syntheticLocalObjectSlotCount(synthetic);
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode =
      ManualObjectDranAssociationMode::GraphConsistencyHard;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationMetadataSemanticAccuracy = synthetic.semanticAccuracy;
  options.associationMetadataDescriptorNoise = synthetic.descriptorNoise;
  options.associationMetadataShapeNoise = synthetic.shapeNoise;
  options.associationMetadataRepeatedObjectRatio =
      synthetic.repeatedObjectRatio;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;

  ManualObjectDranOptions diagonalOptions = options;
  diagonalOptions.associationMode =
      ManualObjectDranAssociationMode::IdentityOracle;
  const ManualObjectDranRunResult diagonal =
      runManualObjectDran(dir.string(), diagonalOptions);
  ASSERT_EQ(diagonal.iterations.size(), 1u);

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations.front().activeConsensusPairs, 0u);
  EXPECT_TRUE(std::isfinite(result.iterations.front().associationF1));
  EXPECT_GE(result.iterations.front().associationF1, 0.0);
  EXPECT_LE(result.iterations.front().associationF1, 1.0);
  EXPECT_TRUE(std::isfinite(
      result.iterations.front().associationFalseMergeRate));
  EXPECT_GE(result.iterations.front().associationFalseMergeRate, 0.0);
  EXPECT_LE(result.iterations.front().associationFalseMergeRate, 1.0);
  EXPECT_GT(result.iterations.front().associationF1,
            diagonal.iterations.front().associationF1);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsManualObjectDranFixedFileAssociation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_fixed_pi");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t objectId = 0; objectId < synthetic.numObjects;
         ++objectId) {
      out << "0,1," << objectId << "," << objectId << ",1\n";
      out << "1,0," << objectId << "," << objectId << ",1\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRecall, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationF1, 1.0);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount,
            result.iterations.front().associationCandidateCount);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationRecall, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationF1, 1.0);
  EXPECT_TRUE(std::isfinite(result.iterations.front().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.front().consensusCost));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOFixedFileCycleConsistencyPrunesAliasConflict) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_fixed_cycle_alias");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 3;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,0,1\n";
    out << "0,1,1,2,1\n";
    out << "0,2,0,0,1\n";
    out << "0,2,1,1,1\n";
    out << "2,1,0,0,1\n";
    out << "2,1,1,1,1\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();

  const ManualObjectDranRunResult unfiltered =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(unfiltered.iterations.empty());
  EXPECT_GT(unfiltered.iterations.back().associationFalseMergeMass, 0.0);

  options.associationCycleConsistencyEnabled = true;
  options.associationCycleConsistencyMinDirectWeight = 0.5;
  options.associationCycleConsistencyMinComposedWeight = 0.25;
  options.associationCycleConsistencyMinTwoHopSupport = 1;
  options.associationCycleConsistencyKeepWithoutEvidence = true;

  const ManualObjectDranRunResult filtered =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(filtered.iterations.empty());
  const ManualObjectDranIterationSummary &row = filtered.iterations.back();
  EXPECT_EQ(row.associationCyclePrunedEntries, 1u);
  EXPECT_DOUBLE_EQ(row.associationFalseMergeMass, 0.0);
  EXPECT_LT(row.associationCandidateCount,
            unfiltered.iterations.back().associationCandidateCount);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsManualObjectDranTransformSoftAssociation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_transform_soft");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.numCategories = 6;
  synthetic.descriptorDim = 8;
  synthetic.permuteLocalObjectIds = true;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.4;
  synthetic.objectInitialDriftPerRobot = 0.4;
  synthetic.semanticAccuracy = 1.0;
  synthetic.descriptorNoise = 0.0;
  synthetic.shapeNoise = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::TransformSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 1;
  options.associationTransformMinInliers = 4;
  options.associationTransformMaxResidual = 10.0;
  options.associationTransformMinScore = 0.1;
  options.associationTransformPlaneAxes = "xy";

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_GE(row.associationCandidateCount,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects - 2);
  EXPECT_DOUBLE_EQ(row.associationPrecision, 1.0);
  EXPECT_GT(row.associationRecall, 0.9);
  EXPECT_GT(row.associationF1, 0.94);
  EXPECT_DOUBLE_EQ(row.associationFalseMergeRate, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOReportsUpperTrackingAndGaugeStationarity) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_upper_tracking_gauge");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 2;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 3;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.permuteLocalObjectIds = true;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.2;
  synthetic.semanticAccuracy = 1.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 3;
  options.associationSinkhornIterations = 10;
  options.associationSupportSharpnessWeight = 0.1;
  options.associationUpperTrackingEnabled = true;
  options.associationUpperTrackingConsensusWeight = 0.5;
  options.associationUpperTrackingCorrectionWeight = 0.25;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_GE(result.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_GT(row.associationUpperTrackingGradientNorm, 0.0);
  EXPECT_GE(row.associationUpperTrackingCorrectionNorm, 0.0);
  EXPECT_GE(row.associationUpperTrackingConsensusResidualNorm, 0.0);
  EXPECT_GE(row.lowerStationarityGradientNorm, 0.0);
  EXPECT_GE(row.lowerGaugeMarginalizedStationarityNorm, 0.0);
  EXPECT_LE(row.lowerGaugeMarginalizedStationarityNorm,
            row.lowerStationarityGradientNorm + 1e-9);
  EXPECT_GE(row.lowerGaugeComponentRatio, 0.0);
  EXPECT_LE(row.lowerGaugeComponentRatio, 1.0 + 1e-9);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOReportsStateValidationHypergradient) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_state_validation");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.permuteLocalObjectIds = true;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.25;
  synthetic.objectInitialDriftPerRobot = 0.35;
  synthetic.semanticAccuracy = 1.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 3;
  options.associationSinkhornIterations = 10;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationResidualUseTransformAlignment = true;
  options.associationStateValidationHypergradientWeight = 0.1;
  options.associationStateValidationMeritTrustScale = 1.0;
  options.associationStateValidationUseEdgeTransformConfidence = true;
  options.associationStateValidationTransformConfidenceDistanceScale = 2.0;
  options.associationEdgeLatentConsensusWeight = 0.05;
  options.associationEdgeLatentConsensusUseTransformConfidence = true;
  options.associationEdgeLatentConsensusConfidenceDistanceScale = 2.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_GE(result.iterations.size(), 2u);
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_GE(row.associationStateValidationHypergradientGradientNorm, 0.0);
  EXPECT_GE(row.associationEdgeLatentConsensusGradientNorm, 0.0);
  EXPECT_GE(row.associationEdgeLatentConsensusResidualNorm, 0.0);
  if (row.associationEdgeLatentConsensusResidualNorm > 1e-12) {
    EXPECT_LE(row.associationEdgeLatentConsensusGradientNorm,
              options.associationEdgeLatentConsensusWeight *
                      row.associationEdgeLatentConsensusResidualNorm *
                      (1.0 + 1e-9) +
                  1e-12);
  }
  EXPECT_GE(row.associationEdgeLatentConsensusConfidenceMean, 0.0);
  EXPECT_LE(row.associationEdgeLatentConsensusConfidenceMean, 1.0);
  EXPECT_TRUE(std::isfinite(row.associationEdgeLatentConsensusObjectiveCost));
  EXPECT_GE(row.associationStateValidationTransformConfidenceCost, 0.0);
  EXPECT_GE(row.associationStateValidationTransformConfidenceMean, 0.0);
  EXPECT_LE(row.associationStateValidationTransformConfidenceMean, 1.0);
  EXPECT_GE(row.associationStateValidationMeritGate, 0.0);
  EXPECT_LE(row.associationStateValidationMeritGate, 1.0);
  EXPECT_TRUE(std::isfinite(row.associationStateValidationMeritRatio));
  EXPECT_GT(row.associationStateValidationUnifiedObjectiveCost, 0.0);
  EXPECT_TRUE(std::isfinite(
      row.associationStateValidationHypergradientObjectiveCost));
  if (row.associationStateValidationMeritGate > 0.0) {
    EXPECT_GT(row.associationStateValidationHypergradientGradientNorm, 0.0);
    EXPECT_GT(std::abs(
                  row.associationStateValidationHypergradientObjectiveCost),
              0.0);
    EXPECT_GT(row.associationStateValidationTrajectoryResidualCost, 0.0);
    EXPECT_GT(row.associationStateValidationPairwiseTrajectoryCost, 0.0);
    EXPECT_GT(row.associationStateValidationTransformConfidenceCost, 0.0);
    EXPECT_LE(row.associationStateValidationTransformConfidenceCost,
              row.associationStateValidationTrajectoryResidualCost *
                      (1.0 + 1e-9) +
                  1e-12);
    EXPECT_GT(row.associationStateValidationObjectConsistencyCost, 0.0);
    EXPECT_GT(row.associationStateValidationGaugeStationarityCost, 0.0);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOFixedFileCanSkipUnobservedCandidates) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_fixed_observed_only");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 3;
  synthetic.numObjects = 20;
  synthetic.posesPerRobot = 20;
  synthetic.observeAllObjects = false;
  synthetic.detectionRange = 20.0;
  synthetic.missDetectionRate = 0.2;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  std::vector<std::vector<bool>> observed(
      synthetic.numRobots, std::vector<bool>(synthetic.numObjects, false));
  {
    std::ifstream input(dir / "observations.csv");
    ASSERT_TRUE(input.is_open());
    std::string line;
    ASSERT_TRUE(std::getline(input, line));
    while (std::getline(input, line)) {
      if (line.empty()) {
        continue;
      }
      const std::vector<std::string> tokens = splitCsvTestLine(line);
      ASSERT_GE(tokens.size(), 4u);
      const std::size_t robotId =
          static_cast<std::size_t>(std::stoul(tokens[0]));
      const std::size_t objectId =
          static_cast<std::size_t>(std::stoul(tokens[2]));
      ASSERT_LT(robotId, synthetic.numRobots);
      ASSERT_LT(objectId, synthetic.numObjects);
      observed[robotId][objectId] = true;
    }
  }
  std::size_t expectedObservedDirectedPairs = 0;
  for (std::size_t receiver = 0; receiver < synthetic.numRobots; ++receiver) {
    for (std::size_t sender = 0; sender < synthetic.numRobots; ++sender) {
      if (receiver == sender) {
        continue;
      }
      for (std::size_t objectId = 0; objectId < synthetic.numObjects;
           ++objectId) {
        if (observed[receiver][objectId] && observed[sender][objectId]) {
          ++expectedObservedDirectedPairs;
        }
      }
    }
  }
  ASSERT_GT(expectedObservedDirectedPairs, 0u);
  ASSERT_LT(expectedObservedDirectedPairs,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t receiver = 0; receiver < synthetic.numRobots;
         ++receiver) {
      for (std::size_t sender = 0; sender < synthetic.numRobots; ++sender) {
        if (receiver == sender) {
          continue;
        }
        for (std::size_t objectId = 0; objectId < synthetic.numObjects;
             ++objectId) {
          out << receiver << "," << sender << "," << objectId << ","
              << objectId << ",1\n";
        }
      }
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationRequireObservedObjects = true;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs,
            expectedObservedDirectedPairs);
  EXPECT_EQ(result.iterations.front().associationCandidateCount,
            expectedObservedDirectedPairs);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount,
            expectedObservedDirectedPairs);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRecall, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationF1, 1.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationMinConsensusWeightFiltersTerms) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_min_pi");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t objectId = 0; objectId < synthetic.numObjects;
         ++objectId) {
      out << "0,1," << objectId << "," << objectId << ",0.4\n";
      out << "1,0," << objectId << "," << objectId << ",0.4\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationMinConsensusWeight = 0.5;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 0u);
  EXPECT_GT(result.iterations.front().associationCandidateCount, 0u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationWeightMass, 0.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationF1, 0.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRecall, 0.4);
  EXPECT_TRUE(std::isfinite(result.iterations.front().measurementCost));
  EXPECT_DOUBLE_EQ(result.iterations.front().consensusCost, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationConsensusPowerReportsFactorMass) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_factor_mass");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  synthetic.observeAllObjects = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t objectId = 0; objectId < synthetic.numObjects;
         ++objectId) {
      out << "0,1," << objectId << "," << objectId << ",0.5\n";
      out << "1,0," << objectId << "," << objectId << ",0.5\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationRequireObservedObjects = true;
  options.associationConsensusWeightPower = 2.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  const ManualObjectDranIterationSummary &row = result.iterations.front();
  const double expectedRawMass =
      static_cast<double>(synthetic.numRobots * (synthetic.numRobots - 1) *
                          synthetic.numObjects) *
      0.5;
  const double expectedFactorMass =
      static_cast<double>(synthetic.numRobots * (synthetic.numRobots - 1) *
                          synthetic.numObjects) *
      0.25;
  EXPECT_NEAR(row.activeAssociationWeightMass, expectedRawMass, 1e-12);
  EXPECT_NEAR(row.activeAssociationCorrectMass, expectedRawMass, 1e-12);
  EXPECT_NEAR(row.activeAssociationFactorWeightMass, expectedFactorMass,
              1e-12);
  EXPECT_NEAR(row.activeAssociationFactorCorrectMass, expectedFactorMass,
              1e-12);
  EXPECT_DOUBLE_EQ(row.activeAssociationFactorFalseMergeMass, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationConsensusMarginRejectsAmbiguity) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_margin_pi");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 2;
  synthetic.posesPerRobot = 4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    for (std::size_t receiver = 0; receiver < synthetic.numRobots;
         ++receiver) {
      const std::size_t sender = 1 - receiver;
      out << receiver << "," << sender << ",0,0,0.60\n";
      out << receiver << "," << sender << ",0,1,0.55\n";
      out << receiver << "," << sender << ",1,0,0.55\n";
      out << receiver << "," << sender << ",1,1,0.60\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationMinConsensusWeight = 0.5;
  options.associationMinConsensusMargin = 0.1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations.front().associationCandidateCount, 0u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 0u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_DOUBLE_EQ(result.iterations.front().activeAssociationWeightMass, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftRequiresStableActiveWindows) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_stable_active");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.0;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationRequireObservedObjects = true;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.05;
  options.associationMaxCandidatesPerEdge = synthetic.numObjects;
  options.associationMinConsensusWeight = 0.5;
  options.associationActiveMinStableUpdates = 2;
  options.beta = 1.0;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_GT(result.iterations.front().associationCandidateCount, 0u);
  EXPECT_GT(result.iterations.front().highConfidenceAssociationCandidateCount,
            0u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 0u);
  EXPECT_GT(result.iterations.back().activeAssociationCandidateCount, 0u);
  EXPECT_GT(result.iterations.back().activeConsensusPairs, 0u);
  EXPECT_EQ(result.iterations.back().highConfidenceAssociationCandidateCount,
            result.iterations.back().activeAssociationCandidateCount);
  EXPECT_DOUBLE_EQ(
      result.iterations.back().highConfidenceAssociationFalseMergeRate,
      result.iterations.back().activeAssociationFalseMergeRate);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOCandidateCapDoesNotSparseResidualSoftSinkhorn) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_decoupled_sinkhorn");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.numCategories = 4;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.robotDriftPerRobot = 0.0;
  synthetic.objectInitialDriftPerRobot = 0.0;
  synthetic.semanticAccuracy = 1.0;
  synthetic.descriptorNoise = 0.0;
  synthetic.shapeNoise = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationMaxCandidatesPerEdge = 1;
  options.associationSinkhornMaxNonzeros = 0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations.front().associationCandidateCount, 2u);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOResidualSoftUsesTentativeWeightBeforeStableActivation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_tentative_weight");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.0;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationRequireObservedObjects = true;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.05;
  options.associationMaxCandidatesPerEdge = synthetic.numObjects;
  options.associationMinConsensusWeight = 0.5;
  options.associationActiveMinStableUpdates = 2;
  options.associationTentativeWeightScale = 0.25;
  options.beta = 1.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations.front().associationCandidateCount, 0u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_GT(result.iterations.front().activeConsensusPairs, 0u);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftActivationRampScalesFirstStableStep) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_activation_ramp");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.robotDriftPerRobot = 0.0;
  synthetic.objectInitialDriftPerRobot = 1.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions baseline;
  baseline.numRobots = synthetic.numRobots;
  baseline.numObjects = synthetic.numObjects;
  baseline.maxIterations = 0;
  baseline.topology = ManualObjectDranTopology::Complete;
  baseline.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  baseline.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  baseline.associationRequireObservedObjects = true;
  baseline.associationSinkhornIterations = 30;
  baseline.associationTemperature = 0.05;
  baseline.associationMaxCandidatesPerEdge = synthetic.numObjects;
  baseline.associationMinConsensusWeight = 0.5;
  baseline.associationActiveMinStableUpdates = 1;
  baseline.beta = 1.0;

  const ManualObjectDranRunResult full =
      runManualObjectDran(dir.string(), baseline);
  ASSERT_EQ(full.iterations.size(), 1u);
  ASSERT_GT(full.iterations.front().activeConsensusPairs, 0u);
  ASSERT_GT(full.iterations.front().activatedAssociationWeightMass, 0.0);

  ManualObjectDranOptions ramped = baseline;
  ramped.associationActivationTrustRampUpdates = 4;
  const ManualObjectDranRunResult rampedResult =
      runManualObjectDran(dir.string(), ramped);

  ASSERT_EQ(rampedResult.iterations.size(), 1u);
  EXPECT_EQ(rampedResult.iterations.front().activeConsensusPairs,
            full.iterations.front().activeConsensusPairs);
  EXPECT_EQ(rampedResult.iterations.front().activatedAssociationCandidateCount,
            full.iterations.front().activatedAssociationCandidateCount);
  EXPECT_LT(rampedResult.iterations.front().activatedAssociationWeightMass,
            0.4 * full.iterations.front().activatedAssociationWeightMass);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationRejectsPersistentBadResidual) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_rejected_assoc");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 2;
  synthetic.posesPerRobot = 4;
  synthetic.observeAllObjects = true;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,1,1\n";
    out << "1,0,0,1,1\n";
  }

  ManualObjectDranOptions baseline;
  baseline.numRobots = synthetic.numRobots;
  baseline.numObjects = synthetic.numObjects;
  baseline.maxIterations = 0;
  baseline.topology = ManualObjectDranTopology::Complete;
  baseline.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  baseline.associationFile = associationFile.string();
  baseline.associationMinConsensusWeight = 0.5;
  baseline.beta = 1.0;

  const ManualObjectDranRunResult unguarded =
      runManualObjectDran(dir.string(), baseline);
  ASSERT_EQ(unguarded.iterations.size(), 1u);
  EXPECT_GT(unguarded.iterations.front().activeConsensusPairs, 0u);

  ManualObjectDranOptions guarded = baseline;
  guarded.associationRejectResidualThreshold = 0.01;
  guarded.associationRejectMinBadUpdates = 1;

  const ManualObjectDranRunResult rejected =
      runManualObjectDran(dir.string(), guarded);
  ASSERT_EQ(rejected.iterations.size(), 1u);
  EXPECT_EQ(rejected.iterations.front().activeConsensusPairs, 0u);
  EXPECT_EQ(rejected.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_EQ(rejected.iterations.front().associationRejectedCandidateCount, 2u);
  EXPECT_DOUBLE_EQ(rejected.iterations.front().associationRejectedWeightMass,
                   2.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationRejectsRelativeBadResidual) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_relative_reject");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 2;
  synthetic.posesPerRobot = 4;
  synthetic.observeAllObjects = true;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,1,1\n";
    out << "1,0,0,1,1\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationMinConsensusWeight = 0.5;
  options.associationRejectResidualBestMargin = 0.01;
  options.associationRejectMinBadUpdates = 1;
  options.beta = 1.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 0u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 0u);
  EXPECT_EQ(result.iterations.front().associationRejectedCandidateCount, 2u);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRejectedWeightMass,
                   2.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOAssociationRejectsQuantileBadResidual) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_quantile_reject");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 2;
  synthetic.posesPerRobot = 4;
  synthetic.observeAllObjects = true;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::filesystem::path associationFile = dir / "association.csv";
  {
    std::ofstream out(associationFile);
    out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
    out << "0,1,0,0,1\n";
    out << "0,1,0,1,1\n";
    out << "1,0,0,0,1\n";
    out << "1,0,0,1,1\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  options.associationFile = associationFile.string();
  options.associationMinConsensusWeight = 0.5;
  options.associationRejectResidualQuantile = 0.25;
  options.associationRejectResidualQuantileScale = 1.0;
  options.associationRejectMinBadUpdates = 1;
  options.beta = 1.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 2u);
  EXPECT_EQ(result.iterations.front().activeAssociationCandidateCount, 2u);
  EXPECT_EQ(result.iterations.front().associationRejectedCandidateCount, 2u);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRejectedWeightMass,
                   2.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOWritesNormalizedOracleAssociationWeights) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_normalized_oracle");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.duplicateObjectRatio = 0.5;
  synthetic.permuteLocalObjectIds = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  std::ifstream input(dir / "gt_associations.csv");
  ASSERT_TRUE(input.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(input, line));
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, double>
      rowWeight;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> tokens = splitCsvTestLine(line);
    ASSERT_GE(tokens.size(), 5u);
    const std::size_t receiverRobot =
        static_cast<std::size_t>(std::stoul(tokens[0]));
    const std::size_t senderRobot =
        static_cast<std::size_t>(std::stoul(tokens[1]));
    const std::size_t receiverObject =
        static_cast<std::size_t>(std::stoul(tokens[2]));
    const double weight = std::stod(tokens[4]);
    rowWeight[{receiverRobot, senderRobot, receiverObject}] += weight;
  }

  ASSERT_FALSE(rowWeight.empty());
  for (const auto &entry : rowWeight) {
    EXPECT_NEAR(entry.second, 1.0, 1e-12);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOResidualSoftTransformAlignmentReducesLocalFrameCost) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_residual_transform_alignment");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  auto writeRobot = [&](std::size_t robotId, bool rotateFrame) {
    std::ofstream out(dir / ("rob_" + std::to_string(robotId) + ".g2o"));
    out << "VERTEX_SE2 0 0 0 0\n";
    const std::array<std::pair<double, double>, 4> localObjects = {
        std::make_pair(0.0, 0.0), std::make_pair(2.0, 0.0),
        std::make_pair(0.3, 1.7), std::make_pair(2.5, 3.0)};
    std::array<std::pair<double, double>, 4> objects;
    for (std::size_t objectId = 0; objectId < localObjects.size();
         ++objectId) {
      const double x = localObjects[objectId].first;
      const double y = localObjects[objectId].second;
      if (rotateFrame) {
        objects[objectId] = std::make_pair(10.0 - y, -3.0 + x);
      } else {
        objects[objectId] = std::make_pair(x, y);
      }
    }
    for (std::size_t objectId = 0; objectId < objects.size(); ++objectId) {
      out << "VERTEX_SE2 " << (objectId + 1) << " "
          << objects[objectId].first << " " << objects[objectId].second
          << " 0\n";
    }
    for (std::size_t objectId = 0; objectId < objects.size(); ++objectId) {
      out << "EDGE_SE2 0 " << (objectId + 1) << " "
          << objects[objectId].first << " " << objects[objectId].second
          << " 0 100 0 0 100 0 100\n";
    }
  };
  writeRobot(0, false);
  writeRobot(1, true);

  {
    std::ofstream out(dir / "local_object_metadata.csv");
    out << "robot_id,local_slot_id,true_object_id,slot_kind,category,"
           "symmetry,semantic_confidence,shape_x,shape_y,shape_z,"
           "descriptor_0,descriptor_1,descriptor_2,descriptor_3\n";
    for (std::size_t robotId = 0; robotId < 2; ++robotId) {
      for (std::size_t objectId = 0; objectId < 4; ++objectId) {
        out << robotId << "," << objectId << "," << objectId
            << ",primary," << objectId
            << ",identity,1,1,1,1";
        for (std::size_t dim = 0; dim < 4; ++dim) {
          out << "," << (dim == objectId ? 1 : 0);
        }
        out << "\n";
      }
    }
  }
  {
    std::ofstream out(dir / "gt_associations.csv");
    out << "receiver_robot,sender_robot,receiver_object,sender_object,"
           "weight,true_object_id\n";
    for (std::size_t objectId = 0; objectId < 4; ++objectId) {
      out << "0,1," << objectId << "," << objectId << ",1," << objectId
          << "\n";
      out << "1,0," << objectId << "," << objectId << ",1," << objectId
          << "\n";
    }
  }

  ManualObjectDranOptions options;
  options.numRobots = 2;
  options.numObjects = 4;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = 4;
  options.associationMetadataDescriptorDim = 4;
  options.associationGeometryScale = 1.0;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 4;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationTransformMinInliers = 4;
  options.associationTransformMaxResidual = 10.0;
  options.associationTransformMinScore = 0.1;
  options.associationTransformPlaneAxes = "xy";

  const ManualObjectDranRunResult direct =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(direct.iterations.empty());
  ASSERT_EQ(direct.finalEstimates.size(), 2u);
  Vector directRobot0ObjectVector(2);
  directRobot0ObjectVector << direct.finalEstimates[0](0, 8) -
                                  direct.finalEstimates[0](0, 5),
      direct.finalEstimates[0](1, 8) - direct.finalEstimates[0](1, 5);
  Vector directRobot1ObjectVector(2);
  directRobot1ObjectVector << direct.finalEstimates[1](0, 8) -
                                  direct.finalEstimates[1](0, 5),
      direct.finalEstimates[1](1, 8) - direct.finalEstimates[1](1, 5);
  EXPECT_LT(directRobot0ObjectVector.normalized().dot(
                directRobot1ObjectVector.normalized()),
            0.5)
      << "v0=" << directRobot0ObjectVector.transpose()
      << " v1=" << directRobot1ObjectVector.transpose()
      << " p0_r1=(" << direct.finalEstimates[1](0, 2) << ","
      << direct.finalEstimates[1](1, 2) << ")";

  options.associationResidualUseTransformAlignment = true;
  const ManualObjectDranRunResult aligned =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(aligned.iterations.empty());

  const double directNorm =
      direct.iterations.front().associationLocalGradientNorm;
  const double alignedNorm =
      aligned.iterations.front().associationLocalGradientNorm;
  EXPECT_GT(directNorm, 0.0);
  EXPECT_LT(alignedNorm, 0.5 * directNorm);
  EXPECT_GT(aligned.iterations.front().associationF1, 0.9);

  options.associationNormalizeUpperCosts = true;
  options.associationUpperCostNormalizationTargetRms = 2.5;
  const ManualObjectDranRunResult normalized =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(normalized.iterations.empty());
  EXPECT_LT(normalized.iterations.front().associationLocalGradientNorm, 25.0);
  EXPECT_LT(normalized.iterations.front().associationLocalGradientNorm,
            0.5 * alignedNorm);
  EXPECT_GT(normalized.iterations.front().associationF1, 0.9);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsManualObjectDranResidualSoftAssociation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_residual_soft");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  synthetic.robotDriftPerRobot = 0.15;
  synthetic.objectInitialDriftPerRobot = 0.15;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  ASSERT_EQ(result.iterations.size(), 3u);
  for (const ManualObjectDranIterationSummary &row : result.iterations) {
    EXPECT_GT(row.activeConsensusPairs, 0u);
    EXPECT_LE(row.activeConsensusPairs,
              synthetic.numRobots * (synthetic.numRobots - 1) *
                  synthetic.numObjects * synthetic.numObjects);
    EXPECT_TRUE(std::isfinite(row.measurementCost));
    EXPECT_TRUE(std::isfinite(row.consensusCost));
    EXPECT_TRUE(std::isfinite(row.gradient));
    EXPECT_GT(row.associationCandidateCount, 0u);
    EXPECT_TRUE(std::isfinite(row.associationWeightMass));
    EXPECT_TRUE(std::isfinite(row.associationEntropy));
    EXPECT_GE(row.associationPrecision, 0.0);
    EXPECT_LE(row.associationPrecision, 1.0);
    EXPECT_GE(row.associationRecall, 0.0);
    EXPECT_LE(row.associationRecall, 1.0);
    EXPECT_GE(row.associationF1, 0.0);
    EXPECT_LE(row.associationF1, 1.0);
    EXPECT_GE(row.associationFalseMergeRate, 0.0);
    EXPECT_LE(row.associationFalseMergeRate, 1.0);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftReportsCycleGradientDiagnostics) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_cycle_gradient_diag");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 3;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.semanticAccuracy = 0.8;
  synthetic.descriptorNoise = 0.05;
  synthetic.repeatedObjectRatio = 0.5;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationPolicy =
      ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1;
  options.associationUpdatePeriod = 1;
  options.associationUseSyntheticMetadata = true;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 3;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationCycleObjectiveWeight = 0.1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();

  EXPECT_GT(row.associationLocalGradientNorm, 0.0);
  EXPECT_GT(row.associationCycleGradientNorm, 0.0);
  EXPECT_GT(row.associationCycleToLocalGradientRatio, 0.0);
  EXPECT_TRUE(std::isfinite(row.associationCycleObjectiveCost));
  EXPECT_GE(row.associationCycleObjectiveCost, 0.0);
  EXPECT_EQ(row.associationCycleObjectiveActiveTriplets, 1u);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftReportsAdaptiveUpperDiagnostics) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_adaptive_upper_diag");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 3;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.semanticAccuracy = 0.8;
  synthetic.descriptorNoise = 0.05;
  synthetic.repeatedObjectRatio = 0.5;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationUseSyntheticMetadata = true;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 3;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationCycleObjectiveWeight = 0.1;
  options.associationAdaptiveUpperEnabled = true;
  options.associationAdaptiveUpperInitialStep = 0.5;
  options.associationAdaptiveUpperMaxStep = 1.0;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();

  EXPECT_GT(row.associationUpperGradientNorm, 0.0);
  EXPECT_GT(row.associationAdaptiveEta, 0.0);
  EXPECT_LE(row.associationAdaptiveEta,
            options.associationAdaptiveUpperMaxStep);
  EXPECT_GT(row.associationActualCycleAlpha, 0.0);
  EXPECT_DOUBLE_EQ(row.associationActualBackendAlpha, 1.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftReportsTemporalKlDiagnostics) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_temporal_kl_diag");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 3;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.semanticAccuracy = 0.8;
  synthetic.descriptorNoise = 0.05;
  synthetic.repeatedObjectRatio = 0.5;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationUseSyntheticMetadata = true;
  options.associationRequireObservedObjects = true;
  options.associationMaxCandidatesPerEdge = 3;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationTemporalKlWeight = 0.2;
  options.associationEntropyRegularizationWeight = 0.05;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();

  EXPECT_GT(row.associationTemporalKlCost, 0.0);
  EXPECT_GT(row.associationTemporalKlGradientNorm, 0.0);
  EXPECT_GT(row.associationEntropyRegularizationGradientNorm, 0.0);
  EXPECT_GT(row.associationChurnL1, 0.0);
  EXPECT_GT(row.associationChurnFrobenius, 0.0);
  EXPECT_DOUBLE_EQ(row.associationActualTemporalAlpha,
                   options.associationTemporalKlWeight);
  EXPECT_DOUBLE_EQ(row.associationActualEntropyAlpha,
                   options.associationEntropyRegularizationWeight);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftBackendFeedback) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_backend_feedback");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendFeedbackGradientScale = 1.0;
  options.associationBackendFeedbackMinTrust = 0.2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));
  EXPECT_GT(row.associationBackendGradientNorm, 0.0);
  EXPECT_GT(row.associationBackendToLocalGradientRatio, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftBackendResponse) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_backend_response");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendResponseGain = 1.0;
  options.associationBackendResponseDamping = 1.0;
  options.associationBackendResponseMaxReductionFraction = 0.5;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftBackendAdjoint) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_backend_adjoint");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendAdjointGain = 1.0;
  options.associationBackendAdjointDamping = 1.0;
  options.associationBackendAdjointMaxReductionFraction = 0.5;
  options.associationBackendAdjointMaxIncreaseFraction = 0.5;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOResidualSoftBackendAdjointReportsSnapshotCapAndTrust) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_backend_adjoint_snapshot_cap_trust");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendAdjointGain = 100.0;
  options.associationBackendAdjointDamping = 1.0;
  options.associationBackendAdjointMaxReductionFraction = 0.5;
  options.associationBackendAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendSnapshotPeriod = 2;
  options.associationBackendMaxEffectiveRatio = 1e-6;
  options.associationBackendStationarityTrustMaxGradientNorm = 1e100;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 3u);
  const ManualObjectDranIterationSummary &inactive = result.iterations[1];
  EXPECT_EQ(inactive.associationBackendSnapshotActive, 0u);
  EXPECT_DOUBLE_EQ(inactive.associationActualBackendAlpha, 0.0);
  EXPECT_DOUBLE_EQ(inactive.associationBackendGradientNorm, 0.0);

  const ManualObjectDranIterationSummary &active = result.iterations.back();
  EXPECT_EQ(active.associationBackendSnapshotActive, 1u);
  EXPECT_EQ(active.associationBackendStationarityTrusted, 1u);
  EXPECT_GT(active.associationBackendStationarityGradientNorm, 0.0);
  EXPECT_GT(active.associationBackendRawGradientNorm, 0.0);
  EXPECT_GE(active.associationBackendRawGradientNorm + 1e-12,
            active.associationBackendGradientNorm);
  EXPECT_LE(active.associationBackendCapScale, 1.0);
  EXPECT_GT(active.associationActualBackendAlpha, 0.0);
  EXPECT_LE(active.associationActualBackendAlpha,
            active.associationBackendCapScale + 1e-12);
  EXPECT_LE(active.associationBackendToLocalGradientRatio, 1.1e-6);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOResidualSoftBackendAdjointDownweightsUntrustedSnapshot) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_backend_adjoint_trust");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendAdjointGain = 100.0;
  options.associationBackendAdjointDamping = 1.0;
  options.associationBackendAdjointMaxReductionFraction = 0.5;
  options.associationBackendAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendStationarityTrustMaxGradientNorm = 1e-12;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_EQ(row.associationBackendSnapshotActive, 1u);
  EXPECT_EQ(row.associationBackendStationarityTrusted, 0u);
  EXPECT_GT(row.associationBackendStationarityGradientNorm, 0.0);
  EXPECT_GT(row.associationBackendRawGradientNorm, 0.0);
  EXPECT_GT(row.associationBackendGradientNorm, 0.0);
  EXPECT_GT(row.associationBackendToLocalGradientRatio, 0.0);
  EXPECT_GT(row.associationActualBackendAlpha, 0.0);
  EXPECT_LT(row.associationActualBackendAlpha, 1e-6);
  EXPECT_LT(row.associationBackendGradientNorm,
            row.associationBackendRawGradientNorm);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftBackendCoupledAdjoint) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_backend_coupled_adjoint");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendCoupledAdjointGain = 1.0;
  options.associationBackendCoupledAdjointDamping = 1.0;
  options.associationBackendCoupledAdjointMaxReductionFraction = 0.5;
  options.associationBackendCoupledAdjointMaxIncreaseFraction = 0.5;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftPenaltyBackendCoupledAdjoint) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_penalty_backend_coupled_adjoint");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationBackendCoupledAdjointGain = 1.0;
  options.associationBackendCoupledAdjointDamping = 1.0;
  options.associationBackendCoupledAdjointMaxReductionFraction = 0.5;
  options.associationBackendCoupledAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendAdjointAllowCostReduction = false;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsResidualSoftGuardedBackendCoupledAdjoint) {
  const std::filesystem::path dir = syntheticTestDirectory(
      "dpgo_synthetic_object_pgo_guarded_backend_coupled_adjoint");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.numCategories = 2;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.4;
  synthetic.semanticAccuracy = 0.75;
  synthetic.descriptorNoise = 0.05;
  synthetic.shapeNoise = 0.02;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUpdatePeriod = 1;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.associationUseSyntheticMetadata = true;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 0.1;
  options.associationDescriptorWeight = 0.1;
  options.associationBackendCoupledAdjointGain = 1.0;
  options.associationBackendCoupledAdjointDamping = 1.0;
  options.associationBackendCoupledAdjointMaxReductionFraction = 0.5;
  options.associationBackendCoupledAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendAdjointReductionMetadataMargin = 0.25;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationFalseMergeRate));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));
  EXPECT_GT(row.associationBackendAdjointReductionCandidateCount, 0u);
  EXPECT_LE(row.associationBackendAdjointReductionAllowedCandidateCount,
            row.associationBackendAdjointReductionCandidateCount);
  EXPECT_GE(row.associationBackendAdjointReductionWeightMass,
            row.associationBackendAdjointReductionAllowedWeightMass);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftCanSkipRelayOnlyCandidates) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_observed_only_soft");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = false;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationRequireObservedObjects = true;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_EQ(result.iterations.front().observedObjectCopies, 4u);
  EXPECT_EQ(result.iterations.front().relayOnlyObjectCopies, 4u);
  EXPECT_EQ(result.iterations.front().associationCandidateCount, 8u);
  EXPECT_LT(result.iterations.front().associationCandidateCount,
            synthetic.numRobots * (synthetic.numRobots - 1) *
                synthetic.numObjects * synthetic.numObjects);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOResidualSoftUsesMetadataWarmStart) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_metadata_warm_start");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 5;
  synthetic.posesPerRobot = 8;
  synthetic.numCategories = synthetic.numObjects;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.semanticAccuracy = 1.0;
  synthetic.descriptorNoise = 0.0;
  synthetic.shapeNoise = 0.0;
  synthetic.repeatedObjectRatio = 0.0;
  synthetic.robotDriftPerRobot = 1.0;
  synthetic.objectInitialDriftPerRobot = 1.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = syntheticLocalObjectSlotCount(synthetic);
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationGeometryScale = 1.0e6;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.05;
  options.associationMaxCandidatesPerEdge = synthetic.numObjects;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations.front().associationPrecision, 0.8);
  EXPECT_GT(result.iterations.front().associationRecall, 0.8);
  EXPECT_LT(result.iterations.front().associationFalseMergeRate, 0.2);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOResidualSoftCanUseGeometryWarmStartWhenRequested) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_geometry_warm_start");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 5;
  synthetic.posesPerRobot = 8;
  synthetic.numCategories = 1;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.semanticAccuracy = 1.0;
  synthetic.descriptorNoise = 0.0;
  synthetic.shapeNoise = 0.0;
  synthetic.repeatedObjectRatio = 0.0;
  synthetic.robotDriftPerRobot = 0.0;
  synthetic.objectInitialDriftPerRobot = 0.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = syntheticLocalObjectSlotCount(synthetic);
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationResidualWarmStartUseGeometry = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationMetadataSemanticAccuracy = synthetic.semanticAccuracy;
  options.associationMetadataDescriptorNoise = synthetic.descriptorNoise;
  options.associationMetadataShapeNoise = synthetic.shapeNoise;
  options.associationMetadataRepeatedObjectRatio =
      synthetic.repeatedObjectRatio;
  options.associationSemanticWeight = 0.0;
  options.associationShapeWeight = 0.0;
  options.associationDescriptorWeight = 0.0;
  options.associationGeometryScale = 1.0;
  options.associationRequireObservedObjects = true;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.01;
  options.associationMaxCandidatesPerEdge = 0;

  ManualObjectDranOptions metadataOnlyOptions = options;
  metadataOnlyOptions.associationResidualWarmStartUseGeometry = false;
  const ManualObjectDranRunResult metadataOnlyResult =
      runManualObjectDran(dir.string(), metadataOnlyOptions);
  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  ASSERT_EQ(metadataOnlyResult.iterations.size(), 1u);
  const ManualObjectDranIterationSummary &geometryRow =
      result.iterations.front();
  const ManualObjectDranIterationSummary &metadataOnlyRow =
      metadataOnlyResult.iterations.front();
  EXPECT_GT(geometryRow.associationPrecision,
            metadataOnlyRow.associationPrecision + 0.2);
  EXPECT_GT(geometryRow.associationRecall,
            metadataOnlyRow.associationRecall + 0.2);
  EXPECT_LT(geometryRow.associationFalseMergeRate,
            metadataOnlyRow.associationFalseMergeRate - 0.2);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOEvaluationReportsGlobalGaugeError) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_global_gauge_eval");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.beta = 0.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.finalEstimates.size(), synthetic.numRobots);

  std::vector<Matrix> shifted = result.finalEstimates;
  const std::size_t localBlocks =
      synthetic.posesPerRobot + syntheticLocalObjectSlotCount(synthetic);
  const std::size_t blockWidth = synthetic.dimension + 1;
  for (std::size_t block = 0; block < localBlocks; ++block) {
    shifted[1].block(0, static_cast<int>(block * blockWidth + synthetic.dimension),
                     static_cast<int>(synthetic.dimension), 1)
        .array() += 5.0;
  }

  const SyntheticObjectPGOEvaluation evaluation =
      evaluateSyntheticObjectPGOEstimates(synthetic, shifted);
  EXPECT_LT(evaluation.trajectoryTranslationRmse, 1e-3);
  EXPECT_GT(evaluation.globalTrajectoryTranslationRmse, 1.0);
  EXPECT_GT(evaluation.globalTrajectoryTranslationRmse,
            1000.0 * evaluation.trajectoryTranslationRmse);
  EXPECT_GT(evaluation.globalObjectTranslationRmse, 1.0);
  EXPECT_GT(evaluation.objectCopyTranslationSpreadRmse, 1.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOEvaluationReportsObservedObjectSubset) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_observed_eval");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = false;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = syntheticLocalObjectSlotCount(synthetic);
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::IdentityOracle;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  const SyntheticObjectPGOEvaluation evaluation =
      evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);

  EXPECT_GT(evaluation.observedObjectCopyCount, 0u);
  EXPECT_LT(evaluation.observedObjectCopyCount, evaluation.objectCopyCount);
  EXPECT_TRUE(std::isfinite(evaluation.observedObjectTranslationRmse));
  EXPECT_TRUE(std::isfinite(evaluation.observedGlobalObjectTranslationRmse));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOEvaluationReportsQuotientObjectRotation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_quotient_eval");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 1;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 0;
  options.associationMode = ManualObjectDranAssociationMode::IdentityOracle;
  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);

  std::vector<Matrix> rotated = result.finalEstimates;
  ASSERT_FALSE(rotated.empty());
  const std::size_t blockWidth = synthetic.dimension + 1;
  const std::size_t c4ObjectIndex = synthetic.posesPerRobot + 2;
  Matrix yaw = Matrix::Identity(synthetic.dimension, synthetic.dimension);
  yaw(0, 0) = 0.0;
  yaw(0, 1) = -1.0;
  yaw(1, 0) = 1.0;
  yaw(1, 1) = 0.0;
  const Matrix shiftedRotation =
      rotated[0].block(0, static_cast<int>(c4ObjectIndex * blockWidth),
                       synthetic.dimension, synthetic.dimension) *
      yaw;
  rotated[0].block(0, static_cast<int>(c4ObjectIndex * blockWidth),
                   synthetic.dimension, synthetic.dimension) =
      shiftedRotation;

  const SyntheticObjectPGOEvaluation base =
      evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);
  const SyntheticObjectPGOEvaluation branchShifted =
      evaluateSyntheticObjectPGOEstimates(synthetic, rotated);

  EXPECT_GT(branchShifted.globalObjectRotationFrobeniusRmse,
            base.globalObjectRotationFrobeniusRmse + 0.1);
  EXPECT_NEAR(branchShifted.globalObjectQuotientRotationFrobeniusRmse,
              base.globalObjectQuotientRotationFrobeniusRmse, 1e-12);
  EXPECT_NEAR(
      branchShifted.observedGlobalObjectQuotientRotationFrobeniusRmse,
      base.observedGlobalObjectQuotientRotationFrobeniusRmse, 1e-12);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGONoCrossObjectBetaZeroKeepsConsensusOff) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_no_cross_object");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 4;
  synthetic.posesPerRobot = 6;
  synthetic.observeAllObjects = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::GeometrySoft;
  options.beta = 0.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  EXPECT_EQ(result.iterations.back().activeConsensusPairs, 0u);
  EXPECT_DOUBLE_EQ(result.iterations.back().consensusCost, 0.0);

  const SyntheticObjectPGOEvaluation evaluation =
      evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);
  EXPECT_TRUE(std::isfinite(evaluation.globalTrajectoryTranslationRmse));
  EXPECT_TRUE(std::isfinite(evaluation.objectCopyTranslationSpreadRmse));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsQuotientMetricObjectConsistency) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_quotient_metric");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 5;
  synthetic.observeAllObjects = true;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::GeometrySoft;
  options.associationUseSyntheticMetadata = true;
  options.associationUseQuotientMetric = true;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  EXPECT_GT(result.iterations.back().activeConsensusPairs, 0u);
  EXPECT_TRUE(std::isfinite(result.iterations.back().consensusCost));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsFirstBaselineTableMethods) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_baseline_table");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 3;
  synthetic.posesPerRobot = 4;
  synthetic.robotDriftPerRobot = 0.2;
  synthetic.objectInitialDriftPerRobot = 0.2;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  std::vector<ManualObjectDranAssociationMode> modes = {
      ManualObjectDranAssociationMode::IdentityOracle,
      ManualObjectDranAssociationMode::GeometryHard,
      ManualObjectDranAssociationMode::GeometrySoft,
      ManualObjectDranAssociationMode::ResidualSoft,
  };
  for (ManualObjectDranAssociationMode mode : modes) {
    ManualObjectDranOptions options;
    options.numRobots = synthetic.numRobots;
    options.numObjects = synthetic.numObjects;
    options.maxIterations = 2;
    options.topology = ManualObjectDranTopology::Complete;
    options.associationMode = mode;
    options.associationSinkhornIterations = 30;
    options.associationTemperature = 0.5;
    options.localTrustRegionIterations = 1;
    options.localTrustRegionMaxInnerIterations = 2;

    const ManualObjectDranRunResult result =
        runManualObjectDran(dir.string(), options);
    ASSERT_FALSE(result.iterations.empty());
    const SyntheticObjectPGOEvaluation evaluation =
        evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);
    const ManualObjectDranIterationSummary &row = result.iterations.back();
    EXPECT_EQ(evaluation.trajectoryPoseCount,
              synthetic.numRobots * synthetic.posesPerRobot);
    EXPECT_EQ(evaluation.objectCopyCount,
              synthetic.numRobots * synthetic.numObjects);
    EXPECT_TRUE(std::isfinite(evaluation.trajectoryTranslationRmse));
    EXPECT_TRUE(std::isfinite(evaluation.objectTranslationRmse));
    EXPECT_TRUE(std::isfinite(row.measurementCost));
    EXPECT_TRUE(std::isfinite(row.consensusCost));
    EXPECT_TRUE(std::isfinite(row.gradient));
    EXPECT_TRUE(std::isfinite(row.associationF1));
    EXPECT_GE(row.associationPrecision, 0.0);
    EXPECT_LE(row.associationPrecision, 1.0);
    EXPECT_GE(row.associationRecall, 0.0);
    EXPECT_LE(row.associationRecall, 1.0);
    EXPECT_GE(row.associationFalseMergeRate, 0.0);
    EXPECT_LE(row.associationFalseMergeRate, 1.0);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsMetadataWeightedAssociation) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_metadata_weighted");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.numCategories = 2;
  synthetic.descriptorDim = 5;
  synthetic.semanticAccuracy = 0.8;
  synthetic.descriptorNoise = 0.03;
  synthetic.shapeNoise = 0.02;
  synthetic.repeatedObjectRatio = 0.6;
  synthetic.robotDriftPerRobot = 0.25;
  synthetic.objectInitialDriftPerRobot = 0.25;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = synthetic.numObjects;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ResidualSoft;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationMetadataDescriptorDim = synthetic.descriptorDim;
  options.associationMetadataSemanticAccuracy = synthetic.semanticAccuracy;
  options.associationMetadataDescriptorNoise = synthetic.descriptorNoise;
  options.associationMetadataShapeNoise = synthetic.shapeNoise;
  options.associationMetadataRepeatedObjectRatio =
      synthetic.repeatedObjectRatio;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 1.0;
  options.associationDescriptorWeight = 1.0;
  options.associationSinkhornIterations = 30;
  options.associationTemperature = 0.5;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_FALSE(result.iterations.empty());
  const ManualObjectDranIterationSummary &row = result.iterations.back();
  const SyntheticObjectPGOEvaluation evaluation =
      evaluateSyntheticObjectPGOEstimates(synthetic, result.finalEstimates);

  EXPECT_TRUE(std::isfinite(row.associationF1));
  EXPECT_TRUE(std::isfinite(row.associationEntropy));
  EXPECT_GE(row.associationPrecision, 0.0);
  EXPECT_LE(row.associationPrecision, 1.0);
  EXPECT_GE(row.associationRecall, 0.0);
  EXPECT_LE(row.associationRecall, 1.0);
  EXPECT_TRUE(std::isfinite(evaluation.objectTranslationRmse));
  EXPECT_TRUE(std::isfinite(evaluation.trajectoryTranslationRmse));

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGORunsTraditionalHardAssociationBaselines) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_traditional_hard");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.numRobots = 2;
  synthetic.numObjects = 6;
  synthetic.posesPerRobot = 5;
  synthetic.numCategories = 3;
  synthetic.descriptorDim = 4;
  synthetic.semanticAccuracy = 0.0;
  synthetic.descriptorNoise = 0.0;
  synthetic.shapeNoise = 0.0;
  synthetic.repeatedObjectRatio = 0.5;
  synthetic.robotDriftPerRobot = 0.25;
  synthetic.objectInitialDriftPerRobot = 0.25;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  std::vector<ManualObjectDranAssociationMode> modes = {
      ManualObjectDranAssociationMode::ClassOnlyHard,
      ManualObjectDranAssociationMode::MetadataHard,
  };
  for (ManualObjectDranAssociationMode mode : modes) {
    ManualObjectDranOptions options;
    options.numRobots = synthetic.numRobots;
    options.numObjects = synthetic.numObjects;
    options.maxIterations = 0;
    options.topology = ManualObjectDranTopology::Complete;
    options.associationMode = mode;
    options.associationUseSyntheticMetadata = true;
    options.associationMetadataNumCategories = synthetic.numCategories;
    options.associationMetadataDescriptorDim = synthetic.descriptorDim;
    options.associationMetadataSemanticAccuracy = synthetic.semanticAccuracy;
    options.associationMetadataDescriptorNoise = synthetic.descriptorNoise;
    options.associationMetadataShapeNoise = synthetic.shapeNoise;
    options.associationMetadataRepeatedObjectRatio =
        synthetic.repeatedObjectRatio;
    options.associationSemanticWeight = 1.0;
    options.associationShapeWeight = 0.0;
    options.associationDescriptorWeight = 0.0;

    const ManualObjectDranRunResult result =
        runManualObjectDran(dir.string(), options);
    ASSERT_EQ(result.iterations.size(), 1u);
    const ManualObjectDranIterationSummary &row = result.iterations.front();
    EXPECT_EQ(row.associationCandidateCount,
              synthetic.numRobots * (synthetic.numRobots - 1) *
                  synthetic.numObjects);
    EXPECT_LT(row.associationF1, 1.0);
    EXPECT_GT(row.associationFalseMergeRate, 0.0);
  }

  std::filesystem::remove_all(dir);
}

TEST(testDPGO,
     SyntheticObjectPGOUsesLocalSlotMetadataForPermutedClassAssociations) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_local_metadata");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 5;
  synthetic.posesPerRobot = 8;
  synthetic.numCategories = synthetic.numObjects;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.semanticAccuracy = 1.0;
  synthetic.repeatedObjectRatio = 0.0;
  synthetic.worldLength = 14.0;
  synthetic.worldWidth = 8.0;
  synthetic.worldHeight = 3.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  ManualObjectDranOptions options;
  options.numRobots = synthetic.numRobots;
  options.numObjects = syntheticLocalObjectSlotCount(synthetic);
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Complete;
  options.associationMode = ManualObjectDranAssociationMode::ClassOnlyHard;
  options.associationUseSyntheticMetadata = true;
  options.associationMetadataNumCategories = synthetic.numCategories;
  options.associationSemanticWeight = 1.0;
  options.associationShapeWeight = 0.0;
  options.associationDescriptorWeight = 0.0;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dir.string(), options);
  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationRecall, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationF1, 1.0);
  EXPECT_DOUBLE_EQ(result.iterations.front().associationFalseMergeRate, 0.0);

  std::filesystem::remove_all(dir);
}

TEST(testDPGO, SyntheticObjectPGOUsesMappedOracleAssociationDiagnostics) {
  const std::filesystem::path dir =
      syntheticTestDirectory("dpgo_synthetic_object_pgo_mapped_oracle");
  std::filesystem::remove_all(dir);

  SyntheticObjectPGOOptions synthetic;
  synthetic.dimension = 3;
  synthetic.numRobots = 2;
  synthetic.numObjects = 5;
  synthetic.posesPerRobot = 8;
  synthetic.numCategories = 3;
  synthetic.observeAllObjects = true;
  synthetic.permuteLocalObjectIds = true;
  synthetic.objectObservationsPerObject = 2;
  synthetic.worldLength = 14.0;
  synthetic.worldWidth = 8.0;
  synthetic.worldHeight = 3.0;
  writeSyntheticObjectPGODirectory(dir.string(), synthetic);

  const std::size_t localSlots = syntheticLocalObjectSlotCount(synthetic);

  ManualObjectDranOptions diagonalOptions;
  diagonalOptions.numRobots = synthetic.numRobots;
  diagonalOptions.numObjects = localSlots;
  diagonalOptions.maxIterations = 0;
  diagonalOptions.topology = ManualObjectDranTopology::Complete;
  diagonalOptions.associationMode = ManualObjectDranAssociationMode::IdentityOracle;

  const ManualObjectDranRunResult diagonal =
      runManualObjectDran(dir.string(), diagonalOptions);
  ASSERT_EQ(diagonal.iterations.size(), 1u);
  EXPECT_LT(diagonal.iterations.front().associationF1, 1.0);
  EXPECT_GT(diagonal.iterations.front().associationFalseMergeRate, 0.0);

  ManualObjectDranOptions oracleOptions = diagonalOptions;
  oracleOptions.associationMode = ManualObjectDranAssociationMode::FixedFileSoft;
  oracleOptions.associationFile = (dir / "gt_associations.csv").string();

  const ManualObjectDranRunResult oracle =
      runManualObjectDran(dir.string(), oracleOptions);
  ASSERT_EQ(oracle.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(oracle.iterations.front().associationPrecision, 1.0);
  EXPECT_DOUBLE_EQ(oracle.iterations.front().associationRecall, 1.0);
  EXPECT_DOUBLE_EQ(oracle.iterations.front().associationF1, 1.0);
  EXPECT_DOUBLE_EQ(oracle.iterations.front().associationFalseMergeRate, 0.0);

  std::filesystem::remove_all(dir);
}
