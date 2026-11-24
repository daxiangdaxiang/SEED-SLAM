/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef DPGO_SYNTHETIC_OBJECT_PGO_H
#define DPGO_SYNTHETIC_OBJECT_PGO_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace DPGO {

struct SyntheticObjectPGOOptions {
  std::size_t instanceSeed{0};
  std::size_t dimension{2};
  std::size_t numRobots{2};
  std::size_t numObjects{3};
  std::size_t posesPerRobot{4};
  double poseSpacing{1.0};
  double robotSpacing{2.0};
  double objectSpacing{1.25};
  double objectYOffset{0.75};
  double worldLength{50.0};
  double worldWidth{20.0};
  double worldHeight{5.0};
  double detectionRange{12.0};
  double horizontalFovDeg{150.0};
  double verticalFovDeg{100.0};
  double missDetectionRate{0.0};
  double observationTranslationNoise{0.0};
  double observationRotationNoise{0.0};
  std::size_t objectObservationsPerObject{1};
  std::size_t falsePositiveObjectsPerRobot{0};
  double duplicateObjectRatio{0.0};
  bool permuteLocalObjectIds{false};
  double robotDriftPerRobot{0.0};
  double objectInitialDriftPerRobot{0.0};
  bool observeAllObjects{true};
  std::size_t numCategories{3};
  std::size_t descriptorDim{4};
  double semanticAccuracy{1.0};
  double descriptorNoise{0.0};
  double shapeNoise{0.0};
  double repeatedObjectRatio{0.0};
  double symmetryYawAmbiguityRate{0.0};
};

std::size_t syntheticLocalObjectSlotCount(
    const SyntheticObjectPGOOptions &options);

enum class SyntheticObjectSymmetry {
  Identity,
  C2,
  C4,
};

struct SyntheticObjectMetadata {
  std::size_t category{0};
  SyntheticObjectSymmetry symmetry{SyntheticObjectSymmetry::Identity};
  Vector shape;
  Vector descriptor;
};

struct SyntheticObjectObservationMetadata {
  std::size_t robotId{0};
  std::size_t objectId{0};
  std::size_t category{0};
  SyntheticObjectSymmetry symmetry{SyntheticObjectSymmetry::Identity};
  double semanticConfidence{1.0};
  Vector shape;
  Vector descriptor;
};

struct SyntheticObjectMetadataSet {
  std::vector<SyntheticObjectMetadata> objects;
  std::vector<std::vector<SyntheticObjectObservationMetadata>> observations;
};

SyntheticObjectMetadataSet generateSyntheticObjectMetadata(
    const SyntheticObjectPGOOptions &options);

void writeSyntheticObjectPGODirectory(
    const std::string &directory,
    const SyntheticObjectPGOOptions &options = SyntheticObjectPGOOptions());

struct SyntheticObjectPGOEvaluation {
  std::size_t trajectoryPoseCount{0};
  std::size_t objectCopyCount{0};
  std::size_t observedObjectCopyCount{0};
  double trajectoryTranslationRmse{0.0};
  double trajectoryRotationFrobeniusRmse{0.0};
  double objectTranslationRmse{0.0};
  double objectRotationFrobeniusRmse{0.0};
  double objectQuotientRotationFrobeniusRmse{0.0};
  double observedObjectTranslationRmse{0.0};
  double observedObjectRotationFrobeniusRmse{0.0};
  double observedObjectQuotientRotationFrobeniusRmse{0.0};
  double globalTrajectoryTranslationRmse{0.0};
  double globalTrajectoryRotationFrobeniusRmse{0.0};
  double globalObjectTranslationRmse{0.0};
  double globalObjectRotationFrobeniusRmse{0.0};
  double globalObjectQuotientRotationFrobeniusRmse{0.0};
  double observedGlobalObjectTranslationRmse{0.0};
  double observedGlobalObjectRotationFrobeniusRmse{0.0};
  double observedGlobalObjectQuotientRotationFrobeniusRmse{0.0};
  std::size_t objectCopySpreadCount{0};
  double objectCopyTranslationSpreadRmse{0.0};
};

SyntheticObjectPGOEvaluation evaluateSyntheticObjectPGOEstimates(
    const SyntheticObjectPGOOptions &options,
    const std::vector<Matrix> &estimates);

}  // namespace DPGO

#endif
