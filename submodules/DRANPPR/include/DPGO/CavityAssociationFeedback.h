/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef CAVITY_ASSOCIATION_FEEDBACK_H
#define CAVITY_ASSOCIATION_FEEDBACK_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace DPGO {

struct CavityPairBlockEvidence {
  bool valid{false};
  std::string status;
  double score{0.0};
  double profileChiSquared{0.0};
  double covarianceLogDet{0.0};
  double gaugeInformationLogDet{0.0};
  std::size_t gaugeRank{0};
  std::size_t numMatches{0};
};

CavityPairBlockEvidence profileCavityPairBlockEvidence(
    const Matrix &firstPoints, const Matrix &secondPoints,
    const Matrix &firstCovariance, const Matrix &secondCovariance);

CavityPairBlockEvidence profileDiagonalCavityPairBlockEvidence(
    const Matrix &firstPoints, const Matrix &secondPoints,
    const Vector &firstVariance, const Vector &secondVariance);

struct CavityComponentAssignment {
  std::vector<std::pair<std::size_t, std::size_t>> assignments;
  double totalCost{0.0};
};

CavityComponentAssignment solveCavityComponentAssignment(
    const std::vector<std::size_t> &candidateFirstObjects,
    const std::vector<std::size_t> &activeSecondObjects,
    const Matrix &costs);

struct CavityRobotAssociationState {
  std::vector<std::size_t> objectIds;
  std::vector<std::size_t> categories;
  Matrix points;
  Vector covarianceDiagonal;
};

struct CavityPairReassignment {
  bool accepted{false};
  double scoreBefore{0.0};
  double scoreAfterCandidate{0.0};
  std::vector<std::pair<std::size_t, std::size_t>> assignmentsBefore;
  std::vector<std::pair<std::size_t, std::size_t>> assignmentsAfter;
  std::size_t changedEdgeCount{0};
  std::size_t componentCount{0};
  std::size_t candidateEvaluationCount{0};
  std::size_t payloadBytes{0};
  std::size_t configuredScoreWeights{0};
  std::size_t configuredAcceptanceThresholds{0};
};

CavityPairReassignment reassignCavityPairIdentities(
    const CavityRobotAssociationState &first,
    const CavityRobotAssociationState &second,
    const std::vector<std::pair<std::size_t, std::size_t>>
        &activeAssignments);

std::vector<std::pair<std::size_t, std::size_t>>
extractCavityActiveAssignments(const Matrix &weights);

}  // namespace DPGO

#endif  // CAVITY_ASSOCIATION_FEEDBACK_H
