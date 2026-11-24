#ifndef DPGO_GAUGE_COVARIANT_ASSOCIATION_H
#define DPGO_GAUGE_COVARIANT_ASSOCIATION_H

#include <DPGO/BilevelAssociation.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace DPGO {

struct GaugeCovariantTransformEstimate {
  bool valid{false};
  Matrix transform;
  Matrix information;
  Matrix covariance;
  double fitTranslationRmse{std::numeric_limits<double>::infinity()};
};

struct GaugeCovariantAssociationEdgeState {
  GaugeCovariantTransformEstimate direct;
  GaugeCovariantTransformEstimate network;
  double reliability{0.0};
  double normalizedDisagreement{
      std::numeric_limits<double>::infinity()};
};

using DirectedGaugeCovariantAssociationEdgeStates =
    std::vector<std::vector<GaugeCovariantAssociationEdgeState>>;

struct GaugeCovariantSynchronizationEdge {
  std::size_t firstRobotId{0};
  std::size_t secondRobotId{0};
  double topologyWeight{1.0};
  GaugeCovariantTransformEstimate directFirstFromSecond;
};

struct GaugeCovariantSynchronizationResult {
  std::vector<Matrix> gauges;
  std::vector<GaugeCovariantAssociationEdgeState> edges;
  std::size_t rounds{0};
  std::size_t payloadBytes{0};
};

GaugeCovariantTransformEstimate estimateGaugeCovariantTransform(
    const Matrix &receiverPoints, const Matrix &senderPoints,
    const std::vector<SparsePoseAlignmentWeightEntry> &matches);

GaugeCovariantTransformEstimate reciprocalGaugeCovariantTransform(
    const GaugeCovariantTransformEstimate &estimate);

double gaugeCovariantPredictiveTranslationVariance(
    const GaugeCovariantTransformEstimate &estimate,
    const Vector &senderPoint);

double gaugeCovariantNormalizedDisagreement(
    const GaugeCovariantTransformEstimate &directFirstFromSecond,
    const Matrix &firstGauge, const Matrix &secondGauge);

Matrix gaugeCovariantLowerViewTransform(
    const GaugeCovariantAssociationEdgeState &edge,
    bool preserveLocalStates);

GaugeCovariantSynchronizationResult synchronizeGaugeCovariantRobotGraph(
    std::size_t numRobots,
    const std::vector<GaugeCovariantSynchronizationEdge> &edges,
    const std::vector<Matrix> &warmStartGauges = {});

}  // namespace DPGO

#endif
