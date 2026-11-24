/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef DECENTRALIZED_BILEVEL_ASSOCIATION_H
#define DECENTRALIZED_BILEVEL_ASSOCIATION_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <functional>
#include <map>
#include <vector>

namespace DPGO {

struct SharedEdgeCommand {
  std::size_t firstRobotId{0};
  std::size_t secondRobotId{0};
  Matrix association;
  Matrix alignment;
};

SharedEdgeCommand reciprocalSharedEdgeCommand(
    const SharedEdgeCommand &command);

// The lower MM solve holds the edge command fixed while updating only local
// pose-object state, its robust latent, and the chosen gauge representative.
struct FixedCommandLowerContract {
  bool freezeAssociationCommand{true};
  bool freezeAlignmentCommand{true};
  bool updateRobustLatent{true};
  bool projectGaugeRepresentative{true};
};

FixedCommandLowerContract sharedEdgeInexactBilevelLowerContract();

struct LocalInexactnessCertificate {
  double quotientStationarityResidual{0.0};
  double robustLatentFixedPointResidual{0.0};
  double normalizedResidual{0.0};
};

LocalInexactnessCertificate evaluateLocalInexactnessCertificate(
    double quotientStationarityResidual,
    double robustLatentFixedPointResidual, double stationarityScale,
    double robustLatentScale);

bool localLowerContinuationRequired(
    const LocalInexactnessCertificate &certificate, double tolerance);

// A robot can stop its fixed-command lower block only when both its local
// quotient/MM residual and every incident shared-object ADMM residual are
// small. The incident values are computed by edge owners and piggybacked on
// the corresponding target messages.
struct LocalInexactAdmmCertificate {
  LocalInexactnessCertificate local;
  double stationarityRatio{0.0};
  double robustLatentRatio{0.0};
  double incidentPrimalResidual{0.0};
  double incidentDualResidual{0.0};
  double normalizedResidual{0.0};
};

LocalInexactAdmmCertificate evaluateLocalInexactAdmmCertificate(
    const LocalInexactnessCertificate &local,
    double stationarityScale, double robustLatentScale,
    double incidentPrimalResidual, double incidentDualResidual,
    double absoluteTolerance, double relativeTolerance);

bool localInexactAdmmContinuationRequired(
    const LocalInexactAdmmCertificate &certificate);

struct LocalCommandForcingCertificate {
  double stationarityResidual{0.0};
  double commandInducedResidual{0.0};
  double forcingRatio{0.0};
};

LocalCommandForcingCertificate evaluateLocalCommandForcingCertificate(
    const Matrix &currentKktResidual,
    const Matrix &previousCommandKktResidual,
    const Matrix &currentCommandKktResidual);

bool localCommandForcingContinuationRequired(
    const LocalCommandForcingCertificate &certificate);

// One undirected robot edge owns this scaled-form ADMM state for a shared
// object hypothesis. Both endpoint poses are expressed in the edge owner's
// frame before the update.
struct SharedObjectConsensusState {
  bool initialized{false};
  bool dualCertificatePrimed{false};
  std::size_t commandEpoch{0};
  Matrix latent;
  Matrix firstScaledDual;
  Matrix secondScaledDual;
};

struct SharedObjectConsensusPenalty {
  double rotation{1.0};
  double translation{1.0};
};

struct SharedObjectEdgePenaltyBalanceParameters {
  bool enabled{false};
  double initialBeta{1.0};
  double minimumBeta{0.05};
  double maximumBeta{0.5};
  double imbalanceRatio{5.0};
  double updateFactor{1.25};
  double emaDecay{0.8};
  std::size_t updatePeriod{5};
  double absoluteTolerance{1e-3};
  double relativeTolerance{1e-3};
};

struct SharedObjectEdgePenaltyBalanceState {
  bool initialized{false};
  bool midpointPrimed{false};
  bool residualRatioPrimed{false};
  std::size_t commandEpoch{0};
  std::size_t updateCount{0};
  std::size_t increaseCount{0};
  std::size_t decreaseCount{0};
  double beta{1.0};
  double logResidualRatioEma{0.0};
  Matrix previousMidpoint;
};

struct SharedObjectEdgePenaltyBalanceUpdate {
  double beta{1.0};
  double normalizedPrimalResidual{0.0};
  double normalizedMidpointMotionResidual{0.0};
  bool penaltyChanged{false};
  bool penaltyIncreased{false};
  bool penaltyDecreased{false};
};

// Balances one undirected edge penalty from endpoint disagreement and the
// penalty-scaled motion of a locally derived virtual midpoint. The midpoint
// is diagnostic state, not a communicated or optimized shared variable.
SharedObjectEdgePenaltyBalanceUpdate updateSharedObjectEdgePenaltyBalance(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectEdgePenaltyBalanceState &state,
    const SharedObjectEdgePenaltyBalanceParameters &parameters,
    const SharedObjectConsensusPenalty &information = {},
    double edgeWeight = 1.0, std::size_t iteration = 0,
    bool advanceState = true, std::size_t commandEpoch = 0);

SharedObjectConsensusPenalty eliminateSharedObjectConsensusLatentPenalty(
    const SharedObjectConsensusPenalty &splitPenalty);

Matrix buildSharedObjectDirectEdgeDualProximalTarget(
    const Matrix &currentEndpoint, const Matrix &biasedNeighborTarget);

Matrix buildSharedObjectDirectEdgeDualProximalTarget(
    const Matrix &currentEndpoint, const Matrix &biasedNeighborTarget,
    double curvatureScale);

struct SharedObjectConsensusUpdate {
  Matrix firstTarget;
  Matrix secondTarget;
  double primalResidual{0.0};
  double dualResidual{0.0};
  double normalizedPrimalResidual{0.0};
  double normalizedDualResidual{0.0};
  double rotationDualResidual{0.0};
  double translationDualResidual{0.0};
  double normalizedRotationDualResidual{0.0};
  double normalizedTranslationDualResidual{0.0};
  double rotationPhysicalMultiplierNorm{0.0};
  double translationPhysicalMultiplierNorm{0.0};
};

SharedObjectConsensusUpdate updateSharedObjectConsensusState(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectConsensusState &state,
    const SharedObjectConsensusPenalty &penalty = {},
    double absoluteTolerance = 1e-3,
    double relativeTolerance = 1e-3,
    std::size_t commandEpoch = 0);

// One undirected edge owns a single physical multiplier for the direct
// equality constraint. Unlike SharedObjectConsensusState, this state has no
// auxiliary consensus latent.
struct SharedObjectDirectEdgeDualState {
  bool initialized{false};
  bool dualCertificatePrimed{false};
  bool endpointSequencePrimed{false};
  bool dualRelaxationSettled{false};
  bool spectralStepPrimed{false};
  bool inertialHistoryPrimed{false};
  bool momentumStepHistoryPrimed{false};
  std::size_t commandEpoch{0};
  std::size_t firstEndpointSequence{0};
  std::size_t secondEndpointSequence{0};
  Matrix physicalMultiplier;
  Matrix previousFirstEndpoint;
  Matrix previousSecondEndpoint;
  Matrix previousPhysicalMultiplier;
  Matrix previousFirstEndpointStep;
  Matrix previousSecondEndpointStep;
  Matrix previousSpectralScaledMultiplier;
  Matrix previousSpectralWhitenedResidual;
  double spectralStep{1.0};
  SharedObjectEdgePenaltyBalanceState penaltyBalance;
};

struct SharedObjectDirectEdgeDualSpectralContribution {
  double numerator{0.0};
  double denominator{0.0};
  bool valid{false};
  double rotationNumerator{0.0};
  double rotationDenominator{0.0};
  bool rotationValid{false};
  double translationNumerator{0.0};
  double translationDenominator{0.0};
  bool translationValid{false};
};

struct SharedObjectDirectEdgeMomentumContribution {
  double numerator{0.0};
  double denominator{0.0};
  bool valid{false};
};

SharedObjectDirectEdgeMomentumContribution
sharedObjectDirectEdgeEndpointMomentumContribution(
    const Matrix &previousStep, const Matrix &currentStep);

SharedObjectDirectEdgeMomentumContribution
updateSharedObjectDirectEdgeEndpointMomentumHistory(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state);

double reduceSharedObjectDirectEdgeMomentum(
    const std::vector<SharedObjectDirectEdgeMomentumContribution>
        &contributions,
    double fallbackMomentum, double previousMomentum,
    double minimumMomentum, double maximumMomentum, double emaDecay,
    std::size_t *validContributions = nullptr);

SharedObjectDirectEdgeDualSpectralContribution
updateSharedObjectDirectEdgeDualSpectralHistory(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty);

double reduceSharedObjectDirectEdgeDualSpectralStep(
    const std::vector<SharedObjectDirectEdgeDualSpectralContribution>
        &contributions,
    double fallbackStep, double previousStep, double minimumStep,
    double maximumStep, double emaDecay,
    std::size_t *validContributions = nullptr);

double estimateSharedObjectDirectEdgeDualSpectralStep(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double fallbackStep,
    double minimumStep, double maximumStep, double emaDecay);

// Re-express an owner-frame physical multiplier after a left gauge rotation.
// The translation of the gauge cancels from the endpoint equality residual.
void transportSharedObjectDirectEdgeDualState(
    SharedObjectDirectEdgeDualState &state,
    const Matrix &ownerGaugeRotation,
    const Matrix &ownerGaugeTranslation = Matrix());

SharedObjectConsensusUpdate updateSharedObjectDirectEdgeDualState(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty = {},
    double dualStepSize = 1.0,
    double absoluteTolerance = 1e-3,
    double relativeTolerance = 1e-3,
    std::size_t commandEpoch = 0,
    double robustReliability = 1.0,
    double associationProbability = 1.0,
    bool advanceDual = true);

struct SharedObjectDirectEdgeDualInertialUpdate {
  SharedObjectConsensusUpdate consensus;
  bool attempted{false};
  bool accepted{false};
  bool restarted{false};
  double baselineMerit{0.0};
  double acceleratedMerit{0.0};
};

// Applies local primal-dual inertia using only endpoint and multiplier
// histories already owned by the edge. The guarded choice commits either the
// complete baseline state or the complete extrapolated state.
SharedObjectDirectEdgeDualInertialUpdate
updateSharedObjectDirectEdgeDualStateInertial(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty = {},
    double dualStepSize = 1.0, double momentum = 0.25,
    double absoluteTolerance = 1e-3,
    double relativeTolerance = 1e-3,
    std::size_t commandEpoch = 0,
    double robustReliability = 1.0,
    double associationProbability = 1.0,
    bool advanceDual = true);

SharedObjectConsensusUpdate
updateSharedObjectDirectEdgeDualStateWithChannelSteps(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double rotationStepSize,
    double translationStepSize, double absoluteTolerance = 1e-3,
    double relativeTolerance = 1e-3, std::size_t commandEpoch = 0,
    double robustReliability = 1.0,
    double associationProbability = 1.0, bool advanceDual = true);

struct SharedObjectLowerMessageKey {
  std::size_t receiverRobotId{0};
  std::size_t senderRobotId{0};
  std::size_t receiverObjectId{0};
  std::size_t senderObjectId{0};

  bool operator==(const SharedObjectLowerMessageKey &other) const;
  bool operator<(const SharedObjectLowerMessageKey &other) const;
};

struct SharedObjectLowerMessage {
  SharedObjectLowerMessageKey key;
  std::size_t ownerRobotId{0};
  std::size_t messageSequence{0};
  std::size_t commandEpoch{0};
  std::size_t ownerEndpointSequence{0};
  std::size_t receiverEndpointSequence{0};
  Matrix target;
  double robustReliability{1.0};
  double normalizedPrimalResidual{0.0};
  double normalizedDualResidual{0.0};
};

class SharedObjectLowerMessageQueue {
 public:
  SharedObjectLowerMessage send(
      std::size_t round, int deliveryDelay, bool dropAttempt,
      SharedObjectLowerMessage message);
  bool retry(std::size_t messageSequence, std::size_t round,
             int deliveryDelay);
  std::vector<SharedObjectLowerMessage> deliverDueMessages(
      std::size_t receiverRobotId, std::size_t round);
  std::vector<SharedObjectLowerMessage> deliverDueMessages(
      std::size_t receiverRobotId, std::size_t round,
      const std::vector<std::size_t> &reachableSenderRobotIds);
  // Transport-level receiver ACK: delivery alone does not clear sender
  // outstanding state, and dropped attempts stay outstanding until retry/ACK.
  bool acknowledge(std::size_t receiverRobotId,
                   std::size_t messageSequence);

  std::vector<SharedObjectLowerMessage> retryableMessagesBySender(
      std::size_t senderRobotId) const;
  std::size_t outstandingCountBySender(std::size_t senderRobotId) const;
  std::size_t pendingDeliveryAttempts() const;
  std::size_t droppedAttempts() const;
  bool empty() const;
  bool hasOutstandingMessages() const;

 private:
  struct PendingAttempt {
    std::size_t deliveryRound{0};
    std::size_t attemptSequence{0};
    std::size_t messageSequence{0};
  };

  void enqueueAttempt(std::size_t messageSequence, std::size_t round,
                      int deliveryDelay);
  std::vector<SharedObjectLowerMessage> deliverDueMessagesWithReachability(
      std::size_t receiverRobotId, std::size_t round,
      const std::vector<std::size_t> *reachableSenderRobotIds);

  std::size_t nextMessageSequence_{1};
  std::size_t nextAttemptSequence_{1};
  std::size_t droppedAttempts_{0};
  std::map<std::size_t, SharedObjectLowerMessage> outstanding_;
  std::vector<PendingAttempt> pending_;
};

enum class SharedObjectLowerMessageStatus {
  Accepted,
  StaleEpoch,
  StaleEndpoint,
  StaleSequence,
  KeyMismatch,
  InactiveStream,
  InvalidPayload
};

struct SharedObjectLowerMessageReceiverState {
  bool initialized{false};
  bool streamActive{false};
  SharedObjectLowerMessageKey streamKey;
  std::size_t expectedCommandEpoch{0};
  std::size_t currentEndpointSequence{0};
  std::size_t lastAcceptedMessageSequence{0};
};

SharedObjectLowerMessageStatus classifySharedObjectLowerMessage(
    const SharedObjectLowerMessage &message,
    const SharedObjectLowerMessageReceiverState &receiverState);

struct LocalSharedObjectLowerQuiescenceState {
  bool localLowerCertificateConverged{false};
  std::size_t currentIncidentCertificates{0};
  std::size_t expectedIncidentCertificates{0};
  std::size_t senderOutstandingMessages{0};
};

bool sharedObjectLowerLocallyQuiescent(
    const LocalSharedObjectLowerQuiescenceState &state);

// Matrix-free action M = J P R A for the translation part of a fixed
// split-consensus lower quadratic. Edge vectors use sqrt-information-scaled
// owner frames. Each robot callback receives all of its incident edge blocks
// at once so its local response retains cross-edge compliance.
class SharedObjectSplitLatentOperator {
 public:
  using LocalResponse =
      std::function<Matrix(std::size_t robotId,
                           const Matrix &scaledEndpointDirections)>;

  SharedObjectSplitLatentOperator(
      std::size_t numEdges, std::size_t blockDimension,
      std::vector<std::vector<std::size_t>> incidentEdgesByRobot,
      LocalResponse localResponse);

  Matrix applyResponse(const Matrix &scaledEdgeDirections) const;
  Matrix applySystem(const Matrix &scaledEdgeLatents) const;

  // One theorem-scaled Richardson step for (I - M) z = rhs. The unit step is
  // nonexpansive when every local scaled compliance is bounded by identity,
  // as it is for the matching augmented lower Hessian.
  Matrix richardsonStep(const Matrix &scaledEdgeLatents,
                        const Matrix &rhs) const;

 private:
  void validateEdgeMatrix(const Matrix &values, const char *name) const;

  std::size_t numEdges_{0};
  std::size_t blockDimension_{0};
  std::vector<std::vector<std::size_t>> incidentEdgesByRobot_;
  LocalResponse localResponse_;
};


}  // namespace DPGO

#endif  // DECENTRALIZED_BILEVEL_ASSOCIATION_H
