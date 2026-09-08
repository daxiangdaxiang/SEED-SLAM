/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef MANUAL_OBJECT_DRAN_H
#define MANUAL_OBJECT_DRAN_H

#include <DPGO/BilevelAssociation.h>
#include <DPGO/DPGO_types.h>
#include <DPGO/GaugeCovariantAssociation.h>
#include <DPGO/ManualDpgoMm.h>
#include <DPGO/ObjectPGOData.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace DPGO {

enum class ManualObjectDranTopology {
  Ring,
  Complete,
};

enum class ManualObjectDranCommunicationPolicy {
  All,
  Triggered,
};

enum class ManualObjectDranObjectInitialization {
  CentralizedChordal,
  NeighborAverage,
  InputState,
};

enum class ManualObjectDranObjectInitializationAnchorMode {
  Constant,
  Information,
};

enum class ManualObjectDranObjectInterfaceMode {
  CopyBaseline,
  ReducedSchurResponse,
  CompleteGraphCoarseResponse,
  MatrixFreeObjectPose3Pcg,
};

enum class ManualObjectDranObjectInterfacePcgPreconditioner {
  InterfaceHessian,
  ExactSchur,
  PrivateBlockJacobi,
};

enum class ManualObjectDranObjectInterfacePcgPrivateSolver {
  SparseLdlt,
  BlockTridiagonalLdlt,
};

enum class ManualObjectDranObjectInterfaceResponseStrategy {
  Serial,
  Batch,
};

enum class ManualObjectDranObjectInterfaceResponseTrigger {
  Periodic,
  PredictedDecrease,
  LocalInnovation,
  AccumulatedInnovation,
};

enum class ManualObjectDranAssociationMode {
  IdentityOracle,
  ClassOnlyHard,
  MetadataHard,
  GeometryHard,
  GraphConsistencyHard,
  GeometrySoft,
  TransformSoft,
  FixedFileSoft,
  ResidualSoft,
};

enum class ManualObjectDranAssociationPolicy {
  Manual,
  TwoTimeScaleBilevelV1,
  SingleLoopBilevelV2,
  SingleLoopRotoBilevelV3,
  FinalDbaP4203,
  SharedEdgeDbaV1,
  SharedEdgeJointDbaV2,
  DecentralizedFeedbackUotDbaV1,
};

enum class ManualObjectDranFinalAblation {
  None,
  StateValidation,
  EdgeConsensus,
  ReciprocalSupport,
  PairSupportGate,
  PairSupportContinuation,
  SharedReliability,
  ObservationPredictiveGate,
};

enum class ManualObjectDranDiagnosticProfile {
  Auto,
  Full,
  Lightweight,
};

enum class ManualObjectDranObjectCommunicationRepresentation {
  LiftedPose,
  CompactPoseFloat64,
  CompactPoseFloat32,
  Point2,
};

enum class ManualObjectDranSharedLatentMode {
  Auto,
  Disabled,
  ExactFixed,
  ExactWeighted,
  KktActiveHybrid,
  DirectEdgeDual,
  ExactPoint2Owner,
  ExactPoint2SchurOracle,
  ExactPoint2OwnerPcg,
};

struct ManualObjectDranExactPoint2Statistic {
  std::size_t robot{0};
  std::size_t object{0};
  double information{0.0};
  Vector weightedPosition;
};

struct ManualObjectDranExactPoint2Packet {
  std::size_t sender{0};
  std::size_t owner{0};
  std::size_t object{0};
  bool carriesInformation{false};
  double information{0.0};
  Vector weightedPosition;
};

using ManualObjectDranExactPoint2InformationCache =
    std::vector<std::vector<double>>;

struct ManualObjectDranExactPoint2Update {
  bool valid{false};
  std::string status;
  std::vector<std::size_t> owners;
  std::vector<Vector> points;
  std::size_t statisticMessages{0};
  std::size_t statisticPayloadBytes{0};
  std::size_t broadcastMessages{0};
  std::size_t broadcastPayloadBytes{0};
};

std::vector<std::size_t> manualObjectDranExactPoint2Owners(
    std::size_t numRobots,
    const std::vector<std::vector<std::size_t>> &observers);

ManualObjectDranExactPoint2Statistic
manualObjectDranExactPoint2LocalStatistic(
    std::size_t robot, std::size_t object,
    const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const std::vector<Vector> &measurements,
    const std::vector<double> &information);

ManualObjectDranExactPoint2Update manualObjectDranAggregateExactPoint2(
    std::size_t numRobots,
    const std::vector<std::vector<std::size_t>> &observers,
    const std::vector<ManualObjectDranExactPoint2Statistic>
        &ownerLocalStatistics,
    const std::vector<ManualObjectDranExactPoint2Packet> &deliveredPackets,
    ManualObjectDranExactPoint2InformationCache &remoteInformationCache);

struct ManualObjectDranConditionalQuadratic {
  bool valid{false};
  std::string status;
  SparseMatrix Q;
  SparseMatrix G;
  Matrix initialState;
};

ManualObjectDranConditionalQuadratic
manualObjectDranConditionTrajectoryQuadratic(
    const SparseMatrix &fullQ, const SparseMatrix &fullG,
    const Matrix &fullState, std::size_t trajectoryPoses,
    unsigned dimension);

struct ManualObjectDranPoint2SchurModel {
  bool valid{false};
  std::string status;
  Matrix hessian;
  Vector gradient;
  std::size_t responseSolves{0};
  double maxResponseRelativeResidual{0.0};
  double symmetryError{0.0};
};

struct ManualObjectDranPoint2SchurSolve {
  bool valid{false};
  std::string status;
  Vector pointStep;
  double spectralShift{0.0};
  double relativeResidual{0.0};
};

struct ManualObjectDranPoint2SchurProduct {
  bool valid{false};
  std::string status;
  Vector product;
  std::size_t responseSolves{0};
  double responseRelativeResidual{0.0};
};

struct ManualObjectDranPoint2OwnerPcgSolve {
  bool valid{false};
  std::string status;
  Vector pointStep;
  std::size_t iterations{0};
  double relativeResidual{0.0};
  std::size_t gradientPayloadBytes{0};
  std::size_t directionPayloadBytes{0};
  std::size_t productPayloadBytes{0};
  std::size_t updatePayloadBytes{0};
  std::size_t scalarPayloadBytes{0};
  std::size_t totalPayloadBytes{0};
  std::size_t gradientMessages{0};
  std::size_t directionMessages{0};
  std::size_t productMessages{0};
  std::size_t updateMessages{0};
  std::size_t scalarMessages{0};
  std::size_t totalMessages{0};
  double warmStartScale{0.0};
};

struct ManualObjectDranChordalPose3FactorLinearization {
  Vector rotationResidual;
  Vector translationResidual;
  Matrix firstRotationJacobian;
  Matrix secondRotationJacobian;
  Matrix firstTranslationJacobian;
  Matrix secondTranslationJacobian;
};

struct ManualObjectDranBlockTridiagonalSolve {
  bool valid{false};
  std::string status;
  Matrix solution;
};

struct ManualObjectDranSparseCouplingEntry {
  std::size_t privateBlock{0};
  std::size_t objectBlock{0};
  Matrix value;
};

struct ManualObjectDranSparseCouplingResult {
  bool valid{false};
  std::string status;
  Matrix forwardProduct;
  Matrix transposeProduct;
  Matrix objectColumns;
  std::size_t aggregateBlocks{0};
};

ManualObjectDranSparseCouplingResult
manualObjectDranSparseCouplingOperationsForTest(
    std::size_t numPrivateBlocks, std::size_t numObjectBlocks,
    std::size_t blockDimension,
    const std::vector<ManualObjectDranSparseCouplingEntry> &entries,
    const Matrix &objectValues, const Matrix &privateValues,
    std::size_t selectedObject);

ManualObjectDranBlockTridiagonalSolve
manualObjectDranSolveBlockTridiagonalForTest(
    const Matrix &hessian, const Matrix &rhs, std::size_t blockDimension);

ManualObjectDranBlockTridiagonalSolve
manualObjectDranSolveBlockTridiagonalBlocksForTest(
    const std::vector<Matrix> &diagonalBlocks,
    const std::vector<Matrix> &lowerBlocks, const Matrix &rhs);

ManualObjectDranChordalPose3FactorLinearization
manualObjectDranLinearizeChordalPose3FactorForTest(
    const Matrix &first, const Matrix &measurement, const Matrix &second);

ManualObjectDranPoint2SchurModel
manualObjectDranPoint2SchurFromDenseForTest(
    const Matrix &hessian, const Vector &gradient,
    std::size_t privateVariables);

ManualObjectDranPoint2SchurSolve manualObjectDranSolvePoint2SchurForTest(
    const std::vector<ManualObjectDranPoint2SchurModel> &models);

bool manualObjectDranCoupledPoint2SchurCorrectionActiveForTest(
    bool enabled, std::size_t startRound, std::size_t period,
    std::size_t round);

bool manualObjectDranCoupledPoint2SchurUsesMatrixFreeOwnerPcgForTest(
    bool correctionEnabled, bool matrixFreeOwnerPcgEnabled);

bool manualObjectDranPoint2OwnerLedgerRequiredForTest(
    bool exactPoint2FamilyMode, bool coupledPoint2SchurOwnerPcgMode);

bool manualObjectDranCoupledPoint2SchurCorrectionAcceptedForTest(
    double costBefore, double costAfter,
    double relativeTolerance = 0.0);

Matrix manualObjectDranExtrapolateIntegratedPoint2StateForTest(
    const Matrix &current, const Matrix &previous,
    std::size_t trajectoryPoseCount, unsigned dimension, double momentum);

bool manualObjectDranIntegratedPoint2AmmSelectsAcceleratedForTest(
    double anchorCost, double baselineCost, double acceleratedCost);

bool manualObjectDranIntegratedPoint2AmmCommonAcceptanceForTest(
    double anchorCost, double baselineCost, double acceleratedCost,
    std::size_t edgeAttempts, double edgeBaselineMerit,
    double edgeAcceleratedMerit);

bool manualObjectDranIntegratedPoint2EdgeMeritAcceptsForTest(
    std::size_t edgeAttempts, double edgeBaselineMerit,
    double edgeAcceleratedMerit);

bool manualObjectDranIntegratedPoint2ObjectFactorMeritAcceptsForTest(
    const std::vector<double> &baselineCosts,
    const std::vector<double> &acceleratedCosts);

bool manualObjectDranIntegratedPoint2CopyMeritAcceptsForTest(
    double baselineInnovation, double acceleratedInnovation);

double manualObjectDranIntegratedPoint2EdgeMomentumForTest(
    double ownerMomentum, const std::vector<double> &objectMomenta,
    bool objectWise, std::size_t objectId);

double manualObjectDranIntegratedPoint2OwnerSpectralStepForTest(
    const std::vector<std::pair<double, double>> &validSecants,
    double fallbackStep, double previousStep, double minimumStep,
    double maximumStep, double emaDecay);

void manualObjectDranResizeOwnerSpectralStateForTest(
    std::vector<bool> &primed, std::vector<double> &steps,
    std::size_t numRobots);

ManualObjectDranPoint2SchurProduct
manualObjectDranPoint2SchurProductFromDenseForTest(
    const Matrix &hessian, std::size_t privateVariables,
    const Vector &value);

Matrix manualObjectDranSchurBlockDiagonalFromDenseForTest(
    const Matrix &privateHessian, const Matrix &privateInterfaceHessian,
    const Matrix &interfaceHessian, std::size_t blockDimension,
    double damping);

Matrix manualObjectDranPrivateBlockJacobiSchurDiagonalFromDenseForTest(
    const Matrix &privateHessian, const Matrix &privateInterfaceHessian,
    const Matrix &interfaceHessian, std::size_t privateBlockDimension,
    std::size_t interfaceBlockDimension, double damping);

bool manualObjectDranObjectInterfaceTerminalPolishActiveForTest(
    ManualObjectDranObjectInterfaceMode mode, bool pcgActivated,
    unsigned iter, unsigned maxIterations, unsigned polishIterations);

bool manualObjectDranObjectInterfaceReplaceFinalRoundForTest(
    ManualObjectDranObjectInterfaceMode mode, bool pcgActivated,
    unsigned iter, unsigned maxIterations, unsigned polishIterations,
    bool replaceFinalRound);

ManualObjectDranPoint2OwnerPcgSolve
manualObjectDranSolvePoint2OwnerPcgForTest(
    const std::vector<ManualObjectDranPoint2SchurModel> &models,
    const std::vector<std::vector<std::size_t>> &observers,
    const std::vector<std::size_t> &owners, const Vector &warmStart,
    std::size_t maxIterations, double relativeTolerance = 1e-10,
    bool optimizeWarmStart = false);

ManualObjectDranPoint2OwnerPcgSolve
manualObjectDranSolveInterfaceOwnerPcgForTest(
    const std::vector<ManualObjectDranPoint2SchurModel> &models,
    const std::vector<std::vector<std::size_t>> &observers,
    const std::vector<std::size_t> &owners, std::size_t blockDimension,
    const Vector &warmStart, std::size_t maxIterations,
    double relativeTolerance = 1e-10, bool optimizeWarmStart = false);

ManualObjectDranPoint2OwnerPcgSolve
manualObjectDranSolveInterfaceOwnerPcgRoutedForTest(
    const std::vector<ManualObjectDranPoint2SchurModel> &models,
    const std::vector<std::vector<std::size_t>> &observers,
    const std::vector<std::size_t> &owners, std::size_t blockDimension,
    const Vector &warmStart, std::size_t maxIterations,
    const std::vector<std::vector<std::size_t>> &hopDistances,
    double relativeTolerance = 1e-10, bool optimizeWarmStart = false);

Vector manualObjectDranRecoverPrivatePoint2StepForTest(
    const Matrix &hessian, const Vector &gradient,
    std::size_t privateVariables, const Vector &pointStep);

using ManualObjectDranDirectedAssociationWeights =
    std::vector<std::vector<Matrix>>;

struct ManualObjectDranSparseAssociationObjective {
  int rows{0};
  int cols{0};
  std::vector<SparseAssignmentCostEntry> entries;
};

using ManualObjectDranDirectedSparseAssociationObjectives =
    std::vector<std::vector<ManualObjectDranSparseAssociationObjective>>;

struct ManualObjectDranLocalLowerFeedbackState {
  bool stationarityScaleInitialized{false};
  bool robustLatentScaleInitialized{false};
  double stationarityScale{0.0};
  double robustLatentScale{0.0};
  double stationarityResidual{0.0};
  double robustLatentFixedPointResidual{0.0};
  double readiness{0.0};
};

struct ManualObjectDranEdgeFeedbackState {
  bool responseInitialized{false};
  std::size_t commandEpoch{0};
  double lastMannStep{1.0};
  int fixedObjectiveRows{0};
  int fixedObjectiveCols{0};
  std::vector<SparseAssignmentCostEntry> fixedObjectiveEntries;
};

double updateManualObjectDranLocalLowerFeedback(
    double stationarityResidual, double robustLatentFixedPointResidual,
    ManualObjectDranLocalLowerFeedbackState &state);

void invalidateManualObjectDranLocalLowerFeedbackCommand(
    ManualObjectDranLocalLowerFeedbackState &state);

struct ManualObjectDranAssociationTransform {
  bool valid{false};
  Matrix transform;
  double translationRmse{std::numeric_limits<double>::infinity()};
  double marginalTranslationRmse{std::numeric_limits<double>::infinity()};
  double synchronizationResidual{0.0};
  double referenceSynchronizationResidual{0.0};
};

using ManualObjectDranDirectedAssociationTransforms =
    std::vector<std::vector<ManualObjectDranAssociationTransform>>;

struct ManualObjectDranEvent {
  std::size_t event{0};
  std::size_t robot{0};
  double logicalTime{0.0};
  std::size_t delayEvents{0};
};

struct ManualObjectDranOptions {
  std::size_t numRobots{0};
  std::size_t numObjects{0};
  unsigned maxIterations{20};
  double beta{1.0};
  ManualObjectDranTopology topology{ManualObjectDranTopology::Ring};
  unsigned ringHops{1};
  std::string topologyFile;
  std::string topologyWeightMode{"unit"};
  ManualObjectDranCommunicationPolicy communicationPolicy{
      ManualObjectDranCommunicationPolicy::Triggered};
  ManualObjectDranObjectInitialization objectInitialization{
      ManualObjectDranObjectInitialization::CentralizedChordal};
  ManualObjectDranObjectInitializationAnchorMode
      objectInitializationAnchorMode{
          ManualObjectDranObjectInitializationAnchorMode::Constant};
  std::size_t objectInitializationConsensusRounds{0};
  double objectInitializationObservedAnchorWeight{1.0};
  double objectInitializationRelayAnchorWeight{0.0};
  bool objectOrientationObserved{true};
  ManualObjectDranObjectCommunicationRepresentation
      objectCommunicationRepresentation{
          ManualObjectDranObjectCommunicationRepresentation::LiftedPose};
  bool objectCommunicationLinkIncidence{false};
  bool associationAwareTedCciInitialization{false};
  ManualObjectDranObjectInterfaceMode objectInterfaceMode{
      ManualObjectDranObjectInterfaceMode::CopyBaseline};
  ManualObjectDranObjectInterfaceResponseStrategy
      objectInterfaceResponseStrategy{
          ManualObjectDranObjectInterfaceResponseStrategy::Serial};
  ManualObjectDranObjectInterfaceResponseTrigger objectInterfaceResponseTrigger{
      ManualObjectDranObjectInterfaceResponseTrigger::Periodic};
  ManualObjectDranAssociationMode associationMode{
      ManualObjectDranAssociationMode::IdentityOracle};
  ManualObjectDranAssociationPolicy associationPolicy{
      ManualObjectDranAssociationPolicy::Manual};
  ManualObjectDranFinalAblation associationFinalAblation{
      ManualObjectDranFinalAblation::None};
  ManualObjectDranDiagnosticProfile diagnosticProfile{
      ManualObjectDranDiagnosticProfile::Auto};
  std::string associationFile;
  std::size_t associationUpdatePeriod{1};
  bool associationFreezeUpperVariables{false};
  bool associationLocalCavityFeedbackSweepEnabled{false};
  bool associationComponentResponseGateEnabled{false};
  bool associationNetworkPartitionGateEnabled{false};
  bool associationNetworkPartitionMulticolorFeedbackEnabled{false};
  bool associationPairCavityDiagnosticEnabled{false};
  bool associationCavityExportLeavePairStates{false};
  std::size_t associationPairCavityExcludeFirstRobot{
      std::numeric_limits<std::size_t>::max()};
  std::size_t associationPairCavityExcludeSecondRobot{
      std::numeric_limits<std::size_t>::max()};
  std::size_t associationSinkhornIterations{20};
  std::size_t associationMaxCandidatesPerEdge{0};
  double associationTemperature{1.0};
  double associationRowCapacity{1.0};
  double associationColumnCapacity{1.0};
  double associationGeometryScale{1.0};
  std::size_t associationSinkhornMaxNonzeros{0};
  double associationUnmatchedCost{std::numeric_limits<double>::infinity()};
  bool associationUseSyntheticMetadata{false};
  std::size_t associationMetadataNumCategories{3};
  std::size_t associationMetadataDescriptorDim{4};
  double associationMetadataSemanticAccuracy{1.0};
  double associationMetadataDescriptorNoise{0.0};
  double associationMetadataShapeNoise{0.0};
  double associationMetadataRepeatedObjectRatio{0.0};
  double associationSemanticWeight{0.0};
  double associationShapeWeight{0.0};
  double associationDescriptorWeight{0.0};
  bool associationRequireObservedObjects{false};
  bool associationUseQuotientMetric{false};
  std::size_t associationTransformMinInliers{3};
  double associationTransformMaxResidual{1.0};
  double associationTransformMinScore{0.0};
  double associationTransformWeightScale{1.0};
  std::size_t associationTransformMaxSeedCandidates{80};
  unsigned associationTransformDimension{0};
  std::string associationTransformPlaneAxes{"xy"};
  bool associationTransformUpdateInLowerFixedPoint{false};
  bool associationTransformLowerFixedPointUseReliability{false};
  bool associationTransformLowerFixedPointUseReliabilityTrust{false};
  bool associationTransformUseProjectedWeights{false};
  bool associationTransformUseInitialObjectPoses{false};
  bool associationGaugeSynchronizationEnabled{false};
  bool associationGaugeSynchronizationApplyToStates{false};
  bool associationGaugeSynchronizationApplyOnceToInitialStates{false};
  bool associationGaugeSynchronizationAsynchronousGaussSeidel{false};
  std::size_t associationGaugeSynchronizationMaxRoundsPerPhase{0};
  double associationGaugeSynchronizationTolerance{0.0};
  bool associationGaugeCovariantPreserveLocalStates{false};
  bool associationCycleConsistencyEnabled{false};
  double associationCycleConsistencyMinDirectWeight{1e-12};
  double associationCycleConsistencyMinComposedWeight{1e-12};
  std::size_t associationCycleConsistencyMinTwoHopSupport{1};
  bool associationCycleConsistencyKeepWithoutEvidence{true};
  double associationCycleObjectiveWeight{0.0};
  double associationCycleObjectiveMinEdgeMass{1e-12};
  bool associationAdaptiveUpperEnabled{false};
  double associationAdaptiveUpperInitialStep{1.0};
  double associationAdaptiveUpperEpsilon{1e-12};
  double associationAdaptiveUpperMaxStep{1.0};
  bool associationUpperTrackingEnabled{false};
  double associationUpperTrackingConsensusWeight{0.0};
  double associationUpperTrackingCorrectionWeight{1.0};
  bool associationUpperTrackingCenterRows{true};
  bool associationUpperTrackingCenterColumns{true};
  bool associationUpperTrackingProjectAgainstLowerFeedback{false};
  bool associationUpperTrackingUseLowerStationarityGate{false};
  double associationUpperTrackingLowerValidationMeritTrustScale{0.0};
  double associationStateValidationHypergradientWeight{0.0};
  double associationStateValidationMeritTrustScale{0.0};
  unsigned associationStateValidationMeritLowerIterations{1};
  bool associationStateValidationMeritUseMeasurementCost{false};
  bool associationStateValidationMeritGateFullUpdate{false};
  bool associationStateValidationUseEdgeTransformConfidence{false};
  double associationStateValidationTransformConfidenceDistanceScale{2.0};
  bool associationStateValidationTrajectoryGuardEnabled{true};
  bool associationLowerValidationSelectBestMeasurementState{false};
  bool associationLowerValidationMeasurementIncumbentProx{false};
  bool associationPairValidationTrustEnabled{false};
  double associationEdgeLatentConsensusWeight{0.0};
  bool associationEdgeLatentConsensusUseTransformConfidence{false};
  double associationEdgeLatentConsensusConfidenceDistanceScale{2.0};
  bool associationSharedEdgeVariableEnabled{false};
  double associationReciprocalFreshSupportBlend{0.0};
  double associationReciprocalFreshSupportUniquenessScale{0.0};
  bool associationReciprocalFreshSupportUseTransformConfidence{false};
  double associationReciprocalFreshSupportConfidenceDistanceScale{2.0};
  double associationPairSupportGatePower{0.0};
  double associationPairSupportGateMinEffectiveSupport{0.0};
  bool associationPairSupportLatentContinuation{false};
  double associationObjectReliabilityResidualScale{0.0};
  double associationObjectReliabilityMin{0.0};
  double associationObjectReliabilityUpperCostWeight{0.0};
  bool associationObjectReliabilityUseSharedState{false};
  double associationObjectObservationRobustResidualScale{0.0};
  double associationObjectObservationRobustMinInformationScale{0.0};
  bool associationObjectObservationRobustNormalizeInformation{false};
  double associationObjectObservationUncertaintyCountPrior{0.0};
  double associationObjectObservationUncertaintyDistanceScale{0.0};
  double associationObjectObservationUncertaintyMinReliability{0.0};
  bool associationObjectObservationUncertaintyNormalizeInformation{false};
  bool associationObjectObservationTrackEmpiricalInformation{false};
  double associationTemporalKlWeight{0.0};
  double associationTemporalKlEpsilon{1e-12};
  double associationEntropyRegularizationWeight{0.0};
  double associationBackendFeedbackGradientScale{0.0};
  double associationBackendFeedbackMinTrust{0.0};
  double associationBackendResponseGain{0.0};
  double associationBackendResponseDamping{1.0};
  double associationBackendResponseMaxReductionFraction{0.5};
  double associationBackendAdjointGain{0.0};
  double associationBackendAdjointDamping{1.0};
  double associationBackendAdjointMaxReductionFraction{0.5};
  double associationBackendAdjointMaxIncreaseFraction{0.5};
  double associationBackendCoupledAdjointGain{0.0};
  double associationBackendCoupledAdjointDamping{1.0};
  double associationBackendCoupledAdjointMaxReductionFraction{0.5};
  double associationBackendCoupledAdjointMaxIncreaseFraction{0.5};
  double associationBackendAdjointReductionMetadataMargin{-1.0};
  bool associationBackendAdjointAllowCostReduction{true};
  bool associationBackendScaleNegativeDeltaByTransformConfidence{false};
  double associationSignedFreshSupportBlend{0.0};
  double associationSignedFreshSupportSigmoidScale{1.0};
  double associationSignedFreshSupportStateValidationMeritTrustScale{0.0};
  std::size_t associationBackendSnapshotPeriod{1};
  double associationBackendMaxEffectiveRatio{0.0};
  bool associationBackendNormalizeActiveRms{false};
  double associationBackendTargetRms{1.0};
  double associationBackendTargetRmsUnmatchedFraction{0.0};
  bool associationBackendCapPositiveDeltaAtUnmatched{false};
  bool associationBackendCapPositiveDeltaUseTransformConfidence{false};
  bool associationBackendCenterRows{true};
  bool associationBackendCenterColumns{true};
  double associationBackendStationarityTrustMaxGradientNorm{0.0};
  std::size_t associationBackendMaxActiveFactorsPerEdge{0};
  double associationTemporalBlend{0.0};
  bool associationRenormalizeSparseWeights{false};
  bool associationResidualWarmStartUseGeometry{false};
  bool associationResidualUseTransformAlignment{false};
  bool associationResidualTransformUpdateFromWeights{false};
  bool associationLowerObjectCopyUseTransformAlignment{true};
  bool associationLowerObjectCopyDirectEnabled{true};
  bool associationLowerObjectCopyUseObservationImpliedTarget{false};
  bool associationLowerTrajectoryObjectCopyFeedback{false};
  bool associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust{false};
  bool associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass{false};
  bool associationLowerTrajectoryObjectCopyFeedbackMassConserving{false};
  bool associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor{false};
  bool associationLowerSharedLatentFusion{false};
  ManualObjectDranSharedLatentMode associationLowerSharedLatentMode{
      ManualObjectDranSharedLatentMode::Auto};
  bool exactPoint2OwnerOptimizeTrajectories{true};
  std::size_t exactPoint2OwnerPcgIterations{1};
  bool exactPoint2OwnerPcgWarmStart{true};
  double exactPoint2OwnerPcgWarmStartScale{1.0};
  std::size_t exactPoint2OwnerPcgLateWarmStartRound{0};
  double exactPoint2OwnerPcgLateWarmStartScale{1.0};
  bool exactPoint2OwnerPcgMatrixFree{false};
  bool exactPoint2OwnerPcgQuadraticWarmStart{false};
  double exactPoint2OwnerPcgPreconditionerRefreshResidual{0.0};
  std::size_t exactPoint2OwnerPcgPreconditionerRefreshMinRounds{5};
  bool coupledPoint2SchurOracleCorrection{false};
  bool coupledPoint2SchurMatrixFreeOwnerPcg{false};
  std::size_t coupledPoint2SchurOracleStartRound{3};
  std::size_t coupledPoint2SchurOraclePeriod{1};
  double coupledPoint2SchurOracleRelativeMeritTolerance{1e-8};
  bool integratedPoint2Mm{false};
  bool integratedPoint2LocalAmm{false};
  bool integratedPoint2CommonAcceptance{true};
  double integratedPoint2AmmMomentum{0.10};
  bool integratedPoint2AdaptiveMomentum{false};
  double integratedPoint2AdaptiveMomentumMin{0.0};
  double integratedPoint2AdaptiveMomentumMax{0.30};
  double integratedPoint2AdaptiveMomentumEmaDecay{0.5};
  bool integratedPoint2ObjectWiseAdaptiveMomentum{false};
  std::size_t exactPoint2GaugeSynchronizationRounds{0};
  double associationLowerObjectCopyRobustResidualScale{0.0};
  double associationLowerObjectCopyRobustMinWeight{0.0};
  bool associationLowerObjectCopyObservationPredictiveGate{false};
  bool associationLowerObjectCopyNormalizeBySupport{false};
  bool associationLowerObjectCopySupportChurnGate{false};
  double associationLowerObjectCopySupportNormalizationScale{1.0};
  double associationLowerTrajectoryTranslationProxWeight{0.0};
  bool associationResidualTransformBootstrap{false};
  bool associationResidualTrustPreviousSupport{false};
  bool associationResidualAllowSupportExpansion{false};
  bool associationSparseUpperSupport{false};
  double associationLowerProjectionPower{1.0};
  bool associationLowerProjectionHard{false};
  unsigned associationTwoTimeScaleLowerInnerRounds{1};
  unsigned associationTwoTimeScaleLowerInnerIterations{0};
  bool associationTwoTimeScaleLowerInnerExchange{true};
  bool associationTwoTimeScaleFreezeUpperInLowerInner{true};
  bool associationLowerPhysicalCopyMajorization{false};
  bool associationLowerPhysicalCopyColoredBcd{false};
  unsigned associationTerminalLowerRounds{0};
  std::size_t associationLowerMessageDelayRounds{0};
  bool associationLowerDropFirstAttempt{false};
  bool associationLowerDirectEdgeSynchronousSharedOnly{false};
  bool associationLowerDirectEdgePreserveInactiveDuals{true};
  bool associationLowerDirectEdgeFixedUnionModel{false};
  bool associationLowerDirectEdgeExactProximalCurvature{false};
  double associationLowerDirectEdgeProximalCurvatureScale{1.0};
  double associationLowerDirectEdgeDualStepSize{1.0};
  double associationLowerDirectEdgeDualRelaxationResidualThreshold{0.0};
  bool associationLowerDirectEdgeSpectralDualStep{false};
  bool associationLowerDirectEdgeGlobalSpectralDualStep{false};
  double associationLowerDirectEdgeSpectralDualStepMin{0.75};
  double associationLowerDirectEdgeSpectralDualStepMax{1.5};
  double associationLowerDirectEdgeSpectralDualStepEmaDecay{0.5};
  bool associationLowerDirectEdgePoint2ConsensusProjection{false};
  double associationLowerDirectEdgePoint2ConsensusProjectionRelaxation{1.0};
  std::size_t associationLowerDirectEdgePoint2ConsensusProjectionRounds{0};
  bool associationLowerDirectEdgePoint2ConsensusResetDualAfterWarmup{false};
  bool associationLowerDirectEdgeAdaptiveBeta{false};
  double associationLowerDirectEdgeBetaMin{0.05};
  double associationLowerDirectEdgeBetaMax{0.5};
  double associationLowerDirectEdgeBetaImbalanceRatio{5.0};
  double associationLowerDirectEdgeBetaUpdateFactor{1.25};
  double associationLowerDirectEdgeBetaEmaDecay{0.8};
  std::size_t associationLowerDirectEdgeBetaUpdatePeriod{5};
  double associationLowerHighConfidenceAnchorGain{0.0};
  double associationLowerContinuousSupportGatePower{0.0};
  bool associationLowerUseCurrentResidualReliability{false};
  bool associationLowerUseBoundedCurrentResidualReliability{false};
  bool associationLowerUseSupportConfidenceReliability{false};
  double associationReferenceTrustWeight{0.0};
  bool associationReferenceTrustScaleWithUpperCost{false};
  double associationReferenceTrustMinimumCostRmsScale{1.0};
  double associationReferenceMassTrustWeight{0.0};
  bool associationReferenceMassTrustScaleWithUpperCost{false};
  double associationReferenceMassTrustMinimumCostRmsScale{1.0};
  double associationRelationalGeometryWeight{0.0};
  double associationRelationalGeometryDistanceScale{0.0};
  bool associationRelationalGeometryNormalizeBySquaredMass{false};
  bool associationRelationalGeometryCenterRows{false};
  bool associationRelationalNullEvidence{false};
  AssociationRelationalNullEvidenceOptions
      associationRelationalNullEvidenceOptions{};
  double associationRotoTranslationAlignmentWeight{0.0};
  double associationRotoTranslationAlignmentDistanceScale{0.0};
  std::size_t associationRotoTranslationHypothesisTopK{0};
  std::size_t associationRotoTranslationHypothesisAngleBins{36};
  double associationRotoTranslationHypothesisSoftminTemperature{1.0};
  bool associationSaveIterationEstimateSnapshots{false};
  std::size_t diagnosticSummaryPeriod{1};
  bool excludeSummaryDiagnosticsFromSolverTime{false};
  bool recordObjectInterfaceDiagnostics{true};
  bool recordDetailedAssociationDiagnostics{true};
  bool associationRecordUpperResponseDiagnostics{false};
  bool recordPredictiveMeasurementCost{true};
  double associationLowerValidationWeight{0.0};
  bool associationLowerValidationCenterRows{true};
  bool associationLowerValidationCenterColumns{true};
  bool associationLowerValidationNormalizeActiveRms{true};
  double associationLowerValidationTargetRms{1.0};
  bool associationLowerValidationScaleWithUpperCost{true};
  double associationLowerValidationMinimumCostRmsScale{1.0};
  double associationLowerValidationStationarityTrustMaxGradientNorm{0.0};
  double associationLowerValidationMeritTrustScale{0.0};
  double associationLowerValidationPairwiseWeight{0.0};
  double associationLowerValidationPairwiseDistanceScale{0.0};
  bool associationLowerValidationPairwiseOnly{false};
  double associationLowerValidationReferenceTrustWeight{0.0};
  double associationValidationRiskWeight{0.0};
  double associationValidationRiskTargetRms{1.0};
  double associationValidationRiskTargetRmsUnmatchedFraction{0.0};
  double associationImplicitValidationAdjointWeight{0.0};
  double associationImplicitValidationAdjointDamping{1.0};
  double associationImplicitValidationAdjointMaxEffectiveRatio{0.0};
  double associationImplicitValidationAdjointTargetRms{1.0};
  double associationImplicitValidationAdjointTargetRmsUnmatchedFraction{0.0};
  bool associationImplicitValidationAdjointUseCoupledSolve{false};
  std::size_t associationImplicitValidationAdjointMaxActiveObjects{0};
  double associationImplicitValidationAdjointObservationRhsWeight{0.0};
  bool associationImplicitValidationAdjointProjectAgainstBackend{false};
  bool associationImplicitValidationAdjointProjectAgainstUpperPrefix{false};
  double associationImplicitValidationAdjointMeritTrustScale{0.0};
  double associationCoveragePriorWeight{0.0};
  bool associationCoveragePriorMatchRowColumn{false};
  bool associationCoveragePriorNormalizeByReferenceMass{false};
  double associationSupportSharpnessWeight{0.0};
  double associationSupportSharpnessTargetRatio{0.0};
  double associationSupportSharpnessValidationBlend{0.0};
  double associationSupportSharpnessCoverageBlend{0.0};
  bool associationSupportSharpnessProjectAgainstLowerFeedback{false};
  double associationStructuralExpansionWeight{0.0};
  double associationStructuralExpansionDistanceScale{0.0};
  bool associationStructuralExpansionUseProposalGate{true};
  double associationReferenceSupportSharpeningWeight{0.0};
  double associationReferenceSupportSharpeningTarget{1.0};
  bool associationPostUpdateLocalRefine{false};
  unsigned associationPostUpdateLocalRefineIterations{1};
  bool associationPostUpdateLocalRefineExchange{true};
  unsigned associationPostUpdateLowerFixedPointRounds{1};
  bool associationNormalizeUpperCosts{false};
  double associationUpperCostNormalizationTargetRms{1.0};
  double associationMinConsensusWeight{0.0};
  double associationMinConsensusMargin{0.0};
  double associationConsensusWeightPower{1.0};
  std::size_t associationActiveMinStableUpdates{0};
  std::size_t associationActivationTrustRampUpdates{0};
  double associationTentativeWeightScale{0.0};
  double associationRejectResidualThreshold{0.0};
  double associationRejectResidualBestMargin{0.0};
  double associationRejectResidualQuantile{0.0};
  double associationRejectResidualQuantileScale{1.0};
  std::size_t associationRejectMinBadUpdates{0};
  std::size_t objectInterfaceResponsePeriod{1};
  double objectInterfaceMinPredictedDecrease{0.0};
  double objectInterfaceMinInnovationScore{0.0};
  double objectInterfaceSchurDamping{1e-2};
  std::size_t objectInterfacePcgIterations{1};
  double objectInterfacePcgRelativeTolerance{1e-8};
  bool objectInterfacePcgWarmStart{false};
  ManualObjectDranObjectInterfacePcgPreconditioner
      objectInterfacePcgPreconditioner{
          ManualObjectDranObjectInterfacePcgPreconditioner::
              PrivateBlockJacobi};
  ManualObjectDranObjectInterfacePcgPrivateSolver
      objectInterfacePcgPrivateSolver{
          ManualObjectDranObjectInterfacePcgPrivateSolver::SparseLdlt};
  unsigned objectInterfacePcgPostActivationLocalIterations{1};
  unsigned objectInterfacePcgTerminalLocalPolishIterations{0};
  bool objectInterfacePcgReplaceFinalRoundWithTerminalPolish{false};
  bool objectInterfacePcgPersistentFeasibleResponse{false};
  double objectInterfacePcgMinTranslationSpread{0.06};
  double objectInterfacePcgMinRotationSpreadDegrees{1.0};
  double objectInterfacePcgMinInitialRelativeDecrease{1e-2};
  double objectInterfacePcgMinContinuationRelativeDecrease{5e-4};
  double objectInterfaceStepGain{1.0};
  std::size_t objectInterfaceMaxObjectsPerRobot{0};
  unsigned objectInterfaceMaxPrivateCols{64};
  double objectInterfaceMaxBlockStepNorm{0.25};
  bool objectInterfaceRequireDecrease{true};
  double objectPoseTolerance{1e-3};
  std::size_t objectMaxAge{5};
  std::size_t associationHeldoutObjectObservationModulo{0};
  std::size_t associationHeldoutObjectObservationRemainder{0};
  double translationProxWeight{0.0};
  bool projectToSEAfterLocalSolve{true};
  bool coupledTranslationTrustRegion{true};
  double translationEliminationProxWeight{0.0};
  ManualDpgoMmLocalSolver localSolver{
      ManualDpgoMmLocalSolver::ReducedRotation};
  unsigned fullSolverOuterIterations{0};
  unsigned localTrustRegionIterations{1};
  unsigned localInitialTrustRegionIterations{0};
  unsigned localInitialTrustRegionRounds{0};
  unsigned localPeriodicTrustRegionIterations{0};
  unsigned localPeriodicTrustRegionPeriod{0};
  unsigned localFinalTrustRegionIterations{0};
  unsigned localFinalTrustRegionRounds{0};
  int localTrustRegionMaxInnerIterations{10};
  double localTrustRegionTolerance{1e-3};
  double localTrustRegionInitialRadius{10.0};
  bool profileReducedRotationRuntime{false};
  bool reducedRotationGradientBoundaryCandidate{true};
  ManualDpgoMmReducedRotationPreconditioner reducedRotationPreconditioner{
      ManualDpgoMmReducedRotationPreconditioner::None};
  std::string eventScheduleFile;
};

std::vector<ManualObjectDranEvent> loadManualObjectDranEventScheduleForTest(
    const std::string &path, std::size_t numRobots,
    std::size_t expectedEvents);

bool manualObjectDranLocalSolveNeedsProblemPreconditioner(
    const ManualObjectDranOptions &options, unsigned outerIteration);

unsigned manualObjectDranLocalIterationsForOuterRound(
    const ManualObjectDranOptions &options, unsigned outerIteration);

std::vector<std::size_t> colorManualObjectDranRobotGraph(
    const std::vector<std::vector<std::size_t>> &adjacency);

struct ManualObjectDranIterationSummary {
  unsigned iter{0};
  std::size_t eventRobot{std::numeric_limits<std::size_t>::max()};
  double eventLogicalTime{0.0};
  double equivalentSweep{0.0};
  bool queueDrain{false};
  double time{0.0};
  double profileRunSetupSeconds{0.0};
  double profileDataLoadSeconds{0.0};
  double profileGraphInitializationSeconds{0.0};
  double profileInitialAssociationSeconds{0.0};
  double profileInitialAssociationBootstrapSeconds{0.0};
  double profileInitialAssociationStateWindowSeconds{0.0};
  double profileInitialGaugeReliabilitySeconds{0.0};
  double profileLowerInnerRoundsSeconds{0.0};
  double profileLowerFactorAssemblySeconds{0.0};
  double profileLowerSolveSeconds{0.0};
  double profileReducedTranslationFactorizationSeconds{0.0};
  double profileReducedTranslationRecoverySeconds{0.0};
  double profileReducedHessianProductSeconds{0.0};
  double profileReducedObjectiveSeconds{0.0};
  double profileReducedGradientBuildSeconds{0.0};
  std::size_t profileReducedTranslationFactorizationCount{0};
  std::size_t profileReducedTranslationRecoveryCount{0};
  std::size_t profileReducedHessianProductCount{0};
  double profileLowerFeedbackCertificateSeconds{0.0};
  double profileLowerConsensusRefreshSeconds{0.0};
  double profileAssociationUpdateSeconds{0.0};
  double profileAssociationUnprofiledSeconds{0.0};
  double profileAssociationWeightUpdateSeconds{0.0};
  double profileAssociationWeightUpdateUnprofiledSeconds{0.0};
  double profileAssociationLocalCostSeconds{0.0};
  double profileAssociationStateValidationPrecomputeSeconds{0.0};
  double profileAssociationStateValidationCostSeconds{0.0};
  double profileAssociationEdgeLatentConsensusSeconds{0.0};
  double profileAssociationReferenceMassTrustSeconds{0.0};
  double profileAssociationTemporalStatsSeconds{0.0};
  double profileAssociationSupportStatsSeconds{0.0};
  double profileAssociationTransformRefreshSeconds{0.0};
  double profileAssociationCycleConsistencySeconds{0.0};
  double profileAssociationStateWindowSeconds{0.0};
  double profileObjectInterfaceResponseSeconds{0.0};
  double profilePairValidationSeconds{0.0};
  double profileAssociationRotoTranslationSeconds{0.0};
  double profileAssociationProjectionSeconds{0.0};
  double profileAssociationSupportPostprocessSeconds{0.0};
  double profileAssociationSharedObjectiveSeconds{0.0};
  double profileAssociationSharedResponseSeconds{0.0};
  std::size_t profileAssociationSupportEntries{0};
  std::size_t profileAssociationDenseEntries{0};
  double profileAssociationSupportDensity{0.0};
  double profileAssociationSupportMassCoverage{0.0};
  double profileAssociationPreCostSupportSeconds{0.0};
  std::size_t profileAssociationPreCostSupportEntries{0};
  std::size_t profileAssociationPreCostDenseEntries{0};
  double profileAssociationPreCostSupportDensity{0.0};
  double profileAssociationPreCostSupportMassCoverage{0.0};
  double profileObjectExchangeSeconds{0.0};
  double profileGaugeReliabilityUpdateSeconds{0.0};
  double profilePostUpdateRefineSeconds{0.0};
  std::size_t postUpdateLowerRoundsExecuted{0};
  std::size_t postUpdateLowerRobotSolves{0};
  std::size_t postUpdateLowerConvergedRobots{0};
  double postUpdateLowerMaxNormalizedResidual{0.0};
  double postUpdateLowerMaxStationarityResidual{0.0};
  double postUpdateLowerMaxOuterContinuationResidual{0.0};
  double postUpdateLowerMaxIncidentPrimalResidual{0.0};
  double postUpdateLowerMaxIncidentDualResidual{0.0};
  double profileSummaryFactorAssemblySeconds{0.0};
  double profileSummaryAssociationDiagnosticsSeconds{0.0};
  double measurementCost{0.0};
  double heldoutMeasurementCost{0.0};
  double predictiveMeasurementCost{0.0};
  double consensusCost{0.0};
  double gradient{0.0};
  std::size_t knownObjectCopies{0};
  std::size_t relayObjectCopies{0};
  std::size_t observedObjectCopies{0};
  std::size_t relayOnlyObjectCopies{0};
  std::size_t initializedFromNeighborObjectCopies{0};
  std::size_t staleObjectCopies{0};
  std::size_t staleObjectCommTriggers{0};
  std::size_t activeConsensusPairs{0};
  std::size_t associationCandidateCount{0};
  double associationWeightMass{0.0};
  double associationCorrectMass{0.0};
  double associationFalseMergeMass{0.0};
  double associationEntropy{0.0};
  double associationPrecision{0.0};
  double associationRecall{0.0};
  double associationF1{0.0};
  double associationFalseMergeRate{0.0};
  double associationFactorWeightMass{0.0};
  double associationFactorCorrectMass{0.0};
  double associationFactorFalseMergeMass{0.0};
  std::size_t associationPairValidationTrustActivePairs{0};
  double associationPairValidationTrustMean{1.0};
  double associationPairValidationTrustMin{1.0};
  double associationPairValidationRiskRms{0.0};
  std::size_t highConfidenceAssociationCandidateCount{0};
  double highConfidenceAssociationWeightMass{0.0};
  double highConfidenceAssociationCorrectMass{0.0};
  double highConfidenceAssociationFalseMergeMass{0.0};
  double highConfidenceAssociationPrecision{0.0};
  double highConfidenceAssociationRecall{0.0};
  double highConfidenceAssociationF1{0.0};
  double highConfidenceAssociationFalseMergeRate{0.0};
  double highConfidenceAssociationFactorWeightMass{0.0};
  double highConfidenceAssociationFactorCorrectMass{0.0};
  double highConfidenceAssociationFactorFalseMergeMass{0.0};
  std::size_t activatedAssociationCandidateCount{0};
  double activatedAssociationWeightMass{0.0};
  double activatedAssociationCorrectMass{0.0};
  double activatedAssociationFalseMergeMass{0.0};
  double activatedAssociationPrecision{0.0};
  double activatedAssociationRecall{0.0};
  double activatedAssociationF1{0.0};
  double activatedAssociationFalseMergeRate{0.0};
  double activatedAssociationFactorWeightMass{0.0};
  double activatedAssociationFactorCorrectMass{0.0};
  double activatedAssociationFactorFalseMergeMass{0.0};
  std::size_t activeAssociationCandidateCount{0};
  double activeAssociationWeightMass{0.0};
  double activeAssociationCorrectMass{0.0};
  double activeAssociationFalseMergeMass{0.0};
  double activeAssociationPrecision{0.0};
  double activeAssociationRecall{0.0};
  double activeAssociationF1{0.0};
  double activeAssociationFalseMergeRate{0.0};
  double activeAssociationFactorWeightMass{0.0};
  double activeAssociationFactorCorrectMass{0.0};
  double activeAssociationFactorFalseMergeMass{0.0};
  std::size_t associationRejectedCandidateCount{0};
  double associationRejectedWeightMass{0.0};
  std::size_t associationCycleCheckedEntries{0};
  std::size_t associationCycleKeptEntries{0};
  std::size_t associationCyclePrunedEntries{0};
  double associationLocalGradientNorm{0.0};
  double associationCycleGradientNorm{0.0};
  double associationCycleToLocalGradientRatio{0.0};
  double associationCycleObjectiveCost{0.0};
  std::size_t associationCycleObjectiveActiveTriplets{0};
  double associationRelationalGeometryGradientNorm{0.0};
  double associationRelationalGeometryToLocalGradientRatio{0.0};
  double associationRelationalGeometryObjectiveCost{0.0};
  double associationActualRelationalGeometryAlpha{0.0};
  double associationRotoTranslationAlignmentGradientNorm{0.0};
  double associationRotoTranslationAlignmentToLocalGradientRatio{0.0};
  double associationRotoTranslationAlignmentObjectiveCost{0.0};
  double associationActualRotoTranslationAlignmentAlpha{0.0};
  double associationLowerValidationGradientNorm{0.0};
  double associationLowerValidationToLocalGradientRatio{0.0};
  double associationLowerValidationObjectiveCost{0.0};
  double associationActualLowerValidationAlpha{0.0};
  double associationLowerValidationMeritGate{1.0};
  double associationLowerValidationMeritDecrease{0.0};
  double associationLowerValidationPairwiseGradientNorm{0.0};
  double associationLowerValidationPairwiseObjectiveCost{0.0};
  double associationValidationRiskGradientNorm{0.0};
  double associationValidationRiskToLocalGradientRatio{0.0};
  double associationValidationRiskObjectiveCost{0.0};
  double associationActualValidationRiskAlpha{0.0};
  double associationImplicitValidationAdjointGradientNorm{0.0};
  double associationImplicitValidationAdjointToLocalGradientRatio{0.0};
  double associationImplicitValidationAdjointObjectiveCost{0.0};
  double associationImplicitValidationAdjointRhsGradientNorm{0.0};
  double associationActualImplicitValidationAdjointAlpha{0.0};
  std::size_t associationImplicitValidationAdjointActiveObjectCount{0};
  double associationImplicitValidationAdjointMeritGate{1.0};
  double associationImplicitValidationAdjointMeritDecrease{0.0};
  double associationImplicitValidationAdjointBackendAlignment{0.0};
  double associationImplicitValidationAdjointUpperPrefixAlignment{0.0};
  double associationImplicitValidationAdjointProjectedConflictNorm{0.0};
  double associationImplicitValidationAdjointUpperPrefixProjectedConflictNorm{
      0.0};
  double associationCoveragePriorGradientNorm{0.0};
  double associationCoveragePriorToLocalGradientRatio{0.0};
  double associationCoveragePriorObjectiveCost{0.0};
  double associationActualCoveragePriorAlpha{0.0};
  double associationSupportSharpnessGradientNorm{0.0};
  double associationSupportSharpnessToLocalGradientRatio{0.0};
  double associationSupportSharpnessObjectiveCost{0.0};
  double associationActualSupportSharpnessAlpha{0.0};
  double associationSupportSharpnessValidationGateMean{1.0};
  double associationSupportSharpnessCoverageGateMean{1.0};
  double associationSupportSharpnessLowerFeedbackAlignment{0.0};
  double associationSupportSharpnessProjectedConflictNorm{0.0};
  double associationStructuralExpansionGradientNorm{0.0};
  double associationStructuralExpansionToLocalGradientRatio{0.0};
  double associationStructuralExpansionObjectiveCost{0.0};
  double associationActualStructuralExpansionAlpha{0.0};
  double associationReferenceSupportSharpeningGradientNorm{0.0};
  double associationReferenceSupportSharpeningToLocalGradientRatio{0.0};
  double associationReferenceSupportSharpeningObjectiveCost{0.0};
  double associationActualReferenceSupportSharpeningAlpha{0.0};
  double associationReferenceTrustGradientNorm{0.0};
  double associationReferenceTrustToLocalGradientRatio{0.0};
  double associationReferenceTrustObjectiveCost{0.0};
  double associationActualReferenceTrustAlpha{0.0};
  double associationBackendGradientNorm{0.0};
  double associationBackendRawGradientNorm{0.0};
  double associationBackendToLocalGradientRatio{0.0};
  double associationBackendRawToLocalGradientRatio{0.0};
  double associationBackendCapScale{1.0};
  double associationBackendStationarityGradientNorm{0.0};
  std::size_t associationBackendSnapshotActive{1};
  std::size_t associationBackendStationarityTrusted{1};
  double associationUpperGradientNorm{0.0};
  double associationAccumulatedUpperGradientNorm{0.0};
  double associationAdaptiveEta{1.0};
  double associationUpperTrackingGradientNorm{0.0};
  double associationUpperTrackingCorrectionNorm{0.0};
  double associationUpperTrackingConsensusResidualNorm{0.0};
  double associationUpperTrackingLocalGradientDeltaNorm{0.0};
  double associationUpperTrackingLowerValidationMeritGate{1.0};
  double associationUpperTrackingLowerValidationMeritDecrease{0.0};
  double associationStateValidationHypergradientGradientNorm{0.0};
  double associationStateValidationHypergradientObjectiveCost{0.0};
  double associationStateValidationUnifiedObjectiveCost{0.0};
  double associationStateValidationMeritGate{1.0};
  double associationStateValidationMeritRatio{0.0};
  double associationStateValidationMeritDecrease{0.0};
  double associationStateValidationTrajectoryResidualCost{0.0};
  double associationStateValidationPairwiseTrajectoryCost{0.0};
  double associationStateValidationTransformConfidenceCost{0.0};
  double associationStateValidationTransformConfidenceMean{1.0};
  double associationStateValidationObjectConsistencyCost{0.0};
  double associationStateValidationGaugeStationarityCost{0.0};
  double associationEdgeLatentConsensusGradientNorm{0.0};
  double associationEdgeLatentConsensusObjectiveCost{0.0};
  double associationEdgeLatentConsensusResidualNorm{0.0};
  double associationEdgeLatentConsensusConfidenceMean{1.0};
  std::size_t associationSharedEdgeSynchronizedEdgeCount{0};
  double associationSharedEdgePreProjectionDisagreementNorm{0.0};
  double associationSharedEdgeProjectionDisplacementNorm{0.0};
  std::size_t associationSharedEdgeCandidateCount{0};
  std::size_t associationSharedEdgeObjectiveScalarCount{0};
  std::size_t associationSharedEdgePayloadBytes{0};
  double associationSharedEdgePayloadMb{0.0};
  double associationSharedEdgeCumulativeMb{0.0};
  double associationSharedEdgeCumulativeTotalMb{0.0};
  double associationFreshUotMarginalResidualNorm{0.0};
  double associationFreshUotResponseMass{0.0};
  double associationUpperFixedPointResidualNorm{0.0};
  double associationLowerFeedbackReadinessMean{1.0};
  double associationMannStepMean{0.0};
  double associationActualCycleAlpha{0.0};
  double associationActualBackendAlpha{1.0};
  double associationTemporalKlCost{0.0};
  double associationTemporalKlGradientNorm{0.0};
  double associationEntropyRegularizationCost{0.0};
  double associationEntropyRegularizationGradientNorm{0.0};
  double associationChurnL1{0.0};
  double associationChurnFrobenius{0.0};
  double associationActualTemporalAlpha{0.0};
  double associationActualEntropyAlpha{0.0};
  double lowerStationarityGradientNorm{0.0};
  double lowerGaugeMarginalizedStationarityNorm{0.0};
  double lowerGaugeComponentNorm{0.0};
  double lowerGaugeComponentRatio{0.0};
  std::size_t lowerSharedObjectConsensusActiveHypotheses{0};
  double lowerSharedObjectConsensusPrimalResidual{0.0};
  double lowerSharedObjectConsensusDualResidual{0.0};
  double lowerSharedObjectConsensusNormalizedPrimalResidual{0.0};
  double lowerSharedObjectConsensusNormalizedDualResidual{0.0};
  std::size_t lowerSharedObjectConsensusAdaptiveBetaEdges{0};
  double lowerSharedObjectConsensusAdaptiveBetaMean{0.0};
  double lowerSharedObjectConsensusAdaptiveBetaMin{0.0};
  double lowerSharedObjectConsensusAdaptiveBetaMax{0.0};
  std::size_t lowerSharedObjectConsensusAdaptiveBetaIncreases{0};
  std::size_t lowerSharedObjectConsensusAdaptiveBetaDecreases{0};
  double lowerSharedObjectConsensusAdaptiveMidpointResidual{0.0};
  double lowerSharedObjectConsensusGlobalSpectralStep{0.0};
  std::size_t lowerSharedObjectConsensusGlobalSpectralContributions{0};
  double lowerSharedObjectConsensusGlobalSpectralNumerator{0.0};
  double lowerSharedObjectConsensusGlobalSpectralDenominator{0.0};
  std::size_t lowerSharedObjectConsensusMaxDualFirstRobot{0};
  std::size_t lowerSharedObjectConsensusMaxDualSecondRobot{0};
  std::size_t lowerSharedObjectConsensusMaxDualFirstObject{0};
  std::size_t lowerSharedObjectConsensusMaxDualSecondObject{0};
  bool lowerSharedObjectConsensusMaxDualTranslation{false};
  double lowerSharedObjectConsensusMaxDualRawResidual{0.0};
  double lowerSharedObjectConsensusMaxDualPenalty{0.0};
  double lowerSharedObjectConsensusMaxDualPhysicalMultiplierNorm{0.0};
  std::size_t lowerSharedObjectConsensusTargetMessages{0};
  std::size_t lowerSharedObjectConsensusDeliveredMessages{0};
  std::size_t lowerSharedObjectConsensusCertificateMessages{0};
  std::size_t lowerSharedObjectConsensusStaleCertificates{0};
  std::size_t lowerSharedObjectConsensusCurrentIncidentCertificates{0};
  std::size_t lowerSharedObjectConsensusExpectedIncidentCertificates{0};
  std::size_t lowerSharedObjectConsensusOutstandingMessages{0};
  std::size_t lowerSharedObjectConsensusDroppedAttempts{0};
  std::size_t lowerSharedObjectConsensusRetryAttempts{0};
  std::size_t lowerSharedObjectConsensusPayloadBytes{0};
  std::size_t lowerSharedObjectConsensusCertificatePayloadBytes{0};
  double lowerSharedObjectConsensusPayloadMb{0.0};
  double lowerSharedObjectConsensusCumulativeMb{0.0};
  std::size_t sharedEndpointStateQueuedMessages{0};
  std::size_t sharedEndpointStateDeliveredMessages{0};
  std::size_t sharedEndpointStateStaleMessages{0};
  std::size_t sharedEndpointStateOutstandingMessages{0};
  std::size_t sharedEndpointStatePayloadBytes{0};
  double cumulativeTotalCommMb{0.0};
  std::size_t associationBackendAdjointReductionCandidateCount{0};
  std::size_t associationBackendAdjointReductionAllowedCandidateCount{0};
  double associationBackendAdjointReductionWeightMass{0.0};
  double associationBackendAdjointReductionAllowedWeightMass{0.0};
  std::size_t commPoseCount{0};
  double iterCommMb{0.0};
  std::size_t cumulativeCommPoseCount{0};
  double cumulativeCommMb{0.0};
  std::size_t integratedPoint2AmmAttempts{0};
  std::size_t integratedPoint2AmmAccepts{0};
  std::size_t integratedPoint2AmmRestarts{0};
  std::size_t integratedPoint2EdgeInertiaAttempts{0};
  std::size_t integratedPoint2EdgeInertiaAccepts{0};
  std::size_t integratedPoint2EdgeInertiaRestarts{0};
  double integratedPoint2EdgeBaselineMerit{0.0};
  double integratedPoint2EdgeAcceleratedMerit{0.0};
  std::size_t integratedPoint2OwnerMomentumValidContributions{0};
  double integratedPoint2OwnerMomentumMean{0.0};
  double integratedPoint2OwnerMomentumMin{0.0};
  double integratedPoint2OwnerMomentumMax{0.0};
  double exactPoint2TrajectoryConditionalCostBefore{0.0};
  double exactPoint2TrajectoryConditionalCostAfter{0.0};
  std::size_t exactPoint2TrajectoryAcceptedSolves{0};
  std::size_t exactPoint2TrajectoryRejectedSolves{0};
  double exactPoint2TrajectorySolverSeconds{0.0};
  double exactPoint2UpdateSeconds{0.0};
  std::size_t exactPoint2StatisticMessages{0};
  std::size_t exactPoint2StatisticPayloadBytes{0};
  std::size_t exactPoint2BroadcastMessages{0};
  std::size_t exactPoint2BroadcastPayloadBytes{0};
  std::size_t exactPoint2IterationPayloadBytes{0};
  std::size_t exactPoint2CumulativePayloadBytes{0};
  double exactPoint2MaxCopySpread{0.0};
  double exactPoint2RmsCopySpread{0.0};
  double exactPoint2PhysicalCostAfterTrajectory{0.0};
  double exactPoint2PhysicalCostAfterUpdate{0.0};
  std::string exactPoint2Status{"inactive"};
  bool point2SchurOracleActive{false};
  double point2SchurOracleAssemblySeconds{0.0};
  std::size_t point2SchurOracleResponseSolves{0};
  double point2SchurOracleMaxResponseRelativeResidual{0.0};
  double point2SchurOracleSymmetryError{0.0};
  double point2SchurOracleSpectralShift{0.0};
  double point2SchurOracleLinearRelativeResidual{0.0};
  std::size_t point2SchurOracleBacktrackingTrials{0};
  double point2SchurOracleAcceptedStep{0.0};
  std::size_t point2SchurOracleHypotheticalDensePayloadBytes{0};
  bool point2OwnerPcgActive{false};
  std::size_t point2OwnerPcgIterations{0};
  double point2OwnerPcgRelativeResidual{0.0};
  std::size_t point2OwnerPcgGradientPayloadBytes{0};
  std::size_t point2OwnerPcgDirectionPayloadBytes{0};
  std::size_t point2OwnerPcgProductPayloadBytes{0};
  std::size_t point2OwnerPcgUpdatePayloadBytes{0};
  std::size_t point2OwnerPcgScalarPayloadBytes{0};
};

struct ManualObjectDranObjectInterfaceDiagnostic {
  unsigned iter{0};
  std::size_t robotId{0};
  std::size_t objectId{0};
  bool observed{false};
  bool relayOnly{false};
  std::size_t age{0};
  double residualNorm{0.0};
  double gradientNorm{0.0};
  double stiffness{0.0};
  double neighborDisagreement{0.0};
  bool selectedForSend{false};
};

struct ManualObjectDranObjectInterfaceResponseSummary {
  unsigned iter{0};
  bool scheduled{false};
  std::size_t activeRobots{0};
  std::size_t candidateObjectBlocks{0};
  std::size_t acceptedObjectBlocks{0};
  std::size_t rejectedObjectBlocks{0};
  std::size_t batchResponseRobots{0};
  std::size_t acceptedRobotBatches{0};
  std::size_t rejectedRobotBatches{0};
  std::size_t triggerCandidateRobots{0};
  std::size_t triggeredRobots{0};
  std::size_t skippedTriggerRobots{0};
  double modelCostBefore{0.0};
  double modelCostAfter{0.0};
  double interfaceInnovationScore{0.0};
  double accumulatedInnovationScore{0.0};
  double predictedModelDecrease{0.0};
  double acceptedModelDecrease{0.0};
  double stepNorm{0.0};
  std::size_t pcgIterations{0};
  double pcgRelativeResidual{0.0};
  std::size_t communicationPayloadBytes{0};
  std::size_t communicationMessages{0};
  double copyTranslationRms{0.0};
  double copyRotationRmsDegrees{0.0};
  double modelAssemblySeconds{0.0};
  double privateFactorizationSeconds{0.0};
  double privateGradientSolveSeconds{0.0};
  double schurApplySeconds{0.0};
  double privateRecoverySeconds{0.0};
  std::size_t blockTridiagonalFallbacks{0};
  std::size_t privateHessianBlocks{0};
  std::size_t privateInterfaceCouplingBlocks{0};
  std::size_t privateInterfaceDenseEquivalentBytes{0};
  std::size_t privateInterfaceStoredBytes{0};
  std::size_t sparseCompatibilityMaterializations{0};
  std::size_t sparseStructureFallbacks{0};
  std::string status;
};

struct ManualObjectDranAssociationMetricSummary {
  std::size_t candidateCount{0};
  double weightMass{0.0};
  double correctMass{0.0};
  double falseMergeMass{0.0};
  double recallDenominator{0.0};
  double precision{0.0};
  double recall{0.0};
  double f1{0.0};
  double falseMergeRate{0.0};
};

struct ManualObjectDranAssociationEntry {
  std::size_t row{0};
  std::size_t col{0};
  double weight{0.0};
};

struct ManualObjectDranAssociationCandidateEdge {
  std::size_t row{0};
  std::size_t col{0};
  double cost{0.0};
};

struct ManualObjectDranAssociationUpperResponseDiagnostic {
  std::size_t firstRobot{0};
  std::size_t secondRobot{0};
  std::size_t row{0};
  std::size_t col{0};
  double fixedCost{0.0};
  double currentCost{0.0};
  double feedbackCost{0.0};
  double rawScore{0.0};
  double responseWeight{0.0};
  double proposalWeight{0.0};
  double rowResponseMass{0.0};
  double columnResponseMass{0.0};
};

struct ManualObjectDranAssociationSupportStats {
  std::size_t supportEntries{0};
  std::size_t denseEntries{0};
  double assignmentMass{0.0};
  double coveredAssignmentMass{0.0};
  double density{0.0};
  double massCoverage{0.0};
};

struct ManualObjectDranPairCavityCovarianceDiagnostic {
  std::size_t robotId{0};
  bool valid{false};
  std::string status;
  std::vector<std::size_t> objectIds;
  Matrix translationCovariance;
  Matrix conditionalTranslationMeans;
  double conditionalTranslationResponseNorm{0.0};
};

struct ManualObjectDranLeavePairCavityDiagnostic {
  std::size_t firstRobot{0};
  std::size_t secondRobot{0};
  std::vector<ManualObjectDranPairCavityCovarianceDiagnostic> robots;
};

struct ManualObjectDranCavityFeedbackSummary {
  std::size_t firstRobot{0};
  std::size_t secondRobot{0};
  bool valid{false};
  std::string status;
  bool accepted{false};
  bool pairScoreAccepted{false};
  bool responseGateEvaluated{false};
  bool responseGateAccepted{false};
  double scoreBefore{0.0};
  double scoreAfterCandidate{0.0};
  double responsePredictedPhysicalDelta{0.0};
  double responseNumericalErrorBound{0.0};
  std::size_t activeCardinality{0};
  std::size_t changedEdgeCount{0};
  std::size_t componentCount{0};
  std::size_t candidateEvaluationCount{0};
  std::size_t payloadBytes{0};
  std::size_t responsePayloadBytes{0};
  double computeSeconds{0.0};
};

struct ManualObjectDranObjectCopyNode {
  std::size_t robot{0};
  std::size_t object{0};

  bool operator==(const ManualObjectDranObjectCopyNode &other) const;
  bool operator<(const ManualObjectDranObjectCopyNode &other) const;
};

struct ManualObjectDranObjectCopyEdge {
  ManualObjectDranObjectCopyNode first;
  ManualObjectDranObjectCopyNode second;

  bool operator==(const ManualObjectDranObjectCopyEdge &other) const;
  bool operator<(const ManualObjectDranObjectCopyEdge &other) const;
};

struct ManualObjectDranNetworkEditZone {
  std::size_t id{0};
  std::vector<ManualObjectDranObjectCopyNode> nodes;
  std::vector<ManualObjectDranObjectCopyEdge> removedEdges;
  std::vector<ManualObjectDranObjectCopyEdge> addedEdges;
};

struct ManualObjectDranDistributedEditZoneResult {
  bool complete{false};
  bool queueQuiescent{false};
  std::string status;
  std::vector<ManualObjectDranNetworkEditZone> zones;
  std::size_t labelMessageCount{0};
  std::size_t labelPayloadBytes{0};
};

ManualObjectDranDistributedEditZoneResult
manualObjectDranDistributedNetworkEditZonesForTest(
    const std::vector<ManualObjectDranObjectCopyEdge> &incumbent,
    const std::vector<ManualObjectDranObjectCopyEdge> &candidate,
    std::size_t numRobots,
    const std::vector<std::pair<std::size_t, std::size_t>> &topologyEdges);

struct ManualObjectDranDistributedFeasibilityResult {
  bool complete{false};
  bool feasible{false};
  std::string status;
  std::size_t labelMessageCount{0};
  std::size_t labelPayloadBytes{0};
  std::size_t reductionMessageCount{0};
  std::size_t reductionPayloadBytes{0};
};

ManualObjectDranDistributedFeasibilityResult
manualObjectDranDistributedNetworkPartitionFeasibleForTest(
    const std::vector<ManualObjectDranObjectCopyEdge> &edges,
    std::size_t numRobots,
    const std::vector<std::pair<std::size_t, std::size_t>> &topologyEdges);

struct ManualObjectDranTranslationSchurPacket {
  std::size_t robot{0};
  Matrix hessian;
  Matrix gradient;
  Matrix positions;
  Vector information;
  std::size_t payloadBytes{0};
};

struct ManualObjectDranTranslationSchurReduction {
  bool valid{false};
  std::string status;
  std::size_t rootRobot{0};
  Matrix hessian;
  Matrix gradient;
  Matrix positions;
  Vector information;
  std::size_t reductionMessageCount{0};
  std::size_t reductionPayloadBytes{0};
  std::size_t broadcastMessageCount{0};
  std::size_t broadcastPayloadBytes{0};
};

ManualObjectDranTranslationSchurReduction
manualObjectDranReduceTranslationSchurPacketsForTest(
    std::size_t numRobots,
    const std::vector<std::pair<std::size_t, std::size_t>> &topologyEdges,
    const std::vector<std::size_t> &participatingRobots,
    std::size_t rootRobot,
    const std::vector<ManualObjectDranTranslationSchurPacket> &packets);

std::vector<std::size_t> manualObjectDranNetworkEditZoneColorsForTest(
    const std::vector<ManualObjectDranNetworkEditZone> &zones);

struct ManualObjectDranNetworkPartitionZoneSummary {
  std::size_t sweep{0};
  std::size_t zoneId{0};
  std::size_t color{0};
  std::size_t lowerStateEpoch{0};
  bool lowerFeedbackRefreshAfterColor{false};
  bool valid{false};
  bool partitionFeasible{false};
  bool responseAccepted{false};
  bool committed{false};
  std::string status;
  std::size_t nodeCount{0};
  std::size_t robotCount{0};
  std::size_t removedEdgeCount{0};
  std::size_t addedEdgeCount{0};
  double predictedPhysicalDelta{0.0};
  double numericalErrorBound{0.0};
  std::size_t responsePayloadBytes{0};
  double computeSeconds{0.0};
};

struct ManualObjectDranNetworkPartitionProtocolSummary {
  std::size_t sweep{0};
  bool complete{false};
  std::string status;
  std::size_t zoneCount{0};
  std::size_t labelMessageCount{0};
  std::size_t labelPayloadBytes{0};
  std::size_t feasibilityReductionMessageCount{0};
  std::size_t feasibilityReductionPayloadBytes{0};
  std::size_t responseReductionMessageCount{0};
  std::size_t responseReductionPayloadBytes{0};
  std::size_t decisionBroadcastMessageCount{0};
  std::size_t decisionBroadcastPayloadBytes{0};
};

std::vector<ManualObjectDranNetworkEditZone>
manualObjectDranNetworkEditZonesForTest(
    const std::vector<ManualObjectDranObjectCopyEdge> &incumbent,
    const std::vector<ManualObjectDranObjectCopyEdge> &candidate);

bool manualObjectDranNetworkPartitionFeasibleForTest(
    const std::vector<ManualObjectDranObjectCopyEdge> &edges);

struct ManualObjectDranTranslationAssociationEdge {
  std::size_t first{0};
  std::size_t second{0};
  double information{0.0};
};

struct ManualObjectDranTranslationResponse {
  bool valid{false};
  double predictedPhysicalDelta{0.0};
  double currentPhysicalChange{0.0};
  double candidatePhysicalChange{0.0};
  double currentLinearResidual{0.0};
  double candidateLinearResidual{0.0};
  std::size_t currentNullity{0};
  std::size_t candidateNullity{0};
  double numericalErrorBound{0.0};
  std::size_t configuredDampingParameters{0};
  std::size_t configuredAcceptanceThresholds{0};
};

struct ManualObjectDranNetworkPartitionPlan {
  std::vector<ManualObjectDranObjectCopyEdge> committedEdges;
  std::vector<ManualObjectDranNetworkPartitionZoneSummary> zoneSummaries;
};

ManualObjectDranNetworkPartitionPlan
manualObjectDranPlanNetworkPartitionForTest(
    const std::vector<ManualObjectDranObjectCopyEdge> &incumbent,
    const std::vector<ManualObjectDranObjectCopyEdge> &candidate,
    const std::vector<ManualObjectDranTranslationResponse> &responses);

struct ManualObjectDranTranslationSchurModel {
  bool valid{false};
  Matrix hessian;
  Matrix gradient;
  std::size_t configuredAnchors{0};
  std::size_t configuredDampingParameters{0};
};

ManualObjectDranTranslationSchurModel
manualObjectDranTranslationQuadraticForTest(
    const std::vector<RelativeSEMeasurement> &measurements,
    const Matrix &state, std::size_t numPoses, unsigned d);

ManualObjectDranTranslationSchurModel
manualObjectDranTranslationSchurForTest(
    const Matrix &hessian, const Matrix &gradient,
    const std::vector<std::size_t> &boundary);

std::size_t manualObjectDranProjectCompletePoint2ConsensusForTest(
    const std::vector<std::size_t> &objectOffsets, std::size_t numObjects,
    unsigned dimension, double relaxation, std::vector<Matrix> &states,
    std::vector<std::vector<Matrix>> &neighborStateCache);

std::vector<std::vector<Matrix>>
manualObjectDranCompleteObjectPoseMeansForTest(
    const std::vector<std::size_t> &objectOffsets, std::size_t numObjects,
    unsigned dimension, const std::vector<Matrix> &states,
    const std::vector<std::vector<Matrix>> &neighborStateCache);

Matrix manualObjectDranCompleteObjectPoseGaugeForTest(
    const Matrix &state, std::size_t objectOffset,
    const std::vector<Matrix> &objectMeans, unsigned dimension);

bool manualObjectDranPoint2ConsensusProjectionActiveForTest(
    bool enabled, std::size_t projectionRounds, unsigned iteration);

bool manualObjectDranPoint2ConsensusShouldResetDualForTest(
    bool enabled, std::size_t projectionRounds, unsigned iteration);

bool manualObjectDranDirectEdgeDualRelaxationCanSettleForTest(
    bool projectionEnabled, std::size_t projectionRounds,
    std::size_t refreshRound);

bool manualObjectDranShouldPruneInactiveDirectEdgeDualForTest(
    bool preserveInactiveDuals, bool active);

bool manualObjectDranShouldPruneInactiveDirectEdgeTargetForTest(
    bool fixedUnionModel, bool active);

ManualObjectDranTranslationResponse
manualObjectDranTranslationResponseForTest(
    const Matrix &physicalHessian, const Matrix &physicalGradient,
    const Matrix &positions,
    const std::vector<ManualObjectDranTranslationAssociationEdge>
        &currentEdges,
    const std::vector<ManualObjectDranTranslationAssociationEdge>
        &candidateEdges);

ManualObjectDranAssociationMetricSummary
evaluateManualObjectDranAssociationMatrixMetrics(
    const Matrix &association, const Matrix &groundTruth,
    bool groundTruthAvailable);

struct ManualObjectDranTrackEmpiricalInformationSummary {
  bool enabled{false};
  std::size_t trackCount{0};
  std::size_t observationCount{0};
  std::size_t singletonTrackCount{0};
  std::size_t overdispersedTrackCount{0};
  double singletonTrackFraction{0.0};
  double overdispersedTrackFraction{0.0};
  double scaleMin{1.0};
  double scaleMedian{1.0};
  double scaleMean{1.0};
  double scaleP90{1.0};
  double scaleMax{1.0};
  double effectiveSampleRatioMedian{1.0};
  double effectiveSampleRatioMean{1.0};
};

ManualObjectDranTrackEmpiricalInformationSummary
summarizeManualObjectDranTrackEmpiricalInformation(
    const std::vector<TrackEmpiricalInformationEstimate> &estimates,
    bool enabled);

struct ManualObjectDranRunResult {
  std::size_t numRobots{0};
  std::size_t numObjects{0};
  std::size_t knownObjectCopies{0};
  std::size_t relayObjectCopies{0};
  std::size_t observedObjectCopies{0};
  std::size_t relayOnlyObjectCopies{0};
  std::size_t initializedFromNeighborObjectCopies{0};
  std::size_t staleObjectCopies{0};
  ManualObjectDranTrackEmpiricalInformationSummary
      trackEmpiricalInformationSummary;
  bool associationAwareTedCciInitializationApplied{false};
  std::size_t associationAwareTedCciActiveVertices{0};
  std::size_t associationAwareTedCciLocalMeasurements{0};
  std::size_t associationAwareTedCciCrossMeasurements{0};
  std::size_t associationAwareTedCciFactorMessages{0};
  std::size_t associationAwareTedCciSolutionMessages{0};
  std::size_t associationAwareTedCciBytesSent{0};
  double associationAwareTedCciSeconds{0.0};
  std::vector<ManualObjectDranIterationSummary> iterations;
  std::vector<std::vector<std::size_t>> localObjectStateSequences;
  std::vector<ManualObjectDranObjectInterfaceDiagnostic>
      objectInterfaceDiagnostics;
  std::vector<ManualObjectDranObjectInterfaceResponseSummary>
      objectInterfaceResponseSummaries;
  std::vector<std::vector<Matrix>> iterationEstimateSnapshots;
  std::vector<Matrix> finalEstimates;
  bool finalEstimatesAreRobotLocal{false};
  std::vector<Matrix> finalRobotGauges;
  ManualObjectDranDirectedAssociationWeights finalAssociationWeights;
  ManualObjectDranDirectedAssociationTransforms finalAssociationTransforms;
  DirectedGaugeCovariantAssociationEdgeStates finalAssociationEdgeStates;
  std::vector<ManualObjectDranAssociationUpperResponseDiagnostic>
      finalAssociationUpperResponseDiagnostics;
  std::vector<ManualObjectDranPairCavityCovarianceDiagnostic>
      finalPairCavityCovariances;
  std::vector<ManualObjectDranLeavePairCavityDiagnostic>
      finalLeavePairCavities;
  std::vector<ManualObjectDranCavityFeedbackSummary>
      cavityFeedbackSummaries;
  std::vector<ManualObjectDranNetworkPartitionZoneSummary>
      networkPartitionZoneSummaries;
  std::vector<ManualObjectDranNetworkPartitionProtocolSummary>
      networkPartitionProtocolSummaries;
  bool lowerValidationSelectedBestMeasurementState{false};
  unsigned selectedIteration{0};
  unsigned evaluatedIteration{0};
  std::size_t selectedIterationCount{0};
  std::size_t evaluatedIterationCount{0};
  std::size_t selectedCumulativeCommPoseCount{0};
  std::size_t evaluatedCumulativeCommPoseCount{0};
  double selectedCumulativeCommMb{0.0};
  double evaluatedCumulativeCommMb{0.0};
  double selectedMeasurementCost{0.0};
  double evaluatedMeasurementCost{0.0};
  double elapsedSeconds{0.0};
};

ManualObjectDranOptions
applyManualObjectDranAssociationPolicy(
    const ManualObjectDranOptions &options);

ManualObjectDranOptions
makeFinalDbaObjectAssociationOptions(
    const ManualObjectDranOptions &options);

struct ManualObjectDranSharedEdgeSyncDiagnostics {
  std::size_t synchronizedEdgeCount{0};
  std::size_t edgeResponseTaskCount{0};
  double preProjectionDisagreementNorm{0.0};
  double projectionDisplacementNorm{0.0};
  std::size_t candidateCount{0};
  std::size_t sparseResponseCandidateCount{0};
  std::size_t objectiveScalarCount{0};
  std::size_t payloadBytes{0};
  double payloadMb{0.0};
  double freshUotMarginalResidualNorm{0.0};
  double freshUotResponseMass{0.0};
  double upperFixedPointResidualNorm{0.0};
  double lowerFeedbackReadinessMean{1.0};
  double mannStepMean{0.0};
  std::vector<bool> commandChangedRobots;
};

ManualObjectDranSharedEdgeSyncDiagnostics
synchronizeManualObjectDranSharedEdgeWeights(
    ManualObjectDranDirectedAssociationWeights &weights,
    const std::vector<std::vector<bool>> &undirectedEdges,
    const PartialSinkhornOptions &options);

ManualObjectDranSharedEdgeSyncDiagnostics
synchronizeManualObjectDranSharedEdgeJointWeights(
    ManualObjectDranDirectedAssociationWeights &weights,
    const ManualObjectDranDirectedAssociationWeights &previousWeights,
    const ManualObjectDranDirectedAssociationWeights &effectiveObjectives,
    const std::vector<std::vector<bool>> &undirectedEdges,
    bool usePrevious,
    const PartialSinkhornOptions &options);

ManualObjectDranSharedEdgeSyncDiagnostics
synchronizeManualObjectDranDecentralizedFeedbackUotWeights(
    ManualObjectDranDirectedAssociationWeights &weights,
    const ManualObjectDranDirectedAssociationWeights &effectiveObjectives,
    const std::vector<std::vector<bool>> &undirectedEdges,
    const std::vector<double> &robotReadiness,
    std::vector<std::vector<ManualObjectDranEdgeFeedbackState>>
        &edgeFeedbackStates,
    const PartialSinkhornOptions &options);

ManualObjectDranSharedEdgeSyncDiagnostics
synchronizeManualObjectDranDecentralizedFeedbackUotSparseWeights(
    ManualObjectDranDirectedAssociationWeights &weights,
    const ManualObjectDranDirectedSparseAssociationObjectives
        &effectiveObjectives,
    const std::vector<std::vector<bool>> &undirectedEdges,
    const std::vector<double> &robotReadiness,
    std::vector<std::vector<ManualObjectDranEdgeFeedbackState>>
        &edgeFeedbackStates,
    const PartialSinkhornOptions &options,
    std::vector<ManualObjectDranAssociationUpperResponseDiagnostic>
        *responseDiagnostics = nullptr);

ManualObjectDranRunResult
runManualObjectDran(const std::string &directory,
                    const ManualObjectDranOptions &options);

double automaticLowerTrajectoryTranslationProxWeight(
    double requestedWeight, std::size_t activeObjectCopyPairs,
    std::size_t numTrajectoryPoses);

double sharedObjectLowerIncidentFixedPointResidual(
    ManualObjectDranSharedLatentMode mode,
    double normalizedPrimalResidual,
    double normalizedDualResidual);

double objectAssociationInformationVolumeCorrection(
    double information, double referenceInformation);

double automaticObjectCopyTargetPrecisionScale(double targetPrecision,
                                               double activeMeanPrecision);

std::size_t estimatedObjectAssociationTermReserveCapacity(
    std::size_t numObjects, const ManualObjectDranOptions &options);

void reserveAdditionalObjectAssociationTerms(
    std::vector<ObjectCopyConsistencyTerm> &terms,
    std::size_t numObjects, const ManualObjectDranOptions &options);

std::vector<std::size_t> objectObservationCounts(
    const ObjectPGORobotData &robot, std::size_t numObjects);

std::vector<ManualObjectDranAssociationEntry> activeAssociationEntries(
    const Matrix &weights, double minWeight);

std::vector<ManualObjectDranAssociationCandidateEdge>
finiteAssociationCandidateEdges(const Matrix &supportCosts);

ManualObjectDranAssociationSupportStats associationFiniteSupportStats(
    const Matrix &supportCosts, const Matrix &assignment);

double semanticAssociationMismatchLowerBoundCost(
    std::size_t localCategory, double localConfidence,
    std::size_t targetCategory, double targetConfidence,
    const ManualObjectDranOptions &options);

bool semanticAssociationMismatchCannotPassMinScore(
    std::size_t localCategory, double localConfidence,
    std::size_t targetCategory, double targetConfidence,
    const ManualObjectDranOptions &options);

std::vector<double> objectBlockDiagonalStiffnesses(
    const SparseMatrix &Q, std::size_t objectOffset,
    std::size_t numObjects, unsigned d);

std::vector<ObjectCopyChannelInformation> objectBlockDiagonalChannelInformation(
    const SparseMatrix &Q, std::size_t objectOffset,
    std::size_t numObjects, unsigned d);

bool associationStateValidationTrajectoryGuardRejects(
    double acceptedTrajectoryCost, double candidateTrajectoryCost);

bool associationStateWindowsCanAffectConsensus(
    const ManualObjectDranOptions &options);

}  // namespace DPGO

#endif
