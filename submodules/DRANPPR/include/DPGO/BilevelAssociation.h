/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef DPGO_BILEVEL_ASSOCIATION_H
#define DPGO_BILEVEL_ASSOCIATION_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace DPGO {

class QuadraticProblem;

struct SparseAssignmentCostEntry;
struct SparseAssignmentWeightEntry;

struct CandidateAssociation {
  std::size_t sourceRobotId{0};
  std::size_t sourceObjectId{0};
  std::size_t targetRobotId{0};
  std::size_t targetObjectId{0};
};

struct AssociationCostWeights {
  double semantic{1.0};
  double shape{1.0};
  double descriptor{1.0};
  double geometry{1.0};
  double temporal{1.0};
};

struct AssociationCostTerms {
  double semantic{0.0};
  double shape{0.0};
  double descriptor{0.0};
  double geometry{0.0};
  double temporal{0.0};

  double weightedCost(const AssociationCostWeights &weights) const;
};

struct TrackEmpiricalInformationEstimate {
  std::size_t sampleCount{0};
  double informationScale{1.0};
  double overdispersion{1.0};
  double effectiveSampleSize{0.0};
  double effectiveSampleRatio{1.0};
  bool empirical{false};
};

TrackEmpiricalInformationEstimate estimateTrackEmpiricalInformation(
    const std::vector<Vector> &centers,
    const std::vector<double> &translationInformation);

struct TransformAssociationCandidate {
  std::size_t localObjectId{0};
  std::size_t targetObjectId{0};
  Vector localPoint;
  Vector targetPoint;
  double score{1.0};
};

struct TransformAssociationVerificationOptions {
  unsigned dimension{2};
  std::size_t minInliers{3};
  double maxResidual{1.0};
  double minScore{0.0};
  double weightScale{1.0};
  std::size_t maxSeedCandidates{80};
};

struct TransformAssociationVerificationResult {
  bool accepted{false};
  std::size_t inlierCount{0};
  std::size_t seedHypothesesEvaluated{0};
  std::size_t seedHypothesesRejectedByDistance{0};
  double rmse{0.0};
  Matrix transform;
  Matrix weights;
  std::vector<TransformAssociationCandidate> inliers;
};

TransformAssociationVerificationResult verifyTransformAwareAssociation(
    std::size_t numLocalObjects, std::size_t numTargetObjects,
    const std::vector<TransformAssociationCandidate> &candidates,
    const TransformAssociationVerificationOptions &options);

struct AssociationCycleConsistencyOptions {
  double minDirectWeight{1e-12};
  double minComposedWeight{1e-12};
  std::size_t minTwoHopSupport{1};
  bool keepPairsWithoutTwoHopEvidence{true};
};

struct AssociationCycleConsistencyStats {
  std::size_t checkedEntries{0};
  std::size_t keptEntries{0};
  std::size_t prunedEntries{0};
};

AssociationCycleConsistencyStats pruneAssociationCycleInconsistencies(
    std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleConsistencyOptions &options);

struct AssociationCycleObjectiveOptions {
  double weight{1.0};
  double minEdgeMass{1e-12};
};

struct AssociationCycleObjectiveResult {
  double cost{0.0};
  std::size_t activeTriplets{0};
  std::vector<std::vector<Matrix>> gradient;
  std::vector<std::vector<double>> blockLipschitz;
};

AssociationCycleObjectiveResult evaluateAssociationCycleObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options);

AssociationCycleObjectiveResult evaluateAssociationPartialCycleAffinityObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options);

AssociationCycleObjectiveResult evaluateAssociationPartialCycleClosureObjective(
    const std::vector<std::vector<Matrix>> &associationWeights,
    const AssociationCycleObjectiveOptions &options);

struct AssociationRelationalGeometryOptions {
  double weight{1.0};
  double distanceScale{0.0};
  bool normalizeBySquaredAssignmentMass{false};
};

struct AssociationRelationalGeometryResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationRelationalGeometryResult evaluateAssociationRelationalGeometryObjective(
    const Matrix &sourceDistances, const Matrix &targetDistances,
    const Matrix &assignment,
    const AssociationRelationalGeometryOptions &options);

enum class AssociationRelationalNullCalibration {
  FixedStudentT,
  AdaptiveEmpiricalRank,
};

struct AssociationRelationalNullEvidenceOptions {
  AssociationRelationalNullCalibration calibration{
      AssociationRelationalNullCalibration::FixedStudentT};
  double fixedAnchorThreshold{0.5};
  std::size_t fixedMinimumAnchors{8};
  double fixedStudentTDegreesOfFreedom{3.0};
  std::size_t geometricDimension{3};
};

struct AssociationRelationalNullEvidenceResult {
  bool valid{false};
  std::size_t anchorCount{0};
  std::size_t nullSampleCount{0};
  double distanceScale{0.0};
  double logNullMedian{0.0};
  double logNullScale{0.0};
  double centeredNullLogMean{0.0};
  double empiricalRankResolution{0.0};
  Matrix costOffset;
};

AssociationRelationalNullEvidenceResult
evaluateAssociationRelationalNullEvidence(
    const Matrix &sourceDistances, const Matrix &targetDistances,
    const Matrix &assignment, const Matrix &supportCosts,
    const std::vector<std::size_t> &sourceCategories,
    const std::vector<std::size_t> &targetCategories,
    const AssociationRelationalNullEvidenceOptions &options = {});

double partialAssignmentSupportCoupling(
    double rowMass, double columnMass, double entryMass,
    double rowCapacity, double columnCapacity);

struct AssociationRotoTranslationAlignmentOptions {
  double weight{1.0};
  double distanceScale{0.0};
};

struct AssociationRotoTranslationAlignmentResult {
  double cost{0.0};
  Matrix gradient;
  Matrix rotation;
  Vector translation;
  double mass{0.0};
};

struct AssociationRotoTranslationEstimate {
  Matrix rotation;
  Vector translation;
  double mass{0.0};
};

struct SparsePoseAlignmentWeightEntry {
  int row{0};
  int col{0};
  double associationWeight{0.0};
  double rotationInformation{0.0};
  double translationInformation{0.0};
};

AssociationRotoTranslationEstimate
estimateSparseAssociationRotoTranslationAlignment(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const std::vector<SparseAssignmentWeightEntry> &entries);

AssociationRotoTranslationEstimate estimateSparseAssociationPoseAlignment(
    const std::vector<Matrix> &sourcePoses,
    const std::vector<Matrix> &targetPoses,
    const std::vector<SparsePoseAlignmentWeightEntry> &entries);

AssociationRotoTranslationAlignmentResult
evaluateAssociationRotoTranslationAlignmentObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const AssociationRotoTranslationAlignmentOptions &options);

struct AssociationRotoTranslationHypothesisMixtureOptions {
  double weight{1.0};
  double distanceScale{0.0};
  double softminTemperature{1.0};
};

struct AssociationRotoTranslationHypothesisMixtureResult {
  double cost{0.0};
  Matrix gradient;
  Vector hypothesisWeights;
};

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options);

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &gradientSupport,
    const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options);

AssociationRotoTranslationHypothesisMixtureResult
evaluateAssociationRotoTranslationHypothesisMixtureObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const std::vector<SparseAssignmentCostEntry> &supportEntries,
    const std::vector<Matrix> &rotations,
    const std::vector<Vector> &translations,
    const AssociationRotoTranslationHypothesisMixtureOptions &options);

struct AssociationPairwiseTransformValidationOptions {
  double weight{1.0};
  double distanceScale{0.0};
};

struct AssociationPairwiseTransformValidationResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationPairwiseTransformValidationResult
evaluateAssociationPairwiseTransformValidationObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &rotation,
    const AssociationPairwiseTransformValidationOptions &options);

AssociationPairwiseTransformValidationResult
evaluateAssociationPairwiseTransformValidationObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &rotation,
    const Matrix &gradientSupport,
    const AssociationPairwiseTransformValidationOptions &options);

struct AssociationRotoTranslationRingHypothesisOptions {
  std::size_t maxHypotheses{0};
  std::size_t angleBins{36};
};

struct AssociationRotoTranslationRingHypothesisResult {
  std::vector<Matrix> rotations;
  std::vector<Vector> translations;
  Vector scores;
};

AssociationRotoTranslationRingHypothesisResult
generateAssociationRotoTranslationRingHypotheses(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const AssociationRotoTranslationRingHypothesisOptions &options);

AssociationRotoTranslationRingHypothesisResult
generateAssociationRotoTranslationRingHypotheses(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment,
    const std::vector<SparseAssignmentCostEntry> &supportEntries,
    const AssociationRotoTranslationRingHypothesisOptions &options);

Matrix centerAssignmentGradientRowsOnFiniteSupport(const Matrix &gradient,
                                                   const Matrix &supportCosts);

Matrix centerAndNormalizeAssignmentGradientOnFiniteSupport(
    const Matrix &gradient, const Matrix &supportCosts, bool centerRows,
    bool centerColumns, double targetRms);

Matrix centerAndNormalizeAssignmentGradientOnFiniteSupport(
    const Matrix &gradient, const Matrix &centeringSupportCosts,
    const Matrix &normalizationSupportCosts, bool centerRows,
    bool centerColumns, double targetRms);

double associationMeritGateFromDecrease(double meritDecrease,
                                        double trustScale);

double associationFreshSupportStateValidationMeritGate(double baseMerit,
                                                       double candidateMerit,
                                                       double trustScale);

double stateValidationTransformConfidencePenalty(double weightedConfidence);

double objectObservationUncertaintyReliability(std::size_t observationCount,
                                               double meanDistance,
                                               double countPrior,
                                               double distanceScale,
                                               double minReliability);

struct AssociationCoveragePriorOptions {
  double weight{1.0};
  bool matchRowColumnMarginals{false};
  bool normalizeByReferenceMass{false};
};

struct AssociationCoveragePriorResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationCoveragePriorResult evaluateAssociationCoveragePrior(
    const Matrix &assignment, const Matrix &reference,
    const AssociationCoveragePriorOptions &options);

Matrix associationConfidenceGatedSignedRecoveryGate(
    const Matrix &signedGradient, const Matrix &confidence,
    double sigmoidScale, double maxBlend);

Matrix associationReciprocalFreshSupportGate(
    const Matrix &localWeights, const Matrix &reverseTransposedWeights,
    const Matrix &confidence, double maxBlend);

Matrix associationReciprocalUniquenessConfidence(
    const Matrix &localWeights, const Matrix &reverseTransposedWeights,
    double competitionScale);

struct AssociationPairSupportGateOptions {
  double minEffectiveSupport{0.0};
  double power{0.0};
};

struct AssociationPairSupportGateResult {
  double effectiveSupport{0.0};
  double gate{1.0};
  Matrix weights;
};

AssociationPairSupportGateResult applyAssociationPairSupportGate(
    const Matrix &assignment, const AssociationPairSupportGateOptions &options);

double associationPairSupportConfidence(
    const Matrix &assignment, const AssociationPairSupportGateOptions &options);

struct AssociationPairSupportLatentContinuationResult {
  double referenceConfidence{0.0};
  double candidateConfidence{0.0};
  double latentConfidence{0.0};
  double blend{0.0};
  Matrix weights;
};

AssociationPairSupportLatentContinuationResult
applyAssociationPairSupportLatentContinuation(
    const Matrix &candidate, const Matrix &reference, double latentConfidence,
    const AssociationPairSupportGateOptions &options);

bool associationPairSupportReferenceShouldUpdate(
    const Matrix &candidate, const Matrix &reference,
    const AssociationPairSupportGateOptions &options);

double objectCopyResidualReliability(double residualNorm,
                                     double residualScale,
                                     double minReliability);

double objectCopyResidualReliabilityFromActiveRms(
    double residualNorm, double activeResidualRms, double minReliability);

Matrix objectCopyResidualReliabilityMatrixFromActiveRms(
    const Matrix &residualNorms, const Matrix &assignment,
    double minReliability);

std::vector<double> objectResidualReliabilityVectorFromActiveRms(
    const std::vector<double> &residualNorms, double minReliability);

double objectCopyReliabilityUpperCost(double reliability, double weight);

struct AssociationSupportSharpnessOptions {
  double weight{1.0};
};

struct AssociationSupportSharpnessResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationSupportSharpnessResult evaluateAssociationSupportSharpnessObjective(
    const Matrix &assignment,
    const AssociationSupportSharpnessOptions &options);

struct AssociationSupportSharpnessAdaptiveOptions {
  double maxWeight{0.0};
  double targetGradientRatio{0.0};
  double epsilon{1e-12};
};

double adaptiveAssociationSupportSharpnessWeight(
    double unitSharpnessGradientNorm, double localGradientNorm,
    const AssociationSupportSharpnessAdaptiveOptions &options);

struct AssociationStructuralExpansionOptions {
  double weight{1.0};
  double distanceScale{0.0};
};

struct AssociationStructuralExpansionResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationStructuralExpansionResult
evaluateAssociationStructuralExpansionObjective(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &assignment, const Matrix &reference,
    const Matrix &rotation, const Vector &translation,
    const AssociationStructuralExpansionOptions &options);

Matrix associationProposalSupportGate(const Matrix &proposal);

Matrix associationUnionSupportGate(const Matrix &first, const Matrix &second);

Matrix associationResidualConfidenceGate(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation,
    double distanceScale);

Matrix associationRelativeResidualConfidenceGate(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation,
    double distanceScale);

Matrix associationRelativeResidualConfidenceGateOnSupport(
    const Matrix &sourcePoints, const Matrix &targetPoints,
    const Matrix &rotation, const Vector &translation, double distanceScale,
    const Matrix &support);

struct AssociationReferenceTrustOptions {
  double weight{1.0};
};

struct AssociationReferenceTrustResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationReferenceTrustResult evaluateAssociationReferenceTrust(
    const Matrix &assignment, const Matrix &reference,
    const AssociationReferenceTrustOptions &options);

struct AssociationReferenceSupportSharpeningOptions {
  double weight{1.0};
  double target{1.0};
};

struct AssociationReferenceSupportSharpeningResult {
  double cost{0.0};
  Matrix gradient;
};

AssociationReferenceSupportSharpeningResult
evaluateAssociationReferenceSupportSharpening(
    const Matrix &assignment, const Matrix &reference,
    const AssociationReferenceSupportSharpeningOptions &options);

double scaleAssociationReferenceTrustWeightByCostRms(
    double baseWeight, const Matrix &costs, double minimumCostRmsScale);

double associationConsensusWeightPower(double associationWeight,
                                       double power);

double associationHighConfidenceAnchorScale(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition,
    double transformConfidence, double gain);

double associationContinuousSupportGate(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition,
    double transformConfidence, double reliability, double power);

double associationBoundedResidualReliabilityBlend(
    double sharedReliability, double currentResidualReliability);

double associationSupportConfidenceReliability(
    double associationWeight, double reciprocalAssociationWeight,
    double rowCompetition, double columnCompetition);

double associationTrajectoryFeedbackTrustScale(double poseTrust);

double associationTrajectoryFeedbackObjectMassNormalizer(double objectMass);

Matrix projectAssociationForLowerGraph(const Matrix &assignment, double power);

Matrix projectAssociationForLowerGraphHard(const Matrix &assignment);

struct BackendAssociationFeedbackOptions {
  double gradientScale{0.0};
  double minTrust{0.0};
};

double backendFeedbackGeometryCost(
    double geometryCost, double localGradientNorm, double targetGradientNorm,
    const BackendAssociationFeedbackOptions &options);

struct BackendAssociationResponseOptions {
  double responseGain{0.0};
  double curvatureDamping{1.0};
  double maxCostReductionFraction{0.5};
};

double backendResponseGeometryCost(
    double geometryCost, double localGradientDotResidual,
    double targetGradientDotResidual, double localStiffness,
    double targetStiffness,
    const BackendAssociationResponseOptions &options);

struct BackendAssociationAdjointOptions {
  double adjointGain{0.0};
  double curvatureDamping{1.0};
  double maxCostReductionFraction{0.5};
  double maxCostIncreaseFraction{0.5};
  bool allowCostReduction{true};
  bool allowCostIncrease{true};
};

double backendAdjointGeometryCost(
    double geometryCost, double localGradientDotResidual,
    double targetGradientDotResidual, double localStiffness,
    double targetStiffness,
    const BackendAssociationAdjointOptions &options);

Matrix solveBackendCoupledAdjoint(const SparseMatrix &hessian,
                                  const Matrix &gradient,
                                  double damping);

struct PartialSinkhornOptions {
  std::size_t iterations{20};
  double stepSize{1.0};
  double temperature{1.0};
  double rowCapacity{1.0};
  double columnCapacity{1.0};
  std::size_t maxNonzeros{0};
  double unmatchedCost{std::numeric_limits<double>::infinity()};
  double temporalBlend{0.0};
  double temporalKlWeight{0.0};
  double temporalKlEpsilon{1e-12};
  double entropyRegularizationWeight{0.0};
  bool renormalizeAfterSparsify{false};
  bool restrictToPreviousSupport{false};
  double previousSupportThreshold{1e-12};
};

struct SparseAssignmentCostEntry {
  int row{0};
  int col{0};
  double cost{0.0};
};

struct SparseAssignmentWeightEntry {
  int row{0};
  int col{0};
  double value{0.0};
};

struct FreshUnbalancedTransportEdgeDiagnostic {
  int row{0};
  int col{0};
  double cost{0.0};
  double rawScore{0.0};
  double projectedWeight{0.0};
  double rowMass{0.0};
  double columnMass{0.0};
};

double reciprocalFiniteCostMean(double firstCost, double secondCost);

std::vector<SparseAssignmentCostEntry> normalizeSparseAssignmentCostEntries(
    int rows, const std::vector<SparseAssignmentCostEntry> &entries,
    std::size_t rowTopK, double targetRms);

struct FreshUnbalancedTransportResult {
  Matrix assignment;
  Vector rowUnmatchedMass;
  Vector columnUnmatchedMass;
  double marginalResidualNorm{0.0};
  std::size_t activeCandidateCount{0};
};

struct DistributedUpperTrackingOptions {
  double consensusWeight{0.0};
  double correctionWeight{1.0};
  bool centerRows{true};
  bool centerColumns{true};
};

struct DistributedUpperTrackingState {
  Matrix tracker;
  Matrix previousLocalGradient;
  bool initialized{false};
};

struct DistributedUpperTrackingResult {
  Matrix trackedGradient;
  Matrix correction;
  double trackerNorm{0.0};
  double correctionNorm{0.0};
  double consensusResidualNorm{0.0};
  double localGradientDeltaNorm{0.0};
};

DistributedUpperTrackingResult updateDistributedUpperTracker(
    DistributedUpperTrackingState &state, const Matrix &localGradient,
    const std::vector<Matrix> &neighborTrackers,
    const DistributedUpperTrackingOptions &options);

struct LowerGaugeStationarityDiagnostics {
  double rawGradientNorm{0.0};
  double quotientGradientNorm{0.0};
  double gaugeComponentNorm{0.0};
  double gaugeComponentRatio{0.0};
};

LowerGaugeStationarityDiagnostics evaluateLowerGaugeStationarity(
    const Matrix &gradient, unsigned dimension);

LowerGaugeStationarityDiagnostics evaluateLowerGaugeStationarity(
    const Matrix &state, const Matrix &gradient, unsigned dimension);

Matrix projectLowerGaugeTangent(const Matrix &state, const Matrix &tangent,
                                unsigned dimension);

struct LowerValidationAdjointOptions {
  std::size_t maxIterations{100};
  double relativeTolerance{1e-10};
  bool projectGauge{true};
  bool useGaussNewtonLinearization{false};
};

struct LowerValidationAdjointResult {
  Matrix adjoint;
  std::size_t iterations{0};
  double damping{0.0};
  double initialResidualNorm{0.0};
  double finalResidualNorm{0.0};
  double operatorSymmetryError{0.0};
  double minimumRayleighQuotient{std::numeric_limits<double>::infinity()};
  double maximumRayleighQuotient{0.0};
  bool converged{false};
  bool negativeCurvature{false};
};

LowerValidationAdjointResult solveLowerValidationAdjoint(
    const QuadraticProblem &lowerProblem, const Matrix &state,
    const Matrix &validationGradient, unsigned dimension,
    const LowerValidationAdjointOptions &options = {});

double implicitLowerValidationHypergradient(
    const Matrix &adjoint, const Matrix &factorGradient);

class SoftAssignmentOptimizer {
 public:
  explicit SoftAssignmentOptimizer(PartialSinkhornOptions options = {});

  Matrix updateFromCosts(const Matrix &costs) const;
  Matrix updateFromCosts(const Matrix &costs, const Matrix &previous) const;
  Matrix updateFromCostsAndGradient(const Matrix &costs,
                                    const Matrix &extraGradient,
                                    const Matrix &previous) const;
  Matrix updateFromCostEntries(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries) const;
  Matrix updateFromCostEntries(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries,
      const Matrix &previous) const;
  Matrix updateFromCostEntriesAndGradient(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries,
      const Matrix &extraGradient, const Matrix &previous) const;
  FreshUnbalancedTransportResult solveFreshUnbalancedTransport(
      const Matrix &costs) const;
  FreshUnbalancedTransportResult solveFreshUnbalancedTransport(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries) const;
  FreshUnbalancedTransportResult solveFreshUnbalancedTransport(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries,
      std::vector<FreshUnbalancedTransportEdgeDiagnostic> *edgeDiagnostics)
      const;
  Matrix updateFromCostsAndReferenceTrust(const Matrix &costs,
                                          const Matrix &reference,
                                          double referenceWeight,
                                          const Matrix &previous) const;
  Matrix updateFromCostsGradientAndReferenceTrust(
      const Matrix &costs, const Matrix &extraGradient,
      const Matrix &reference, double referenceWeight,
      const Matrix &previous) const;
  Matrix applyReferenceMassTrust(const Matrix &proposal,
                                 double referenceMass,
                                 double referenceWeight,
                                 const Matrix &previous) const;
  Matrix applyConfidenceWeightedReferenceMassTrust(
      const Matrix &proposal, const Matrix &reference,
      const Matrix &confidence, double referenceWeight,
      const Matrix &previous) const;
  Matrix applyConfidenceWeightedReferenceMassTrustOnEntries(
      const Matrix &proposal, const Matrix &reference,
      const Matrix &confidence,
      const std::vector<SparseAssignmentCostEntry> &supportEntries,
      double referenceWeight, const Matrix &previous) const;

  Matrix projectWeights(const Matrix &weights,
                        bool applySparsity = true) const;
  Matrix projectWeightsSparseSupport(const Matrix &weights,
                                     bool applySparsity = true) const;

  const PartialSinkhornOptions &options() const { return options_; }

 private:
  PartialSinkhornOptions options_;

  Matrix initializeWeights(const Matrix &costs, const Matrix *previous) const;
  Matrix applyRegularizationGradients(const Matrix &proposal,
                                      const Matrix *previous,
                                      bool applySparsity = true) const;
  Matrix applyReferenceTrust(const Matrix &proposal, const Matrix &reference,
                             double referenceWeight,
                             const Matrix &previous) const;
  Matrix applyPreviousSupportMask(const Matrix &weights,
                                  const Matrix &previous) const;
  Matrix materializeSparseCosts(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries) const;
  std::vector<SparseAssignmentWeightEntry> initializeWeightEntries(
      int rows, int cols,
      const std::vector<SparseAssignmentCostEntry> &entries,
      const Matrix *previous, const Matrix *extraGradient) const;
  Matrix projectPartialEntries(
      int rows, int cols, std::vector<SparseAssignmentWeightEntry> entries,
      bool applySparsity = true) const;
  Matrix projectPartial(const Matrix &weights,
                        bool applySparsity = true) const;
  Matrix projectPartialSparseSupport(const Matrix &weights,
                                     bool applySparsity = true) const;
  bool shouldUseSparseSupportProjection(const Matrix &weights) const;
};

struct ObjectCopyChannelInformation {
  double rotation{0.0};
  double translation{0.0};
};

struct ObjectCopyConsistencyTerm {
  std::size_t localObjectId{0};
  std::size_t targetRobotId{0};
  std::size_t targetObjectId{0};
  Matrix targetPose;
  double associationProbability{1.0};
  double consensusPenaltyScale{1.0};
  double weight{1.0};
  double reliability{1.0};
  double supportReliabilityScale{1.0};
  double localPrecisionScale{1.0};
  double targetPrecisionScale{1.0};
  ObjectCopyChannelInformation localInformation;
  ObjectCopyChannelInformation targetInformation;
  double alignmentReliability{1.0};
  double alignmentTranslationVariance{0.0};
  bool useChannelInformation{false};
  double robustResidualScale{0.0};
  double robustMinWeight{0.0};
  double robustLatentTarget{1.0};
  double robustLatentValue{1.0};
  std::size_t yawSymmetryOrder{1};
  bool targetFrameVerified{true};
};

ObjectCopyChannelInformation independentRelativeObjectCopyInformation(
    const ObjectCopyChannelInformation &local,
    const ObjectCopyChannelInformation &target);

ObjectCopyChannelInformation
marginalizeObjectCopyAlignmentTranslationVariance(
    const ObjectCopyChannelInformation &information,
    double alignmentTranslationVariance);

void configureExactSharedObjectConsensusPenalty(
    ObjectCopyConsistencyTerm &term);

void configureWeightedSharedObjectConsensusPenalty(
    ObjectCopyConsistencyTerm &term);

void configureDirectEdgeDualConsensusProximalFactor(
    ObjectCopyConsistencyTerm &term);

void configureExactDirectEdgeDualConsensusProximalFactor(
    ObjectCopyConsistencyTerm &term);

bool sharedObjectConsensusHypothesisIsActive(double associationWeight,
                                             bool boundOnly);

double whitenedObjectCopyInnovationSquared(
    double rotationResidualSquared, double translationResidualSquared,
    const ObjectCopyChannelInformation &relativeInformation);

double dimensionCalibratedCauchyLatent(double whitenedInnovationSquared,
                                       double residualDimension);

double dimensionCalibratedStudentTAlignmentCost(
    double translationResidualSquared, double translationRmseSquared,
    double residualDimension);

double configureExpectedObjectCopyFactor(ObjectCopyConsistencyTerm &term);

void configurePhysicalObjectCopyMajorizationFactor(
    ObjectCopyConsistencyTerm &term);

double objectCopyMetropolisDegreeScale(double localDegree,
                                       double targetDegree);

struct ObjectCopyConsistencyStats {
  std::size_t activeTerms{0};
  double totalWeight{0.0};
  double cost{0.0};
};

std::size_t applyObjectCopySharedBarycenter(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    std::vector<ObjectCopyConsistencyTerm> &terms);

std::size_t normalizeObjectCopyConsistencyTermsBySupportDegree(
    std::vector<ObjectCopyConsistencyTerm> &terms);

double objectCopyRobustLatentFixedPointResidual(
    const std::vector<ObjectCopyConsistencyTerm> &terms);

ObjectCopyConsistencyStats addSoftObjectCopyConsistencyTerms(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms, SparseMatrix &Q,
    SparseMatrix &G);

void addSoftObjectCopyJacobiMajorizationTerms(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms, SparseMatrix &Q,
    SparseMatrix &G);

ObjectCopyConsistencyStats
accumulateSoftObjectCopyConsistencyEuclideanGradient(
    std::size_t objectOffset, unsigned dimension, const Matrix &localState,
    const std::vector<ObjectCopyConsistencyTerm> &terms,
    Matrix &euclideanGradient);

}  // namespace DPGO

#endif
