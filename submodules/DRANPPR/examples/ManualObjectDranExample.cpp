#include <DPGO/ManualObjectDran.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/ObjectPGOData.h>

#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool parseBool(const std::string &value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "on";
}

DPGO::AssociationRelationalNullCalibration
parseAssociationRelationalNullCalibration(const std::string &value) {
  if (value == "fixed" || value == "student_t") {
    return DPGO::AssociationRelationalNullCalibration::FixedStudentT;
  }
  if (value == "adaptive" || value == "empirical_rank") {
    return DPGO::AssociationRelationalNullCalibration::AdaptiveEmpiricalRank;
  }
  throw std::invalid_argument("Unknown relational null calibration: " +
                              value);
}

DPGO::ManualObjectDranTopology parseTopology(const std::string &value) {
  if (value == "ring") {
    return DPGO::ManualObjectDranTopology::Ring;
  }
  if (value == "complete") {
    return DPGO::ManualObjectDranTopology::Complete;
  }
  throw std::invalid_argument("Unknown topology: " + value);
}

DPGO::ManualObjectDranCommunicationPolicy parseCommunicationPolicy(
    const std::string &value) {
  if (value == "all" || value == "full") {
    return DPGO::ManualObjectDranCommunicationPolicy::All;
  }
  if (value == "triggered" || value == "event_triggered") {
    return DPGO::ManualObjectDranCommunicationPolicy::Triggered;
  }
  throw std::invalid_argument("Unknown communication policy: " + value);
}

DPGO::ManualObjectDranObjectInitialization parseObjectInitialization(
    const std::string &value) {
  if (value == "centralized_chordal" || value == "centralized") {
    return DPGO::ManualObjectDranObjectInitialization::CentralizedChordal;
  }
  if (value == "neighbor_average" || value == "neighbor") {
    return DPGO::ManualObjectDranObjectInitialization::NeighborAverage;
  }
  if (value == "input_state" || value == "input" ||
      value == "preserve_input") {
    return DPGO::ManualObjectDranObjectInitialization::InputState;
  }
  throw std::invalid_argument("Unknown object initialization: " + value);
}

DPGO::ManualObjectDranObjectInitializationAnchorMode
parseObjectInitializationAnchorMode(const std::string &value) {
  if (value == "constant" || value == "uniform") {
    return DPGO::ManualObjectDranObjectInitializationAnchorMode::Constant;
  }
  if (value == "information" || value == "info" ||
      value == "measurement_information") {
    return DPGO::ManualObjectDranObjectInitializationAnchorMode::Information;
  }
  throw std::invalid_argument("Unknown object initialization anchor mode: " +
                              value);
}

DPGO::ManualObjectDranObjectInterfaceMode parseObjectInterfaceMode(
    const std::string &value) {
  if (value == "copy_baseline" || value == "baseline" || value == "copy") {
    return DPGO::ManualObjectDranObjectInterfaceMode::CopyBaseline;
  }
  if (value == "reduced_schur_response" || value == "schur_response" ||
      value == "object_schur_response") {
    return DPGO::ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  }
  if (value == "complete_graph_coarse_response" ||
      value == "coarse_response") {
    return DPGO::ManualObjectDranObjectInterfaceMode::
        CompleteGraphCoarseResponse;
  }
  if (value == "matrix_free_objectpose3_pcg" ||
      value == "objectpose3_pcg") {
    return DPGO::ManualObjectDranObjectInterfaceMode::
        MatrixFreeObjectPose3Pcg;
  }
  throw std::invalid_argument("Unknown object interface mode: " + value);
}

DPGO::ManualObjectDranObjectInterfacePcgPreconditioner
parseObjectInterfacePcgPreconditioner(const std::string &value) {
  if (value == "interface_hessian" || value == "hpp") {
    return DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::
        InterfaceHessian;
  }
  if (value == "exact_schur") {
    return DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::ExactSchur;
  }
  if (value == "private_block_jacobi" || value == "pose_block_jacobi") {
    return DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::
        PrivateBlockJacobi;
  }
  throw std::invalid_argument(
      "Unknown object interface PCG preconditioner: " + value);
}

DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver
parseObjectInterfacePcgPrivateSolver(const std::string &value) {
  if (value == "sparse_ldlt") {
    return DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver::SparseLdlt;
  }
  if (value == "block_tridiagonal_ldlt" || value == "block_tridiagonal") {
    return DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver::
        BlockTridiagonalLdlt;
  }
  throw std::invalid_argument(
      "Unknown object interface PCG private solver: " + value);
}

DPGO::ManualObjectDranObjectInterfaceResponseStrategy
parseObjectInterfaceResponseStrategy(const std::string &value) {
  if (value == "serial") {
    return DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Serial;
  }
  if (value == "batch" || value == "cached_batch") {
    return DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  }
  throw std::invalid_argument("Unknown object interface response strategy: " +
                              value);
}

DPGO::ManualObjectDranObjectInterfaceResponseTrigger
parseObjectInterfaceResponseTrigger(const std::string &value) {
  if (value == "periodic") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::Periodic;
  }
  if (value == "predicted_decrease" || value == "adaptive" ||
      value == "model_decrease") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        PredictedDecrease;
  }
  if (value == "local_innovation" || value == "innovation") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        LocalInnovation;
  }
  if (value == "accumulated_innovation" || value == "innovation_bucket" ||
      value == "accumulated") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        AccumulatedInnovation;
  }
  throw std::invalid_argument("Unknown object interface response trigger: " +
                              value);
}

DPGO::ManualObjectDranAssociationMode parseAssociationMode(
    const std::string &value) {
  if (value == "identity_oracle" || value == "oracle" ||
      value == "identity") {
    return DPGO::ManualObjectDranAssociationMode::IdentityOracle;
  }
  if (value == "class_only_hard" || value == "class_only" ||
      value == "semantic_class_only") {
    return DPGO::ManualObjectDranAssociationMode::ClassOnlyHard;
  }
  if (value == "metadata_hard" ||
      value == "semantic_shape_descriptor_hard") {
    return DPGO::ManualObjectDranAssociationMode::MetadataHard;
  }
  if (value == "geometry_hard" || value == "hard_geometry") {
    return DPGO::ManualObjectDranAssociationMode::GeometryHard;
  }
  if (value == "graph_consistency_hard" ||
      value == "slide_graph_hard" || value == "slide_slam_hard" ||
      value == "clipper_graph_hard") {
    return DPGO::ManualObjectDranAssociationMode::GraphConsistencyHard;
  }
  if (value == "geometry_soft" || value == "soft_geometry" ||
      value == "sinkhorn_geometry") {
    return DPGO::ManualObjectDranAssociationMode::GeometrySoft;
  }
  if (value == "transform_soft" || value == "transform_aware_soft" ||
      value == "local_frame_soft") {
    return DPGO::ManualObjectDranAssociationMode::TransformSoft;
  }
  if (value == "fixed_file_soft" || value == "file_soft" ||
      value == "fixed_file") {
    return DPGO::ManualObjectDranAssociationMode::FixedFileSoft;
  }
  if (value == "residual_soft" || value == "online_residual_soft" ||
      value == "dba_residual") {
    return DPGO::ManualObjectDranAssociationMode::ResidualSoft;
  }
  throw std::invalid_argument("Unknown association mode: " + value);
}

DPGO::ManualObjectDranAssociationPolicy parseAssociationPolicy(
    const std::string &value) {
  if (value == "manual" || value == "none") {
    return DPGO::ManualObjectDranAssociationPolicy::Manual;
  }
  if (value == "two_time_scale_bilevel_v1" ||
      value == "two_time_scale" || value == "bilevel_v1") {
    return DPGO::ManualObjectDranAssociationPolicy::TwoTimeScaleBilevelV1;
  }
  if (value == "single_loop_bilevel_v2" ||
      value == "single_loop" || value == "bilevel_v2") {
    return DPGO::ManualObjectDranAssociationPolicy::SingleLoopBilevelV2;
  }
  if (value == "single_loop_roto_bilevel_v3" ||
      value == "single_loop_roto" || value == "bilevel_v3") {
    return DPGO::ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3;
  }
  if (value == "final_dba_p4_203" || value == "dba_final" ||
      value == "dba_paper" || value == "p4_203") {
    return DPGO::ManualObjectDranAssociationPolicy::FinalDbaP4203;
  }
  if (value == "shared_edge_bilevel_v1") {
    return DPGO::ManualObjectDranAssociationPolicy::SharedEdgeDbaV1;
  }
  if (value == "shared_edge_joint_bilevel_v2") {
    return DPGO::ManualObjectDranAssociationPolicy::SharedEdgeJointDbaV2;
  }
  if (value == "decentralized_feedback_uot_bilevel_v1" ||
      value == "feedback_uot_dba_v1" || value == "dba_uot_v1") {
    return DPGO::ManualObjectDranAssociationPolicy::
        DecentralizedFeedbackUotDbaV1;
  }
  throw std::invalid_argument("Unknown association policy: " + value);
}

DPGO::ManualObjectDranFinalAblation parseAssociationFinalAblation(
    const std::string &value) {
  if (value == "none") {
    return DPGO::ManualObjectDranFinalAblation::None;
  }
  if (value == "state_validation") {
    return DPGO::ManualObjectDranFinalAblation::StateValidation;
  }
  if (value == "edge_consensus") {
    return DPGO::ManualObjectDranFinalAblation::EdgeConsensus;
  }
  if (value == "reciprocal_support") {
    return DPGO::ManualObjectDranFinalAblation::ReciprocalSupport;
  }
  if (value == "pair_support_gate") {
    return DPGO::ManualObjectDranFinalAblation::PairSupportGate;
  }
  if (value == "pair_support_continuation") {
    return DPGO::ManualObjectDranFinalAblation::PairSupportContinuation;
  }
  if (value == "shared_reliability") {
    return DPGO::ManualObjectDranFinalAblation::SharedReliability;
  }
  if (value == "observation_predictive_gate") {
    return DPGO::ManualObjectDranFinalAblation::ObservationPredictiveGate;
  }
  throw std::invalid_argument("Unknown association final ablation: " + value);
}

DPGO::ManualObjectDranSharedLatentMode parseSharedLatentMode(
    const std::string &value) {
  if (value == "auto") {
    return DPGO::ManualObjectDranSharedLatentMode::Auto;
  }
  if (value == "disabled" || value == "none") {
    return DPGO::ManualObjectDranSharedLatentMode::Disabled;
  }
  if (value == "exact_fixed") {
    return DPGO::ManualObjectDranSharedLatentMode::ExactFixed;
  }
  if (value == "exact_weighted") {
    return DPGO::ManualObjectDranSharedLatentMode::ExactWeighted;
  }
  if (value == "kkt_active_hybrid") {
    return DPGO::ManualObjectDranSharedLatentMode::KktActiveHybrid;
  }
  if (value == "direct_edge_dual") {
    return DPGO::ManualObjectDranSharedLatentMode::DirectEdgeDual;
  }
  if (value == "exact_point2_owner") {
    return DPGO::ManualObjectDranSharedLatentMode::ExactPoint2Owner;
  }
  if (value == "exact_point2_schur_oracle") {
    return DPGO::ManualObjectDranSharedLatentMode::ExactPoint2SchurOracle;
  }
  if (value == "exact_point2_owner_pcg") {
    return DPGO::ManualObjectDranSharedLatentMode::ExactPoint2OwnerPcg;
  }
  throw std::invalid_argument("Unknown shared latent mode: " + value);
}

DPGO::ManualObjectDranObjectCommunicationRepresentation
parseObjectCommunicationRepresentation(const std::string &value) {
  if (value == "lifted_pose" || value == "full") {
    return DPGO::ManualObjectDranObjectCommunicationRepresentation::LiftedPose;
  }
  if (value == "compact_pose_float64" || value == "compact_pose_double") {
    return DPGO::ManualObjectDranObjectCommunicationRepresentation::
        CompactPoseFloat64;
  }
  if (value == "compact_pose_float32" || value == "compact_pose") {
    return DPGO::ManualObjectDranObjectCommunicationRepresentation::
        CompactPoseFloat32;
  }
  if (value == "point2") {
    return DPGO::ManualObjectDranObjectCommunicationRepresentation::Point2;
  }
  throw std::invalid_argument("Unknown object communication representation: " +
                              value);
}

std::string objectCommunicationRepresentationName(
    DPGO::ManualObjectDranObjectCommunicationRepresentation representation) {
  switch (representation) {
    case DPGO::ManualObjectDranObjectCommunicationRepresentation::LiftedPose:
      return "lifted_pose";
    case DPGO::ManualObjectDranObjectCommunicationRepresentation::
        CompactPoseFloat64:
      return "compact_pose_float64";
    case DPGO::ManualObjectDranObjectCommunicationRepresentation::
        CompactPoseFloat32:
      return "compact_pose_float32";
    case DPGO::ManualObjectDranObjectCommunicationRepresentation::Point2:
      return "point2";
  }
  return "lifted_pose";
}

DPGO::ManualObjectDranDiagnosticProfile parseDiagnosticProfile(
    const std::string &value) {
  if (value == "auto" || value == "default") {
    return DPGO::ManualObjectDranDiagnosticProfile::Auto;
  }
  if (value == "full" || value == "analysis_full" || value == "debug_full") {
    return DPGO::ManualObjectDranDiagnosticProfile::Full;
  }
  if (value == "lightweight" || value == "paper_final" ||
      value == "paper" || value == "final") {
    return DPGO::ManualObjectDranDiagnosticProfile::Lightweight;
  }
  throw std::invalid_argument("Unknown diagnostic profile: " + value);
}

std::string topologyName(DPGO::ManualObjectDranTopology topology) {
  switch (topology) {
    case DPGO::ManualObjectDranTopology::Ring:
      return "ring";
    case DPGO::ManualObjectDranTopology::Complete:
      return "complete";
  }
  return "ring";
}

std::string associationRelationalNullCalibrationName(
    DPGO::AssociationRelationalNullCalibration calibration) {
  switch (calibration) {
    case DPGO::AssociationRelationalNullCalibration::FixedStudentT:
      return "fixed";
    case DPGO::AssociationRelationalNullCalibration::AdaptiveEmpiricalRank:
      return "adaptive";
  }
  return "fixed";
}

std::string communicationPolicyName(
    DPGO::ManualObjectDranCommunicationPolicy policy) {
  switch (policy) {
    case DPGO::ManualObjectDranCommunicationPolicy::All:
      return "all";
    case DPGO::ManualObjectDranCommunicationPolicy::Triggered:
      return "triggered";
  }
  return "triggered";
}

std::string objectInitializationName(
    DPGO::ManualObjectDranObjectInitialization initialization) {
  switch (initialization) {
    case DPGO::ManualObjectDranObjectInitialization::CentralizedChordal:
      return "centralized_chordal";
    case DPGO::ManualObjectDranObjectInitialization::NeighborAverage:
      return "neighbor_average";
    case DPGO::ManualObjectDranObjectInitialization::InputState:
      return "input_state";
  }
  return "centralized_chordal";
}

std::string objectInitializationAnchorModeName(
    DPGO::ManualObjectDranObjectInitializationAnchorMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranObjectInitializationAnchorMode::Constant:
      return "constant";
    case DPGO::ManualObjectDranObjectInitializationAnchorMode::Information:
      return "information";
  }
  return "constant";
}

std::string objectInterfaceModeName(
    DPGO::ManualObjectDranObjectInterfaceMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranObjectInterfaceMode::CopyBaseline:
      return "copy_baseline";
    case DPGO::ManualObjectDranObjectInterfaceMode::ReducedSchurResponse:
      return "reduced_schur_response";
    case DPGO::ManualObjectDranObjectInterfaceMode::
        CompleteGraphCoarseResponse:
      return "complete_graph_coarse_response";
    case DPGO::ManualObjectDranObjectInterfaceMode::
        MatrixFreeObjectPose3Pcg:
      return "matrix_free_objectpose3_pcg";
  }
  return "copy_baseline";
}

std::string objectInterfacePcgPreconditionerName(
    DPGO::ManualObjectDranObjectInterfacePcgPreconditioner preconditioner) {
  switch (preconditioner) {
    case DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::
        InterfaceHessian:
      return "interface_hessian";
    case DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::ExactSchur:
      return "exact_schur";
    case DPGO::ManualObjectDranObjectInterfacePcgPreconditioner::
        PrivateBlockJacobi:
      return "private_block_jacobi";
  }
  return "private_block_jacobi";
}

std::string objectInterfacePcgPrivateSolverName(
    DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver solver) {
  switch (solver) {
    case DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver::SparseLdlt:
      return "sparse_ldlt";
    case DPGO::ManualObjectDranObjectInterfacePcgPrivateSolver::
        BlockTridiagonalLdlt:
      return "block_tridiagonal_ldlt";
  }
  return "sparse_ldlt";
}

std::string objectInterfaceResponseStrategyName(
    DPGO::ManualObjectDranObjectInterfaceResponseStrategy strategy) {
  switch (strategy) {
    case DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Serial:
      return "serial";
    case DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Batch:
      return "batch";
  }
  return "serial";
}

std::string objectInterfaceResponseTriggerName(
    DPGO::ManualObjectDranObjectInterfaceResponseTrigger trigger) {
  switch (trigger) {
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::Periodic:
      return "periodic";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        PredictedDecrease:
      return "predicted_decrease";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        LocalInnovation:
      return "local_innovation";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        AccumulatedInnovation:
      return "accumulated_innovation";
  }
  return "periodic";
}

std::string associationModeName(DPGO::ManualObjectDranAssociationMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranAssociationMode::IdentityOracle:
      return "identity_oracle";
    case DPGO::ManualObjectDranAssociationMode::ClassOnlyHard:
      return "class_only_hard";
    case DPGO::ManualObjectDranAssociationMode::MetadataHard:
      return "metadata_hard";
    case DPGO::ManualObjectDranAssociationMode::GeometryHard:
      return "geometry_hard";
    case DPGO::ManualObjectDranAssociationMode::GraphConsistencyHard:
      return "graph_consistency_hard";
    case DPGO::ManualObjectDranAssociationMode::GeometrySoft:
      return "geometry_soft";
    case DPGO::ManualObjectDranAssociationMode::TransformSoft:
      return "transform_soft";
    case DPGO::ManualObjectDranAssociationMode::FixedFileSoft:
      return "fixed_file_soft";
    case DPGO::ManualObjectDranAssociationMode::ResidualSoft:
      return "residual_soft";
  }
  return "identity_oracle";
}

std::string associationPolicyName(
    DPGO::ManualObjectDranAssociationPolicy policy) {
  switch (policy) {
    case DPGO::ManualObjectDranAssociationPolicy::Manual:
      return "manual";
    case DPGO::ManualObjectDranAssociationPolicy::TwoTimeScaleBilevelV1:
      return "two_time_scale_bilevel_v1";
    case DPGO::ManualObjectDranAssociationPolicy::SingleLoopBilevelV2:
      return "single_loop_bilevel_v2";
    case DPGO::ManualObjectDranAssociationPolicy::SingleLoopRotoBilevelV3:
      return "single_loop_roto_bilevel_v3";
    case DPGO::ManualObjectDranAssociationPolicy::FinalDbaP4203:
      return "final_dba_p4_203";
    case DPGO::ManualObjectDranAssociationPolicy::SharedEdgeDbaV1:
      return "shared_edge_bilevel_v1";
    case DPGO::ManualObjectDranAssociationPolicy::SharedEdgeJointDbaV2:
      return "shared_edge_joint_bilevel_v2";
    case DPGO::ManualObjectDranAssociationPolicy::
        DecentralizedFeedbackUotDbaV1:
      return "decentralized_feedback_uot_bilevel_v1";
  }
  return "manual";
}

std::string sharedLatentModeName(
    DPGO::ManualObjectDranSharedLatentMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranSharedLatentMode::Auto:
      return "auto";
    case DPGO::ManualObjectDranSharedLatentMode::Disabled:
      return "disabled";
    case DPGO::ManualObjectDranSharedLatentMode::ExactFixed:
      return "exact_fixed";
    case DPGO::ManualObjectDranSharedLatentMode::ExactWeighted:
      return "exact_weighted";
    case DPGO::ManualObjectDranSharedLatentMode::KktActiveHybrid:
      return "kkt_active_hybrid";
    case DPGO::ManualObjectDranSharedLatentMode::DirectEdgeDual:
      return "direct_edge_dual";
    case DPGO::ManualObjectDranSharedLatentMode::ExactPoint2Owner:
      return "exact_point2_owner";
    case DPGO::ManualObjectDranSharedLatentMode::ExactPoint2SchurOracle:
      return "exact_point2_schur_oracle";
    case DPGO::ManualObjectDranSharedLatentMode::ExactPoint2OwnerPcg:
      return "exact_point2_owner_pcg";
  }
  return "auto";
}

std::string associationFinalAblationName(
    DPGO::ManualObjectDranFinalAblation ablation) {
  switch (ablation) {
    case DPGO::ManualObjectDranFinalAblation::None:
      return "none";
    case DPGO::ManualObjectDranFinalAblation::StateValidation:
      return "state_validation";
    case DPGO::ManualObjectDranFinalAblation::EdgeConsensus:
      return "edge_consensus";
    case DPGO::ManualObjectDranFinalAblation::ReciprocalSupport:
      return "reciprocal_support";
    case DPGO::ManualObjectDranFinalAblation::PairSupportGate:
      return "pair_support_gate";
    case DPGO::ManualObjectDranFinalAblation::PairSupportContinuation:
      return "pair_support_continuation";
    case DPGO::ManualObjectDranFinalAblation::SharedReliability:
      return "shared_reliability";
    case DPGO::ManualObjectDranFinalAblation::ObservationPredictiveGate:
      return "observation_predictive_gate";
  }
  return "none";
}

std::string diagnosticProfileName(
    DPGO::ManualObjectDranDiagnosticProfile profile) {
  switch (profile) {
    case DPGO::ManualObjectDranDiagnosticProfile::Auto:
      return "auto";
    case DPGO::ManualObjectDranDiagnosticProfile::Full:
      return "full";
    case DPGO::ManualObjectDranDiagnosticProfile::Lightweight:
      return "lightweight";
  }
  return "full";
}

void printUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --data_dir path --num_robots N [options]\n"
            << "Options:\n"
            << "  --num_objects N\n"
            << "  --iters N\n"
            << "  --quiet true|false\n"
            << "  --beta value\n"
            << "  --topology ring|complete\n"
            << "  --ring_hops N\n"
            << "  --topology_file path\n"
            << "  --topology_weight_mode unit|matrix\n"
            << "  --communication_policy all|triggered\n"
            << "  --event_schedule_file path\n"
            << "  --object_initialization centralized_chordal|neighbor_average|input_state\n"
            << "  --object_initialization_anchor_mode constant|information\n"
            << "  --object_initialization_consensus_rounds N\n"
            << "  --object_initialization_observed_anchor_weight value\n"
            << "  --object_initialization_relay_anchor_weight value\n"
            << "  --object_orientation_observed true|false\n"
            << "  --object_communication_representation lifted_pose|compact_pose_float64|compact_pose_float32|point2\n"
            << "  --object_communication_link_incidence true|false\n"
            << "  --association_aware_ted_cci_initialization true|false\n"
            << "  --association_lower_physical_copy_majorization true|false\n"
            << "  --association_lower_physical_copy_colored_bcd true|false\n"
            << "  --object_interface_mode copy_baseline|reduced_schur_response|complete_graph_coarse_response|matrix_free_objectpose3_pcg\n"
            << "  --object_interface_response_strategy serial|batch\n"
            << "  --object_interface_response_trigger periodic|predicted_decrease|local_innovation|accumulated_innovation\n"
            << "  --association_mode identity_oracle|class_only_hard|metadata_hard|geometry_hard|graph_consistency_hard|geometry_soft|transform_soft|fixed_file_soft|residual_soft\n"
            << "  --association_policy manual|two_time_scale_bilevel_v1|single_loop_bilevel_v2|single_loop_roto_bilevel_v3|final_dba_p4_203|shared_edge_bilevel_v1|shared_edge_joint_bilevel_v2|decentralized_feedback_uot_bilevel_v1\n"
            << "  --association_final_ablation none|state_validation|edge_consensus|reciprocal_support|pair_support_gate|pair_support_continuation|shared_reliability|observation_predictive_gate\n"
            << "  --diagnostic_profile auto|full|lightweight\n"
            << "  --diagnostic_summary_period N\n"
            << "  --association_file path\n"
            << "  --association_update_period N\n"
            << "  --association_freeze_upper_variables true|false\n"
            << "  --association_pair_cavity_diagnostic true|false\n"
            << "  --association_cavity_export_leave_pair_states true|false\n"
            << "  --association_local_cavity_feedback_sweep true|false\n"
            << "  --association_component_response_gate true|false\n"
            << "  --association_network_partition_gate true|false\n"
            << "  --association_network_partition_multicolor_feedback true|false\n"
            << "  --association_pair_cavity_exclude_first_robot N\n"
            << "  --association_pair_cavity_exclude_second_robot N\n"
            << "  --association_sinkhorn_iterations N\n"
            << "  --association_max_candidates_per_edge N\n"
            << "  --association_temperature value\n"
            << "  --association_row_capacity value\n"
            << "  --association_column_capacity value\n"
            << "  --association_sinkhorn_max_nonzeros N\n"
            << "  --association_unmatched_cost value\n"
            << "  --association_geometry_scale value\n"
            << "  --association_use_synthetic_metadata true|false\n"
            << "  --association_metadata_num_categories N\n"
            << "  --association_metadata_descriptor_dim N\n"
            << "  --association_metadata_semantic_accuracy value\n"
            << "  --association_metadata_descriptor_noise value\n"
            << "  --association_metadata_shape_noise value\n"
            << "  --association_metadata_repeated_object_ratio value\n"
            << "  --association_semantic_weight value\n"
            << "  --association_shape_weight value\n"
            << "  --association_descriptor_weight value\n"
            << "  --association_require_observed_objects true|false\n"
            << "  --association_use_quotient_metric true|false\n"
            << "  --association_transform_min_inliers N\n"
            << "  --association_transform_max_residual value\n"
            << "  --association_transform_min_score value\n"
            << "  --association_transform_weight_scale value\n"
            << "  --association_transform_max_seed_candidates N\n"
            << "  --association_transform_plane_axes xy|xz|yz\n"
            << "  --association_transform_update_in_lower_fixed_point true|false\n"
            << "  --association_transform_lower_fixed_point_use_reliability true|false\n"
            << "  --association_transform_lower_fixed_point_use_reliability_trust true|false\n"
            << "  --association_transform_use_projected_weights true|false\n"
            << "  --association_transform_use_initial_object_poses true|false\n"
            << "  --association_gauge_synchronization_enabled true|false\n"
            << "  --association_gauge_synchronization_apply_to_states true|false\n"
            << "  --association_gauge_synchronization_apply_once_to_initial_states true|false\n"
            << "  --association_gauge_synchronization_async_gauss_seidel true|false\n"
            << "  --association_gauge_synchronization_max_rounds_per_phase N\n"
            << "  --association_gauge_synchronization_tolerance VALUE\n"
            << "  --association_gauge_covariant_preserve_local_states true|false\n"
            << "  --association_cycle_consistency_enabled true|false\n"
            << "  --association_cycle_consistency_min_direct_weight value\n"
            << "  --association_cycle_consistency_min_composed_weight value\n"
            << "  --association_cycle_consistency_min_two_hop_support N\n"
            << "  --association_cycle_consistency_keep_without_evidence true|false\n"
            << "  --association_cycle_objective_weight value\n"
            << "  --association_cycle_objective_min_edge_mass value\n"
            << "  --association_adaptive_upper_enabled true|false\n"
            << "  --association_adaptive_upper_initial_step value\n"
            << "  --association_adaptive_upper_epsilon value\n"
            << "  --association_adaptive_upper_max_step value\n"
            << "  --association_upper_tracking_enabled true|false\n"
            << "  --association_upper_tracking_consensus_weight value\n"
            << "  --association_upper_tracking_correction_weight value\n"
            << "  --association_upper_tracking_center_rows true|false\n"
            << "  --association_upper_tracking_center_columns true|false\n"
            << "  --association_upper_tracking_project_against_lower_feedback true|false\n"
            << "  --association_upper_tracking_use_lower_stationarity_gate true|false\n"
            << "  --association_upper_tracking_lower_validation_merit_trust_scale value\n"
            << "  --association_state_validation_hypergradient_weight value\n"
            << "  --association_state_validation_merit_trust_scale value\n"
            << "  --association_state_validation_merit_lower_iterations N\n"
            << "  --association_state_validation_merit_use_measurement_cost true|false\n"
            << "  --association_state_validation_merit_gate_full_update true|false\n"
            << "  --association_state_validation_use_edge_transform_confidence true|false\n"
            << "  --association_state_validation_transform_confidence_distance_scale value\n"
            << "  --association_state_validation_trajectory_guard_enabled true|false\n"
            << "  --association_lower_validation_select_best_measurement_state true|false\n"
            << "  --association_lower_validation_measurement_incumbent_prox true|false\n"
            << "  --association_save_iteration_estimate_snapshots true|false\n"
            << "  --exclude_summary_diagnostics_from_solver_time true|false\n"
            << "  --record_object_interface_diagnostics true|false\n"
            << "  --record_detailed_association_diagnostics true|false\n"
            << "  --association_record_upper_response_diagnostics true|false\n"
            << "  --record_predictive_measurement_cost true|false\n"
            << "  --association_pair_validation_trust_enabled true|false\n"
            << "  --association_edge_latent_consensus_weight value\n"
            << "  --association_edge_latent_consensus_use_transform_confidence true|false\n"
            << "  --association_edge_latent_consensus_confidence_distance_scale value\n"
            << "  --association_reciprocal_fresh_support_blend value\n"
            << "  --association_reciprocal_fresh_support_uniqueness_scale value\n"
            << "  --association_reciprocal_fresh_support_use_transform_confidence true|false\n"
            << "  --association_reciprocal_fresh_support_confidence_distance_scale value\n"
            << "  --association_pair_support_gate_power value\n"
            << "  --association_pair_support_gate_min_effective_support value\n"
            << "  --association_pair_support_latent_continuation true|false\n"
            << "  --association_object_reliability_residual_scale value\n"
            << "  --association_object_reliability_min value\n"
            << "  --association_object_reliability_upper_cost_weight value\n"
            << "  --association_object_reliability_use_shared_state true|false\n"
            << "  --association_object_observation_robust_residual_scale value\n"
            << "  --association_object_observation_robust_min_information_scale value\n"
            << "  --association_object_observation_robust_normalize_information true|false\n"
            << "  --association_object_observation_uncertainty_count_prior value\n"
            << "  --association_object_observation_uncertainty_distance_scale value\n"
            << "  --association_object_observation_uncertainty_min_reliability value\n"
            << "  --association_object_observation_uncertainty_normalize_information true|false\n"
            << "  --association_object_observation_track_empirical_information true|false\n"
            << "  --association_backend_feedback_gradient_scale value\n"
            << "  --association_backend_feedback_min_trust value\n"
            << "  --association_backend_response_gain value\n"
            << "  --association_backend_response_damping value\n"
            << "  --association_backend_response_max_reduction_fraction value\n"
            << "  --association_backend_adjoint_gain value\n"
            << "  --association_backend_adjoint_damping value\n"
            << "  --association_backend_adjoint_max_reduction_fraction value\n"
            << "  --association_backend_adjoint_max_increase_fraction value\n"
            << "  --association_backend_coupled_adjoint_gain value\n"
            << "  --association_backend_coupled_adjoint_damping value\n"
            << "  --association_backend_coupled_adjoint_max_reduction_fraction value\n"
            << "  --association_backend_coupled_adjoint_max_increase_fraction value\n"
            << "  --association_backend_adjoint_reduction_metadata_margin value\n"
            << "  --association_backend_adjoint_allow_cost_reduction true|false\n"
            << "  --association_backend_scale_negative_delta_by_transform_confidence true|false\n"
            << "  --association_signed_fresh_support_blend value\n"
            << "  --association_signed_fresh_support_sigmoid_scale value\n"
            << "  --association_signed_fresh_support_state_validation_merit_trust_scale value\n"
            << "  --association_backend_snapshot_period N\n"
            << "  --association_backend_max_effective_ratio value\n"
            << "  --association_backend_stationarity_trust_max_gradient_norm value\n"
            << "  --association_backend_max_active_factors_per_edge N\n"
            << "  --association_temporal_blend value\n"
            << "  --association_temporal_kl_weight value\n"
            << "  --association_temporal_kl_epsilon value\n"
            << "  --association_entropy_regularization_weight value\n"
            << "  --association_renormalize_sparse_weights true|false\n"
            << "  --association_residual_warm_start_use_geometry true|false\n"
            << "  --association_residual_use_transform_alignment true|false\n"
            << "  --association_residual_transform_update_from_weights true|false\n"
            << "  --association_lower_object_copy_use_transform_alignment true|false\n"
            << "  --association_lower_object_copy_direct_enabled true|false\n"
            << "  --association_lower_object_copy_use_observation_implied_target true|false\n"
            << "  --association_lower_trajectory_object_copy_feedback true|false\n"
            << "  --association_lower_trajectory_object_copy_feedback_quadratic_trust true|false\n"
            << "  --association_lower_trajectory_object_copy_feedback_normalize_object_mass true|false\n"
            << "  --association_lower_trajectory_object_copy_feedback_mass_conserving true|false\n"
            << "  --association_lower_trajectory_object_copy_feedback_zero_trust_floor true|false\n"
            << "  --association_lower_shared_latent_fusion true|false\n"
            << "  --association_lower_shared_latent_mode auto|disabled|exact_fixed|exact_weighted|kkt_active_hybrid|direct_edge_dual\n"
            << "  --coupled_point2_schur_oracle_correction true|false\n"
            << "  --coupled_point2_schur_matrix_free_owner_pcg true|false\n"
            << "  --coupled_point2_schur_oracle_start_round N\n"
            << "  --coupled_point2_schur_oracle_period N\n"
            << "  --coupled_point2_schur_oracle_relative_merit_tolerance value\n"
            << "  --integrated_point2_mm true|false\n"
            << "  --integrated_point2_local_amm true|false\n"
            << "  --integrated_point2_common_acceptance true|false\n"
            << "  --integrated_point2_amm_momentum value\n"
            << "  --integrated_point2_adaptive_momentum true|false\n"
            << "  --integrated_point2_adaptive_momentum_min value\n"
            << "  --integrated_point2_adaptive_momentum_max value\n"
            << "  --integrated_point2_adaptive_momentum_ema_decay value\n"
            << "  --integrated_point2_object_wise_adaptive_momentum true|false\n"
            << "  --association_lower_object_copy_robust_residual_scale value\n"
            << "  --association_lower_object_copy_robust_min_weight value\n"
            << "  --association_lower_object_copy_observation_predictive_gate true|false\n"
            << "  --association_lower_object_copy_normalize_by_support true|false\n"
            << "  --association_lower_object_copy_support_churn_gate true|false\n"
            << "  --association_lower_trajectory_translation_prox_weight value\n"
            << "  --association_lower_projection_power value\n"
            << "  --association_lower_projection_hard true|false\n"
            << "  --association_two_time_scale_lower_inner_rounds N\n"
            << "  --association_two_time_scale_lower_inner_iterations N\n"
            << "  --association_two_time_scale_lower_inner_exchange true|false\n"
            << "  --association_two_time_scale_freeze_upper_in_lower_inner true|false\n"
            << "  --association_lower_message_delay_rounds N (network emulation)\n"
            << "  --association_lower_direct_edge_synchronous_shared_only true|false\n"
            << "  --association_lower_direct_edge_preserve_inactive_duals true|false\n"
            << "  --association_lower_direct_edge_fixed_union_model true|false\n"
            << "  --association_lower_direct_edge_exact_proximal_curvature true|false\n"
            << "  --association_lower_direct_edge_proximal_curvature_scale value\n"
            << "  --association_lower_direct_edge_dual_step_size value\n"
            << "  --association_lower_direct_edge_dual_relaxation_residual_threshold value\n"
            << "  --association_lower_direct_edge_spectral_dual_step true|false\n"
            << "  --association_lower_direct_edge_global_spectral_dual_step true|false\n"
            << "  --association_lower_direct_edge_spectral_dual_step_min value\n"
            << "  --association_lower_direct_edge_spectral_dual_step_max value\n"
            << "  --association_lower_direct_edge_spectral_dual_step_ema_decay value\n"
            << "  --association_lower_direct_edge_point2_consensus_projection true|false\n"
            << "  --association_lower_direct_edge_point2_consensus_projection_relaxation value\n"
            << "  --association_lower_direct_edge_point2_consensus_projection_rounds N (0 means all rounds)\n"
            << "  --association_lower_direct_edge_point2_consensus_reset_dual_after_warmup true|false\n"
            << "  --association_lower_direct_edge_adaptive_beta true|false\n"
            << "  --association_lower_direct_edge_beta_min value\n"
            << "  --association_lower_direct_edge_beta_max value\n"
            << "  --association_lower_direct_edge_beta_imbalance_ratio value\n"
            << "  --association_lower_direct_edge_beta_update_factor value\n"
            << "  --association_lower_direct_edge_beta_ema_decay value\n"
            << "  --association_lower_direct_edge_beta_update_period N\n"
            << "  --association_lower_high_confidence_anchor_gain value\n"
            << "  --association_lower_continuous_support_gate_power value\n"
            << "  --association_lower_use_current_residual_reliability true|false\n"
            << "  --association_lower_use_bounded_current_residual_reliability true|false\n"
            << "  --association_lower_use_support_confidence_reliability true|false\n"
            << "  --association_residual_transform_bootstrap true|false\n"
            << "  --association_residual_trust_previous_support true|false\n"
            << "  --association_residual_allow_support_expansion true|false\n"
            << "  --association_sparse_upper_support true|false\n"
            << "  --association_transform_dimension N (0=full dimension, 2=plane)\n"
            << "  --association_transform_update_in_lower_fixed_point true|false\n"
            << "  --association_transform_lower_fixed_point_use_reliability true|false\n"
            << "  --association_transform_lower_fixed_point_use_reliability_trust true|false\n"
            << "  --association_gauge_synchronization_enabled true|false\n"
            << "  --association_gauge_synchronization_apply_to_states true|false\n"
            << "  --association_gauge_synchronization_apply_once_to_initial_states true|false\n"
            << "  --association_gauge_synchronization_async_gauss_seidel true|false\n"
            << "  --association_gauge_synchronization_max_rounds_per_phase N\n"
            << "  --association_gauge_synchronization_tolerance VALUE\n"
            << "  --association_gauge_covariant_preserve_local_states true|false\n"
            << "  --association_relational_geometry_weight value\n"
            << "  --association_relational_geometry_distance_scale value\n"
            << "  --association_relational_geometry_normalize_by_squared_mass true|false\n"
            << "  --association_relational_geometry_center_rows true|false\n"
            << "  --association_relational_null_evidence true|false\n"
            << "  --association_relational_null_calibration fixed|adaptive\n"
            << "  --association_relational_null_anchor_threshold value\n"
            << "  --association_relational_null_minimum_anchors N\n"
            << "  --association_relational_null_student_t_dof value\n"
            << "  --association_roto_translation_alignment_weight value\n"
            << "  --association_roto_translation_alignment_distance_scale value\n"
            << "  --association_roto_translation_hypothesis_top_k N\n"
            << "  --association_roto_translation_hypothesis_angle_bins N\n"
            << "  --association_roto_translation_hypothesis_softmin_temperature value\n"
            << "  --association_lower_validation_weight value\n"
            << "  --association_lower_validation_center_rows true|false\n"
            << "  --association_lower_validation_center_columns true|false\n"
            << "  --association_lower_validation_normalize_active_rms true|false\n"
            << "  --association_lower_validation_target_rms value\n"
            << "  --association_lower_validation_scale_with_upper_cost true|false\n"
            << "  --association_lower_validation_minimum_cost_rms_scale value\n"
            << "  --association_lower_validation_stationarity_trust_max_gradient_norm value\n"
            << "  --association_lower_validation_merit_trust_scale value\n"
            << "  --association_lower_validation_pairwise_weight value\n"
            << "  --association_lower_validation_pairwise_distance_scale value\n"
            << "  --association_lower_validation_pairwise_only true|false\n"
            << "  --association_lower_validation_reference_trust_weight value\n"
            << "  --association_validation_risk_weight value\n"
            << "  --association_validation_risk_target_rms value\n"
            << "  --association_validation_risk_target_rms_unmatched_fraction value\n"
            << "  --association_implicit_validation_adjoint_weight value\n"
            << "  --association_implicit_validation_adjoint_damping value\n"
            << "  --association_implicit_validation_adjoint_max_effective_ratio value\n"
            << "  --association_implicit_validation_adjoint_target_rms value\n"
            << "  --association_implicit_validation_adjoint_target_rms_unmatched_fraction value\n"
            << "  --association_implicit_validation_adjoint_use_coupled_solve true|false\n"
            << "  --association_implicit_validation_adjoint_max_active_objects N\n"
            << "  --association_implicit_validation_adjoint_observation_rhs_weight value\n"
            << "  --association_implicit_validation_adjoint_project_against_backend true|false\n"
            << "  --association_implicit_validation_adjoint_project_against_upper_prefix true|false\n"
            << "  --association_implicit_validation_adjoint_merit_trust_scale value\n"
            << "  --association_coverage_prior_weight value\n"
            << "  --association_coverage_prior_match_row_column true|false\n"
            << "  --association_coverage_prior_normalize_by_reference_mass true|false\n"
            << "  --association_support_sharpness_weight value\n"
            << "  --association_support_sharpness_target_ratio value\n"
            << "  --association_support_sharpness_validation_blend value\n"
            << "  --association_support_sharpness_coverage_blend value\n"
            << "  --association_support_sharpness_project_against_lower_feedback true|false\n"
            << "  --association_structural_expansion_weight value\n"
            << "  --association_structural_expansion_distance_scale value\n"
            << "  --association_structural_expansion_use_proposal_gate true|false\n"
            << "  --association_reference_support_sharpening_weight value\n"
            << "  --association_reference_support_sharpening_target value\n"
            << "  --association_post_update_local_refine true|false\n"
            << "  --association_post_update_local_refine_iterations value\n"
            << "  --association_post_update_local_refine_exchange true|false\n"
            << "  --association_post_update_lower_fixed_point_rounds value\n"
            << "  --association_terminal_lower_rounds value\n"
            << "  --object_interface_pcg_terminal_local_polish_iterations value\n"
            << "  --association_lower_drop_first_attempt true|false\n"
            << "  --association_reference_trust_weight value\n"
            << "  --association_reference_trust_scale_with_upper_cost true|false\n"
            << "  --association_reference_trust_minimum_cost_rms_scale value\n"
            << "  --association_reference_mass_trust_weight value\n"
            << "  --association_reference_mass_trust_scale_with_upper_cost true|false\n"
            << "  --association_reference_mass_trust_minimum_cost_rms_scale value\n"
            << "  --association_normalize_upper_costs true|false\n"
            << "  --association_upper_cost_normalization_target_rms value\n"
            << "  --association_min_consensus_weight value\n"
            << "  --association_min_consensus_margin value\n"
            << "  --association_consensus_weight_power value\n"
            << "  --association_active_min_stable_updates N\n"
            << "  --association_activation_trust_ramp_updates N\n"
            << "  --association_tentative_weight_scale value\n"
            << "  --association_reject_residual_threshold value\n"
            << "  --association_reject_residual_best_margin value\n"
            << "  --association_reject_residual_quantile value\n"
            << "  --association_reject_residual_quantile_scale value\n"
            << "  --association_reject_min_bad_updates N\n"
            << "  --object_interface_response_period N\n"
            << "  --object_interface_min_predicted_decrease value\n"
            << "  --object_interface_min_innovation_score value\n"
            << "  --object_interface_schur_damping value\n"
            << "  --object_interface_pcg_iterations N\n"
            << "  --object_interface_pcg_relative_tolerance value\n"
            << "  --object_interface_pcg_warm_start true|false\n"
            << "  --object_interface_pcg_post_activation_local_iterations N\n"
            << "  --object_interface_pcg_persistent_feasible_response true|false\n"
            << "  --object_interface_pcg_min_translation_spread value\n"
            << "  --object_interface_pcg_min_rotation_spread_deg value\n"
            << "  --object_interface_pcg_min_initial_relative_decrease value\n"
            << "  --object_interface_pcg_min_continuation_relative_decrease value\n"
            << "  --object_interface_step_gain value\n"
            << "  --object_interface_max_objects_per_robot N\n"
            << "  --object_interface_max_private_cols N\n"
            << "  --object_interface_max_block_step_norm value\n"
            << "  --object_interface_require_decrease true|false\n"
            << "  --object_pose_tol value\n"
            << "  --object_max_age N\n"
            << "  --association_heldout_object_observation_modulo N\n"
            << "  --association_heldout_object_observation_remainder N\n"
            << "  --translation_prox_weight value\n"
            << "  --project_to_se_after_local_solve true|false\n"
            << "  --coupled_translation_trust_region true|false\n"
            << "  --translation_elimination_prox_weight value\n"
            << "  --local_solver reduced_rotation|manual_full|full_equiv_hybrid\n"
            << "  --full_solver_outer_iterations N\n"
            << "  --local_max_iterations N\n"
            << "  --local_initial_max_iterations N\n"
            << "  --local_initial_rounds N\n"
            << "  --local_periodic_max_iterations N\n"
            << "  --local_periodic_period N\n"
            << "  --local_final_max_iterations N\n"
            << "  --local_final_rounds N\n"
            << "  --local_max_tcg_iterations N\n"
            << "  --local_grad_norm_tol value\n"
            << "  --trust_region_initial_radius value\n"
            << "  --reduced_rotation_preconditioner none|jacobi|schur_jacobi|cholesky|portfolio\n"
            << "    portfolio currently evaluates none|jacobi only; use schur_jacobi explicitly\n"
            << "  --output_dir path\n"
            << "  --save true|false\n";
}

void writeIterationCsv(const std::filesystem::path &path,
                       const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open CSV output: " + path.string());
  }
  out << "iter,time,"
         "profile_run_setup_seconds,"
         "profile_data_load_seconds,"
         "profile_graph_initialization_seconds,"
         "profile_initial_association_seconds,"
         "profile_initial_association_bootstrap_seconds,"
         "profile_initial_association_state_window_seconds,"
         "profile_initial_gauge_reliability_seconds,"
         "profile_lower_inner_rounds_seconds,"
         "profile_lower_factor_assembly_seconds,"
         "profile_lower_solve_seconds,"
         "profile_reduced_translation_factorization_seconds,"
         "profile_reduced_translation_recovery_seconds,"
         "profile_reduced_hessian_product_seconds,"
         "profile_reduced_objective_seconds,"
         "profile_reduced_gradient_build_seconds,"
         "profile_reduced_translation_factorization_count,"
         "profile_reduced_translation_recovery_count,"
         "profile_reduced_hessian_product_count,"
         "profile_lower_feedback_certificate_seconds,"
         "profile_lower_consensus_refresh_seconds,"
         "profile_association_update_seconds,"
         "profile_association_unprofiled_seconds,"
         "profile_association_weight_update_seconds,"
         "profile_association_weight_update_unprofiled_seconds,"
         "profile_association_local_cost_seconds,"
         "profile_association_state_validation_precompute_seconds,"
         "profile_association_state_validation_cost_seconds,"
         "profile_association_edge_latent_consensus_seconds,"
         "profile_association_reference_mass_trust_seconds,"
         "profile_association_temporal_stats_seconds,"
         "profile_association_support_stats_seconds,"
         "profile_association_transform_refresh_seconds,"
         "profile_association_cycle_consistency_seconds,"
         "profile_association_state_window_seconds,"
         "profile_object_interface_response_seconds,"
         "profile_pair_validation_seconds,"
         "profile_association_roto_translation_seconds,"
         "profile_association_projection_seconds,"
         "profile_association_support_postprocess_seconds,"
         "profile_association_shared_objective_seconds,"
         "profile_association_shared_response_seconds,"
         "profile_association_support_entries,"
         "profile_association_dense_entries,"
         "profile_association_support_density,"
         "profile_association_support_mass_coverage,"
         "profile_association_precost_support_seconds,"
         "profile_association_precost_support_entries,"
         "profile_association_precost_dense_entries,"
         "profile_association_precost_support_density,"
         "profile_association_precost_support_mass_coverage,"
         "profile_object_exchange_seconds,"
         "profile_gauge_reliability_update_seconds,"
         "profile_post_update_refine_seconds,"
         "post_update_lower_rounds_executed,"
         "post_update_lower_robot_solves,"
         "post_update_lower_converged_robots,"
         "post_update_lower_max_normalized_residual,"
         "post_update_lower_max_stationarity_residual,"
         "post_update_lower_max_outer_continuation_residual,"
         "post_update_lower_max_incident_primal_residual,"
         "post_update_lower_max_incident_dual_residual,"
         "profile_summary_factor_assembly_seconds,"
         "profile_summary_association_diagnostics_seconds,"
         "global_cost,measurement_cost,heldout_measurement_cost,"
         "predictive_measurement_cost,consensus_cost,"
         "augmented_cost,gradient,active_consensus_pairs,comm_pose_count,"
         "iter_comm_mb,cumulative_comm_pose_count,cumulative_comm_mb,"
         "known_object_copies,relay_object_copies,observed_object_copies,"
         "relay_only_object_copies,initialized_from_neighbor_object_copies,"
         "stale_object_copies,stale_object_comm_triggers,"
         "association_candidate_count,association_weight_mass,"
         "association_correct_mass,association_false_merge_mass,"
         "association_entropy,association_precision,association_recall,"
         "association_f1,association_false_merge_rate,"
         "association_factor_weight_mass,"
         "association_factor_correct_mass,"
         "association_factor_false_merge_mass,"
         "association_pair_validation_trust_active_pairs,"
         "association_pair_validation_trust_mean,"
         "association_pair_validation_trust_min,"
         "association_pair_validation_risk_rms,"
         "high_confidence_association_candidate_count,"
         "high_confidence_association_weight_mass,"
         "high_confidence_association_correct_mass,"
         "high_confidence_association_false_merge_mass,"
         "high_confidence_association_precision,"
         "high_confidence_association_recall,"
         "high_confidence_association_f1,"
         "high_confidence_association_false_merge_rate,"
         "high_confidence_association_factor_weight_mass,"
         "high_confidence_association_factor_correct_mass,"
         "high_confidence_association_factor_false_merge_mass,"
         "activated_association_candidate_count,"
         "activated_association_weight_mass,"
         "activated_association_correct_mass,"
         "activated_association_false_merge_mass,"
         "activated_association_precision,"
         "activated_association_recall,"
         "activated_association_f1,"
         "activated_association_false_merge_rate,"
         "activated_association_factor_weight_mass,"
         "activated_association_factor_correct_mass,"
         "activated_association_factor_false_merge_mass,"
         "active_association_candidate_count,"
         "active_association_weight_mass,"
         "active_association_correct_mass,"
         "active_association_false_merge_mass,"
         "active_association_precision,"
         "active_association_recall,"
         "active_association_f1,"
         "active_association_false_merge_rate,"
         "active_association_factor_weight_mass,"
         "active_association_factor_correct_mass,"
         "active_association_factor_false_merge_mass,"
         "association_rejected_candidate_count,"
         "association_rejected_weight_mass,"
         "association_backend_adjoint_reduction_candidate_count,"
         "association_backend_adjoint_reduction_allowed_candidate_count,"
         "association_backend_adjoint_reduction_weight_mass,"
         "association_backend_adjoint_reduction_allowed_weight_mass,"
         "association_cycle_checked_entries,"
         "association_cycle_kept_entries,"
         "association_cycle_pruned_entries,"
         "association_local_gradient_norm,"
         "association_cycle_gradient_norm,"
         "association_cycle_to_local_gradient_ratio,"
         "association_cycle_objective_cost,"
         "association_cycle_objective_active_triplets,"
         "association_relational_geometry_gradient_norm,"
         "association_relational_geometry_to_local_gradient_ratio,"
         "association_relational_geometry_objective_cost,"
         "association_actual_relational_geometry_alpha,"
         "association_roto_translation_alignment_gradient_norm,"
         "association_roto_translation_alignment_to_local_gradient_ratio,"
         "association_roto_translation_alignment_objective_cost,"
         "association_actual_roto_translation_alignment_alpha,"
         "association_lower_validation_gradient_norm,"
         "association_lower_validation_to_local_gradient_ratio,"
         "association_lower_validation_objective_cost,"
         "association_actual_lower_validation_alpha,"
         "association_lower_validation_merit_gate,"
         "association_lower_validation_merit_decrease,"
         "association_lower_validation_pairwise_gradient_norm,"
         "association_lower_validation_pairwise_objective_cost,"
         "association_validation_risk_gradient_norm,"
         "association_validation_risk_to_local_gradient_ratio,"
         "association_validation_risk_objective_cost,"
         "association_actual_validation_risk_alpha,"
         "association_implicit_validation_adjoint_gradient_norm,"
         "association_implicit_validation_adjoint_to_local_gradient_ratio,"
         "association_implicit_validation_adjoint_objective_cost,"
         "association_implicit_validation_adjoint_rhs_gradient_norm,"
         "association_actual_implicit_validation_adjoint_alpha,"
         "association_implicit_validation_adjoint_active_object_count,"
         "association_implicit_validation_adjoint_merit_gate,"
         "association_implicit_validation_adjoint_merit_decrease,"
         "association_implicit_validation_adjoint_backend_alignment,"
         "association_implicit_validation_adjoint_upper_prefix_alignment,"
         "association_implicit_validation_adjoint_projected_conflict_norm,"
         "association_implicit_validation_adjoint_upper_prefix_projected_conflict_norm,"
         "association_coverage_prior_gradient_norm,"
         "association_coverage_prior_to_local_gradient_ratio,"
         "association_coverage_prior_objective_cost,"
         "association_actual_coverage_prior_alpha,"
         "association_support_sharpness_gradient_norm,"
         "association_support_sharpness_to_local_gradient_ratio,"
         "association_support_sharpness_objective_cost,"
         "association_actual_support_sharpness_alpha,"
         "association_support_sharpness_validation_gate_mean,"
         "association_support_sharpness_coverage_gate_mean,"
         "association_support_sharpness_lower_feedback_alignment,"
         "association_support_sharpness_projected_conflict_norm,"
         "association_structural_expansion_gradient_norm,"
         "association_structural_expansion_to_local_gradient_ratio,"
         "association_structural_expansion_objective_cost,"
         "association_actual_structural_expansion_alpha,"
         "association_reference_support_sharpening_gradient_norm,"
         "association_reference_support_sharpening_to_local_gradient_ratio,"
         "association_reference_support_sharpening_objective_cost,"
         "association_actual_reference_support_sharpening_alpha,"
         "association_reference_trust_gradient_norm,"
         "association_reference_trust_to_local_gradient_ratio,"
         "association_reference_trust_objective_cost,"
         "association_actual_reference_trust_alpha,"
         "association_backend_gradient_norm,"
         "association_backend_raw_gradient_norm,"
         "association_backend_to_local_gradient_ratio,"
         "association_backend_raw_to_local_gradient_ratio,"
         "association_backend_cap_scale,"
         "association_backend_stationarity_gradient_norm,"
         "association_backend_snapshot_active,"
         "association_backend_stationarity_trusted,"
         "association_upper_gradient_norm,"
         "association_accumulated_upper_gradient_norm,"
         "association_adaptive_eta,"
         "association_upper_tracking_gradient_norm,"
         "association_upper_tracking_correction_norm,"
         "association_upper_tracking_consensus_residual_norm,"
         "association_upper_tracking_local_gradient_delta_norm,"
         "association_upper_tracking_lower_validation_merit_gate,"
         "association_upper_tracking_lower_validation_merit_decrease,"
         "association_state_validation_hypergradient_gradient_norm,"
         "association_state_validation_hypergradient_objective_cost,"
         "association_state_validation_unified_objective_cost,"
         "association_state_validation_merit_gate,"
         "association_state_validation_merit_ratio,"
         "association_state_validation_merit_decrease,"
         "association_state_validation_trajectory_residual_cost,"
         "association_state_validation_pairwise_trajectory_cost,"
         "association_state_validation_transform_confidence_cost,"
         "association_state_validation_transform_confidence_mean,"
         "association_state_validation_object_consistency_cost,"
         "association_state_validation_gauge_stationarity_cost,"
         "association_edge_latent_consensus_gradient_norm,"
         "association_edge_latent_consensus_objective_cost,"
         "association_edge_latent_consensus_residual_norm,"
         "association_edge_latent_consensus_confidence_mean,"
         "association_shared_edge_synchronized_edge_count,"
         "association_shared_edge_pre_projection_disagreement_norm,"
         "association_shared_edge_projection_displacement_norm,"
         "association_shared_edge_candidate_count,"
         "association_shared_edge_objective_scalar_count,"
         "association_shared_edge_payload_bytes,"
         "association_shared_edge_payload_mb,"
         "association_shared_edge_cumulative_mb,"
         "association_shared_edge_cumulative_total_mb,"
         "association_fresh_uot_marginal_residual_norm,"
         "association_fresh_uot_response_mass,"
         "association_upper_fixed_point_residual_norm,"
         "association_lower_feedback_readiness_mean,"
         "association_mann_step_mean,"
         "association_actual_cycle_alpha,"
         "association_actual_backend_alpha,"
         "association_temporal_kl_cost,"
         "association_temporal_kl_gradient_norm,"
         "association_entropy_regularization_cost,"
         "association_entropy_regularization_gradient_norm,"
         "association_churn_l1,"
         "association_churn_frobenius,"
         "association_actual_temporal_alpha,"
         "association_actual_entropy_alpha,"
         "lower_stationarity_gradient_norm,"
         "lower_gauge_marginalized_stationarity_norm,"
         "lower_gauge_component_norm,"
         "lower_gauge_component_ratio,"
         "lower_shared_object_consensus_active_hypotheses,"
         "lower_shared_object_consensus_primal_residual,"
         "lower_shared_object_consensus_dual_residual,"
         "lower_shared_object_consensus_normalized_primal_residual,"
         "lower_shared_object_consensus_normalized_dual_residual,"
         "lower_shared_object_consensus_adaptive_beta_edges,"
         "lower_shared_object_consensus_adaptive_beta_mean,"
         "lower_shared_object_consensus_adaptive_beta_min,"
         "lower_shared_object_consensus_adaptive_beta_max,"
         "lower_shared_object_consensus_adaptive_beta_increases,"
         "lower_shared_object_consensus_adaptive_beta_decreases,"
         "lower_shared_object_consensus_adaptive_midpoint_residual,"
         "lower_shared_object_consensus_global_spectral_step,"
         "lower_shared_object_consensus_global_spectral_contributions,"
         "lower_shared_object_consensus_global_spectral_numerator,"
         "lower_shared_object_consensus_global_spectral_denominator,"
         "lower_shared_object_consensus_max_dual_first_robot,"
         "lower_shared_object_consensus_max_dual_second_robot,"
         "lower_shared_object_consensus_max_dual_first_object,"
         "lower_shared_object_consensus_max_dual_second_object,"
         "lower_shared_object_consensus_max_dual_translation,"
         "lower_shared_object_consensus_max_dual_raw_residual,"
         "lower_shared_object_consensus_max_dual_penalty,"
         "lower_shared_object_consensus_max_dual_physical_multiplier_norm,"
         "lower_shared_object_consensus_target_messages,"
         "lower_shared_object_consensus_delivered_messages,"
         "lower_shared_object_consensus_certificate_messages,"
         "lower_shared_object_consensus_stale_certificates,"
         "lower_shared_object_consensus_current_incident_certificates,"
         "lower_shared_object_consensus_expected_incident_certificates,"
         "lower_shared_object_consensus_outstanding_messages,"
         "lower_shared_object_consensus_dropped_attempts,"
         "lower_shared_object_consensus_retry_attempts,"
         "lower_shared_object_consensus_payload_bytes,"
         "lower_shared_object_consensus_certificate_payload_bytes,"
         "lower_shared_object_consensus_payload_mb,"
         "lower_shared_object_consensus_cumulative_mb,"
         "cumulative_total_comm_mb,"
         "integrated_point2_amm_attempts,"
         "integrated_point2_amm_accepts,"
         "integrated_point2_amm_restarts,"
         "integrated_point2_edge_inertia_attempts,"
         "integrated_point2_edge_inertia_accepts,"
         "integrated_point2_edge_inertia_restarts,"
         "integrated_point2_edge_baseline_merit,"
         "integrated_point2_edge_accelerated_merit,"
         "integrated_point2_owner_momentum_valid_contributions,"
         "integrated_point2_owner_momentum_mean,"
         "integrated_point2_owner_momentum_min,"
         "integrated_point2_owner_momentum_max,"
         "exact_point2_trajectory_conditional_cost_before,"
         "exact_point2_trajectory_conditional_cost_after,"
         "exact_point2_trajectory_accepted_solves,"
         "exact_point2_trajectory_rejected_solves,"
         "exact_point2_trajectory_solver_seconds,"
         "exact_point2_update_seconds,"
         "exact_point2_statistic_messages,"
         "exact_point2_statistic_payload_bytes,"
         "exact_point2_broadcast_messages,"
         "exact_point2_broadcast_payload_bytes,"
         "exact_point2_iteration_payload_bytes,"
         "exact_point2_cumulative_payload_bytes,"
         "exact_point2_max_copy_spread,"
         "exact_point2_rms_copy_spread,"
         "exact_point2_physical_cost_after_trajectory,"
         "exact_point2_physical_cost_after_update,"
         "point2_schur_oracle_active,"
         "point2_schur_oracle_assembly_seconds,"
         "point2_schur_oracle_response_solves,"
         "point2_schur_oracle_max_response_relative_residual,"
         "point2_schur_oracle_symmetry_error,"
         "point2_schur_oracle_spectral_shift,"
         "point2_schur_oracle_linear_relative_residual,"
         "point2_schur_oracle_backtracking_trials,"
         "point2_schur_oracle_accepted_step,"
         "point2_schur_oracle_hypothetical_dense_payload_bytes,"
         "point2_owner_pcg_active,"
         "point2_owner_pcg_iterations,"
         "point2_owner_pcg_relative_residual,"
         "point2_owner_pcg_gradient_payload_bytes,"
         "point2_owner_pcg_direction_payload_bytes,"
         "point2_owner_pcg_product_payload_bytes,"
         "point2_owner_pcg_update_payload_bytes,"
         "point2_owner_pcg_scalar_payload_bytes,"
         "exact_point2_status,"
         "event_robot,event_logical_time,equivalent_sweep,queue_drain,"
         "shared_endpoint_state_queued_messages,"
         "shared_endpoint_state_delivered_messages,"
         "shared_endpoint_state_stale_messages,"
         "shared_endpoint_state_outstanding_messages,"
         "shared_endpoint_state_payload_bytes\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranIterationSummary &row :
       result.iterations) {
    out << row.iter << "," << row.time << ","
        << row.profileRunSetupSeconds << ","
        << row.profileDataLoadSeconds << ","
        << row.profileGraphInitializationSeconds << ","
        << row.profileInitialAssociationSeconds << ","
        << row.profileInitialAssociationBootstrapSeconds << ","
        << row.profileInitialAssociationStateWindowSeconds << ","
        << row.profileInitialGaugeReliabilitySeconds << ","
        << row.profileLowerInnerRoundsSeconds << ","
        << row.profileLowerFactorAssemblySeconds << ","
        << row.profileLowerSolveSeconds << ","
        << row.profileReducedTranslationFactorizationSeconds << ","
        << row.profileReducedTranslationRecoverySeconds << ","
        << row.profileReducedHessianProductSeconds << ","
        << row.profileReducedObjectiveSeconds << ","
        << row.profileReducedGradientBuildSeconds << ","
        << row.profileReducedTranslationFactorizationCount << ","
        << row.profileReducedTranslationRecoveryCount << ","
        << row.profileReducedHessianProductCount << ","
        << row.profileLowerFeedbackCertificateSeconds << ","
        << row.profileLowerConsensusRefreshSeconds << ","
        << row.profileAssociationUpdateSeconds << ","
        << row.profileAssociationUnprofiledSeconds << ","
        << row.profileAssociationWeightUpdateSeconds << ","
        << row.profileAssociationWeightUpdateUnprofiledSeconds << ","
        << row.profileAssociationLocalCostSeconds << ","
        << row.profileAssociationStateValidationPrecomputeSeconds << ","
        << row.profileAssociationStateValidationCostSeconds << ","
        << row.profileAssociationEdgeLatentConsensusSeconds << ","
        << row.profileAssociationReferenceMassTrustSeconds << ","
        << row.profileAssociationTemporalStatsSeconds << ","
        << row.profileAssociationSupportStatsSeconds << ","
        << row.profileAssociationTransformRefreshSeconds << ","
        << row.profileAssociationCycleConsistencySeconds << ","
        << row.profileAssociationStateWindowSeconds << ","
        << row.profileObjectInterfaceResponseSeconds << ","
        << row.profilePairValidationSeconds << ","
        << row.profileAssociationRotoTranslationSeconds << ","
        << row.profileAssociationProjectionSeconds << ","
        << row.profileAssociationSupportPostprocessSeconds << ","
        << row.profileAssociationSharedObjectiveSeconds << ","
        << row.profileAssociationSharedResponseSeconds << ","
        << row.profileAssociationSupportEntries << ","
        << row.profileAssociationDenseEntries << ","
        << row.profileAssociationSupportDensity << ","
        << row.profileAssociationSupportMassCoverage << ","
        << row.profileAssociationPreCostSupportSeconds << ","
        << row.profileAssociationPreCostSupportEntries << ","
        << row.profileAssociationPreCostDenseEntries << ","
        << row.profileAssociationPreCostSupportDensity << ","
        << row.profileAssociationPreCostSupportMassCoverage << ","
        << row.profileObjectExchangeSeconds << ","
        << row.profileGaugeReliabilityUpdateSeconds << ","
        << row.profilePostUpdateRefineSeconds << ","
        << row.postUpdateLowerRoundsExecuted << ","
        << row.postUpdateLowerRobotSolves << ","
        << row.postUpdateLowerConvergedRobots << ","
        << row.postUpdateLowerMaxNormalizedResidual << ","
        << row.postUpdateLowerMaxStationarityResidual << ","
        << row.postUpdateLowerMaxOuterContinuationResidual << ","
        << row.postUpdateLowerMaxIncidentPrimalResidual << ","
        << row.postUpdateLowerMaxIncidentDualResidual << ","
        << row.profileSummaryFactorAssemblySeconds << ","
        << row.profileSummaryAssociationDiagnosticsSeconds << ","
        << row.measurementCost << ","
        << row.measurementCost << "," << row.heldoutMeasurementCost << ","
        << row.predictiveMeasurementCost << "," << row.consensusCost << ","
        << row.measurementCost + row.consensusCost << "," << row.gradient
        << "," << row.activeConsensusPairs << "," << row.commPoseCount << ","
        << row.iterCommMb << "," << row.cumulativeCommPoseCount << ","
        << row.cumulativeCommMb << "," << row.knownObjectCopies << ","
        << row.relayObjectCopies << "," << row.observedObjectCopies << ","
        << row.relayOnlyObjectCopies << ","
        << row.initializedFromNeighborObjectCopies << ","
        << row.staleObjectCopies << ","
        << row.staleObjectCommTriggers << ","
        << row.associationCandidateCount << ","
        << row.associationWeightMass << ","
        << row.associationCorrectMass << ","
        << row.associationFalseMergeMass << ","
        << row.associationEntropy << ","
        << row.associationPrecision << ","
        << row.associationRecall << ","
        << row.associationF1 << ","
        << row.associationFalseMergeRate << ","
        << row.associationFactorWeightMass << ","
        << row.associationFactorCorrectMass << ","
        << row.associationFactorFalseMergeMass << ","
        << row.associationPairValidationTrustActivePairs << ","
        << row.associationPairValidationTrustMean << ","
        << row.associationPairValidationTrustMin << ","
        << row.associationPairValidationRiskRms << ","
        << row.highConfidenceAssociationCandidateCount << ","
        << row.highConfidenceAssociationWeightMass << ","
        << row.highConfidenceAssociationCorrectMass << ","
        << row.highConfidenceAssociationFalseMergeMass << ","
        << row.highConfidenceAssociationPrecision << ","
        << row.highConfidenceAssociationRecall << ","
        << row.highConfidenceAssociationF1 << ","
        << row.highConfidenceAssociationFalseMergeRate << ","
        << row.highConfidenceAssociationFactorWeightMass << ","
        << row.highConfidenceAssociationFactorCorrectMass << ","
        << row.highConfidenceAssociationFactorFalseMergeMass << ","
        << row.activatedAssociationCandidateCount << ","
        << row.activatedAssociationWeightMass << ","
        << row.activatedAssociationCorrectMass << ","
        << row.activatedAssociationFalseMergeMass << ","
        << row.activatedAssociationPrecision << ","
        << row.activatedAssociationRecall << ","
        << row.activatedAssociationF1 << ","
        << row.activatedAssociationFalseMergeRate << ","
        << row.activatedAssociationFactorWeightMass << ","
        << row.activatedAssociationFactorCorrectMass << ","
        << row.activatedAssociationFactorFalseMergeMass << ","
        << row.activeAssociationCandidateCount << ","
        << row.activeAssociationWeightMass << ","
        << row.activeAssociationCorrectMass << ","
        << row.activeAssociationFalseMergeMass << ","
        << row.activeAssociationPrecision << ","
        << row.activeAssociationRecall << ","
        << row.activeAssociationF1 << ","
        << row.activeAssociationFalseMergeRate << ","
        << row.activeAssociationFactorWeightMass << ","
        << row.activeAssociationFactorCorrectMass << ","
        << row.activeAssociationFactorFalseMergeMass << ","
        << row.associationRejectedCandidateCount << ","
        << row.associationRejectedWeightMass << ","
        << row.associationBackendAdjointReductionCandidateCount << ","
        << row.associationBackendAdjointReductionAllowedCandidateCount << ","
        << row.associationBackendAdjointReductionWeightMass << ","
        << row.associationBackendAdjointReductionAllowedWeightMass << ","
        << row.associationCycleCheckedEntries << ","
        << row.associationCycleKeptEntries << ","
        << row.associationCyclePrunedEntries << ","
        << row.associationLocalGradientNorm << ","
        << row.associationCycleGradientNorm << ","
        << row.associationCycleToLocalGradientRatio << ","
        << row.associationCycleObjectiveCost << ","
        << row.associationCycleObjectiveActiveTriplets << ","
        << row.associationRelationalGeometryGradientNorm << ","
        << row.associationRelationalGeometryToLocalGradientRatio << ","
        << row.associationRelationalGeometryObjectiveCost << ","
        << row.associationActualRelationalGeometryAlpha << ","
        << row.associationRotoTranslationAlignmentGradientNorm << ","
        << row.associationRotoTranslationAlignmentToLocalGradientRatio << ","
        << row.associationRotoTranslationAlignmentObjectiveCost << ","
        << row.associationActualRotoTranslationAlignmentAlpha << ","
        << row.associationLowerValidationGradientNorm << ","
        << row.associationLowerValidationToLocalGradientRatio << ","
        << row.associationLowerValidationObjectiveCost << ","
        << row.associationActualLowerValidationAlpha << ","
        << row.associationLowerValidationMeritGate << ","
        << row.associationLowerValidationMeritDecrease << ","
        << row.associationLowerValidationPairwiseGradientNorm << ","
        << row.associationLowerValidationPairwiseObjectiveCost << ","
        << row.associationValidationRiskGradientNorm << ","
        << row.associationValidationRiskToLocalGradientRatio << ","
        << row.associationValidationRiskObjectiveCost << ","
        << row.associationActualValidationRiskAlpha << ","
        << row.associationImplicitValidationAdjointGradientNorm << ","
        << row.associationImplicitValidationAdjointToLocalGradientRatio << ","
        << row.associationImplicitValidationAdjointObjectiveCost << ","
        << row.associationImplicitValidationAdjointRhsGradientNorm << ","
        << row.associationActualImplicitValidationAdjointAlpha << ","
        << row.associationImplicitValidationAdjointActiveObjectCount << ","
        << row.associationImplicitValidationAdjointMeritGate << ","
        << row.associationImplicitValidationAdjointMeritDecrease << ","
        << row.associationImplicitValidationAdjointBackendAlignment << ","
        << row.associationImplicitValidationAdjointUpperPrefixAlignment << ","
        << row.associationImplicitValidationAdjointProjectedConflictNorm << ","
        << row.associationImplicitValidationAdjointUpperPrefixProjectedConflictNorm
        << ","
        << row.associationCoveragePriorGradientNorm << ","
        << row.associationCoveragePriorToLocalGradientRatio << ","
        << row.associationCoveragePriorObjectiveCost << ","
        << row.associationActualCoveragePriorAlpha << ","
        << row.associationSupportSharpnessGradientNorm << ","
        << row.associationSupportSharpnessToLocalGradientRatio << ","
        << row.associationSupportSharpnessObjectiveCost << ","
        << row.associationActualSupportSharpnessAlpha << ","
        << row.associationSupportSharpnessValidationGateMean << ","
        << row.associationSupportSharpnessCoverageGateMean << ","
        << row.associationSupportSharpnessLowerFeedbackAlignment << ","
        << row.associationSupportSharpnessProjectedConflictNorm << ","
        << row.associationStructuralExpansionGradientNorm << ","
        << row.associationStructuralExpansionToLocalGradientRatio << ","
        << row.associationStructuralExpansionObjectiveCost << ","
        << row.associationActualStructuralExpansionAlpha << ","
        << row.associationReferenceSupportSharpeningGradientNorm << ","
        << row.associationReferenceSupportSharpeningToLocalGradientRatio << ","
        << row.associationReferenceSupportSharpeningObjectiveCost << ","
        << row.associationActualReferenceSupportSharpeningAlpha << ","
        << row.associationReferenceTrustGradientNorm << ","
        << row.associationReferenceTrustToLocalGradientRatio << ","
        << row.associationReferenceTrustObjectiveCost << ","
        << row.associationActualReferenceTrustAlpha << ","
        << row.associationBackendGradientNorm << ","
        << row.associationBackendRawGradientNorm << ","
        << row.associationBackendToLocalGradientRatio << ","
        << row.associationBackendRawToLocalGradientRatio << ","
        << row.associationBackendCapScale << ","
        << row.associationBackendStationarityGradientNorm << ","
        << row.associationBackendSnapshotActive << ","
        << row.associationBackendStationarityTrusted << ","
        << row.associationUpperGradientNorm << ","
        << row.associationAccumulatedUpperGradientNorm << ","
        << row.associationAdaptiveEta << ","
        << row.associationUpperTrackingGradientNorm << ","
        << row.associationUpperTrackingCorrectionNorm << ","
        << row.associationUpperTrackingConsensusResidualNorm << ","
        << row.associationUpperTrackingLocalGradientDeltaNorm << ","
        << row.associationUpperTrackingLowerValidationMeritGate << ","
        << row.associationUpperTrackingLowerValidationMeritDecrease << ","
        << row.associationStateValidationHypergradientGradientNorm << ","
        << row.associationStateValidationHypergradientObjectiveCost << ","
        << row.associationStateValidationUnifiedObjectiveCost << ","
        << row.associationStateValidationMeritGate << ","
        << row.associationStateValidationMeritRatio << ","
        << row.associationStateValidationMeritDecrease << ","
        << row.associationStateValidationTrajectoryResidualCost << ","
        << row.associationStateValidationPairwiseTrajectoryCost << ","
        << row.associationStateValidationTransformConfidenceCost << ","
        << row.associationStateValidationTransformConfidenceMean << ","
        << row.associationStateValidationObjectConsistencyCost << ","
        << row.associationStateValidationGaugeStationarityCost << ","
        << row.associationEdgeLatentConsensusGradientNorm << ","
        << row.associationEdgeLatentConsensusObjectiveCost << ","
        << row.associationEdgeLatentConsensusResidualNorm << ","
        << row.associationEdgeLatentConsensusConfidenceMean << ","
        << row.associationSharedEdgeSynchronizedEdgeCount << ","
        << row.associationSharedEdgePreProjectionDisagreementNorm << ","
        << row.associationSharedEdgeProjectionDisplacementNorm << ","
        << row.associationSharedEdgeCandidateCount << ","
        << row.associationSharedEdgeObjectiveScalarCount << ","
        << row.associationSharedEdgePayloadBytes << ","
        << row.associationSharedEdgePayloadMb << ","
        << row.associationSharedEdgeCumulativeMb << ","
        << row.associationSharedEdgeCumulativeTotalMb << ","
        << row.associationFreshUotMarginalResidualNorm << ","
        << row.associationFreshUotResponseMass << ","
        << row.associationUpperFixedPointResidualNorm << ","
        << row.associationLowerFeedbackReadinessMean << ","
        << row.associationMannStepMean << ","
        << row.associationActualCycleAlpha << ","
        << row.associationActualBackendAlpha << ","
        << row.associationTemporalKlCost << ","
        << row.associationTemporalKlGradientNorm << ","
        << row.associationEntropyRegularizationCost << ","
        << row.associationEntropyRegularizationGradientNorm << ","
        << row.associationChurnL1 << ","
        << row.associationChurnFrobenius << ","
        << row.associationActualTemporalAlpha << ","
        << row.associationActualEntropyAlpha << ","
        << row.lowerStationarityGradientNorm << ","
        << row.lowerGaugeMarginalizedStationarityNorm << ","
        << row.lowerGaugeComponentNorm << ","
        << row.lowerGaugeComponentRatio << ","
        << row.lowerSharedObjectConsensusActiveHypotheses << ","
        << row.lowerSharedObjectConsensusPrimalResidual << ","
        << row.lowerSharedObjectConsensusDualResidual << ","
        << row.lowerSharedObjectConsensusNormalizedPrimalResidual << ","
        << row.lowerSharedObjectConsensusNormalizedDualResidual << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaEdges << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaMean << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaMin << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaMax << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaIncreases << ","
        << row.lowerSharedObjectConsensusAdaptiveBetaDecreases << ","
        << row.lowerSharedObjectConsensusAdaptiveMidpointResidual << ","
        << row.lowerSharedObjectConsensusGlobalSpectralStep << ","
        << row.lowerSharedObjectConsensusGlobalSpectralContributions << ","
        << row.lowerSharedObjectConsensusGlobalSpectralNumerator << ","
        << row.lowerSharedObjectConsensusGlobalSpectralDenominator << ","
        << row.lowerSharedObjectConsensusMaxDualFirstRobot << ","
        << row.lowerSharedObjectConsensusMaxDualSecondRobot << ","
        << row.lowerSharedObjectConsensusMaxDualFirstObject << ","
        << row.lowerSharedObjectConsensusMaxDualSecondObject << ","
        << row.lowerSharedObjectConsensusMaxDualTranslation << ","
        << row.lowerSharedObjectConsensusMaxDualRawResidual << ","
        << row.lowerSharedObjectConsensusMaxDualPenalty << ","
        << row.lowerSharedObjectConsensusMaxDualPhysicalMultiplierNorm << ","
        << row.lowerSharedObjectConsensusTargetMessages << ","
        << row.lowerSharedObjectConsensusDeliveredMessages << ","
        << row.lowerSharedObjectConsensusCertificateMessages << ","
        << row.lowerSharedObjectConsensusStaleCertificates << ","
        << row.lowerSharedObjectConsensusCurrentIncidentCertificates << ","
        << row.lowerSharedObjectConsensusExpectedIncidentCertificates << ","
        << row.lowerSharedObjectConsensusOutstandingMessages << ","
        << row.lowerSharedObjectConsensusDroppedAttempts << ","
        << row.lowerSharedObjectConsensusRetryAttempts << ","
        << row.lowerSharedObjectConsensusPayloadBytes << ","
        << row.lowerSharedObjectConsensusCertificatePayloadBytes << ","
        << row.lowerSharedObjectConsensusPayloadMb << ","
        << row.lowerSharedObjectConsensusCumulativeMb << ","
        << row.cumulativeTotalCommMb << ","
        << row.integratedPoint2AmmAttempts << ","
        << row.integratedPoint2AmmAccepts << ","
        << row.integratedPoint2AmmRestarts << ","
        << row.integratedPoint2EdgeInertiaAttempts << ","
        << row.integratedPoint2EdgeInertiaAccepts << ","
        << row.integratedPoint2EdgeInertiaRestarts << ","
        << row.integratedPoint2EdgeBaselineMerit << ","
        << row.integratedPoint2EdgeAcceleratedMerit << ","
        << row.integratedPoint2OwnerMomentumValidContributions << ","
        << row.integratedPoint2OwnerMomentumMean << ","
        << row.integratedPoint2OwnerMomentumMin << ","
        << row.integratedPoint2OwnerMomentumMax << ","
        << row.exactPoint2TrajectoryConditionalCostBefore << ","
        << row.exactPoint2TrajectoryConditionalCostAfter << ","
        << row.exactPoint2TrajectoryAcceptedSolves << ","
        << row.exactPoint2TrajectoryRejectedSolves << ","
        << row.exactPoint2TrajectorySolverSeconds << ","
        << row.exactPoint2UpdateSeconds << ","
        << row.exactPoint2StatisticMessages << ","
        << row.exactPoint2StatisticPayloadBytes << ","
        << row.exactPoint2BroadcastMessages << ","
        << row.exactPoint2BroadcastPayloadBytes << ","
        << row.exactPoint2IterationPayloadBytes << ","
        << row.exactPoint2CumulativePayloadBytes << ","
        << row.exactPoint2MaxCopySpread << ","
        << row.exactPoint2RmsCopySpread << ","
        << row.exactPoint2PhysicalCostAfterTrajectory << ","
        << row.exactPoint2PhysicalCostAfterUpdate << ","
        << row.point2SchurOracleActive << ","
        << row.point2SchurOracleAssemblySeconds << ","
        << row.point2SchurOracleResponseSolves << ","
        << row.point2SchurOracleMaxResponseRelativeResidual << ","
        << row.point2SchurOracleSymmetryError << ","
        << row.point2SchurOracleSpectralShift << ","
        << row.point2SchurOracleLinearRelativeResidual << ","
        << row.point2SchurOracleBacktrackingTrials << ","
        << row.point2SchurOracleAcceptedStep << ","
        << row.point2SchurOracleHypotheticalDensePayloadBytes << ","
        << row.point2OwnerPcgActive << ","
        << row.point2OwnerPcgIterations << ","
        << row.point2OwnerPcgRelativeResidual << ","
        << row.point2OwnerPcgGradientPayloadBytes << ","
        << row.point2OwnerPcgDirectionPayloadBytes << ","
        << row.point2OwnerPcgProductPayloadBytes << ","
        << row.point2OwnerPcgUpdatePayloadBytes << ","
        << row.point2OwnerPcgScalarPayloadBytes << ","
        << row.exactPoint2Status << "," << row.eventRobot << ","
        << row.eventLogicalTime << "," << row.equivalentSweep << ","
        << (row.queueDrain ? 1 : 0) << ","
        << row.sharedEndpointStateQueuedMessages << ","
        << row.sharedEndpointStateDeliveredMessages << ","
        << row.sharedEndpointStateStaleMessages << ","
        << row.sharedEndpointStateOutstandingMessages << ","
        << row.sharedEndpointStatePayloadBytes << "\n";
  }
}

void writeSelectionSummaryCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open selection summary CSV output: " +
                             path.string());
  }
  out << "lower_validation_selected_best_measurement_state,"
         "selected_iter,evaluated_iter,"
         "selected_iteration_count,evaluated_iteration_count,"
         "selected_cumulative_comm_pose_count,"
         "evaluated_cumulative_comm_pose_count,"
         "selected_cumulative_comm_mb,evaluated_cumulative_comm_mb,"
         "selected_measurement_cost,evaluated_measurement_cost\n";
  out << std::setprecision(20)
      << (result.lowerValidationSelectedBestMeasurementState ? "true"
                                                             : "false")
      << "," << result.selectedIteration
      << "," << result.evaluatedIteration
      << "," << result.selectedIterationCount
      << "," << result.evaluatedIterationCount
      << "," << result.selectedCumulativeCommPoseCount
      << "," << result.evaluatedCumulativeCommPoseCount
      << "," << result.selectedCumulativeCommMb
      << "," << result.evaluatedCumulativeCommMb
      << "," << result.selectedMeasurementCost
      << "," << result.evaluatedMeasurementCost << "\n";
}

void writeTrackEmpiricalInformationSummaryCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open track empirical information summary CSV output: " +
        path.string());
  }
  const DPGO::ManualObjectDranTrackEmpiricalInformationSummary &summary =
      result.trackEmpiricalInformationSummary;
  out << "enabled,track_count,observation_count,singleton_track_count,"
         "overdispersed_track_count,singleton_track_fraction,"
         "overdispersed_track_fraction,scale_min,scale_median,scale_mean,"
         "scale_p90,scale_max,effective_sample_ratio_median,"
         "effective_sample_ratio_mean\n";
  out << std::setprecision(20) << (summary.enabled ? "true" : "false")
      << "," << summary.trackCount << "," << summary.observationCount << ","
      << summary.singletonTrackCount << ","
      << summary.overdispersedTrackCount << ","
      << summary.singletonTrackFraction << ","
      << summary.overdispersedTrackFraction << "," << summary.scaleMin << ","
      << summary.scaleMedian << "," << summary.scaleMean << ","
      << summary.scaleP90 << "," << summary.scaleMax << ","
      << summary.effectiveSampleRatioMedian << ","
      << summary.effectiveSampleRatioMean << "\n";
}

void writeObjectInterfaceResponseCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open object interface response CSV output: " +
        path.string());
  }
  out << "iter,scheduled,active_robots,candidate_object_blocks,"
         "accepted_object_blocks,rejected_object_blocks,"
         "batch_response_robots,accepted_robot_batches,"
         "rejected_robot_batches,trigger_candidate_robots,"
         "triggered_robots,skipped_trigger_robots,model_cost_before,"
         "model_cost_after,interface_innovation_score,"
         "accumulated_innovation_score,predicted_model_decrease,"
         "accepted_model_decrease,step_norm,pcg_iterations,"
         "pcg_relative_residual,communication_payload_bytes,"
         "communication_messages,copy_translation_rms,"
         "copy_rotation_rms_deg,model_assembly_seconds,"
         "private_factorization_seconds,private_gradient_solve_seconds,"
         "schur_apply_seconds,private_recovery_seconds,"
         "block_tridiagonal_fallbacks,private_hessian_blocks,"
         "private_interface_coupling_blocks,"
         "private_interface_dense_equivalent_bytes,"
         "private_interface_stored_bytes,"
         "sparse_compatibility_materializations,"
         "sparse_structure_fallbacks,status\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranObjectInterfaceResponseSummary &row :
       result.objectInterfaceResponseSummaries) {
    out << row.iter << "," << (row.scheduled ? 1 : 0) << ","
        << row.activeRobots << ","
        << row.candidateObjectBlocks << "," << row.acceptedObjectBlocks
        << "," << row.rejectedObjectBlocks << ","
        << row.batchResponseRobots << "," << row.acceptedRobotBatches
        << "," << row.rejectedRobotBatches << ","
        << row.triggerCandidateRobots << "," << row.triggeredRobots
        << "," << row.skippedTriggerRobots << "," << row.modelCostBefore
        << "," << row.modelCostAfter << "," << row.interfaceInnovationScore
        << "," << row.accumulatedInnovationScore
        << "," << row.predictedModelDecrease
        << "," << row.acceptedModelDecrease
        << "," << row.stepNorm << "," << row.pcgIterations
        << "," << row.pcgRelativeResidual
        << "," << row.communicationPayloadBytes
        << "," << row.communicationMessages
        << "," << row.copyTranslationRms
        << "," << row.copyRotationRmsDegrees
        << "," << row.modelAssemblySeconds
        << "," << row.privateFactorizationSeconds
        << "," << row.privateGradientSolveSeconds
        << "," << row.schurApplySeconds
        << "," << row.privateRecoverySeconds
        << "," << row.blockTridiagonalFallbacks
        << "," << row.privateHessianBlocks
        << "," << row.privateInterfaceCouplingBlocks
        << "," << row.privateInterfaceDenseEquivalentBytes
        << "," << row.privateInterfaceStoredBytes
        << "," << row.sparseCompatibilityMaterializations
        << "," << row.sparseStructureFallbacks
        << "," << row.status << "\n";
  }
}

void writeObjectInterfaceCsv(const std::filesystem::path &path,
                             const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open object interface CSV output: " +
                             path.string());
  }
  out << "iter,robot,object_id,observed,relay_only,age,residual_norm,"
         "gradient_norm,stiffness,neighbor_disagreement,selected_for_send\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranObjectInterfaceDiagnostic &row :
       result.objectInterfaceDiagnostics) {
    out << row.iter << "," << row.robotId << "," << row.objectId << ","
        << (row.observed ? 1 : 0) << "," << (row.relayOnly ? 1 : 0)
        << "," << row.age << "," << row.residualNorm << ","
        << row.gradientNorm << "," << row.stiffness << ","
        << row.neighborDisagreement << ","
        << (row.selectedForSend ? 1 : 0) << "\n";
  }
}

void writeFinalAssociationWeightsCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open final association CSV output: " +
                             path.string());
  }
  out << "receiver_robot,sender_robot,receiver_object,sender_object,weight\n";
  out << std::setprecision(20);
  for (std::size_t receiver = 0;
       receiver < result.finalAssociationWeights.size(); ++receiver) {
    const auto &senderWeights = result.finalAssociationWeights[receiver];
    for (std::size_t sender = 0; sender < senderWeights.size(); ++sender) {
      const DPGO::Matrix &pi = senderWeights[sender];
      for (int receiverObject = 0; receiverObject < pi.rows();
           ++receiverObject) {
        for (int senderObject = 0; senderObject < pi.cols(); ++senderObject) {
          const double weight = pi(receiverObject, senderObject);
          if (weight <= 0.0) {
            continue;
          }
          out << receiver << "," << sender << "," << receiverObject << ","
              << senderObject << "," << weight << "\n";
        }
      }
    }
  }
}

void writeFinalAssociationUpperResponseCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open final upper response CSV output: " + path.string());
  }
  out << "first_robot,second_robot,first_object,second_object,fixed_cost,"
         "current_cost,feedback_cost,raw_score,response_weight,"
         "proposal_weight,row_response_mass,column_response_mass\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranAssociationUpperResponseDiagnostic &row :
       result.finalAssociationUpperResponseDiagnostics) {
    out << row.firstRobot << "," << row.secondRobot << "," << row.row << ","
        << row.col << "," << row.fixedCost << "," << row.currentCost << ","
        << row.feedbackCost << "," << row.rawScore << ","
        << row.responseWeight << "," << row.proposalWeight << ","
        << row.rowResponseMass << "," << row.columnResponseMass << "\n";
  }
}

void writeFinalAssociationTransformsCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open final transform CSV output: " +
                             path.string());
  }
  out << "receiver_robot,sender_robot,valid,translation_rmse,"
         "marginal_translation_rmse,"
         "synchronization_residual,reference_synchronization_residual,"
         "rows,cols";
  for (std::size_t idx = 0; idx < 12u; ++idx) {
    out << ",value_" << idx;
  }
  out << "\n";
  out << std::setprecision(20);
  for (std::size_t receiver = 0;
       receiver < result.finalAssociationTransforms.size(); ++receiver) {
    const auto &senderTransforms =
        result.finalAssociationTransforms[receiver];
    for (std::size_t sender = 0; sender < senderTransforms.size(); ++sender) {
      const DPGO::ManualObjectDranAssociationTransform &entry =
          senderTransforms[sender];
      const DPGO::Matrix &transform = entry.transform;
      out << receiver << "," << sender << "," << (entry.valid ? 1 : 0)
          << "," << entry.translationRmse << ","
          << entry.marginalTranslationRmse << ","
          << entry.synchronizationResidual << ","
          << entry.referenceSynchronizationResidual << ","
          << transform.rows() << "," << transform.cols();
      std::size_t written = 0;
      for (int row = 0; row < transform.rows(); ++row) {
        for (int col = 0; col < transform.cols(); ++col) {
          if (written < 12u) {
            out << "," << transform(row, col);
            ++written;
          }
        }
      }
      while (written < 12u) {
        out << ",";
        ++written;
      }
      out << "\n";
    }
  }
}

void writeFinalRobotGaugesCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result,
    const std::string &planeAxes) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open final robot gauge CSV output: " +
                             path.string());
  }
  out << "robot,apply_to_estimates,valid,rows,cols,plane_axes";
  for (std::size_t index = 0; index < 12u; ++index) {
    out << ",value_" << index;
  }
  out << "\n";
  out << std::setprecision(20);
  for (std::size_t robotId = 0;
       robotId < result.finalRobotGauges.size(); ++robotId) {
    const DPGO::Matrix &gauge = result.finalRobotGauges[robotId];
    const bool valid =
        (gauge.rows() == 2 || gauge.rows() == 3) &&
        gauge.cols() == gauge.rows() + 1 && gauge.allFinite();
    out << robotId << ","
        << (result.finalEstimatesAreRobotLocal ? 1 : 0) << ","
        << (valid ? 1 : 0) << "," << gauge.rows() << "," << gauge.cols()
        << "," << planeAxes;
    std::size_t written = 0;
    for (int row = 0; row < gauge.rows(); ++row) {
      for (int col = 0; col < gauge.cols(); ++col) {
        if (written < 12u) {
          out << "," << gauge(row, col);
          ++written;
        }
      }
    }
    while (written < 12u) {
      out << ",";
      ++written;
    }
    out << "\n";
  }
}

void writeFinalPairCavityCovarianceCsv(
    const std::filesystem::path &path,
    const std::vector<DPGO::ManualObjectDranPairCavityCovarianceDiagnostic>
        &diagnostics) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open pair-cavity covariance CSV output: " +
        path.string());
  }
  out << "robot,valid,status,row_object,column_object,covariance\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranPairCavityCovarianceDiagnostic &entry :
       diagnostics) {
    if (!entry.valid || entry.translationCovariance.rows() == 0) {
      out << entry.robotId << "," << (entry.valid ? 1 : 0) << ","
          << entry.status << ",,,\n";
      continue;
    }
    for (int row = 0; row < entry.translationCovariance.rows(); ++row) {
      for (int col = 0; col < entry.translationCovariance.cols(); ++col) {
        out << entry.robotId << ",1," << entry.status << ","
            << entry.objectIds[static_cast<std::size_t>(row)] << ","
            << entry.objectIds[static_cast<std::size_t>(col)] << ","
            << entry.translationCovariance(row, col) << "\n";
      }
    }
  }
}

void writeFinalPairCavityMeansCsv(
    const std::filesystem::path &path,
    const std::vector<DPGO::ManualObjectDranPairCavityCovarianceDiagnostic>
        &diagnostics) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open pair-cavity means CSV output: " + path.string());
  }
  out << "robot,valid,status,object_id,x,y,z,response_norm\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranPairCavityCovarianceDiagnostic &entry :
       diagnostics) {
    if (!entry.valid || entry.conditionalTranslationMeans.cols() == 0) {
      out << entry.robotId << "," << (entry.valid ? 1 : 0) << ","
          << entry.status << ",,,,,\n";
      continue;
    }
    for (int column = 0;
         column < entry.conditionalTranslationMeans.cols(); ++column) {
      out << entry.robotId << ",1," << entry.status << ","
          << entry.objectIds[static_cast<std::size_t>(column)];
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        out << ",";
        if (coordinate < entry.conditionalTranslationMeans.rows()) {
          out << entry.conditionalTranslationMeans(coordinate, column);
        }
      }
      out << "," << entry.conditionalTranslationResponseNorm << "\n";
    }
  }
}

void writeCavityFeedbackSummaryCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open cavity feedback summary CSV output: " +
        path.string());
  }
  out << "first_robot,second_robot,valid,status,accepted,"
         "pair_score_accepted,response_gate_evaluated,"
         "response_gate_accepted,score_before,score_after_candidate,"
         "response_predicted_physical_delta,"
         "response_numerical_error_bound,active_cardinality,"
         "changed_edge_count,component_count,candidate_evaluation_count,"
         "payload_bytes,response_payload_bytes,compute_seconds\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranCavityFeedbackSummary &row :
       result.cavityFeedbackSummaries) {
    out << row.firstRobot << "," << row.secondRobot << ","
        << (row.valid ? 1 : 0) << "," << row.status << ","
        << (row.accepted ? 1 : 0) << ","
        << (row.pairScoreAccepted ? 1 : 0) << ","
        << (row.responseGateEvaluated ? 1 : 0) << ","
        << (row.responseGateAccepted ? 1 : 0) << "," << row.scoreBefore
        << "," << row.scoreAfterCandidate << ","
        << row.responsePredictedPhysicalDelta << ","
        << row.responseNumericalErrorBound << "," << row.activeCardinality
        << "," << row.changedEdgeCount << "," << row.componentCount << ","
        << row.candidateEvaluationCount << "," << row.payloadBytes << ","
        << row.responsePayloadBytes << "," << row.computeSeconds << "\n";
  }
}

void writeNetworkPartitionZoneSummaryCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open network partition zone summary CSV output: " +
        path.string());
  }
  out << "sweep,zone_id,color,lower_state_epoch,"
         "lower_feedback_refresh_after_color,valid,partition_feasible,"
         "response_accepted,committed,status,node_count,robot_count,"
         "removed_edge_count,"
         "added_edge_count,predicted_physical_delta,numerical_error_bound,"
         "response_payload_bytes,compute_seconds\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranNetworkPartitionZoneSummary &row :
       result.networkPartitionZoneSummaries) {
    out << row.sweep << "," << row.zoneId << "," << row.color << ","
        << row.lowerStateEpoch << ","
        << (row.lowerFeedbackRefreshAfterColor ? 1 : 0) << ","
        << (row.valid ? 1 : 0) << ","
        << (row.partitionFeasible ? 1 : 0) << ","
        << (row.responseAccepted ? 1 : 0) << ","
        << (row.committed ? 1 : 0) << "," << row.status << ","
        << row.nodeCount << "," << row.robotCount << ","
        << row.removedEdgeCount << "," << row.addedEdgeCount << ","
        << row.predictedPhysicalDelta << ","
        << row.numericalErrorBound << "," << row.responsePayloadBytes
        << "," << row.computeSeconds << "\n";
  }
}

void writeNetworkPartitionProtocolSummaryCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open network partition protocol summary CSV output: " +
        path.string());
  }
  out << "sweep,complete,status,zone_count,label_message_count,"
         "label_payload_bytes,feasibility_reduction_message_count,"
         "feasibility_reduction_payload_bytes,"
         "response_reduction_message_count,response_reduction_payload_bytes,"
         "decision_broadcast_message_count,decision_broadcast_payload_bytes\n";
  for (const DPGO::ManualObjectDranNetworkPartitionProtocolSummary &row :
       result.networkPartitionProtocolSummaries) {
    out << row.sweep << "," << (row.complete ? 1 : 0) << "," << row.status
        << "," << row.zoneCount << "," << row.labelMessageCount << ","
        << row.labelPayloadBytes << ","
        << row.feasibilityReductionMessageCount << ","
        << row.feasibilityReductionPayloadBytes << ","
        << row.responseReductionMessageCount << ","
        << row.responseReductionPayloadBytes << ","
        << row.decisionBroadcastMessageCount << ","
        << row.decisionBroadcastPayloadBytes << "\n";
  }
}

void writeFinalAssociationEdgeStatesCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open final edge-state CSV output: " +
                             path.string());
  }
  out << "receiver_robot,sender_robot,direct_valid,network_valid,"
         "direct_fit_translation_rmse,normalized_disagreement,reliability";
  for (std::size_t idx = 0; idx < 12u; ++idx) {
    out << ",direct_value_" << idx;
  }
  for (std::size_t idx = 0; idx < 12u; ++idx) {
    out << ",network_value_" << idx;
  }
  for (std::size_t idx = 0; idx < 36u; ++idx) {
    out << ",information_value_" << idx;
  }
  out << "\n";
  out << std::setprecision(20);
  const auto writeMatrix = [&out](const DPGO::Matrix &matrix,
                                  std::size_t count) {
    std::size_t written = 0u;
    for (int row = 0; row < matrix.rows() && written < count; ++row) {
      for (int col = 0; col < matrix.cols() && written < count; ++col) {
        out << "," << matrix(row, col);
        ++written;
      }
    }
    while (written < count) {
      out << ",";
      ++written;
    }
  };
  for (std::size_t receiver = 0;
       receiver < result.finalAssociationEdgeStates.size(); ++receiver) {
    for (std::size_t sender = 0;
         sender < result.finalAssociationEdgeStates[receiver].size();
         ++sender) {
      const DPGO::GaugeCovariantAssociationEdgeState &edge =
          result.finalAssociationEdgeStates[receiver][sender];
      out << receiver << "," << sender << ","
          << (edge.direct.valid ? 1 : 0) << ","
          << (edge.network.valid ? 1 : 0) << ","
          << edge.direct.fitTranslationRmse << ","
          << edge.normalizedDisagreement << "," << edge.reliability;
      writeMatrix(edge.direct.transform, 12u);
      writeMatrix(edge.network.transform, 12u);
      writeMatrix(edge.direct.information, 36u);
      out << "\n";
    }
  }
}

void writeObjectAwareEstimateStates(
    const std::filesystem::path &path,
    const DPGO::ObjectPGODirectoryData &dataset,
    const std::vector<DPGO::Matrix> &estimates) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open estimate output: " +
                             path.string());
  }
  if (dataset.dimension != 3) {
    throw std::runtime_error(
        "manual_object_dran object_poses.txt output currently expects SE(3)");
  }
  out << std::setprecision(20);
  out << "type robot_id local_vertex_id object_id x y z qx qy qz qw\n";
  for (std::size_t robotId = 0; robotId < estimates.size(); ++robotId) {
    const DPGO::Matrix &state = estimates[robotId];
    const DPGO::ObjectPGORobotData &robot = dataset.robots[robotId];
    const unsigned d = static_cast<unsigned>(state.rows());
    const unsigned blockWidth = d + 1;
    for (std::size_t localId = 0; localId < robot.trajectoryVertexIds.size();
         ++localId) {
      const Eigen::Matrix3d R = DPGO::projectToRotationGroup(
          state.block(0, localId * blockWidth, d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      const DPGO::Matrix t =
          state.block(0, localId * blockWidth + d, d, 1);
      out << "trajectory " << robotId << ' '
          << robot.trajectoryVertexIds[localId] << " -1 " << t(0) << ' '
          << t(1) << ' ' << t(2) << ' ' << q.x() << ' ' << q.y() << ' '
          << q.z() << ' ' << q.w() << '\n';
    }
    for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      const std::size_t poseIndex = robot.trajectoryVertexIds.size() + objectId;
      const std::size_t localVertexId =
          objectId < robot.objectLocalVertexIds.size()
              ? robot.objectLocalVertexIds[objectId]
              : robot.objectStart + objectId;
      const Eigen::Matrix3d R = DPGO::projectToRotationGroup(
          state.block(0, poseIndex * blockWidth, d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      const DPGO::Matrix t =
          state.block(0, poseIndex * blockWidth + d, d, 1);
      out << "object " << robotId << ' ' << localVertexId << ' ' << objectId
          << ' ' << t(0) << ' ' << t(1) << ' ' << t(2) << ' ' << q.x()
          << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
  }
}

void writeObjectAwareEstimates(const std::filesystem::path &path,
                               const DPGO::ObjectPGODirectoryData &dataset,
                               const DPGO::ManualObjectDranRunResult &result) {
  writeObjectAwareEstimateStates(path, dataset, result.finalEstimates);
}

void writeIterationEstimateSnapshots(
    const std::filesystem::path &root,
    const DPGO::ObjectPGODirectoryData &dataset,
    const DPGO::ManualObjectDranRunResult &result) {
  const std::size_t count = std::min(result.iterations.size(),
                                    result.iterationEstimateSnapshots.size());
  for (std::size_t index = 0; index < count; ++index) {
    std::ostringstream dirname;
    dirname << "iter_" << std::setw(4) << std::setfill('0')
            << result.iterations[index].iter;
    const std::filesystem::path iterDir = root / dirname.str();
    std::filesystem::create_directories(iterDir);
    writeObjectAwareEstimateStates(iterDir / "object_poses.txt", dataset,
                                   result.iterationEstimateSnapshots[index]);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string dataDir;
  std::string outputDir = ".";
  bool save = true;
  bool quiet = false;
  DPGO::ManualObjectDranOptions options;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto requireValue = [&](const std::string &name) -> std::string {
        if (i + 1 >= argc) {
          throw std::invalid_argument("Missing value for " + name);
        }
        return argv[++i];
      };

      if (arg == "--help" || arg == "-h") {
        printUsage(argv[0]);
        return 0;
      } else if (arg == "--data_dir" || arg == "--dataset") {
        dataDir = requireValue(arg);
      } else if (arg == "--num_robots" || arg == "--num_nodes") {
        options.numRobots =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--num_objects") {
        options.numObjects =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--iters") {
        options.maxIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--quiet") {
        quiet = parseBool(requireValue(arg));
      } else if (arg == "--beta") {
        options.beta = std::stod(requireValue(arg));
      } else if (arg == "--topology") {
        options.topology = parseTopology(requireValue(arg));
      } else if (arg == "--ring_hops") {
        options.ringHops =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--topology_file") {
        options.topologyFile = requireValue(arg);
      } else if (arg == "--topology_weight_mode") {
        options.topologyWeightMode = requireValue(arg);
      } else if (arg == "--communication_policy") {
        options.communicationPolicy =
            parseCommunicationPolicy(requireValue(arg));
      } else if (arg == "--event_schedule_file") {
        options.eventScheduleFile = requireValue(arg);
      } else if (arg == "--object_initialization") {
        options.objectInitialization =
            parseObjectInitialization(requireValue(arg));
      } else if (arg == "--object_initialization_anchor_mode") {
        options.objectInitializationAnchorMode =
            parseObjectInitializationAnchorMode(requireValue(arg));
      } else if (arg == "--object_initialization_consensus_rounds") {
        options.objectInitializationConsensusRounds =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_initialization_observed_anchor_weight") {
        options.objectInitializationObservedAnchorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--object_initialization_relay_anchor_weight") {
        options.objectInitializationRelayAnchorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--object_orientation_observed") {
        options.objectOrientationObserved =
            parseBool(requireValue(arg));
      } else if (arg == "--object_communication_representation") {
        options.objectCommunicationRepresentation =
            parseObjectCommunicationRepresentation(requireValue(arg));
      } else if (arg == "--object_communication_link_incidence") {
        options.objectCommunicationLinkIncidence =
            parseBool(requireValue(arg));
      } else if (arg == "--association_aware_ted_cci_initialization") {
        options.associationAwareTedCciInitialization =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_physical_copy_majorization") {
        options.associationLowerPhysicalCopyMajorization =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_physical_copy_colored_bcd") {
        options.associationLowerPhysicalCopyColoredBcd =
            parseBool(requireValue(arg));
      } else if (arg == "--object_interface_mode") {
        options.objectInterfaceMode =
            parseObjectInterfaceMode(requireValue(arg));
      } else if (arg == "--object_interface_response_strategy") {
        options.objectInterfaceResponseStrategy =
            parseObjectInterfaceResponseStrategy(requireValue(arg));
      } else if (arg == "--object_interface_response_trigger") {
        options.objectInterfaceResponseTrigger =
            parseObjectInterfaceResponseTrigger(requireValue(arg));
      } else if (arg == "--association_mode") {
        options.associationMode = parseAssociationMode(requireValue(arg));
      } else if (arg == "--association_policy") {
        options.associationPolicy = parseAssociationPolicy(requireValue(arg));
      } else if (arg == "--association_final_ablation") {
        options.associationFinalAblation =
            parseAssociationFinalAblation(requireValue(arg));
      } else if (arg == "--diagnostic_profile") {
        options.diagnosticProfile = parseDiagnosticProfile(requireValue(arg));
      } else if (arg == "--diagnostic_summary_period") {
        options.diagnosticSummaryPeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_file") {
        options.associationFile = requireValue(arg);
      } else if (arg == "--association_update_period") {
        options.associationUpdatePeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_freeze_upper_variables") {
        options.associationFreezeUpperVariables =
            parseBool(requireValue(arg));
      } else if (arg == "--association_pair_cavity_diagnostic") {
        options.associationPairCavityDiagnosticEnabled =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_cavity_export_leave_pair_states") {
        options.associationCavityExportLeavePairStates =
            parseBool(requireValue(arg));
      } else if (arg == "--association_local_cavity_feedback_sweep") {
        options.associationLocalCavityFeedbackSweepEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_component_response_gate") {
        options.associationComponentResponseGateEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_network_partition_gate") {
        options.associationNetworkPartitionGateEnabled =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_network_partition_multicolor_feedback") {
        options.associationNetworkPartitionMulticolorFeedbackEnabled =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_pair_cavity_exclude_first_robot") {
        options.associationPairCavityExcludeFirstRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg ==
                 "--association_pair_cavity_exclude_second_robot") {
        options.associationPairCavityExcludeSecondRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_sinkhorn_iterations") {
        options.associationSinkhornIterations =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_max_candidates_per_edge") {
        options.associationMaxCandidatesPerEdge =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_temperature") {
        options.associationTemperature = std::stod(requireValue(arg));
      } else if (arg == "--association_row_capacity") {
        options.associationRowCapacity = std::stod(requireValue(arg));
      } else if (arg == "--association_column_capacity") {
        options.associationColumnCapacity = std::stod(requireValue(arg));
      } else if (arg == "--association_sinkhorn_max_nonzeros") {
        options.associationSinkhornMaxNonzeros =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_unmatched_cost") {
        options.associationUnmatchedCost = std::stod(requireValue(arg));
      } else if (arg == "--association_geometry_scale") {
        options.associationGeometryScale = std::stod(requireValue(arg));
      } else if (arg == "--association_use_synthetic_metadata") {
        options.associationUseSyntheticMetadata =
            parseBool(requireValue(arg));
      } else if (arg == "--association_metadata_num_categories") {
        options.associationMetadataNumCategories =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_metadata_descriptor_dim") {
        options.associationMetadataDescriptorDim =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_metadata_semantic_accuracy") {
        options.associationMetadataSemanticAccuracy =
            std::stod(requireValue(arg));
      } else if (arg == "--association_metadata_descriptor_noise") {
        options.associationMetadataDescriptorNoise =
            std::stod(requireValue(arg));
      } else if (arg == "--association_metadata_shape_noise") {
        options.associationMetadataShapeNoise = std::stod(requireValue(arg));
      } else if (arg == "--association_metadata_repeated_object_ratio") {
        options.associationMetadataRepeatedObjectRatio =
            std::stod(requireValue(arg));
      } else if (arg == "--association_semantic_weight") {
        options.associationSemanticWeight = std::stod(requireValue(arg));
      } else if (arg == "--association_shape_weight") {
        options.associationShapeWeight = std::stod(requireValue(arg));
      } else if (arg == "--association_descriptor_weight") {
        options.associationDescriptorWeight = std::stod(requireValue(arg));
      } else if (arg == "--association_require_observed_objects") {
        options.associationRequireObservedObjects =
            parseBool(requireValue(arg));
      } else if (arg == "--association_use_quotient_metric") {
        options.associationUseQuotientMetric =
            parseBool(requireValue(arg));
      } else if (arg == "--association_transform_min_inliers") {
        options.associationTransformMinInliers =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_transform_max_residual") {
        options.associationTransformMaxResidual =
            std::stod(requireValue(arg));
      } else if (arg == "--association_transform_min_score") {
        options.associationTransformMinScore = std::stod(requireValue(arg));
      } else if (arg == "--association_transform_weight_scale") {
        options.associationTransformWeightScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_transform_max_seed_candidates") {
        options.associationTransformMaxSeedCandidates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_transform_plane_axes") {
        options.associationTransformPlaneAxes = requireValue(arg);
      } else if (
          arg == "--association_transform_update_in_lower_fixed_point") {
        options.associationTransformUpdateInLowerFixedPoint =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_transform_lower_fixed_point_use_reliability") {
        options.associationTransformLowerFixedPointUseReliability =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_transform_lower_fixed_point_use_reliability_trust") {
        options.associationTransformLowerFixedPointUseReliabilityTrust =
            parseBool(requireValue(arg));
      } else if (arg == "--association_transform_use_projected_weights") {
        options.associationTransformUseProjectedWeights =
            parseBool(requireValue(arg));
      } else if (arg == "--association_transform_use_initial_object_poses") {
        options.associationTransformUseInitialObjectPoses =
            parseBool(requireValue(arg));
      } else if (arg == "--association_gauge_synchronization_enabled") {
        options.associationGaugeSynchronizationEnabled =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_gauge_synchronization_apply_to_states") {
        options.associationGaugeSynchronizationApplyToStates =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_gauge_synchronization_apply_once_to_initial_states") {
        options.associationGaugeSynchronizationApplyOnceToInitialStates =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_gauge_synchronization_async_gauss_seidel") {
        options.associationGaugeSynchronizationAsynchronousGaussSeidel =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_gauge_synchronization_max_rounds_per_phase") {
        options.associationGaugeSynchronizationMaxRoundsPerPhase =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_gauge_synchronization_tolerance") {
        options.associationGaugeSynchronizationTolerance =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_gauge_covariant_preserve_local_states") {
        options.associationGaugeCovariantPreserveLocalStates =
            parseBool(requireValue(arg));
      } else if (arg == "--association_cycle_consistency_enabled") {
        options.associationCycleConsistencyEnabled =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_direct_weight") {
        options.associationCycleConsistencyMinDirectWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_composed_weight") {
        options.associationCycleConsistencyMinComposedWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_two_hop_support") {
        options.associationCycleConsistencyMinTwoHopSupport =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg ==
                 "--association_cycle_consistency_keep_without_evidence") {
        options.associationCycleConsistencyKeepWithoutEvidence =
            parseBool(requireValue(arg));
      } else if (arg == "--association_cycle_objective_weight") {
        options.associationCycleObjectiveWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_cycle_objective_min_edge_mass") {
        options.associationCycleObjectiveMinEdgeMass =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_enabled") {
        options.associationAdaptiveUpperEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_initial_step") {
        options.associationAdaptiveUpperInitialStep =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_epsilon") {
        options.associationAdaptiveUpperEpsilon =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_max_step") {
        options.associationAdaptiveUpperMaxStep =
            std::stod(requireValue(arg));
      } else if (arg == "--association_upper_tracking_enabled") {
        options.associationUpperTrackingEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_upper_tracking_consensus_weight") {
        options.associationUpperTrackingConsensusWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_upper_tracking_correction_weight") {
        options.associationUpperTrackingCorrectionWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_upper_tracking_center_rows") {
        options.associationUpperTrackingCenterRows =
            parseBool(requireValue(arg));
      } else if (arg == "--association_upper_tracking_center_columns") {
        options.associationUpperTrackingCenterColumns =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_upper_tracking_project_against_lower_feedback") {
        options.associationUpperTrackingProjectAgainstLowerFeedback =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_upper_tracking_use_lower_stationarity_gate") {
        options.associationUpperTrackingUseLowerStationarityGate =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_upper_tracking_lower_validation_merit_trust_scale") {
        options.associationUpperTrackingLowerValidationMeritTrustScale =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_state_validation_hypergradient_weight") {
        options.associationStateValidationHypergradientWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_state_validation_merit_trust_scale") {
        options.associationStateValidationMeritTrustScale =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_state_validation_merit_lower_iterations") {
        options.associationStateValidationMeritLowerIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg ==
                 "--association_state_validation_merit_use_measurement_cost") {
        options.associationStateValidationMeritUseMeasurementCost =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_state_validation_merit_gate_full_update") {
        options.associationStateValidationMeritGateFullUpdate =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_state_validation_use_edge_transform_confidence") {
        options.associationStateValidationUseEdgeTransformConfidence =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_state_validation_transform_confidence_distance_scale") {
        options.associationStateValidationTransformConfidenceDistanceScale =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_state_validation_trajectory_guard_enabled") {
        options.associationStateValidationTrajectoryGuardEnabled =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_select_best_measurement_state") {
        options.associationLowerValidationSelectBestMeasurementState =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_measurement_incumbent_prox") {
        options.associationLowerValidationMeasurementIncumbentProx =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_save_iteration_estimate_snapshots") {
        options.associationSaveIterationEstimateSnapshots =
            parseBool(requireValue(arg));
      } else if (arg == "--record_object_interface_diagnostics") {
        options.recordObjectInterfaceDiagnostics =
            parseBool(requireValue(arg));
      } else if (arg == "--record_detailed_association_diagnostics") {
        options.recordDetailedAssociationDiagnostics =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_record_upper_response_diagnostics") {
        options.associationRecordUpperResponseDiagnostics =
            parseBool(requireValue(arg));
      } else if (arg == "--record_predictive_measurement_cost") {
        options.recordPredictiveMeasurementCost =
            parseBool(requireValue(arg));
      } else if (arg == "--association_pair_validation_trust_enabled") {
        options.associationPairValidationTrustEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_edge_latent_consensus_weight") {
        options.associationEdgeLatentConsensusWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_edge_latent_consensus_use_transform_confidence") {
        options.associationEdgeLatentConsensusUseTransformConfidence =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_edge_latent_consensus_confidence_distance_scale") {
        options.associationEdgeLatentConsensusConfidenceDistanceScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reciprocal_fresh_support_blend") {
        options.associationReciprocalFreshSupportBlend =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reciprocal_fresh_support_uniqueness_scale") {
        options.associationReciprocalFreshSupportUniquenessScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_reciprocal_fresh_support_use_transform_confidence") {
        options.associationReciprocalFreshSupportUseTransformConfidence =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_reciprocal_fresh_support_confidence_distance_scale") {
        options.associationReciprocalFreshSupportConfidenceDistanceScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_pair_support_gate_power") {
        options.associationPairSupportGatePower =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_pair_support_gate_min_effective_support") {
        options.associationPairSupportGateMinEffectiveSupport =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_pair_support_latent_continuation") {
        options.associationPairSupportLatentContinuation =
            parseBool(requireValue(arg));
      } else if (arg == "--association_object_reliability_residual_scale") {
        options.associationObjectReliabilityResidualScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_object_reliability_min") {
        options.associationObjectReliabilityMin =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_object_reliability_upper_cost_weight") {
        options.associationObjectReliabilityUpperCostWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_object_reliability_use_shared_state") {
        options.associationObjectReliabilityUseSharedState =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_object_observation_robust_residual_scale") {
        options.associationObjectObservationRobustResidualScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_robust_min_information_scale") {
        options.associationObjectObservationRobustMinInformationScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_robust_normalize_information") {
        options.associationObjectObservationRobustNormalizeInformation =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_uncertainty_count_prior") {
        options.associationObjectObservationUncertaintyCountPrior =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_uncertainty_distance_scale") {
        options.associationObjectObservationUncertaintyDistanceScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_uncertainty_min_reliability") {
        options.associationObjectObservationUncertaintyMinReliability =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_uncertainty_normalize_information") {
        options.associationObjectObservationUncertaintyNormalizeInformation =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_object_observation_track_empirical_information") {
        options.associationObjectObservationTrackEmpiricalInformation =
            parseBool(requireValue(arg));
      } else if (arg == "--association_backend_feedback_gradient_scale") {
        options.associationBackendFeedbackGradientScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_feedback_min_trust") {
        options.associationBackendFeedbackMinTrust =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_response_gain") {
        options.associationBackendResponseGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_response_damping") {
        options.associationBackendResponseDamping =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_response_max_reduction_fraction") {
        options.associationBackendResponseMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_adjoint_gain") {
        options.associationBackendAdjointGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_adjoint_damping") {
        options.associationBackendAdjointDamping =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_adjoint_max_reduction_fraction") {
        options.associationBackendAdjointMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_adjoint_max_increase_fraction") {
        options.associationBackendAdjointMaxIncreaseFraction =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_coupled_adjoint_gain") {
        options.associationBackendCoupledAdjointGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_coupled_adjoint_damping") {
        options.associationBackendCoupledAdjointDamping =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_coupled_adjoint_max_reduction_fraction") {
        options.associationBackendCoupledAdjointMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_coupled_adjoint_max_increase_fraction") {
        options.associationBackendCoupledAdjointMaxIncreaseFraction =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_adjoint_reduction_metadata_margin") {
        options.associationBackendAdjointReductionMetadataMargin =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_backend_adjoint_allow_cost_reduction") {
        options.associationBackendAdjointAllowCostReduction =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_scale_negative_delta_by_transform_confidence") {
        options.associationBackendScaleNegativeDeltaByTransformConfidence =
            parseBool(requireValue(arg));
      } else if (arg == "--association_signed_fresh_support_blend") {
        options.associationSignedFreshSupportBlend =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_signed_fresh_support_sigmoid_scale") {
        options.associationSignedFreshSupportSigmoidScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_signed_fresh_support_state_validation_merit_trust_scale") {
        options.associationSignedFreshSupportStateValidationMeritTrustScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_snapshot_period") {
        options.associationBackendSnapshotPeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_backend_max_effective_ratio") {
        options.associationBackendMaxEffectiveRatio =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_normalize_active_rms") {
        options.associationBackendNormalizeActiveRms =
            parseBool(requireValue(arg));
      } else if (arg == "--association_backend_target_rms") {
        options.associationBackendTargetRms =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_backend_target_rms_unmatched_fraction") {
        options.associationBackendTargetRmsUnmatchedFraction =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_backend_cap_positive_delta_at_unmatched") {
        options.associationBackendCapPositiveDeltaAtUnmatched =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_cap_positive_delta_use_transform_confidence") {
        options.associationBackendCapPositiveDeltaUseTransformConfidence =
            parseBool(requireValue(arg));
      } else if (arg == "--association_backend_center_rows") {
        options.associationBackendCenterRows = parseBool(requireValue(arg));
      } else if (arg == "--association_backend_center_columns") {
        options.associationBackendCenterColumns =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_stationarity_trust_max_gradient_norm") {
        options.associationBackendStationarityTrustMaxGradientNorm =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_max_active_factors_per_edge") {
        options.associationBackendMaxActiveFactorsPerEdge =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_temporal_blend") {
        options.associationTemporalBlend = std::stod(requireValue(arg));
      } else if (arg == "--association_temporal_kl_weight") {
        options.associationTemporalKlWeight = std::stod(requireValue(arg));
      } else if (arg == "--association_temporal_kl_epsilon") {
        options.associationTemporalKlEpsilon = std::stod(requireValue(arg));
      } else if (arg == "--association_entropy_regularization_weight") {
        options.associationEntropyRegularizationWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_renormalize_sparse_weights") {
        options.associationRenormalizeSparseWeights =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_warm_start_use_geometry") {
        options.associationResidualWarmStartUseGeometry =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_use_transform_alignment") {
        options.associationResidualUseTransformAlignment =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_residual_transform_update_from_weights") {
        options.associationResidualTransformUpdateFromWeights =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_lower_object_copy_use_transform_alignment") {
        options.associationLowerObjectCopyUseTransformAlignment =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_object_copy_direct_enabled") {
        options.associationLowerObjectCopyDirectEnabled =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_object_copy_use_observation_implied_target") {
        options.associationLowerObjectCopyUseObservationImpliedTarget =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_object_copy_feedback") {
        options.associationLowerTrajectoryObjectCopyFeedback =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_object_copy_feedback_quadratic_trust") {
        options.associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_object_copy_feedback_normalize_object_mass") {
        options.associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_object_copy_feedback_mass_conserving") {
        options.associationLowerTrajectoryObjectCopyFeedbackMassConserving =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_object_copy_feedback_zero_trust_floor") {
        options.associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_shared_latent_fusion") {
        options.associationLowerSharedLatentFusion =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_shared_latent_mode") {
        options.associationLowerSharedLatentMode =
            parseSharedLatentMode(requireValue(arg));
      } else if (arg == "--exact_point2_owner_optimize_trajectories") {
        options.exactPoint2OwnerOptimizeTrajectories =
            parseBool(requireValue(arg));
      } else if (arg == "--exact_point2_owner_pcg_iterations") {
        options.exactPoint2OwnerPcgIterations =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--exact_point2_owner_pcg_warm_start") {
        options.exactPoint2OwnerPcgWarmStart =
            parseBool(requireValue(arg));
      } else if (arg == "--exact_point2_owner_pcg_warm_start_scale") {
        options.exactPoint2OwnerPcgWarmStartScale =
            std::stod(requireValue(arg));
      } else if (arg == "--exact_point2_owner_pcg_late_warm_start_round") {
        options.exactPoint2OwnerPcgLateWarmStartRound =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--exact_point2_owner_pcg_late_warm_start_scale") {
        options.exactPoint2OwnerPcgLateWarmStartScale =
            std::stod(requireValue(arg));
      } else if (arg == "--exact_point2_owner_pcg_matrix_free") {
        options.exactPoint2OwnerPcgMatrixFree =
            parseBool(requireValue(arg));
      } else if (arg == "--exact_point2_owner_pcg_quadratic_warm_start") {
        options.exactPoint2OwnerPcgQuadraticWarmStart =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--exact_point2_owner_pcg_preconditioner_refresh_residual") {
        options.exactPoint2OwnerPcgPreconditionerRefreshResidual =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--exact_point2_owner_pcg_preconditioner_refresh_min_rounds") {
        options.exactPoint2OwnerPcgPreconditionerRefreshMinRounds =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--coupled_point2_schur_oracle_correction") {
        options.coupledPoint2SchurOracleCorrection =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--coupled_point2_schur_matrix_free_owner_pcg") {
        options.coupledPoint2SchurMatrixFreeOwnerPcg =
            parseBool(requireValue(arg));
      } else if (arg == "--coupled_point2_schur_oracle_start_round") {
        options.coupledPoint2SchurOracleStartRound =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--coupled_point2_schur_oracle_period") {
        options.coupledPoint2SchurOraclePeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg ==
          "--coupled_point2_schur_oracle_relative_merit_tolerance") {
        options.coupledPoint2SchurOracleRelativeMeritTolerance =
            std::stod(requireValue(arg));
      } else if (arg == "--integrated_point2_mm") {
        options.integratedPoint2Mm = parseBool(requireValue(arg));
      } else if (arg == "--integrated_point2_local_amm") {
        options.integratedPoint2LocalAmm = parseBool(requireValue(arg));
      } else if (arg == "--integrated_point2_common_acceptance") {
        options.integratedPoint2CommonAcceptance =
            parseBool(requireValue(arg));
      } else if (arg == "--integrated_point2_amm_momentum") {
        options.integratedPoint2AmmMomentum =
            std::stod(requireValue(arg));
      } else if (arg == "--integrated_point2_adaptive_momentum") {
        options.integratedPoint2AdaptiveMomentum =
            parseBool(requireValue(arg));
      } else if (arg == "--integrated_point2_adaptive_momentum_min") {
        options.integratedPoint2AdaptiveMomentumMin =
            std::stod(requireValue(arg));
      } else if (arg == "--integrated_point2_adaptive_momentum_max") {
        options.integratedPoint2AdaptiveMomentumMax =
            std::stod(requireValue(arg));
      } else if (
          arg == "--integrated_point2_adaptive_momentum_ema_decay") {
        options.integratedPoint2AdaptiveMomentumEmaDecay =
            std::stod(requireValue(arg));
      } else if (
          arg == "--integrated_point2_object_wise_adaptive_momentum") {
        options.integratedPoint2ObjectWiseAdaptiveMomentum =
            parseBool(requireValue(arg));
      } else if (arg == "--exact_point2_gauge_sync_rounds") {
        options.exactPoint2GaugeSynchronizationRounds =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_lower_object_copy_robust_residual_scale") {
        options.associationLowerObjectCopyRobustResidualScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_object_copy_robust_min_weight") {
        options.associationLowerObjectCopyRobustMinWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_object_copy_observation_predictive_gate") {
        options.associationLowerObjectCopyObservationPredictiveGate =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_object_copy_normalize_by_support") {
        options.associationLowerObjectCopyNormalizeBySupport =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_object_copy_support_churn_gate") {
        options.associationLowerObjectCopySupportChurnGate =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_trajectory_translation_prox_weight") {
        options.associationLowerTrajectoryTranslationProxWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_projection_power") {
        options.associationLowerProjectionPower =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_projection_hard") {
        options.associationLowerProjectionHard =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_two_time_scale_lower_inner_rounds") {
        options.associationTwoTimeScaleLowerInnerRounds =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_two_time_scale_lower_inner_iterations") {
        options.associationTwoTimeScaleLowerInnerIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_two_time_scale_lower_inner_exchange") {
        options.associationTwoTimeScaleLowerInnerExchange =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_two_time_scale_freeze_upper_in_lower_inner") {
        options.associationTwoTimeScaleFreezeUpperInLowerInner =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_message_delay_rounds") {
        options.associationLowerMessageDelayRounds =
            static_cast<std::size_t>(std::stoull(requireValue(arg)));
      } else if (arg == "--association_lower_drop_first_attempt") {
        options.associationLowerDropFirstAttempt =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_synchronous_shared_only") {
        options.associationLowerDirectEdgeSynchronousSharedOnly =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_preserve_inactive_duals") {
        options.associationLowerDirectEdgePreserveInactiveDuals =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_fixed_union_model") {
        options.associationLowerDirectEdgeFixedUnionModel =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_exact_proximal_curvature") {
        options.associationLowerDirectEdgeExactProximalCurvature =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_proximal_curvature_scale") {
        options.associationLowerDirectEdgeProximalCurvatureScale =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_dual_step_size") {
        options.associationLowerDirectEdgeDualStepSize =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_dual_relaxation_residual_threshold") {
        options.associationLowerDirectEdgeDualRelaxationResidualThreshold =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_spectral_dual_step") {
        options.associationLowerDirectEdgeSpectralDualStep =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_global_spectral_dual_step") {
        options.associationLowerDirectEdgeGlobalSpectralDualStep =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_spectral_dual_step_min") {
        options.associationLowerDirectEdgeSpectralDualStepMin =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_spectral_dual_step_max") {
        options.associationLowerDirectEdgeSpectralDualStepMax =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_spectral_dual_step_ema_decay") {
        options.associationLowerDirectEdgeSpectralDualStepEmaDecay =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_point2_consensus_projection") {
        options.associationLowerDirectEdgePoint2ConsensusProjection =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_point2_consensus_projection_relaxation") {
        options.associationLowerDirectEdgePoint2ConsensusProjectionRelaxation =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_direct_edge_point2_consensus_projection_rounds") {
        options.associationLowerDirectEdgePoint2ConsensusProjectionRounds =
            static_cast<std::size_t>(std::stoull(requireValue(arg)));
      } else if (
          arg ==
          "--association_lower_direct_edge_point2_consensus_reset_dual_after_warmup") {
        options
            .associationLowerDirectEdgePoint2ConsensusResetDualAfterWarmup =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_adaptive_beta") {
        options.associationLowerDirectEdgeAdaptiveBeta =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_direct_edge_beta_min") {
        options.associationLowerDirectEdgeBetaMin =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_direct_edge_beta_max") {
        options.associationLowerDirectEdgeBetaMax =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_beta_imbalance_ratio") {
        options.associationLowerDirectEdgeBetaImbalanceRatio =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_beta_update_factor") {
        options.associationLowerDirectEdgeBetaUpdateFactor =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_beta_ema_decay") {
        options.associationLowerDirectEdgeBetaEmaDecay =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_direct_edge_beta_update_period") {
        options.associationLowerDirectEdgeBetaUpdatePeriod =
            static_cast<std::size_t>(std::stoull(requireValue(arg)));
      } else if (arg ==
                 "--association_lower_high_confidence_anchor_gain") {
        options.associationLowerHighConfidenceAnchorGain =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_continuous_support_gate_power") {
        options.associationLowerContinuousSupportGatePower =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_use_current_residual_reliability") {
        options.associationLowerUseCurrentResidualReliability =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_use_bounded_current_residual_reliability") {
        options.associationLowerUseBoundedCurrentResidualReliability =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_use_support_confidence_reliability") {
        options.associationLowerUseSupportConfidenceReliability =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_transform_bootstrap") {
        options.associationResidualTransformBootstrap =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_trust_previous_support") {
        options.associationResidualTrustPreviousSupport =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_allow_support_expansion") {
        options.associationResidualAllowSupportExpansion =
            parseBool(requireValue(arg));
      } else if (arg == "--association_sparse_upper_support") {
        options.associationSparseUpperSupport = parseBool(requireValue(arg));
      } else if (arg == "--association_transform_dimension") {
        options.associationTransformDimension =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_relational_geometry_weight") {
        options.associationRelationalGeometryWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_relational_geometry_distance_scale") {
        options.associationRelationalGeometryDistanceScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_relational_geometry_normalize_by_squared_mass") {
        options.associationRelationalGeometryNormalizeBySquaredMass =
            parseBool(requireValue(arg));
      } else if (arg == "--association_relational_geometry_center_rows") {
        options.associationRelationalGeometryCenterRows =
            parseBool(requireValue(arg));
      } else if (arg == "--association_relational_null_evidence") {
        options.associationRelationalNullEvidence =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_relational_null_calibration") {
        options.associationRelationalNullEvidenceOptions.calibration =
            parseAssociationRelationalNullCalibration(requireValue(arg));
      } else if (arg ==
                 "--association_relational_null_anchor_threshold") {
        options.associationRelationalNullEvidenceOptions
            .fixedAnchorThreshold = std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_relational_null_minimum_anchors") {
        options.associationRelationalNullEvidenceOptions
            .fixedMinimumAnchors =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg ==
                 "--association_relational_null_student_t_dof") {
        options.associationRelationalNullEvidenceOptions
            .fixedStudentTDegreesOfFreedom =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_roto_translation_alignment_weight") {
        options.associationRotoTranslationAlignmentWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_roto_translation_alignment_distance_scale") {
        options.associationRotoTranslationAlignmentDistanceScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_roto_translation_hypothesis_top_k") {
        options.associationRotoTranslationHypothesisTopK =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_roto_translation_hypothesis_angle_bins") {
        options.associationRotoTranslationHypothesisAngleBins =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg ==
          "--association_roto_translation_hypothesis_softmin_temperature") {
        options.associationRotoTranslationHypothesisSoftminTemperature =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_validation_weight") {
        options.associationLowerValidationWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_lower_validation_center_rows") {
        options.associationLowerValidationCenterRows =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_validation_center_columns") {
        options.associationLowerValidationCenterColumns =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_lower_validation_normalize_active_rms") {
        options.associationLowerValidationNormalizeActiveRms =
            parseBool(requireValue(arg));
      } else if (arg == "--association_lower_validation_target_rms") {
        options.associationLowerValidationTargetRms =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_validation_scale_with_upper_cost") {
        options.associationLowerValidationScaleWithUpperCost =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_minimum_cost_rms_scale") {
        options.associationLowerValidationMinimumCostRmsScale =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_stationarity_trust_max_gradient_norm") {
        options
            .associationLowerValidationStationarityTrustMaxGradientNorm =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_validation_merit_trust_scale") {
        options.associationLowerValidationMeritTrustScale =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_validation_pairwise_weight") {
        options.associationLowerValidationPairwiseWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_pairwise_distance_scale") {
        options.associationLowerValidationPairwiseDistanceScale =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_lower_validation_pairwise_only") {
        options.associationLowerValidationPairwiseOnly =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_lower_validation_reference_trust_weight") {
        options.associationLowerValidationReferenceTrustWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_validation_risk_weight") {
        options.associationValidationRiskWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_validation_risk_target_rms") {
        options.associationValidationRiskTargetRms =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_validation_risk_target_rms_unmatched_fraction") {
        options.associationValidationRiskTargetRmsUnmatchedFraction =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_implicit_validation_adjoint_weight") {
        options.associationImplicitValidationAdjointWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_implicit_validation_adjoint_damping") {
        options.associationImplicitValidationAdjointDamping =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_max_effective_ratio") {
        options.associationImplicitValidationAdjointMaxEffectiveRatio =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_implicit_validation_adjoint_target_rms") {
        options.associationImplicitValidationAdjointTargetRms =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_target_rms_unmatched_fraction") {
        options.associationImplicitValidationAdjointTargetRmsUnmatchedFraction =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_implicit_validation_adjoint_use_coupled_solve") {
        options.associationImplicitValidationAdjointUseCoupledSolve =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_implicit_validation_adjoint_max_active_objects") {
        options.associationImplicitValidationAdjointMaxActiveObjects =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_observation_rhs_weight") {
        options.associationImplicitValidationAdjointObservationRhsWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_project_against_backend") {
        options.associationImplicitValidationAdjointProjectAgainstBackend =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_project_against_upper_prefix") {
        options.associationImplicitValidationAdjointProjectAgainstUpperPrefix =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_implicit_validation_adjoint_merit_trust_scale") {
        options.associationImplicitValidationAdjointMeritTrustScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_coverage_prior_weight") {
        options.associationCoveragePriorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_coverage_prior_match_row_column") {
        options.associationCoveragePriorMatchRowColumn =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_coverage_prior_normalize_by_reference_mass") {
        options.associationCoveragePriorNormalizeByReferenceMass =
            parseBool(requireValue(arg));
      } else if (arg == "--association_support_sharpness_weight") {
        options.associationSupportSharpnessWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_support_sharpness_target_ratio") {
        options.associationSupportSharpnessTargetRatio =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_support_sharpness_validation_blend") {
        options.associationSupportSharpnessValidationBlend =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_support_sharpness_coverage_blend") {
        options.associationSupportSharpnessCoverageBlend =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_support_sharpness_project_against_lower_feedback") {
        options.associationSupportSharpnessProjectAgainstLowerFeedback =
            parseBool(requireValue(arg));
      } else if (arg == "--association_structural_expansion_weight") {
        options.associationStructuralExpansionWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_structural_expansion_distance_scale") {
        options.associationStructuralExpansionDistanceScale =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_structural_expansion_use_proposal_gate") {
        options.associationStructuralExpansionUseProposalGate =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_reference_support_sharpening_weight") {
        options.associationReferenceSupportSharpeningWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reference_support_sharpening_target") {
        options.associationReferenceSupportSharpeningTarget =
            std::stod(requireValue(arg));
      } else if (arg == "--association_post_update_local_refine") {
        options.associationPostUpdateLocalRefine =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_post_update_local_refine_iterations") {
        options.associationPostUpdateLocalRefineIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_post_update_local_refine_exchange") {
        options.associationPostUpdateLocalRefineExchange =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_post_update_lower_fixed_point_rounds") {
        options.associationPostUpdateLowerFixedPointRounds =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_terminal_lower_rounds") {
        options.associationTerminalLowerRounds =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_reference_trust_weight") {
        options.associationReferenceTrustWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reference_trust_scale_with_upper_cost") {
        options.associationReferenceTrustScaleWithUpperCost =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_reference_trust_minimum_cost_rms_scale") {
        options.associationReferenceTrustMinimumCostRmsScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reference_mass_trust_weight") {
        options.associationReferenceMassTrustWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reference_mass_trust_scale_with_upper_cost") {
        options.associationReferenceMassTrustScaleWithUpperCost =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_reference_mass_trust_minimum_cost_rms_scale") {
        options.associationReferenceMassTrustMinimumCostRmsScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_normalize_upper_costs") {
        options.associationNormalizeUpperCosts =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_upper_cost_normalization_target_rms" ||
          arg == "--association_upper_cost_normalization_target_norm") {
        options.associationUpperCostNormalizationTargetRms =
            std::stod(requireValue(arg));
      } else if (arg == "--association_min_consensus_weight") {
        options.associationMinConsensusWeight = std::stod(requireValue(arg));
      } else if (arg == "--association_min_consensus_margin") {
        options.associationMinConsensusMargin = std::stod(requireValue(arg));
      } else if (arg == "--association_consensus_weight_power") {
        options.associationConsensusWeightPower =
            std::stod(requireValue(arg));
      } else if (arg == "--association_active_min_stable_updates") {
        options.associationActiveMinStableUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_activation_trust_ramp_updates") {
        options.associationActivationTrustRampUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_tentative_weight_scale") {
        options.associationTentativeWeightScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_threshold") {
        options.associationRejectResidualThreshold =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_best_margin") {
        options.associationRejectResidualBestMargin =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_quantile") {
        options.associationRejectResidualQuantile =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_quantile_scale") {
        options.associationRejectResidualQuantileScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_min_bad_updates") {
        options.associationRejectMinBadUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_response_period") {
        options.objectInterfaceResponsePeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_min_predicted_decrease") {
        options.objectInterfaceMinPredictedDecrease =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_min_innovation_score") {
        options.objectInterfaceMinInnovationScore =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_schur_damping") {
        options.objectInterfaceSchurDamping = std::stod(requireValue(arg));
      } else if (arg == "--object_interface_pcg_iterations") {
        options.objectInterfacePcgIterations =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_pcg_relative_tolerance") {
        options.objectInterfacePcgRelativeTolerance =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_pcg_warm_start") {
        options.objectInterfacePcgWarmStart = parseBool(requireValue(arg));
      } else if (arg == "--object_interface_pcg_preconditioner") {
        options.objectInterfacePcgPreconditioner =
            parseObjectInterfacePcgPreconditioner(requireValue(arg));
      } else if (arg == "--object_interface_pcg_private_solver") {
        options.objectInterfacePcgPrivateSolver =
            parseObjectInterfacePcgPrivateSolver(requireValue(arg));
      } else if (arg ==
                 "--object_interface_pcg_post_activation_local_iterations") {
        options.objectInterfacePcgPostActivationLocalIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--object_interface_pcg_terminal_local_polish_iterations") {
        options.objectInterfacePcgTerminalLocalPolishIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (
          arg ==
          "--object_interface_pcg_replace_final_round_with_terminal_polish") {
        options.objectInterfacePcgReplaceFinalRoundWithTerminalPolish =
            parseBool(requireValue(arg));
      } else if (
          arg == "--object_interface_pcg_persistent_feasible_response") {
        options.objectInterfacePcgPersistentFeasibleResponse =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--object_interface_pcg_min_translation_spread") {
        options.objectInterfacePcgMinTranslationSpread =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--object_interface_pcg_min_rotation_spread_deg") {
        options.objectInterfacePcgMinRotationSpreadDegrees =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--object_interface_pcg_min_initial_relative_decrease") {
        options.objectInterfacePcgMinInitialRelativeDecrease =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--object_interface_pcg_min_continuation_relative_decrease") {
        options.objectInterfacePcgMinContinuationRelativeDecrease =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_step_gain") {
        options.objectInterfaceStepGain = std::stod(requireValue(arg));
      } else if (arg == "--object_interface_max_objects_per_robot") {
        options.objectInterfaceMaxObjectsPerRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_max_private_cols") {
        options.objectInterfaceMaxPrivateCols =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_max_block_step_norm") {
        options.objectInterfaceMaxBlockStepNorm =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_require_decrease") {
        options.objectInterfaceRequireDecrease = parseBool(requireValue(arg));
      } else if (arg == "--object_pose_tol") {
        options.objectPoseTolerance = std::stod(requireValue(arg));
      } else if (arg == "--object_max_age") {
        options.objectMaxAge =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_heldout_object_observation_modulo") {
        options.associationHeldoutObjectObservationModulo =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_heldout_object_observation_remainder") {
        options.associationHeldoutObjectObservationRemainder =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--translation_prox_weight") {
        options.translationProxWeight = std::stod(requireValue(arg));
      } else if (arg == "--project_to_se_after_local_solve") {
        options.projectToSEAfterLocalSolve = parseBool(requireValue(arg));
      } else if (arg == "--coupled_translation_trust_region") {
        options.coupledTranslationTrustRegion =
            parseBool(requireValue(arg));
      } else if (arg == "--translation_elimination_prox_weight") {
        options.translationEliminationProxWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--local_solver") {
        options.localSolver =
            DPGO::parseManualDpgoMmLocalSolver(requireValue(arg));
      } else if (arg == "--full_solver_outer_iterations") {
        options.fullSolverOuterIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_iterations") {
        options.localTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_initial_max_iterations") {
        options.localInitialTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_initial_rounds") {
        options.localInitialTrustRegionRounds =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_periodic_max_iterations") {
        options.localPeriodicTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_periodic_period") {
        options.localPeriodicTrustRegionPeriod =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_final_max_iterations") {
        options.localFinalTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_final_rounds") {
        options.localFinalTrustRegionRounds =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_tcg_iterations") {
        options.localTrustRegionMaxInnerIterations =
            std::stoi(requireValue(arg));
      } else if (arg == "--local_grad_norm_tol") {
        options.localTrustRegionTolerance = std::stod(requireValue(arg));
      } else if (arg == "--trust_region_initial_radius") {
        options.localTrustRegionInitialRadius = std::stod(requireValue(arg));
      } else if (arg == "--profile_reduced_rotation_runtime") {
        options.profileReducedRotationRuntime =
            parseBool(requireValue(arg));
      } else if (arg == "--exclude_summary_diagnostics_from_solver_time") {
        options.excludeSummaryDiagnosticsFromSolverTime =
            parseBool(requireValue(arg));
      } else if (arg == "--reduced_rotation_gradient_boundary_candidate") {
        options.reducedRotationGradientBoundaryCandidate =
            parseBool(requireValue(arg));
      } else if (arg == "--reduced_rotation_preconditioner") {
        options.reducedRotationPreconditioner =
            DPGO::parseManualDpgoMmReducedRotationPreconditioner(
                requireValue(arg));
      } else if (arg == "--output_dir") {
        outputDir = requireValue(arg);
      } else if (arg == "--save") {
        save = parseBool(requireValue(arg));
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    if (dataDir.empty()) {
      printUsage(argv[0]);
      return 1;
    }
    const DPGO::AssociationRelationalNullEvidenceOptions
        &relationalNullOptions =
        options.associationRelationalNullEvidenceOptions;
    if (relationalNullOptions.calibration ==
            DPGO::AssociationRelationalNullCalibration::FixedStudentT &&
        (!std::isfinite(relationalNullOptions.fixedAnchorThreshold) ||
         relationalNullOptions.fixedAnchorThreshold < 0.0 ||
         relationalNullOptions.fixedMinimumAnchors == 0u ||
         !std::isfinite(
             relationalNullOptions.fixedStudentTDegreesOfFreedom) ||
         relationalNullOptions.fixedStudentTDegreesOfFreedom <= 0.0)) {
      throw std::invalid_argument(
          "fixed relational null options must be finite and positive");
    }

    options = DPGO::applyManualObjectDranAssociationPolicy(options);

    if (!quiet) {
      std::cout << "ABLATION_CONFIG method=manual_object_dran"
                << " base=object_copy_riemannian_local_model"
              << " shared_variable=object_copies"
              << " consensus=penalty"
              << " init=" << objectInitializationName(options.objectInitialization)
              << " topology=" << topologyName(options.topology)
              << " ring_hops=" << options.ringHops
              << " topology_file=" << options.topologyFile
              << " topology_weight_mode=" << options.topologyWeightMode
              << " communication_policy="
              << communicationPolicyName(options.communicationPolicy)
              << " object_initialization="
              << objectInitializationName(options.objectInitialization)
              << " object_initialization_anchor_mode="
              << objectInitializationAnchorModeName(
                     options.objectInitializationAnchorMode)
              << " object_initialization_consensus_rounds="
              << options.objectInitializationConsensusRounds
              << " object_initialization_observed_anchor_weight="
              << options.objectInitializationObservedAnchorWeight
              << " object_initialization_relay_anchor_weight="
              << options.objectInitializationRelayAnchorWeight
              << " object_orientation_observed="
              << (options.objectOrientationObserved ? "true" : "false")
              << " object_communication_representation="
              << objectCommunicationRepresentationName(
                     options.objectCommunicationRepresentation)
              << " association_aware_ted_cci_initialization="
              << (options.associationAwareTedCciInitialization ? "true"
                                                               : "false")
              << " association_lower_physical_copy_majorization="
              << (options.associationLowerPhysicalCopyMajorization ? "true"
                                                                    : "false")
              << " association_lower_physical_copy_colored_bcd="
              << (options.associationLowerPhysicalCopyColoredBcd ? "true"
                                                                 : "false")
              << " object_interface_mode="
              << objectInterfaceModeName(options.objectInterfaceMode)
              << " object_interface_response_strategy="
              << objectInterfaceResponseStrategyName(
                     options.objectInterfaceResponseStrategy)
              << " object_interface_response_trigger="
              << objectInterfaceResponseTriggerName(
                     options.objectInterfaceResponseTrigger)
              << " association_mode="
              << associationModeName(options.associationMode)
              << " association_policy="
              << associationPolicyName(options.associationPolicy)
              << " association_final_ablation="
              << associationFinalAblationName(options.associationFinalAblation)
              << " diagnostic_profile="
              << diagnosticProfileName(options.diagnosticProfile)
              << " association_file=" << options.associationFile
              << " association_update_period="
              << options.associationUpdatePeriod
              << " association_freeze_upper_variables="
              << (options.associationFreezeUpperVariables ? "true" : "false")
              << " association_cavity_export_leave_pair_states="
              << (options.associationCavityExportLeavePairStates ? "true"
                                                                 : "false")
              << " association_local_cavity_feedback_sweep="
              << (options.associationLocalCavityFeedbackSweepEnabled
                      ? "true"
                      : "false")
              << " association_component_response_gate="
              << (options.associationComponentResponseGateEnabled
                      ? "true"
                      : "false")
              << " association_network_partition_gate="
              << (options.associationNetworkPartitionGateEnabled
                      ? "true"
                      : "false")
              << " association_network_partition_multicolor_feedback="
              << (options
                          .associationNetworkPartitionMulticolorFeedbackEnabled
                      ? "true"
                      : "false")
              << " association_sinkhorn_iterations="
              << options.associationSinkhornIterations
              << " association_max_candidates_per_edge="
              << options.associationMaxCandidatesPerEdge
              << " association_temperature="
              << options.associationTemperature
              << " association_row_capacity="
              << options.associationRowCapacity
              << " association_column_capacity="
              << options.associationColumnCapacity
              << " association_sinkhorn_max_nonzeros="
              << options.associationSinkhornMaxNonzeros
              << " association_unmatched_cost="
              << options.associationUnmatchedCost
              << " association_geometry_scale="
              << options.associationGeometryScale
              << " association_use_synthetic_metadata="
              << (options.associationUseSyntheticMetadata ? "true" : "false")
              << " association_metadata_num_categories="
              << options.associationMetadataNumCategories
              << " association_metadata_descriptor_dim="
              << options.associationMetadataDescriptorDim
              << " association_metadata_semantic_accuracy="
              << options.associationMetadataSemanticAccuracy
              << " association_metadata_descriptor_noise="
              << options.associationMetadataDescriptorNoise
              << " association_metadata_shape_noise="
              << options.associationMetadataShapeNoise
              << " association_metadata_repeated_object_ratio="
              << options.associationMetadataRepeatedObjectRatio
              << " association_semantic_weight="
              << options.associationSemanticWeight
              << " association_shape_weight="
              << options.associationShapeWeight
              << " association_descriptor_weight="
              << options.associationDescriptorWeight
              << " association_require_observed_objects="
              << (options.associationRequireObservedObjects ? "true"
                                                            : "false")
              << " association_use_quotient_metric="
              << (options.associationUseQuotientMetric ? "true" : "false")
              << " association_transform_min_inliers="
              << options.associationTransformMinInliers
              << " association_transform_max_residual="
              << options.associationTransformMaxResidual
              << " association_transform_min_score="
              << options.associationTransformMinScore
              << " association_transform_weight_scale="
              << options.associationTransformWeightScale
              << " association_transform_max_seed_candidates="
              << options.associationTransformMaxSeedCandidates
              << " association_transform_plane_axes="
              << options.associationTransformPlaneAxes
              << " association_transform_update_in_lower_fixed_point="
              << (options.associationTransformUpdateInLowerFixedPoint ? "true"
                                                                      : "false")
              << " association_transform_lower_fixed_point_use_reliability="
              << (options.associationTransformLowerFixedPointUseReliability
                      ? "true"
                      : "false")
              << " association_transform_lower_fixed_point_use_reliability_trust="
              << (options.associationTransformLowerFixedPointUseReliabilityTrust
                      ? "true"
                      : "false")
              << " association_transform_use_projected_weights="
              << (options.associationTransformUseProjectedWeights ? "true"
                                                                   : "false")
              << " association_transform_use_initial_object_poses="
              << (options.associationTransformUseInitialObjectPoses ? "true"
                                                                     : "false")
              << " association_gauge_synchronization_enabled="
              << (options.associationGaugeSynchronizationEnabled ? "true"
                                                                  : "false")
              << " association_gauge_synchronization_apply_to_states="
              << (options.associationGaugeSynchronizationApplyToStates
                      ? "true"
                      : "false")
              << " association_gauge_synchronization_apply_once_to_initial_states="
              << (options.associationGaugeSynchronizationApplyOnceToInitialStates
                      ? "true"
                      : "false")
              << " association_gauge_covariant_preserve_local_states="
              << (options.associationGaugeCovariantPreserveLocalStates
                      ? "true"
                      : "false")
              << " association_cycle_consistency_enabled="
              << (options.associationCycleConsistencyEnabled ? "true"
                                                              : "false")
              << " association_cycle_consistency_min_direct_weight="
              << options.associationCycleConsistencyMinDirectWeight
              << " association_cycle_consistency_min_composed_weight="
              << options.associationCycleConsistencyMinComposedWeight
              << " association_cycle_consistency_min_two_hop_support="
              << options.associationCycleConsistencyMinTwoHopSupport
              << " association_cycle_consistency_keep_without_evidence="
              << (options.associationCycleConsistencyKeepWithoutEvidence
                      ? "true"
                      : "false")
              << " association_cycle_objective_weight="
              << options.associationCycleObjectiveWeight
              << " association_cycle_objective_min_edge_mass="
              << options.associationCycleObjectiveMinEdgeMass
              << " association_adaptive_upper_enabled="
              << (options.associationAdaptiveUpperEnabled ? "true" : "false")
              << " association_adaptive_upper_initial_step="
              << options.associationAdaptiveUpperInitialStep
              << " association_adaptive_upper_epsilon="
              << options.associationAdaptiveUpperEpsilon
              << " association_adaptive_upper_max_step="
              << options.associationAdaptiveUpperMaxStep
              << " association_upper_tracking_enabled="
              << (options.associationUpperTrackingEnabled ? "true" : "false")
              << " association_upper_tracking_consensus_weight="
              << options.associationUpperTrackingConsensusWeight
              << " association_upper_tracking_correction_weight="
              << options.associationUpperTrackingCorrectionWeight
              << " association_upper_tracking_center_rows="
              << (options.associationUpperTrackingCenterRows ? "true"
                                                             : "false")
              << " association_upper_tracking_center_columns="
              << (options.associationUpperTrackingCenterColumns ? "true"
                                                                : "false")
              << " association_upper_tracking_project_against_lower_feedback="
              << (options.associationUpperTrackingProjectAgainstLowerFeedback
                      ? "true"
                      : "false")
              << " association_upper_tracking_use_lower_stationarity_gate="
              << (options.associationUpperTrackingUseLowerStationarityGate
                      ? "true"
                      : "false")
              << " association_upper_tracking_lower_validation_merit_trust_scale="
              << options.associationUpperTrackingLowerValidationMeritTrustScale
              << " association_state_validation_hypergradient_weight="
              << options.associationStateValidationHypergradientWeight
              << " association_state_validation_merit_trust_scale="
              << options.associationStateValidationMeritTrustScale
              << " association_state_validation_merit_lower_iterations="
              << options.associationStateValidationMeritLowerIterations
              << " association_state_validation_merit_use_measurement_cost="
              << (options.associationStateValidationMeritUseMeasurementCost
                      ? "true"
                      : "false")
              << " association_state_validation_merit_gate_full_update="
              << (options.associationStateValidationMeritGateFullUpdate
                      ? "true"
                      : "false")
              << " association_state_validation_use_edge_transform_confidence="
              << (options.associationStateValidationUseEdgeTransformConfidence
                      ? "true"
                      : "false")
              << " association_state_validation_transform_confidence_distance_scale="
              << options
                     .associationStateValidationTransformConfidenceDistanceScale
              << " association_state_validation_trajectory_guard_enabled="
              << (options.associationStateValidationTrajectoryGuardEnabled
                      ? "true"
                      : "false")
              << " association_lower_validation_select_best_measurement_state="
              << (options.associationLowerValidationSelectBestMeasurementState
                      ? "true"
                      : "false")
              << " association_lower_validation_measurement_incumbent_prox="
              << (options.associationLowerValidationMeasurementIncumbentProx
                      ? "true"
                      : "false")
              << " association_save_iteration_estimate_snapshots="
              << (options.associationSaveIterationEstimateSnapshots ? "true"
                                                                    : "false")
              << " exclude_summary_diagnostics_from_solver_time="
              << (options.excludeSummaryDiagnosticsFromSolverTime ? "true"
                                                                  : "false")
              << " record_object_interface_diagnostics="
              << (options.recordObjectInterfaceDiagnostics ? "true" : "false")
              << " record_detailed_association_diagnostics="
              << (options.recordDetailedAssociationDiagnostics ? "true"
                                                              : "false")
              << " association_record_upper_response_diagnostics="
              << (options.associationRecordUpperResponseDiagnostics
                      ? "true"
                      : "false")
              << " record_predictive_measurement_cost="
              << (options.recordPredictiveMeasurementCost ? "true" : "false")
              << " association_pair_validation_trust_enabled="
              << (options.associationPairValidationTrustEnabled ? "true"
                                                                : "false")
              << " association_edge_latent_consensus_weight="
              << options.associationEdgeLatentConsensusWeight
              << " association_edge_latent_consensus_use_transform_confidence="
              << (options.associationEdgeLatentConsensusUseTransformConfidence
                      ? "true"
                      : "false")
              << " association_edge_latent_consensus_confidence_distance_scale="
              << options.associationEdgeLatentConsensusConfidenceDistanceScale
              << " association_shared_edge_variable_enabled="
              << (options.associationSharedEdgeVariableEnabled ? "true"
                                                               : "false")
              << " association_reciprocal_fresh_support_blend="
              << options.associationReciprocalFreshSupportBlend
              << " association_reciprocal_fresh_support_uniqueness_scale="
              << options.associationReciprocalFreshSupportUniquenessScale
              << " association_reciprocal_fresh_support_use_transform_confidence="
              << (options
                          .associationReciprocalFreshSupportUseTransformConfidence
                      ? "true"
                      : "false")
              << " association_reciprocal_fresh_support_confidence_distance_scale="
              << options
                     .associationReciprocalFreshSupportConfidenceDistanceScale
              << " association_pair_support_gate_power="
              << options.associationPairSupportGatePower
              << " association_pair_support_gate_min_effective_support="
              << options.associationPairSupportGateMinEffectiveSupport
              << " association_pair_support_latent_continuation="
              << (options.associationPairSupportLatentContinuation ? "true"
                                                                    : "false")
              << " association_object_reliability_residual_scale="
              << options.associationObjectReliabilityResidualScale
              << " association_object_reliability_min="
              << options.associationObjectReliabilityMin
              << " association_object_reliability_upper_cost_weight="
              << options.associationObjectReliabilityUpperCostWeight
              << " association_object_reliability_use_shared_state="
              << (options.associationObjectReliabilityUseSharedState ? "true"
                                                                      : "false")
              << " association_object_observation_robust_residual_scale="
              << options.associationObjectObservationRobustResidualScale
              << " association_object_observation_robust_min_information_scale="
              << options.associationObjectObservationRobustMinInformationScale
              << " association_object_observation_robust_normalize_information="
              << (options.associationObjectObservationRobustNormalizeInformation
                      ? "true"
                      : "false")
              << " association_object_observation_uncertainty_count_prior="
              << options.associationObjectObservationUncertaintyCountPrior
              << " association_object_observation_uncertainty_distance_scale="
              << options.associationObjectObservationUncertaintyDistanceScale
              << " association_object_observation_uncertainty_min_reliability="
              << options.associationObjectObservationUncertaintyMinReliability
              << " association_object_observation_uncertainty_normalize_information="
              << (options
                          .associationObjectObservationUncertaintyNormalizeInformation
                      ? "true"
                      : "false")
              << " association_object_observation_track_empirical_information="
              << (options
                          .associationObjectObservationTrackEmpiricalInformation
                      ? "true"
                      : "false")
              << " association_backend_feedback_gradient_scale="
              << options.associationBackendFeedbackGradientScale
              << " association_backend_feedback_min_trust="
              << options.associationBackendFeedbackMinTrust
              << " association_backend_response_gain="
              << options.associationBackendResponseGain
              << " association_backend_response_damping="
              << options.associationBackendResponseDamping
              << " association_backend_response_max_reduction_fraction="
              << options.associationBackendResponseMaxReductionFraction
              << " association_backend_adjoint_gain="
              << options.associationBackendAdjointGain
              << " association_backend_adjoint_damping="
              << options.associationBackendAdjointDamping
              << " association_backend_adjoint_max_reduction_fraction="
              << options.associationBackendAdjointMaxReductionFraction
              << " association_backend_adjoint_max_increase_fraction="
              << options.associationBackendAdjointMaxIncreaseFraction
              << " association_backend_coupled_adjoint_gain="
              << options.associationBackendCoupledAdjointGain
              << " association_backend_coupled_adjoint_damping="
              << options.associationBackendCoupledAdjointDamping
              << " association_backend_coupled_adjoint_max_reduction_fraction="
              << options.associationBackendCoupledAdjointMaxReductionFraction
              << " association_backend_coupled_adjoint_max_increase_fraction="
              << options.associationBackendCoupledAdjointMaxIncreaseFraction
              << " association_backend_adjoint_reduction_metadata_margin="
              << options.associationBackendAdjointReductionMetadataMargin
              << " association_backend_adjoint_allow_cost_reduction="
              << (options.associationBackendAdjointAllowCostReduction
                      ? "true"
                      : "false")
              << " association_backend_scale_negative_delta_by_transform_confidence="
              << (options
                          .associationBackendScaleNegativeDeltaByTransformConfidence
                      ? "true"
                      : "false")
              << " association_signed_fresh_support_blend="
              << options.associationSignedFreshSupportBlend
              << " association_signed_fresh_support_sigmoid_scale="
              << options.associationSignedFreshSupportSigmoidScale
              << " association_signed_fresh_support_state_validation_merit_trust_scale="
              << options
                     .associationSignedFreshSupportStateValidationMeritTrustScale
              << " association_backend_snapshot_period="
              << options.associationBackendSnapshotPeriod
              << " association_backend_max_effective_ratio="
              << options.associationBackendMaxEffectiveRatio
              << " association_backend_normalize_active_rms="
              << (options.associationBackendNormalizeActiveRms ? "true"
                                                               : "false")
              << " association_backend_target_rms="
              << options.associationBackendTargetRms
              << " association_backend_target_rms_unmatched_fraction="
              << options.associationBackendTargetRmsUnmatchedFraction
              << " association_backend_cap_positive_delta_at_unmatched="
              << (options.associationBackendCapPositiveDeltaAtUnmatched
                      ? "true"
                      : "false")
              << " association_backend_cap_positive_delta_use_transform_confidence="
              << (options
                          .associationBackendCapPositiveDeltaUseTransformConfidence
                      ? "true"
                      : "false")
              << " association_backend_center_rows="
              << (options.associationBackendCenterRows ? "true" : "false")
              << " association_backend_center_columns="
              << (options.associationBackendCenterColumns ? "true" : "false")
              << " association_backend_stationarity_trust_max_gradient_norm="
              << options.associationBackendStationarityTrustMaxGradientNorm
              << " association_backend_max_active_factors_per_edge="
              << options.associationBackendMaxActiveFactorsPerEdge
              << " association_temporal_blend="
              << options.associationTemporalBlend
              << " association_temporal_kl_weight="
              << options.associationTemporalKlWeight
              << " association_temporal_kl_epsilon="
              << options.associationTemporalKlEpsilon
              << " association_entropy_regularization_weight="
              << options.associationEntropyRegularizationWeight
              << " association_renormalize_sparse_weights="
              << (options.associationRenormalizeSparseWeights ? "true"
                                                               : "false")
              << " association_residual_warm_start_use_geometry="
              << (options.associationResidualWarmStartUseGeometry ? "true"
                                                                   : "false")
              << " association_residual_use_transform_alignment="
              << (options.associationResidualUseTransformAlignment ? "true"
                                                                    : "false")
              << " association_residual_transform_update_from_weights="
              << (options.associationResidualTransformUpdateFromWeights
                      ? "true"
                      : "false")
              << " association_lower_object_copy_use_transform_alignment="
              << (options.associationLowerObjectCopyUseTransformAlignment
                      ? "true"
                      : "false")
              << " association_lower_object_copy_direct_enabled="
              << (options.associationLowerObjectCopyDirectEnabled ? "true"
                                                                  : "false")
              << " association_lower_object_copy_use_observation_implied_target="
              << (options.associationLowerObjectCopyUseObservationImpliedTarget
                      ? "true"
                      : "false")
              << " association_lower_trajectory_object_copy_feedback="
              << (options.associationLowerTrajectoryObjectCopyFeedback
                      ? "true"
                      : "false")
              << " association_lower_trajectory_object_copy_feedback_quadratic_trust="
              << (options.associationLowerTrajectoryObjectCopyFeedbackQuadraticTrust
                      ? "true"
                      : "false")
              << " association_lower_trajectory_object_copy_feedback_normalize_object_mass="
              << (options.associationLowerTrajectoryObjectCopyFeedbackNormalizeObjectMass
                      ? "true"
                      : "false")
              << " association_lower_trajectory_object_copy_feedback_mass_conserving="
              << (options.associationLowerTrajectoryObjectCopyFeedbackMassConserving
                      ? "true"
                      : "false")
              << " association_lower_trajectory_object_copy_feedback_zero_trust_floor="
              << (options.associationLowerTrajectoryObjectCopyFeedbackZeroTrustFloor
                      ? "true"
                      : "false")
              << " association_lower_shared_latent_fusion="
              << (options.associationLowerSharedLatentFusion ? "true"
                                                             : "false")
              << " association_lower_shared_latent_mode="
              << sharedLatentModeName(
                     options.associationLowerSharedLatentMode)
              << " association_lower_object_copy_robust_residual_scale="
              << options.associationLowerObjectCopyRobustResidualScale
              << " association_lower_object_copy_robust_min_weight="
              << options.associationLowerObjectCopyRobustMinWeight
              << " association_lower_object_copy_observation_predictive_gate="
              << (options
                          .associationLowerObjectCopyObservationPredictiveGate
                      ? "true"
                      : "false")
              << " association_lower_object_copy_normalize_by_support="
              << (options.associationLowerObjectCopyNormalizeBySupport
                      ? "true"
                      : "false")
              << " association_lower_object_copy_support_churn_gate="
              << (options.associationLowerObjectCopySupportChurnGate ? "true"
                                                                      : "false")
              << " association_lower_trajectory_translation_prox_weight="
              << options.associationLowerTrajectoryTranslationProxWeight
              << " association_lower_projection_power="
              << options.associationLowerProjectionPower
              << " association_lower_projection_hard="
              << (options.associationLowerProjectionHard ? "true" : "false")
              << " association_two_time_scale_lower_inner_rounds="
              << options.associationTwoTimeScaleLowerInnerRounds
              << " association_two_time_scale_lower_inner_iterations="
              << options.associationTwoTimeScaleLowerInnerIterations
              << " association_two_time_scale_lower_inner_exchange="
              << (options.associationTwoTimeScaleLowerInnerExchange ? "true"
                                                                     : "false")
              << " association_two_time_scale_freeze_upper_in_lower_inner="
              << (options.associationTwoTimeScaleFreezeUpperInLowerInner
                      ? "true"
                      : "false")
              << " association_lower_message_delay_rounds="
              << options.associationLowerMessageDelayRounds
              << " event_schedule_file=" << options.eventScheduleFile
              << " association_lower_drop_first_attempt="
              << (options.associationLowerDropFirstAttempt ? "true" : "false")
              << " association_lower_direct_edge_synchronous_shared_only="
              << (options.associationLowerDirectEdgeSynchronousSharedOnly
                      ? "true"
                      : "false")
              << " coupled_point2_schur_oracle_correction="
              << (options.coupledPoint2SchurOracleCorrection ? "true"
                                                               : "false")
              << " coupled_point2_schur_matrix_free_owner_pcg="
              << (options.coupledPoint2SchurMatrixFreeOwnerPcg ? "true"
                                                                : "false")
              << " coupled_point2_schur_oracle_start_round="
              << options.coupledPoint2SchurOracleStartRound
              << " coupled_point2_schur_oracle_period="
              << options.coupledPoint2SchurOraclePeriod
              << " coupled_point2_schur_oracle_relative_merit_tolerance="
              << options.coupledPoint2SchurOracleRelativeMeritTolerance
              << " integrated_point2_mm="
              << (options.integratedPoint2Mm ? "true" : "false")
              << " integrated_point2_local_amm="
              << (options.integratedPoint2LocalAmm ? "true" : "false")
              << " integrated_point2_common_acceptance="
              << (options.integratedPoint2CommonAcceptance ? "true"
                                                            : "false")
              << " integrated_point2_amm_momentum="
              << options.integratedPoint2AmmMomentum
              << " integrated_point2_adaptive_momentum="
              << (options.integratedPoint2AdaptiveMomentum ? "true"
                                                            : "false")
              << " integrated_point2_adaptive_momentum_min="
              << options.integratedPoint2AdaptiveMomentumMin
              << " integrated_point2_adaptive_momentum_max="
              << options.integratedPoint2AdaptiveMomentumMax
              << " integrated_point2_adaptive_momentum_ema_decay="
              << options.integratedPoint2AdaptiveMomentumEmaDecay
              << " integrated_point2_object_wise_adaptive_momentum="
              << (options.integratedPoint2ObjectWiseAdaptiveMomentum
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_exact_proximal_curvature="
              << (options.associationLowerDirectEdgeExactProximalCurvature
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_preserve_inactive_duals="
              << (options.associationLowerDirectEdgePreserveInactiveDuals
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_fixed_union_model="
              << (options.associationLowerDirectEdgeFixedUnionModel ? "true"
                                                                     : "false")
              << " association_lower_direct_edge_proximal_curvature_scale="
              << options.associationLowerDirectEdgeProximalCurvatureScale
              << " association_lower_direct_edge_dual_step_size="
              << options.associationLowerDirectEdgeDualStepSize
              << " association_lower_direct_edge_dual_relaxation_residual_threshold="
              << options
                     .associationLowerDirectEdgeDualRelaxationResidualThreshold
              << " association_lower_direct_edge_spectral_dual_step="
              << (options.associationLowerDirectEdgeSpectralDualStep
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_global_spectral_dual_step="
              << (options.associationLowerDirectEdgeGlobalSpectralDualStep
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_spectral_dual_step_min="
              << options.associationLowerDirectEdgeSpectralDualStepMin
              << " association_lower_direct_edge_spectral_dual_step_max="
              << options.associationLowerDirectEdgeSpectralDualStepMax
              << " association_lower_direct_edge_spectral_dual_step_ema_decay="
              << options.associationLowerDirectEdgeSpectralDualStepEmaDecay
              << " association_lower_direct_edge_point2_consensus_projection="
              << (options
                          .associationLowerDirectEdgePoint2ConsensusProjection
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_point2_consensus_projection_relaxation="
              << options
                     .associationLowerDirectEdgePoint2ConsensusProjectionRelaxation
              << " association_lower_direct_edge_point2_consensus_projection_rounds="
              << options
                     .associationLowerDirectEdgePoint2ConsensusProjectionRounds
              << " association_lower_direct_edge_point2_consensus_reset_dual_after_warmup="
              << (options
                          .associationLowerDirectEdgePoint2ConsensusResetDualAfterWarmup
                      ? "true"
                      : "false")
              << " association_lower_direct_edge_adaptive_beta="
              << (options.associationLowerDirectEdgeAdaptiveBeta ? "true"
                                                                  : "false")
              << " association_lower_direct_edge_beta_min="
              << options.associationLowerDirectEdgeBetaMin
              << " association_lower_direct_edge_beta_max="
              << options.associationLowerDirectEdgeBetaMax
              << " association_lower_direct_edge_beta_imbalance_ratio="
              << options.associationLowerDirectEdgeBetaImbalanceRatio
              << " association_lower_direct_edge_beta_update_factor="
              << options.associationLowerDirectEdgeBetaUpdateFactor
              << " association_lower_direct_edge_beta_ema_decay="
              << options.associationLowerDirectEdgeBetaEmaDecay
              << " association_lower_direct_edge_beta_update_period="
              << options.associationLowerDirectEdgeBetaUpdatePeriod
              << " association_lower_high_confidence_anchor_gain="
              << options.associationLowerHighConfidenceAnchorGain
              << " association_lower_continuous_support_gate_power="
              << options.associationLowerContinuousSupportGatePower
              << " association_lower_use_current_residual_reliability="
              << (options.associationLowerUseCurrentResidualReliability
                      ? "true"
                      : "false")
              << " association_lower_use_bounded_current_residual_reliability="
              << (options.associationLowerUseBoundedCurrentResidualReliability
                      ? "true"
                      : "false")
              << " association_lower_use_support_confidence_reliability="
              << (options.associationLowerUseSupportConfidenceReliability
                      ? "true"
                      : "false")
              << " association_residual_transform_bootstrap="
              << (options.associationResidualTransformBootstrap ? "true"
                                                                : "false")
              << " association_residual_trust_previous_support="
              << (options.associationResidualTrustPreviousSupport ? "true"
                                                                  : "false")
              << " association_residual_allow_support_expansion="
              << (options.associationResidualAllowSupportExpansion ? "true"
                                                                   : "false")
              << " association_sparse_upper_support="
              << (options.associationSparseUpperSupport ? "true" : "false")
              << " association_transform_dimension="
              << options.associationTransformDimension
              << " association_transform_update_in_lower_fixed_point="
              << (options.associationTransformUpdateInLowerFixedPoint ? "true"
                                                                      : "false")
              << " association_transform_lower_fixed_point_use_reliability="
              << (options.associationTransformLowerFixedPointUseReliability
                      ? "true"
                      : "false")
              << " association_transform_lower_fixed_point_use_reliability_trust="
              << (options.associationTransformLowerFixedPointUseReliabilityTrust
                      ? "true"
                      : "false")
              << " association_transform_use_projected_weights="
              << (options.associationTransformUseProjectedWeights ? "true"
                                                                   : "false")
              << " association_transform_use_initial_object_poses="
              << (options.associationTransformUseInitialObjectPoses ? "true"
                                                                     : "false")
              << " association_gauge_synchronization_enabled="
              << (options.associationGaugeSynchronizationEnabled ? "true"
                                                                  : "false")
              << " association_gauge_synchronization_apply_to_states="
              << (options.associationGaugeSynchronizationApplyToStates
                      ? "true"
                      : "false")
              << " association_gauge_synchronization_apply_once_to_initial_states="
              << (options.associationGaugeSynchronizationApplyOnceToInitialStates
                      ? "true"
                      : "false")
              << " association_gauge_covariant_preserve_local_states="
              << (options.associationGaugeCovariantPreserveLocalStates
                      ? "true"
                      : "false")
              << " association_relational_geometry_weight="
              << options.associationRelationalGeometryWeight
              << " association_relational_geometry_distance_scale="
              << options.associationRelationalGeometryDistanceScale
              << " association_relational_geometry_normalize_by_squared_mass="
              << (options
                          .associationRelationalGeometryNormalizeBySquaredMass
                      ? "true"
                      : "false")
              << " association_relational_geometry_center_rows="
              << (options.associationRelationalGeometryCenterRows ? "true"
                                                                 : "false")
              << " association_relational_null_evidence="
              << (options.associationRelationalNullEvidence ? "true"
                                                            : "false")
              << " association_relational_null_calibration="
              << associationRelationalNullCalibrationName(
                     options.associationRelationalNullEvidenceOptions
                         .calibration)
              << " association_relational_null_anchor_threshold="
              << options.associationRelationalNullEvidenceOptions
                     .fixedAnchorThreshold
              << " association_relational_null_minimum_anchors="
              << options.associationRelationalNullEvidenceOptions
                     .fixedMinimumAnchors
              << " association_relational_null_student_t_dof="
              << options.associationRelationalNullEvidenceOptions
                     .fixedStudentTDegreesOfFreedom
              << " association_roto_translation_alignment_weight="
              << options.associationRotoTranslationAlignmentWeight
              << " association_roto_translation_alignment_distance_scale="
              << options.associationRotoTranslationAlignmentDistanceScale
              << " association_roto_translation_hypothesis_top_k="
              << options.associationRotoTranslationHypothesisTopK
              << " association_roto_translation_hypothesis_angle_bins="
              << options.associationRotoTranslationHypothesisAngleBins
              << " association_roto_translation_hypothesis_softmin_temperature="
              << options.associationRotoTranslationHypothesisSoftminTemperature
              << " association_lower_validation_weight="
              << options.associationLowerValidationWeight
              << " association_lower_validation_center_rows="
              << (options.associationLowerValidationCenterRows ? "true"
                                                               : "false")
              << " association_lower_validation_center_columns="
              << (options.associationLowerValidationCenterColumns ? "true"
                                                                  : "false")
              << " association_lower_validation_normalize_active_rms="
              << (options.associationLowerValidationNormalizeActiveRms
                      ? "true"
                      : "false")
              << " association_lower_validation_target_rms="
              << options.associationLowerValidationTargetRms
              << " association_lower_validation_scale_with_upper_cost="
              << (options.associationLowerValidationScaleWithUpperCost
                      ? "true"
                      : "false")
              << " association_lower_validation_minimum_cost_rms_scale="
              << options.associationLowerValidationMinimumCostRmsScale
              << " association_lower_validation_stationarity_trust_max_gradient_norm="
              << options
                     .associationLowerValidationStationarityTrustMaxGradientNorm
              << " association_lower_validation_merit_trust_scale="
              << options.associationLowerValidationMeritTrustScale
              << " association_lower_validation_pairwise_weight="
              << options.associationLowerValidationPairwiseWeight
              << " association_lower_validation_pairwise_distance_scale="
              << options.associationLowerValidationPairwiseDistanceScale
              << " association_lower_validation_pairwise_only="
              << (options.associationLowerValidationPairwiseOnly ? "true"
                                                                 : "false")
              << " association_lower_validation_reference_trust_weight="
              << options.associationLowerValidationReferenceTrustWeight
              << " association_validation_risk_weight="
              << options.associationValidationRiskWeight
              << " association_validation_risk_target_rms="
              << options.associationValidationRiskTargetRms
              << " association_validation_risk_target_rms_unmatched_fraction="
              << options.associationValidationRiskTargetRmsUnmatchedFraction
              << " association_implicit_validation_adjoint_weight="
              << options.associationImplicitValidationAdjointWeight
              << " association_implicit_validation_adjoint_damping="
              << options.associationImplicitValidationAdjointDamping
              << " association_implicit_validation_adjoint_max_effective_ratio="
              << options.associationImplicitValidationAdjointMaxEffectiveRatio
              << " association_implicit_validation_adjoint_target_rms="
              << options.associationImplicitValidationAdjointTargetRms
              << " association_implicit_validation_adjoint_target_rms_unmatched_fraction="
              << options
                     .associationImplicitValidationAdjointTargetRmsUnmatchedFraction
              << " association_implicit_validation_adjoint_use_coupled_solve="
              << (options.associationImplicitValidationAdjointUseCoupledSolve
                      ? "true"
                      : "false")
              << " association_implicit_validation_adjoint_max_active_objects="
              << options.associationImplicitValidationAdjointMaxActiveObjects
              << " association_implicit_validation_adjoint_observation_rhs_weight="
              << options
                     .associationImplicitValidationAdjointObservationRhsWeight
              << " association_implicit_validation_adjoint_project_against_backend="
              << (options
                          .associationImplicitValidationAdjointProjectAgainstBackend
                      ? "true"
                      : "false")
              << " association_implicit_validation_adjoint_project_against_upper_prefix="
              << (options
                          .associationImplicitValidationAdjointProjectAgainstUpperPrefix
                      ? "true"
                      : "false")
              << " association_implicit_validation_adjoint_merit_trust_scale="
              << options.associationImplicitValidationAdjointMeritTrustScale
              << " association_coverage_prior_weight="
              << options.associationCoveragePriorWeight
              << " association_coverage_prior_match_row_column="
              << (options.associationCoveragePriorMatchRowColumn ? "true"
                                                                  : "false")
              << " association_coverage_prior_normalize_by_reference_mass="
              << (options.associationCoveragePriorNormalizeByReferenceMass
                      ? "true"
                      : "false")
              << " association_support_sharpness_weight="
              << options.associationSupportSharpnessWeight
              << " association_support_sharpness_target_ratio="
              << options.associationSupportSharpnessTargetRatio
              << " association_support_sharpness_validation_blend="
              << options.associationSupportSharpnessValidationBlend
              << " association_support_sharpness_coverage_blend="
              << options.associationSupportSharpnessCoverageBlend
              << " association_support_sharpness_project_against_lower_feedback="
              << (options
                          .associationSupportSharpnessProjectAgainstLowerFeedback
                      ? "true"
                      : "false")
              << " association_structural_expansion_weight="
              << options.associationStructuralExpansionWeight
              << " association_structural_expansion_distance_scale="
              << options.associationStructuralExpansionDistanceScale
              << " association_structural_expansion_use_proposal_gate="
              << (options.associationStructuralExpansionUseProposalGate
                      ? "true"
                      : "false")
              << " association_reference_support_sharpening_weight="
              << options.associationReferenceSupportSharpeningWeight
              << " association_reference_support_sharpening_target="
              << options.associationReferenceSupportSharpeningTarget
              << " association_post_update_local_refine="
              << (options.associationPostUpdateLocalRefine ? "true"
                                                            : "false")
              << " association_post_update_local_refine_iterations="
              << options.associationPostUpdateLocalRefineIterations
              << " association_post_update_local_refine_exchange="
              << (options.associationPostUpdateLocalRefineExchange ? "true"
                                                                    : "false")
              << " association_post_update_lower_fixed_point_rounds="
              << options.associationPostUpdateLowerFixedPointRounds
              << " association_terminal_lower_rounds="
              << options.associationTerminalLowerRounds
              << " association_reference_trust_weight="
              << options.associationReferenceTrustWeight
              << " association_reference_trust_scale_with_upper_cost="
              << (options.associationReferenceTrustScaleWithUpperCost
                      ? "true"
                      : "false")
              << " association_reference_trust_minimum_cost_rms_scale="
              << options.associationReferenceTrustMinimumCostRmsScale
              << " association_reference_mass_trust_weight="
              << options.associationReferenceMassTrustWeight
              << " association_reference_mass_trust_scale_with_upper_cost="
              << (options.associationReferenceMassTrustScaleWithUpperCost
                      ? "true"
                      : "false")
              << " association_reference_mass_trust_minimum_cost_rms_scale="
              << options.associationReferenceMassTrustMinimumCostRmsScale
              << " association_normalize_upper_costs="
              << (options.associationNormalizeUpperCosts ? "true" : "false")
              << " association_upper_cost_normalization_target_rms="
              << options.associationUpperCostNormalizationTargetRms
              << " association_min_consensus_weight="
              << options.associationMinConsensusWeight
              << " association_min_consensus_margin="
              << options.associationMinConsensusMargin
              << " association_consensus_weight_power="
              << options.associationConsensusWeightPower
              << " association_active_min_stable_updates="
              << options.associationActiveMinStableUpdates
              << " association_activation_trust_ramp_updates="
              << options.associationActivationTrustRampUpdates
              << " association_tentative_weight_scale="
              << options.associationTentativeWeightScale
              << " association_reject_residual_threshold="
              << options.associationRejectResidualThreshold
              << " association_reject_residual_best_margin="
              << options.associationRejectResidualBestMargin
              << " association_reject_residual_quantile="
              << options.associationRejectResidualQuantile
              << " association_reject_residual_quantile_scale="
              << options.associationRejectResidualQuantileScale
              << " association_reject_min_bad_updates="
              << options.associationRejectMinBadUpdates
              << " object_interface_response_period="
              << options.objectInterfaceResponsePeriod
              << " object_interface_min_predicted_decrease="
              << options.objectInterfaceMinPredictedDecrease
              << " object_interface_min_innovation_score="
              << options.objectInterfaceMinInnovationScore
              << " object_interface_schur_damping="
              << options.objectInterfaceSchurDamping
              << " object_interface_pcg_iterations="
              << options.objectInterfacePcgIterations
              << " object_interface_pcg_relative_tolerance="
              << options.objectInterfacePcgRelativeTolerance
              << " object_interface_pcg_warm_start="
              << (options.objectInterfacePcgWarmStart ? "true" : "false")
              << " object_interface_pcg_preconditioner="
              << objectInterfacePcgPreconditionerName(
                     options.objectInterfacePcgPreconditioner)
              << " object_interface_pcg_private_solver="
              << objectInterfacePcgPrivateSolverName(
                     options.objectInterfacePcgPrivateSolver)
              << " object_interface_pcg_post_activation_local_iterations="
              << options.objectInterfacePcgPostActivationLocalIterations
              << " object_interface_pcg_terminal_local_polish_iterations="
              << options.objectInterfacePcgTerminalLocalPolishIterations
              << " object_interface_pcg_replace_final_round_with_terminal_polish="
              << (options
                          .objectInterfacePcgReplaceFinalRoundWithTerminalPolish
                      ? 1
                      : 0)
              << " object_interface_pcg_persistent_feasible_response="
              << (options.objectInterfacePcgPersistentFeasibleResponse ? 1 : 0)
              << " object_interface_pcg_min_translation_spread="
              << options.objectInterfacePcgMinTranslationSpread
              << " object_interface_pcg_min_rotation_spread_deg="
              << options.objectInterfacePcgMinRotationSpreadDegrees
              << " object_interface_pcg_min_initial_relative_decrease="
              << options.objectInterfacePcgMinInitialRelativeDecrease
              << " object_interface_pcg_min_continuation_relative_decrease="
              << options.objectInterfacePcgMinContinuationRelativeDecrease
              << " object_interface_step_gain="
              << options.objectInterfaceStepGain
              << " object_interface_max_objects_per_robot="
              << options.objectInterfaceMaxObjectsPerRobot
              << " object_interface_max_private_cols="
              << options.objectInterfaceMaxPrivateCols
              << " object_interface_max_block_step_norm="
              << options.objectInterfaceMaxBlockStepNorm
              << " object_interface_require_decrease="
              << (options.objectInterfaceRequireDecrease ? "true" : "false")
              << " object_pose_tol=" << options.objectPoseTolerance
              << " object_communication_link_incidence="
              << (options.objectCommunicationLinkIncidence ? "true" : "false")
              << " object_max_age=" << options.objectMaxAge
              << " association_heldout_object_observation_modulo="
              << options.associationHeldoutObjectObservationModulo
              << " association_heldout_object_observation_remainder="
              << options.associationHeldoutObjectObservationRemainder
              << " translation_prox_weight="
              << options.translationProxWeight
              << " project_to_se_after_local_solve="
              << (options.projectToSEAfterLocalSolve ? "true" : "false")
              << " coupled_translation_trust_region="
              << (options.coupledTranslationTrustRegion ? "true" : "false")
              << " translation_elimination_prox_weight="
              << options.translationEliminationProxWeight
              << " beta=" << options.beta
              << " local_solver="
              << DPGO::manualDpgoMmLocalSolverName(options.localSolver)
              << " full_solver_outer_iterations="
              << options.fullSolverOuterIterations
              << " reduced_rotation_preconditioner="
              << DPGO::manualDpgoMmReducedRotationPreconditionerName(
                     options.reducedRotationPreconditioner)
              << " local_max_iterations="
              << options.localTrustRegionIterations
              << " local_initial_max_iterations="
              << options.localInitialTrustRegionIterations
              << " local_initial_rounds="
              << options.localInitialTrustRegionRounds
              << " local_periodic_max_iterations="
              << options.localPeriodicTrustRegionIterations
              << " local_periodic_period="
              << options.localPeriodicTrustRegionPeriod
              << " local_final_max_iterations="
              << options.localFinalTrustRegionIterations
              << " local_final_rounds="
              << options.localFinalTrustRegionRounds
              << " local_max_tcg_iterations="
                << options.localTrustRegionMaxInnerIterations
                << std::endl;
    }

    const DPGO::ManualObjectDranRunResult result =
        DPGO::runManualObjectDran(dataDir, options);

    if (!quiet) {
      std::cout << std::setprecision(20);
      if (result.associationAwareTedCciInitializationApplied) {
        std::cout << "ASSOCIATION_AWARE_TED_CCI_SUMMARY"
                  << " active_vertices="
                  << result.associationAwareTedCciActiveVertices
                  << " local_measurements="
                  << result.associationAwareTedCciLocalMeasurements
                  << " cross_measurements="
                  << result.associationAwareTedCciCrossMeasurements
                  << " factor_messages="
                  << result.associationAwareTedCciFactorMessages
                  << " solution_messages="
                  << result.associationAwareTedCciSolutionMessages
                  << " bytes_sent="
                  << result.associationAwareTedCciBytesSent
                  << " seconds="
                  << result.associationAwareTedCciSeconds << std::endl;
      }
      const DPGO::ManualObjectDranTrackEmpiricalInformationSummary
          &trackSummary = result.trackEmpiricalInformationSummary;
      std::cout << "TRACK_EMPIRICAL_INFORMATION_SUMMARY"
                << " enabled=" << (trackSummary.enabled ? 1 : 0)
                << " track_count=" << trackSummary.trackCount
                << " observation_count=" << trackSummary.observationCount
                << " singleton_track_fraction="
                << trackSummary.singletonTrackFraction
                << " overdispersed_track_fraction="
                << trackSummary.overdispersedTrackFraction
                << " scale_min=" << trackSummary.scaleMin
                << " scale_median=" << trackSummary.scaleMedian
                << " scale_mean=" << trackSummary.scaleMean
                << " scale_p90=" << trackSummary.scaleP90
                << " scale_max=" << trackSummary.scaleMax
                << " effective_sample_ratio_median="
                << trackSummary.effectiveSampleRatioMedian
                << " effective_sample_ratio_mean="
                << trackSummary.effectiveSampleRatioMean << std::endl;
      for (const DPGO::ManualObjectDranIterationSummary &row :
           result.iterations) {
      std::cout << "ITER_SUMMARY iter=" << row.iter
                << " time=" << row.time
                << " profile_run_setup_seconds="
                << row.profileRunSetupSeconds
                << " profile_data_load_seconds="
                << row.profileDataLoadSeconds
                << " profile_graph_initialization_seconds="
                << row.profileGraphInitializationSeconds
                << " profile_initial_association_seconds="
                << row.profileInitialAssociationSeconds
                << " profile_initial_association_bootstrap_seconds="
                << row.profileInitialAssociationBootstrapSeconds
                << " profile_initial_association_state_window_seconds="
                << row.profileInitialAssociationStateWindowSeconds
                << " profile_initial_gauge_reliability_seconds="
                << row.profileInitialGaugeReliabilitySeconds
                << " profile_lower_inner_rounds_seconds="
                << row.profileLowerInnerRoundsSeconds
                << " profile_lower_factor_assembly_seconds="
                << row.profileLowerFactorAssemblySeconds
                << " profile_lower_solve_seconds="
                << row.profileLowerSolveSeconds
                << " profile_lower_feedback_certificate_seconds="
                << row.profileLowerFeedbackCertificateSeconds
                << " profile_lower_consensus_refresh_seconds="
                << row.profileLowerConsensusRefreshSeconds
                << " profile_association_update_seconds="
                << row.profileAssociationUpdateSeconds
                << " profile_association_unprofiled_seconds="
                << row.profileAssociationUnprofiledSeconds
                << " profile_association_weight_update_seconds="
                << row.profileAssociationWeightUpdateSeconds
                << " profile_association_weight_update_unprofiled_seconds="
                << row.profileAssociationWeightUpdateUnprofiledSeconds
                << " profile_association_local_cost_seconds="
                << row.profileAssociationLocalCostSeconds
                << " profile_association_state_validation_precompute_seconds="
                << row.profileAssociationStateValidationPrecomputeSeconds
                << " profile_association_state_validation_cost_seconds="
                << row.profileAssociationStateValidationCostSeconds
                << " profile_association_edge_latent_consensus_seconds="
                << row.profileAssociationEdgeLatentConsensusSeconds
                << " profile_association_reference_mass_trust_seconds="
                << row.profileAssociationReferenceMassTrustSeconds
                << " profile_association_temporal_stats_seconds="
                << row.profileAssociationTemporalStatsSeconds
                << " profile_association_support_stats_seconds="
                << row.profileAssociationSupportStatsSeconds
                << " profile_association_transform_refresh_seconds="
                << row.profileAssociationTransformRefreshSeconds
                << " profile_association_cycle_consistency_seconds="
                << row.profileAssociationCycleConsistencySeconds
                << " profile_association_state_window_seconds="
                << row.profileAssociationStateWindowSeconds
                << " profile_object_interface_response_seconds="
                << row.profileObjectInterfaceResponseSeconds
                << " profile_pair_validation_seconds="
                << row.profilePairValidationSeconds
                << " profile_association_roto_translation_seconds="
                << row.profileAssociationRotoTranslationSeconds
                << " profile_association_projection_seconds="
                << row.profileAssociationProjectionSeconds
                << " profile_association_support_postprocess_seconds="
                << row.profileAssociationSupportPostprocessSeconds
                << " profile_association_shared_objective_seconds="
                << row.profileAssociationSharedObjectiveSeconds
                << " profile_association_shared_response_seconds="
                << row.profileAssociationSharedResponseSeconds
                << " profile_association_support_entries="
                << row.profileAssociationSupportEntries
                << " profile_association_dense_entries="
                << row.profileAssociationDenseEntries
                << " profile_association_support_density="
                << row.profileAssociationSupportDensity
                << " profile_association_support_mass_coverage="
                << row.profileAssociationSupportMassCoverage
                << " profile_association_precost_support_seconds="
                << row.profileAssociationPreCostSupportSeconds
                << " profile_association_precost_support_entries="
                << row.profileAssociationPreCostSupportEntries
                << " profile_association_precost_dense_entries="
                << row.profileAssociationPreCostDenseEntries
                << " profile_association_precost_support_density="
                << row.profileAssociationPreCostSupportDensity
                << " profile_association_precost_support_mass_coverage="
                << row.profileAssociationPreCostSupportMassCoverage
                << " profile_object_exchange_seconds="
                << row.profileObjectExchangeSeconds
                << " profile_gauge_reliability_update_seconds="
                << row.profileGaugeReliabilityUpdateSeconds
                << " profile_post_update_refine_seconds="
                << row.profilePostUpdateRefineSeconds
                << " post_update_lower_rounds_executed="
                << row.postUpdateLowerRoundsExecuted
                << " post_update_lower_robot_solves="
                << row.postUpdateLowerRobotSolves
                << " post_update_lower_converged_robots="
                << row.postUpdateLowerConvergedRobots
                << " post_update_lower_max_normalized_residual="
                << row.postUpdateLowerMaxNormalizedResidual
                << " post_update_lower_max_stationarity_residual="
                << row.postUpdateLowerMaxStationarityResidual
                << " post_update_lower_max_outer_continuation_residual="
                << row.postUpdateLowerMaxOuterContinuationResidual
                << " profile_summary_factor_assembly_seconds="
                << row.profileSummaryFactorAssemblySeconds
                << " profile_summary_association_diagnostics_seconds="
                << row.profileSummaryAssociationDiagnosticsSeconds
                << " global_cost=" << row.measurementCost
                << " measurement_cost=" << row.measurementCost
                << " heldout_measurement_cost="
                << row.heldoutMeasurementCost
                << " predictive_measurement_cost="
                << row.predictiveMeasurementCost
                << " consensus_cost=" << row.consensusCost
                << " augmented_cost="
                << row.measurementCost + row.consensusCost
                << " gradient=" << row.gradient
                << " known_object_copies=" << row.knownObjectCopies
                << " relay_object_copies=" << row.relayObjectCopies
                << " observed_object_copies=" << row.observedObjectCopies
                << " relay_only_object_copies="
                << row.relayOnlyObjectCopies
                << " initialized_from_neighbor_object_copies="
                << row.initializedFromNeighborObjectCopies
                << " stale_object_copies=" << row.staleObjectCopies
                << " stale_object_comm_triggers="
                << row.staleObjectCommTriggers
                << " active_consensus_pairs=" << row.activeConsensusPairs
                << " association_candidate_count="
                << row.associationCandidateCount
                << " association_weight_mass="
                << row.associationWeightMass
                << " association_correct_mass="
                << row.associationCorrectMass
                << " association_false_merge_mass="
                << row.associationFalseMergeMass
                << " association_entropy=" << row.associationEntropy
                << " association_precision="
                << row.associationPrecision
                << " association_recall=" << row.associationRecall
                << " association_f1=" << row.associationF1
                << " association_false_merge_rate="
                << row.associationFalseMergeRate
                << " association_factor_weight_mass="
                << row.associationFactorWeightMass
                << " association_factor_correct_mass="
                << row.associationFactorCorrectMass
                << " association_factor_false_merge_mass="
                << row.associationFactorFalseMergeMass
                << " high_confidence_association_candidate_count="
                << row.highConfidenceAssociationCandidateCount
                << " high_confidence_association_weight_mass="
                << row.highConfidenceAssociationWeightMass
                << " high_confidence_association_correct_mass="
                << row.highConfidenceAssociationCorrectMass
                << " high_confidence_association_false_merge_mass="
                << row.highConfidenceAssociationFalseMergeMass
                << " high_confidence_association_precision="
                << row.highConfidenceAssociationPrecision
                << " high_confidence_association_recall="
                << row.highConfidenceAssociationRecall
                << " high_confidence_association_f1="
                << row.highConfidenceAssociationF1
                << " high_confidence_association_false_merge_rate="
                << row.highConfidenceAssociationFalseMergeRate
                << " high_confidence_association_factor_weight_mass="
                << row.highConfidenceAssociationFactorWeightMass
                << " high_confidence_association_factor_correct_mass="
                << row.highConfidenceAssociationFactorCorrectMass
                << " high_confidence_association_factor_false_merge_mass="
                << row.highConfidenceAssociationFactorFalseMergeMass
                << " activated_association_candidate_count="
                << row.activatedAssociationCandidateCount
                << " activated_association_weight_mass="
                << row.activatedAssociationWeightMass
                << " activated_association_correct_mass="
                << row.activatedAssociationCorrectMass
                << " activated_association_false_merge_mass="
                << row.activatedAssociationFalseMergeMass
                << " activated_association_precision="
                << row.activatedAssociationPrecision
                << " activated_association_recall="
                << row.activatedAssociationRecall
                << " activated_association_f1="
                << row.activatedAssociationF1
                << " activated_association_false_merge_rate="
                << row.activatedAssociationFalseMergeRate
                << " activated_association_factor_weight_mass="
                << row.activatedAssociationFactorWeightMass
                << " activated_association_factor_correct_mass="
                << row.activatedAssociationFactorCorrectMass
                << " activated_association_factor_false_merge_mass="
                << row.activatedAssociationFactorFalseMergeMass
                << " active_association_candidate_count="
                << row.activeAssociationCandidateCount
                << " active_association_weight_mass="
                << row.activeAssociationWeightMass
                << " active_association_correct_mass="
                << row.activeAssociationCorrectMass
                << " active_association_false_merge_mass="
                << row.activeAssociationFalseMergeMass
                << " active_association_precision="
                << row.activeAssociationPrecision
                << " active_association_recall="
                << row.activeAssociationRecall
                << " active_association_f1="
                << row.activeAssociationF1
                << " active_association_false_merge_rate="
                << row.activeAssociationFalseMergeRate
                << " active_association_factor_weight_mass="
                << row.activeAssociationFactorWeightMass
                << " active_association_factor_correct_mass="
                << row.activeAssociationFactorCorrectMass
                << " active_association_factor_false_merge_mass="
                << row.activeAssociationFactorFalseMergeMass
                << " association_rejected_candidate_count="
                << row.associationRejectedCandidateCount
                << " association_rejected_weight_mass="
                << row.associationRejectedWeightMass
                << " association_cycle_checked_entries="
                << row.associationCycleCheckedEntries
                << " association_cycle_kept_entries="
                << row.associationCycleKeptEntries
                << " association_cycle_pruned_entries="
                << row.associationCyclePrunedEntries
                << " association_local_gradient_norm="
                << row.associationLocalGradientNorm
                << " association_cycle_gradient_norm="
                << row.associationCycleGradientNorm
                << " association_cycle_to_local_gradient_ratio="
                << row.associationCycleToLocalGradientRatio
                << " association_cycle_objective_cost="
                << row.associationCycleObjectiveCost
                << " association_cycle_objective_active_triplets="
                << row.associationCycleObjectiveActiveTriplets
                << " association_relational_geometry_gradient_norm="
                << row.associationRelationalGeometryGradientNorm
                << " association_relational_geometry_to_local_gradient_ratio="
                << row.associationRelationalGeometryToLocalGradientRatio
                << " association_relational_geometry_objective_cost="
                << row.associationRelationalGeometryObjectiveCost
                << " association_actual_relational_geometry_alpha="
                << row.associationActualRelationalGeometryAlpha
                << " association_roto_translation_alignment_gradient_norm="
                << row.associationRotoTranslationAlignmentGradientNorm
                << " association_roto_translation_alignment_to_local_gradient_ratio="
                << row.associationRotoTranslationAlignmentToLocalGradientRatio
                << " association_roto_translation_alignment_objective_cost="
                << row.associationRotoTranslationAlignmentObjectiveCost
                << " association_actual_roto_translation_alignment_alpha="
                << row.associationActualRotoTranslationAlignmentAlpha
                << " association_lower_validation_gradient_norm="
                << row.associationLowerValidationGradientNorm
                << " association_lower_validation_to_local_gradient_ratio="
                << row.associationLowerValidationToLocalGradientRatio
                << " association_lower_validation_objective_cost="
                << row.associationLowerValidationObjectiveCost
                << " association_actual_lower_validation_alpha="
                << row.associationActualLowerValidationAlpha
                << " association_lower_validation_merit_gate="
                << row.associationLowerValidationMeritGate
                << " association_lower_validation_merit_decrease="
                << row.associationLowerValidationMeritDecrease
                << " association_lower_validation_pairwise_gradient_norm="
                << row.associationLowerValidationPairwiseGradientNorm
                << " association_lower_validation_pairwise_objective_cost="
                << row.associationLowerValidationPairwiseObjectiveCost
                << " association_validation_risk_gradient_norm="
                << row.associationValidationRiskGradientNorm
                << " association_validation_risk_to_local_gradient_ratio="
                << row.associationValidationRiskToLocalGradientRatio
                << " association_validation_risk_objective_cost="
                << row.associationValidationRiskObjectiveCost
                << " association_actual_validation_risk_alpha="
                << row.associationActualValidationRiskAlpha
                << " association_implicit_validation_adjoint_gradient_norm="
                << row.associationImplicitValidationAdjointGradientNorm
                << " association_implicit_validation_adjoint_to_local_gradient_ratio="
                << row.associationImplicitValidationAdjointToLocalGradientRatio
                << " association_implicit_validation_adjoint_objective_cost="
                << row.associationImplicitValidationAdjointObjectiveCost
                << " association_implicit_validation_adjoint_rhs_gradient_norm="
                << row.associationImplicitValidationAdjointRhsGradientNorm
                << " association_actual_implicit_validation_adjoint_alpha="
                << row.associationActualImplicitValidationAdjointAlpha
                << " association_implicit_validation_adjoint_active_object_count="
                << row.associationImplicitValidationAdjointActiveObjectCount
                << " association_implicit_validation_adjoint_merit_gate="
                << row.associationImplicitValidationAdjointMeritGate
                << " association_implicit_validation_adjoint_merit_decrease="
                << row.associationImplicitValidationAdjointMeritDecrease
                << " association_implicit_validation_adjoint_backend_alignment="
                << row.associationImplicitValidationAdjointBackendAlignment
                << " association_implicit_validation_adjoint_upper_prefix_alignment="
                << row
                       .associationImplicitValidationAdjointUpperPrefixAlignment
                << " association_implicit_validation_adjoint_projected_conflict_norm="
                << row
                       .associationImplicitValidationAdjointProjectedConflictNorm
                << " association_implicit_validation_adjoint_upper_prefix_projected_conflict_norm="
                << row
                       .associationImplicitValidationAdjointUpperPrefixProjectedConflictNorm
                << " association_coverage_prior_gradient_norm="
                << row.associationCoveragePriorGradientNorm
                << " association_coverage_prior_to_local_gradient_ratio="
                << row.associationCoveragePriorToLocalGradientRatio
                << " association_coverage_prior_objective_cost="
                << row.associationCoveragePriorObjectiveCost
                << " association_actual_coverage_prior_alpha="
                << row.associationActualCoveragePriorAlpha
                << " association_support_sharpness_gradient_norm="
                << row.associationSupportSharpnessGradientNorm
                << " association_support_sharpness_to_local_gradient_ratio="
                << row.associationSupportSharpnessToLocalGradientRatio
                << " association_support_sharpness_objective_cost="
                << row.associationSupportSharpnessObjectiveCost
                << " association_actual_support_sharpness_alpha="
                << row.associationActualSupportSharpnessAlpha
                << " association_support_sharpness_validation_gate_mean="
                << row.associationSupportSharpnessValidationGateMean
                << " association_support_sharpness_coverage_gate_mean="
                << row.associationSupportSharpnessCoverageGateMean
                << " association_support_sharpness_lower_feedback_alignment="
                << row.associationSupportSharpnessLowerFeedbackAlignment
                << " association_support_sharpness_projected_conflict_norm="
                << row.associationSupportSharpnessProjectedConflictNorm
                << " association_structural_expansion_gradient_norm="
                << row.associationStructuralExpansionGradientNorm
                << " association_structural_expansion_to_local_gradient_ratio="
                << row.associationStructuralExpansionToLocalGradientRatio
                << " association_structural_expansion_objective_cost="
                << row.associationStructuralExpansionObjectiveCost
                << " association_actual_structural_expansion_alpha="
                << row.associationActualStructuralExpansionAlpha
                << " association_reference_support_sharpening_gradient_norm="
                << row.associationReferenceSupportSharpeningGradientNorm
                << " association_reference_support_sharpening_to_local_gradient_ratio="
                << row.associationReferenceSupportSharpeningToLocalGradientRatio
                << " association_reference_support_sharpening_objective_cost="
                << row.associationReferenceSupportSharpeningObjectiveCost
                << " association_actual_reference_support_sharpening_alpha="
                << row.associationActualReferenceSupportSharpeningAlpha
                << " association_reference_trust_gradient_norm="
                << row.associationReferenceTrustGradientNorm
                << " association_reference_trust_to_local_gradient_ratio="
                << row.associationReferenceTrustToLocalGradientRatio
                << " association_reference_trust_objective_cost="
                << row.associationReferenceTrustObjectiveCost
                << " association_actual_reference_trust_alpha="
                << row.associationActualReferenceTrustAlpha
                << " association_backend_gradient_norm="
                << row.associationBackendGradientNorm
                << " association_backend_raw_gradient_norm="
                << row.associationBackendRawGradientNorm
                << " association_backend_to_local_gradient_ratio="
                << row.associationBackendToLocalGradientRatio
                << " association_backend_raw_to_local_gradient_ratio="
                << row.associationBackendRawToLocalGradientRatio
                << " association_backend_cap_scale="
                << row.associationBackendCapScale
                << " association_backend_stationarity_gradient_norm="
                << row.associationBackendStationarityGradientNorm
                << " association_backend_snapshot_active="
                << row.associationBackendSnapshotActive
                << " association_backend_stationarity_trusted="
                << row.associationBackendStationarityTrusted
                << " association_upper_gradient_norm="
                << row.associationUpperGradientNorm
                << " association_accumulated_upper_gradient_norm="
                << row.associationAccumulatedUpperGradientNorm
                << " association_adaptive_eta="
                << row.associationAdaptiveEta
                << " association_upper_tracking_gradient_norm="
                << row.associationUpperTrackingGradientNorm
                << " association_upper_tracking_correction_norm="
                << row.associationUpperTrackingCorrectionNorm
                << " association_upper_tracking_consensus_residual_norm="
                << row.associationUpperTrackingConsensusResidualNorm
                << " association_upper_tracking_local_gradient_delta_norm="
                << row.associationUpperTrackingLocalGradientDeltaNorm
                << " association_upper_tracking_lower_validation_merit_gate="
                << row.associationUpperTrackingLowerValidationMeritGate
                << " association_upper_tracking_lower_validation_merit_decrease="
                << row.associationUpperTrackingLowerValidationMeritDecrease
                << " association_state_validation_hypergradient_gradient_norm="
                << row.associationStateValidationHypergradientGradientNorm
                << " association_state_validation_hypergradient_objective_cost="
                << row.associationStateValidationHypergradientObjectiveCost
                << " association_state_validation_unified_objective_cost="
                << row.associationStateValidationUnifiedObjectiveCost
                << " association_state_validation_merit_gate="
                << row.associationStateValidationMeritGate
                << " association_state_validation_merit_ratio="
                << row.associationStateValidationMeritRatio
                << " association_state_validation_merit_decrease="
                << row.associationStateValidationMeritDecrease
                << " association_state_validation_trajectory_residual_cost="
                << row.associationStateValidationTrajectoryResidualCost
                << " association_state_validation_pairwise_trajectory_cost="
                << row.associationStateValidationPairwiseTrajectoryCost
                << " association_state_validation_transform_confidence_cost="
                << row.associationStateValidationTransformConfidenceCost
                << " association_state_validation_transform_confidence_mean="
                << row.associationStateValidationTransformConfidenceMean
                << " association_state_validation_object_consistency_cost="
                << row.associationStateValidationObjectConsistencyCost
                << " association_state_validation_gauge_stationarity_cost="
                << row.associationStateValidationGaugeStationarityCost
                << " association_edge_latent_consensus_gradient_norm="
                << row.associationEdgeLatentConsensusGradientNorm
                << " association_edge_latent_consensus_objective_cost="
                << row.associationEdgeLatentConsensusObjectiveCost
                << " association_edge_latent_consensus_residual_norm="
                << row.associationEdgeLatentConsensusResidualNorm
                << " association_edge_latent_consensus_confidence_mean="
                << row.associationEdgeLatentConsensusConfidenceMean
                << " association_shared_edge_candidate_count="
                << row.associationSharedEdgeCandidateCount
                << " association_shared_edge_objective_scalar_count="
                << row.associationSharedEdgeObjectiveScalarCount
                << " association_shared_edge_payload_bytes="
                << row.associationSharedEdgePayloadBytes
                << " association_shared_edge_payload_mb="
                << row.associationSharedEdgePayloadMb
                << " association_shared_edge_cumulative_mb="
                << row.associationSharedEdgeCumulativeMb
                << " association_shared_edge_cumulative_total_mb="
                << row.associationSharedEdgeCumulativeTotalMb
                << " association_fresh_uot_marginal_residual_norm="
                << row.associationFreshUotMarginalResidualNorm
                << " association_fresh_uot_response_mass="
                << row.associationFreshUotResponseMass
                << " association_upper_fixed_point_residual_norm="
                << row.associationUpperFixedPointResidualNorm
                << " association_lower_feedback_readiness_mean="
                << row.associationLowerFeedbackReadinessMean
                << " association_mann_step_mean="
                << row.associationMannStepMean
                << " association_actual_cycle_alpha="
                << row.associationActualCycleAlpha
                << " association_actual_backend_alpha="
                << row.associationActualBackendAlpha
                << " association_temporal_kl_cost="
                << row.associationTemporalKlCost
                << " association_temporal_kl_gradient_norm="
                << row.associationTemporalKlGradientNorm
                << " association_entropy_regularization_cost="
                << row.associationEntropyRegularizationCost
                << " association_entropy_regularization_gradient_norm="
                << row.associationEntropyRegularizationGradientNorm
                << " association_churn_l1="
                << row.associationChurnL1
                << " association_churn_frobenius="
                << row.associationChurnFrobenius
                << " association_actual_temporal_alpha="
                << row.associationActualTemporalAlpha
                << " association_actual_entropy_alpha="
                << row.associationActualEntropyAlpha
                << " lower_stationarity_gradient_norm="
                << row.lowerStationarityGradientNorm
                << " lower_gauge_marginalized_stationarity_norm="
                << row.lowerGaugeMarginalizedStationarityNorm
                << " lower_gauge_component_norm="
                << row.lowerGaugeComponentNorm
                << " lower_gauge_component_ratio="
                << row.lowerGaugeComponentRatio
                << " lower_shared_object_consensus_active_hypotheses="
                << row.lowerSharedObjectConsensusActiveHypotheses
                << " lower_shared_object_consensus_primal_residual="
                << row.lowerSharedObjectConsensusPrimalResidual
                << " lower_shared_object_consensus_dual_residual="
                << row.lowerSharedObjectConsensusDualResidual
                << " lower_shared_object_consensus_normalized_primal_residual="
                << row.lowerSharedObjectConsensusNormalizedPrimalResidual
                << " lower_shared_object_consensus_normalized_dual_residual="
                << row.lowerSharedObjectConsensusNormalizedDualResidual
                << " lower_shared_object_consensus_adaptive_beta_mean="
                << row.lowerSharedObjectConsensusAdaptiveBetaMean
                << " lower_shared_object_consensus_adaptive_beta_min="
                << row.lowerSharedObjectConsensusAdaptiveBetaMin
                << " lower_shared_object_consensus_adaptive_beta_max="
                << row.lowerSharedObjectConsensusAdaptiveBetaMax
                << " lower_shared_object_consensus_adaptive_beta_increases="
                << row.lowerSharedObjectConsensusAdaptiveBetaIncreases
                << " lower_shared_object_consensus_adaptive_beta_decreases="
                << row.lowerSharedObjectConsensusAdaptiveBetaDecreases
                << " lower_shared_object_consensus_adaptive_midpoint_residual="
                << row.lowerSharedObjectConsensusAdaptiveMidpointResidual
                << " lower_shared_object_consensus_target_messages="
                << row.lowerSharedObjectConsensusTargetMessages
                << " lower_shared_object_consensus_delivered_messages="
                << row.lowerSharedObjectConsensusDeliveredMessages
                << " lower_shared_object_consensus_current_incident_certificates="
                << row.lowerSharedObjectConsensusCurrentIncidentCertificates
                << " lower_shared_object_consensus_expected_incident_certificates="
                << row.lowerSharedObjectConsensusExpectedIncidentCertificates
                << " lower_shared_object_consensus_outstanding_messages="
                << row.lowerSharedObjectConsensusOutstandingMessages
                << " lower_shared_object_consensus_dropped_attempts="
                << row.lowerSharedObjectConsensusDroppedAttempts
                << " lower_shared_object_consensus_retry_attempts="
                << row.lowerSharedObjectConsensusRetryAttempts
                << " lower_shared_object_consensus_payload_mb="
                << row.lowerSharedObjectConsensusPayloadMb
                << " comm_pose_count=" << row.commPoseCount
                << " iter_comm_mb=" << row.iterCommMb
                << " cumulative_comm_pose_count="
                << row.cumulativeCommPoseCount
                << " cumulative_comm_mb=" << row.cumulativeCommMb
                << " cumulative_total_comm_mb="
                << row.cumulativeTotalCommMb
                << " integrated_point2_amm_attempts="
                << row.integratedPoint2AmmAttempts
                << " integrated_point2_amm_accepts="
                << row.integratedPoint2AmmAccepts
                << " integrated_point2_amm_restarts="
                << row.integratedPoint2AmmRestarts
                << " integrated_point2_edge_inertia_attempts="
                << row.integratedPoint2EdgeInertiaAttempts
                << " integrated_point2_edge_inertia_accepts="
                << row.integratedPoint2EdgeInertiaAccepts
                << " integrated_point2_edge_inertia_restarts="
                << row.integratedPoint2EdgeInertiaRestarts
                << " integrated_point2_edge_baseline_merit="
                << row.integratedPoint2EdgeBaselineMerit
                << " integrated_point2_edge_accelerated_merit="
                << row.integratedPoint2EdgeAcceleratedMerit
                << " integrated_point2_owner_momentum_valid_contributions="
                << row.integratedPoint2OwnerMomentumValidContributions
                << " integrated_point2_owner_momentum_mean="
                << row.integratedPoint2OwnerMomentumMean
                << " integrated_point2_owner_momentum_min="
                << row.integratedPoint2OwnerMomentumMin
                << " integrated_point2_owner_momentum_max="
                << row.integratedPoint2OwnerMomentumMax
                << " exact_point2_iteration_payload_bytes="
                << row.exactPoint2IterationPayloadBytes
                << " exact_point2_cumulative_payload_bytes="
                << row.exactPoint2CumulativePayloadBytes
                << " exact_point2_max_copy_spread="
                << row.exactPoint2MaxCopySpread
                << " exact_point2_physical_cost_after_trajectory="
                << row.exactPoint2PhysicalCostAfterTrajectory
                << " exact_point2_physical_cost_after_update="
                << row.exactPoint2PhysicalCostAfterUpdate
                << " point2_schur_oracle_active="
                << row.point2SchurOracleActive
                << " point2_schur_oracle_response_solves="
                << row.point2SchurOracleResponseSolves
                << " point2_schur_oracle_max_response_relative_residual="
                << row.point2SchurOracleMaxResponseRelativeResidual
                << " point2_schur_oracle_symmetry_error="
                << row.point2SchurOracleSymmetryError
                << " point2_schur_oracle_spectral_shift="
                << row.point2SchurOracleSpectralShift
                << " point2_schur_oracle_linear_relative_residual="
                << row.point2SchurOracleLinearRelativeResidual
                << " point2_schur_oracle_backtracking_trials="
                << row.point2SchurOracleBacktrackingTrials
                << " point2_schur_oracle_accepted_step="
                << row.point2SchurOracleAcceptedStep
                << " point2_schur_oracle_hypothetical_dense_payload_bytes="
                << row.point2SchurOracleHypotheticalDensePayloadBytes
                << " point2_owner_pcg_active=" << row.point2OwnerPcgActive
                << " point2_owner_pcg_iterations="
                << row.point2OwnerPcgIterations
                << " point2_owner_pcg_relative_residual="
                << row.point2OwnerPcgRelativeResidual
                << " point2_owner_pcg_gradient_payload_bytes="
                << row.point2OwnerPcgGradientPayloadBytes
                << " point2_owner_pcg_direction_payload_bytes="
                << row.point2OwnerPcgDirectionPayloadBytes
                << " point2_owner_pcg_product_payload_bytes="
                << row.point2OwnerPcgProductPayloadBytes
                << " point2_owner_pcg_update_payload_bytes="
                << row.point2OwnerPcgUpdatePayloadBytes
                << " point2_owner_pcg_scalar_payload_bytes="
                << row.point2OwnerPcgScalarPayloadBytes
                << " exact_point2_status=" << row.exactPoint2Status
                << " event_robot=" << row.eventRobot
                << " event_logical_time=" << row.eventLogicalTime
                << " equivalent_sweep=" << row.equivalentSweep
                << " queue_drain=" << (row.queueDrain ? 1 : 0)
                << " shared_endpoint_state_queued_messages="
                << row.sharedEndpointStateQueuedMessages
                << " shared_endpoint_state_delivered_messages="
                << row.sharedEndpointStateDeliveredMessages
                << " shared_endpoint_state_stale_messages="
                << row.sharedEndpointStateStaleMessages
                << " shared_endpoint_state_outstanding_messages="
                << row.sharedEndpointStateOutstandingMessages
                << " shared_endpoint_state_payload_bytes="
                << row.sharedEndpointStatePayloadBytes
                << std::endl;
      }
      for (const DPGO::ManualObjectDranObjectInterfaceResponseSummary &row :
           result.objectInterfaceResponseSummaries) {
        std::cout << "OBJECT_INTERFACE_RESPONSE iter=" << row.iter
                << " scheduled=" << (row.scheduled ? 1 : 0)
                << " active_robots=" << row.activeRobots
                << " candidate_object_blocks="
                << row.candidateObjectBlocks
                << " accepted_object_blocks=" << row.acceptedObjectBlocks
                << " rejected_object_blocks=" << row.rejectedObjectBlocks
                << " batch_response_robots=" << row.batchResponseRobots
                << " accepted_robot_batches=" << row.acceptedRobotBatches
                << " rejected_robot_batches=" << row.rejectedRobotBatches
                << " trigger_candidate_robots="
                << row.triggerCandidateRobots
                << " triggered_robots=" << row.triggeredRobots
                << " skipped_trigger_robots="
                << row.skippedTriggerRobots
                << " model_cost_before=" << row.modelCostBefore
                << " model_cost_after=" << row.modelCostAfter
                << " interface_innovation_score="
                << row.interfaceInnovationScore
                << " accumulated_innovation_score="
                << row.accumulatedInnovationScore
                << " predicted_model_decrease="
                << row.predictedModelDecrease
                << " accepted_model_decrease="
                << row.acceptedModelDecrease
                << " step_norm=" << row.stepNorm
                << " pcg_iterations=" << row.pcgIterations
                << " pcg_relative_residual=" << row.pcgRelativeResidual
                << " communication_payload_bytes="
                << row.communicationPayloadBytes
                << " communication_messages="
                << row.communicationMessages
                << " copy_translation_rms="
                << row.copyTranslationRms
                << " copy_rotation_rms_deg="
                << row.copyRotationRmsDegrees
                << " model_assembly_seconds="
                << row.modelAssemblySeconds
                << " private_factorization_seconds="
                << row.privateFactorizationSeconds
                << " private_gradient_solve_seconds="
                << row.privateGradientSolveSeconds
                << " schur_apply_seconds=" << row.schurApplySeconds
                << " private_recovery_seconds="
                << row.privateRecoverySeconds
                << " block_tridiagonal_fallbacks="
                << row.blockTridiagonalFallbacks
                << " private_hessian_blocks="
                << row.privateHessianBlocks
                << " private_interface_coupling_blocks="
                << row.privateInterfaceCouplingBlocks
                << " private_interface_dense_equivalent_bytes="
                << row.privateInterfaceDenseEquivalentBytes
                << " private_interface_stored_bytes="
                << row.privateInterfaceStoredBytes
                << " sparse_compatibility_materializations="
                << row.sparseCompatibilityMaterializations
                << " sparse_structure_fallbacks="
                << row.sparseStructureFallbacks
                << " status=" << row.status << std::endl;
      }
    }

    if (save) {
      const std::filesystem::path outputPath(outputDir);
      std::filesystem::create_directories(outputPath);
      writeIterationCsv(outputPath / "iteration_summary.csv", result);
      writeSelectionSummaryCsv(outputPath / "selection_summary.csv",
                               result);
      writeTrackEmpiricalInformationSummaryCsv(
          outputPath / "track_empirical_information_summary.csv", result);
      writeObjectInterfaceCsv(outputPath / "object_interface_summary.csv",
                              result);
      writeObjectInterfaceResponseCsv(
          outputPath / "object_interface_response_summary.csv", result);
      writeFinalAssociationWeightsCsv(
          outputPath / "final_association_weights.csv", result);
      if (!result.finalAssociationUpperResponseDiagnostics.empty()) {
        writeFinalAssociationUpperResponseCsv(
            outputPath / "final_association_upper_response.csv", result);
      }
      writeFinalAssociationTransformsCsv(
          outputPath / "final_association_transforms.csv", result);
      writeFinalRobotGaugesCsv(outputPath / "final_robot_gauges.csv", result,
                               options.associationTransformPlaneAxes);
      if (!result.finalPairCavityCovariances.empty()) {
        writeFinalPairCavityCovarianceCsv(
            outputPath / "final_pair_cavity_covariance.csv",
            result.finalPairCavityCovariances);
        writeFinalPairCavityMeansCsv(
            outputPath / "final_pair_cavity_means.csv",
            result.finalPairCavityCovariances);
      }
      for (const DPGO::ManualObjectDranLeavePairCavityDiagnostic &cavity :
           result.finalLeavePairCavities) {
        const std::filesystem::path cavityPath =
            outputPath /
            ("leave_pair_cavity_" + std::to_string(cavity.firstRobot) + "_" +
             std::to_string(cavity.secondRobot));
        std::filesystem::create_directories(cavityPath);
        writeFinalPairCavityCovarianceCsv(
            cavityPath / "final_pair_cavity_covariance.csv", cavity.robots);
        writeFinalPairCavityMeansCsv(
            cavityPath / "final_pair_cavity_means.csv", cavity.robots);
        writeFinalAssociationTransformsCsv(
            cavityPath / "final_association_transforms.csv", result);
      }
      if (!result.cavityFeedbackSummaries.empty()) {
        writeCavityFeedbackSummaryCsv(
            outputPath / "cavity_feedback_summary.csv", result);
      }
      if (!result.networkPartitionZoneSummaries.empty()) {
        writeNetworkPartitionZoneSummaryCsv(
            outputPath / "network_partition_zone_summary.csv", result);
      }
      if (!result.networkPartitionProtocolSummaries.empty()) {
        writeNetworkPartitionProtocolSummaryCsv(
            outputPath / "network_partition_protocol_summary.csv", result);
      }
      if (!result.finalAssociationEdgeStates.empty()) {
        writeFinalAssociationEdgeStatesCsv(
            outputPath / "final_association_edge_states.csv", result);
      }
      const DPGO::ObjectPGODirectoryData dataset =
          DPGO::loadObjectAwareG2ODirectory(dataDir, options.numRobots,
                                            options.numObjects);
      writeObjectAwareEstimates(outputPath / "object_poses.txt", dataset,
                                result);
      if (options.associationSaveIterationEstimateSnapshots) {
        writeIterationEstimateSnapshots(outputPath / "iteration_estimates",
                                        dataset, result);
      }
    }

    if (!quiet && !result.iterations.empty()) {
      const auto &last = result.iterations.back();
      std::cout << "final measurement_cost: " << last.measurementCost
                << std::endl;
      std::cout << "final heldout_measurement_cost: "
                << last.heldoutMeasurementCost << std::endl;
      std::cout << "final predictive_measurement_cost: "
                << last.predictiveMeasurementCost << std::endl;
      std::cout << "final consensus_cost: " << last.consensusCost
                << std::endl;
      std::cout << "final augmented_cost: "
                << last.measurementCost + last.consensusCost << std::endl;
      std::cout << "final gradient: " << last.gradient << std::endl;
      std::cout << "time: " << result.elapsedSeconds << " s" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "manual_object_dran failed: " << e.what() << std::endl;
    return 2;
  }

  return 0;
}
