#include <DPGO/ManualObjectDran.h>
#include <DPGO/SyntheticObjectPGO.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> splitCommaList(const std::string &value) {
  std::vector<std::string> result;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
}

bool parseBool(const std::string &value) {
  if (value == "1" || value == "true" || value == "TRUE" ||
      value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE" ||
      value == "no" || value == "off") {
    return false;
  }
  throw std::invalid_argument("Expected boolean value, got: " + value);
}

std::string boolString(bool value) { return value ? "true" : "false"; }

std::string csvQuote(const std::string &value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += '"';
  return escaped;
}

template <typename T>
std::string valueString(const T &value) {
  std::ostringstream out;
  out << std::setprecision(20) << value;
  return out.str();
}

std::string shellQuote(const std::string &value) {
  if (value.empty()) {
    return "''";
  }
  const bool needsQuote =
      value.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyz"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "0123456789_@%+=:,./-") != std::string::npos;
  if (!needsQuote) {
    return value;
  }
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

std::string commandLine(int argc, char **argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) {
      out << ' ';
    }
    out << shellQuote(argv[i]);
  }
  return out.str();
}

std::string joinList(const std::vector<std::string> &items) {
  std::ostringstream out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << items[i];
  }
  return out.str();
}

std::string canonicalMethodName(const std::string &method) {
  if (method == "oracle" || method == "identity_oracle" ||
      method == "identity") {
    return "oracle";
  }
  if (method == "no_cross_object" || method == "no_object_fusion" ||
      method == "beta0" || method == "beta_zero") {
    return "no_cross_object";
  }
  if (method == "class_only" || method == "class_only_hard" ||
      method == "semantic_class_only") {
    return "class_only";
  }
  if (method == "metadata_hard" ||
      method == "semantic_shape_descriptor_hard") {
    return "metadata_hard";
  }
  if (method == "metadata_geometry_hard" ||
      method == "semantic_geometry_hard" ||
      method == "semantic_geometry_gate") {
    return "metadata_geometry_hard";
  }
  if (method == "geometry_hard" || method == "hard_geometry") {
    return "geometry_hard";
  }
  if (method == "graph_consistency_hard" ||
      method == "slide_graph_hard" ||
      method == "slide_slam_hard" ||
      method == "clipper_graph_hard") {
    return "graph_consistency_hard";
  }
  if (method == "geometry_soft" || method == "soft_geometry") {
    return "geometry_soft";
  }
  if (method == "transform_soft" || method == "transform_aware_soft" ||
      method == "local_frame_soft") {
    return "transform_soft";
  }
  if (method == "residual_soft" || method == "dba_residual") {
    return "residual_soft";
  }
  if (method == "backend_feedback_soft" ||
      method == "dba_backend_feedback" ||
      method == "residual_backend_feedback") {
    return "backend_feedback_soft";
  }
  if (method == "backend_response_soft" ||
      method == "dba_backend_response" ||
      method == "residual_backend_response") {
    return "backend_response_soft";
  }
  if (method == "backend_adjoint_soft" ||
      method == "dba_backend_adjoint" ||
      method == "backend_signed_adjoint_soft" ||
      method == "dba_hypergradient_proxy") {
    return "backend_adjoint_soft";
  }
  if (method == "backend_coupled_adjoint_soft" ||
      method == "dba_backend_coupled_adjoint" ||
      method == "backend_local_adjoint_soft") {
    return "backend_coupled_adjoint_soft";
  }
  if (method == "backend_penalty_coupled_adjoint_soft" ||
      method == "dba_backend_penalty_coupled_adjoint" ||
      method == "backend_coupled_adjoint_penalty_soft") {
    return "backend_penalty_coupled_adjoint_soft";
  }
  if (method == "backend_guarded_coupled_adjoint_soft" ||
      method == "dba_backend_guarded_coupled_adjoint" ||
      method == "backend_metadata_guarded_coupled_adjoint_soft") {
    return "backend_guarded_coupled_adjoint_soft";
  }
  if (method == "backend_hybrid_soft" ||
      method == "backend_feedback_response_soft" ||
      method == "dba_backend_hybrid") {
    return "backend_hybrid_soft";
  }
  throw std::invalid_argument("Unknown synthetic DBA method: " + method);
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
  return "unknown";
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
    case DPGO::ManualObjectDranAssociationPolicy::DecentralizedFeedbackUotDbaV1:
      return "decentralized_feedback_uot_bilevel_v1";
  }
  return "manual";
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
  if (value == "shared_edge_bilevel_v1" || value == "shared_edge_v1") {
    return DPGO::ManualObjectDranAssociationPolicy::SharedEdgeDbaV1;
  }
  if (value == "shared_edge_joint_bilevel_v2" ||
      value == "shared_edge_joint_v2") {
    return DPGO::ManualObjectDranAssociationPolicy::SharedEdgeJointDbaV2;
  }
  if (value == "decentralized_feedback_uot_bilevel_v1" ||
      value == "feedback_uot_v1") {
    return DPGO::ManualObjectDranAssociationPolicy::
        DecentralizedFeedbackUotDbaV1;
  }
  throw std::invalid_argument("Unknown association policy: " + value);
}

DPGO::ManualObjectDranAssociationMode associationModeForMethod(
    const std::string &method) {
  const std::string canonical = canonicalMethodName(method);
  if (canonical == "no_cross_object") {
    return DPGO::ManualObjectDranAssociationMode::GeometrySoft;
  }
  if (canonical == "oracle") {
    return DPGO::ManualObjectDranAssociationMode::IdentityOracle;
  }
  if (canonical == "class_only") {
    return DPGO::ManualObjectDranAssociationMode::ClassOnlyHard;
  }
  if (canonical == "metadata_hard") {
    return DPGO::ManualObjectDranAssociationMode::MetadataHard;
  }
  if (canonical == "geometry_hard" ||
      canonical == "metadata_geometry_hard") {
    return DPGO::ManualObjectDranAssociationMode::GeometryHard;
  }
  if (canonical == "graph_consistency_hard") {
    return DPGO::ManualObjectDranAssociationMode::GraphConsistencyHard;
  }
  if (canonical == "geometry_soft") {
    return DPGO::ManualObjectDranAssociationMode::GeometrySoft;
  }
  if (canonical == "transform_soft") {
    return DPGO::ManualObjectDranAssociationMode::TransformSoft;
  }
  if (canonical == "residual_soft" ||
      canonical == "backend_feedback_soft" ||
      canonical == "backend_response_soft" ||
      canonical == "backend_adjoint_soft" ||
      canonical == "backend_coupled_adjoint_soft" ||
      canonical == "backend_penalty_coupled_adjoint_soft" ||
      canonical == "backend_guarded_coupled_adjoint_soft" ||
      canonical == "backend_hybrid_soft") {
    return DPGO::ManualObjectDranAssociationMode::ResidualSoft;
  }
  throw std::invalid_argument("Unknown synthetic DBA method: " + method);
}

void copySyntheticMetadataOptions(
    const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
    DPGO::ManualObjectDranOptions &options) {
  options.associationMetadataNumCategories = syntheticOptions.numCategories;
  options.associationMetadataDescriptorDim = syntheticOptions.descriptorDim;
  options.associationMetadataSemanticAccuracy =
      syntheticOptions.semanticAccuracy;
  options.associationMetadataDescriptorNoise =
      syntheticOptions.descriptorNoise;
  options.associationMetadataShapeNoise = syntheticOptions.shapeNoise;
  options.associationMetadataRepeatedObjectRatio =
      syntheticOptions.repeatedObjectRatio;
}

void configureAssociationForMethod(const std::string &method,
                                   DPGO::ManualObjectDranOptions &options) {
  const std::string canonical = canonicalMethodName(method);
  const double requestedFeedbackGradientScale =
      options.associationBackendFeedbackGradientScale;
  const double requestedFeedbackMinTrust =
      options.associationBackendFeedbackMinTrust;
  const double requestedResponseGain =
      options.associationBackendResponseGain;
  const double requestedResponseDamping =
      options.associationBackendResponseDamping;
  const double requestedResponseMaxReductionFraction =
      options.associationBackendResponseMaxReductionFraction;
  const double requestedAdjointGain =
      options.associationBackendAdjointGain;
  const double requestedAdjointDamping =
      options.associationBackendAdjointDamping;
  const double requestedAdjointMaxReductionFraction =
      options.associationBackendAdjointMaxReductionFraction;
  const double requestedAdjointMaxIncreaseFraction =
      options.associationBackendAdjointMaxIncreaseFraction;
  const double requestedCoupledAdjointGain =
      options.associationBackendCoupledAdjointGain;
  const double requestedCoupledAdjointDamping =
      options.associationBackendCoupledAdjointDamping;
  const double requestedCoupledAdjointMaxReductionFraction =
      options.associationBackendCoupledAdjointMaxReductionFraction;
  const double requestedCoupledAdjointMaxIncreaseFraction =
      options.associationBackendCoupledAdjointMaxIncreaseFraction;
  const double requestedAdjointReductionMetadataMargin =
      options.associationBackendAdjointReductionMetadataMargin;
  const bool requestedAdjointAllowCostReduction =
      options.associationBackendAdjointAllowCostReduction;
  const std::size_t requestedBackendSnapshotPeriod =
      options.associationBackendSnapshotPeriod;
  const double requestedBackendMaxEffectiveRatio =
      options.associationBackendMaxEffectiveRatio;
  const double requestedBackendStationarityTrustMaxGradientNorm =
      options.associationBackendStationarityTrustMaxGradientNorm;
  options.associationBackendFeedbackGradientScale = 0.0;
  options.associationBackendFeedbackMinTrust = 0.0;
  options.associationBackendResponseGain = 0.0;
  options.associationBackendResponseDamping = 1.0;
  options.associationBackendResponseMaxReductionFraction = 0.5;
  options.associationBackendAdjointGain = 0.0;
  options.associationBackendAdjointDamping = 1.0;
  options.associationBackendAdjointMaxReductionFraction = 0.5;
  options.associationBackendAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendCoupledAdjointGain = 0.0;
  options.associationBackendCoupledAdjointDamping = 1.0;
  options.associationBackendCoupledAdjointMaxReductionFraction = 0.5;
  options.associationBackendCoupledAdjointMaxIncreaseFraction = 0.5;
  options.associationBackendAdjointReductionMetadataMargin = -1.0;
  options.associationBackendAdjointAllowCostReduction = true;
  options.associationBackendSnapshotPeriod = 1;
  options.associationBackendMaxEffectiveRatio = 0.0;
  options.associationBackendStationarityTrustMaxGradientNorm = 0.0;

  options.associationMode = associationModeForMethod(canonical);
  if (canonical == "no_cross_object") {
    options.associationUseSyntheticMetadata = false;
    options.associationGeometryScale = 1.0;
    return;
  }
  if (canonical == "geometry_hard" || canonical == "geometry_soft") {
    options.associationUseSyntheticMetadata = false;
    options.associationGeometryScale = 1.0;
    options.associationSemanticWeight = 0.0;
    options.associationShapeWeight = 0.0;
    options.associationDescriptorWeight = 0.0;
    return;
  }
  if (canonical == "class_only") {
    options.associationUseSyntheticMetadata = true;
    options.associationGeometryScale = 0.0;
    options.associationSemanticWeight = 1.0;
    options.associationShapeWeight = 0.0;
    options.associationDescriptorWeight = 0.0;
    return;
  }
  if (canonical == "metadata_hard") {
    options.associationUseSyntheticMetadata = true;
    options.associationGeometryScale = 0.0;
    return;
  }
  if (canonical == "transform_soft") {
    options.associationUseSyntheticMetadata = true;
    options.associationGeometryScale = 0.0;
    options.associationMaxCandidatesPerEdge =
        options.associationMaxCandidatesPerEdge > 0
            ? options.associationMaxCandidatesPerEdge
            : 8;
    return;
  }
  if (canonical == "metadata_geometry_hard" ||
      canonical == "graph_consistency_hard" ||
      canonical == "residual_soft" ||
      canonical == "backend_feedback_soft" ||
      canonical == "backend_response_soft" ||
      canonical == "backend_adjoint_soft" ||
      canonical == "backend_coupled_adjoint_soft" ||
      canonical == "backend_penalty_coupled_adjoint_soft" ||
      canonical == "backend_guarded_coupled_adjoint_soft" ||
      canonical == "backend_hybrid_soft") {
    options.associationUseSyntheticMetadata = true;
    options.associationGeometryScale = 1.0;
    options.associationBackendSnapshotPeriod =
        std::max<std::size_t>(1, requestedBackendSnapshotPeriod);
    options.associationBackendMaxEffectiveRatio =
        requestedBackendMaxEffectiveRatio;
    options.associationBackendStationarityTrustMaxGradientNorm =
        requestedBackendStationarityTrustMaxGradientNorm;
    if (canonical == "backend_feedback_soft" ||
        canonical == "backend_hybrid_soft") {
      options.associationBackendFeedbackGradientScale =
          requestedFeedbackGradientScale > 0.0
              ? requestedFeedbackGradientScale
              : 1.0;
      options.associationBackendFeedbackMinTrust =
          requestedFeedbackMinTrust > 0.0 ? requestedFeedbackMinTrust : 0.2;
    }
    if (canonical == "backend_response_soft" ||
        canonical == "backend_hybrid_soft") {
      options.associationBackendResponseGain =
          requestedResponseGain > 0.0 ? requestedResponseGain : 1.0;
      options.associationBackendResponseDamping =
          requestedResponseDamping > 0.0 ? requestedResponseDamping : 1.0;
      options.associationBackendResponseMaxReductionFraction =
          requestedResponseMaxReductionFraction > 0.0 &&
                  requestedResponseMaxReductionFraction <= 1.0
              ? requestedResponseMaxReductionFraction
              : 0.5;
    }
    if (canonical == "backend_adjoint_soft") {
      options.associationBackendAdjointAllowCostReduction =
          requestedAdjointAllowCostReduction;
      options.associationBackendAdjointGain =
          requestedAdjointGain > 0.0 ? requestedAdjointGain : 1.0;
      options.associationBackendAdjointDamping =
          requestedAdjointDamping > 0.0 ? requestedAdjointDamping : 1.0;
      options.associationBackendAdjointMaxReductionFraction =
          requestedAdjointMaxReductionFraction > 0.0 &&
                  requestedAdjointMaxReductionFraction <= 1.0
              ? requestedAdjointMaxReductionFraction
              : 0.5;
      options.associationBackendAdjointMaxIncreaseFraction =
          requestedAdjointMaxIncreaseFraction > 0.0
              ? requestedAdjointMaxIncreaseFraction
              : 0.5;
    }
    if (canonical == "backend_coupled_adjoint_soft" ||
        canonical == "backend_penalty_coupled_adjoint_soft" ||
        canonical == "backend_guarded_coupled_adjoint_soft") {
      options.associationBackendAdjointAllowCostReduction =
          canonical == "backend_penalty_coupled_adjoint_soft"
              ? false
              : requestedAdjointAllowCostReduction;
      options.associationBackendCoupledAdjointGain =
          requestedCoupledAdjointGain > 0.0 ? requestedCoupledAdjointGain
                                            : 1.0;
      options.associationBackendCoupledAdjointDamping =
          requestedCoupledAdjointDamping > 0.0
              ? requestedCoupledAdjointDamping
              : 1.0;
      options.associationBackendCoupledAdjointMaxReductionFraction =
          requestedCoupledAdjointMaxReductionFraction > 0.0 &&
                  requestedCoupledAdjointMaxReductionFraction <= 1.0
              ? requestedCoupledAdjointMaxReductionFraction
              : 0.5;
      options.associationBackendCoupledAdjointMaxIncreaseFraction =
          requestedCoupledAdjointMaxIncreaseFraction > 0.0
              ? requestedCoupledAdjointMaxIncreaseFraction
              : 0.5;
      if (canonical == "backend_guarded_coupled_adjoint_soft") {
        options.associationBackendAdjointReductionMetadataMargin =
            requestedAdjointReductionMetadataMargin >= 0.0
                ? requestedAdjointReductionMetadataMargin
                : 0.25;
      }
    }
    return;
  }
}

DPGO::ManualObjectDranOptions effectiveOptionsForMethod(
    const std::filesystem::path &dataDir, const std::string &method,
    const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
    const DPGO::ManualObjectDranOptions &backendOptions) {
  DPGO::ManualObjectDranOptions options = backendOptions;
  options.numRobots = syntheticOptions.numRobots;
  options.numObjects = DPGO::syntheticLocalObjectSlotCount(syntheticOptions);
  copySyntheticMetadataOptions(syntheticOptions, options);
  configureAssociationForMethod(method, options);
  options = DPGO::applyManualObjectDranAssociationPolicy(options);
  if (canonicalMethodName(method) == "no_cross_object") {
    options.beta = 0.0;
  }
  if (canonicalMethodName(method) == "oracle") {
    const std::filesystem::path gtAssociationFile =
        dataDir / "gt_associations.csv";
    if (std::filesystem::exists(gtAssociationFile)) {
      options.associationMode =
          DPGO::ManualObjectDranAssociationMode::FixedFileSoft;
      options.associationFile = gtAssociationFile.string();
      options.associationRejectResidualThreshold = 0.0;
      options.associationRejectResidualBestMargin = 0.0;
      options.associationRejectResidualQuantile = 0.0;
      options.associationRejectMinBadUpdates = 0;
    }
  }
  return options;
}

bool associationRejectConfigured(const DPGO::ManualObjectDranOptions &options) {
  return options.associationRejectMinBadUpdates > 0 &&
         ((std::isfinite(options.associationRejectResidualThreshold) &&
           options.associationRejectResidualThreshold > 0.0) ||
          (std::isfinite(options.associationRejectResidualBestMargin) &&
           options.associationRejectResidualBestMargin > 0.0) ||
          (std::isfinite(options.associationRejectResidualQuantile) &&
           options.associationRejectResidualQuantile > 0.0 &&
           options.associationRejectResidualQuantile <= 1.0));
}

bool associationRejectApplies(const DPGO::ManualObjectDranOptions &options) {
  return associationRejectConfigured(options) &&
         (options.associationMode ==
              DPGO::ManualObjectDranAssociationMode::ResidualSoft ||
          options.associationMode ==
              DPGO::ManualObjectDranAssociationMode::FixedFileSoft);
}

void writeConfigRow(std::ofstream &out, const std::string &section,
                    const std::string &key, const std::string &value) {
  out << csvQuote(section) << "," << csvQuote(key) << ","
      << csvQuote(value) << "\n";
}

void writeExperimentConfig(
    const std::filesystem::path &outputDir, int argc, char **argv,
    const std::vector<std::string> &methods,
    const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
    const DPGO::ManualObjectDranOptions &backendOptions, bool failureSearch,
    bool generateOnly) {
  std::ofstream out(outputDir / "experiment_config.csv");
  if (!out) {
    throw std::runtime_error("Could not open experiment_config.csv under " +
                             outputDir.string());
  }
  out << "section,key,value\n";
  writeConfigRow(out, "run", "command", commandLine(argc, argv));
  writeConfigRow(out, "run", "methods", joinList(methods));
  writeConfigRow(out, "run", "failure_search", boolString(failureSearch));
  writeConfigRow(out, "run", "generate_only", boolString(generateOnly));
  writeConfigRow(out, "synthetic", "instance_seed",
                 valueString(syntheticOptions.instanceSeed));
  writeConfigRow(out, "synthetic", "dimension",
                 valueString(syntheticOptions.dimension));
  writeConfigRow(out, "synthetic", "num_robots",
                 valueString(syntheticOptions.numRobots));
  writeConfigRow(out, "synthetic", "num_objects",
                 valueString(syntheticOptions.numObjects));
  writeConfigRow(out, "synthetic", "local_slots_per_robot",
                 valueString(DPGO::syntheticLocalObjectSlotCount(
                     syntheticOptions)));
  writeConfigRow(out, "synthetic", "poses_per_robot",
                 valueString(syntheticOptions.posesPerRobot));
  writeConfigRow(out, "synthetic", "pose_spacing",
                 valueString(syntheticOptions.poseSpacing));
  writeConfigRow(out, "synthetic", "robot_spacing",
                 valueString(syntheticOptions.robotSpacing));
  writeConfigRow(out, "synthetic", "object_spacing",
                 valueString(syntheticOptions.objectSpacing));
  writeConfigRow(out, "synthetic", "object_y_offset",
                 valueString(syntheticOptions.objectYOffset));
  writeConfigRow(out, "synthetic", "world_length",
                 valueString(syntheticOptions.worldLength));
  writeConfigRow(out, "synthetic", "world_width",
                 valueString(syntheticOptions.worldWidth));
  writeConfigRow(out, "synthetic", "world_height",
                 valueString(syntheticOptions.worldHeight));
  writeConfigRow(out, "synthetic", "detection_range",
                 valueString(syntheticOptions.detectionRange));
  writeConfigRow(out, "synthetic", "horizontal_fov_deg",
                 valueString(syntheticOptions.horizontalFovDeg));
  writeConfigRow(out, "synthetic", "vertical_fov_deg",
                 valueString(syntheticOptions.verticalFovDeg));
  writeConfigRow(out, "synthetic", "miss_detection_rate",
                 valueString(syntheticOptions.missDetectionRate));
  writeConfigRow(out, "synthetic", "observation_translation_noise",
                 valueString(syntheticOptions.observationTranslationNoise));
  writeConfigRow(out, "synthetic", "observation_rotation_noise",
                 valueString(syntheticOptions.observationRotationNoise));
  writeConfigRow(out, "synthetic", "object_observations_per_object",
                 valueString(syntheticOptions.objectObservationsPerObject));
  writeConfigRow(out, "synthetic", "false_positive_objects_per_robot",
                 valueString(syntheticOptions.falsePositiveObjectsPerRobot));
  writeConfigRow(out, "synthetic", "duplicate_object_ratio",
                 valueString(syntheticOptions.duplicateObjectRatio));
  writeConfigRow(out, "synthetic", "permute_local_object_ids",
                 boolString(syntheticOptions.permuteLocalObjectIds));
  writeConfigRow(out, "synthetic", "observe_all_objects",
                 boolString(syntheticOptions.observeAllObjects));
  writeConfigRow(out, "synthetic", "robot_drift_per_robot",
                 valueString(syntheticOptions.robotDriftPerRobot));
  writeConfigRow(out, "synthetic", "object_initial_drift_per_robot",
                 valueString(syntheticOptions.objectInitialDriftPerRobot));
  writeConfigRow(out, "synthetic", "num_categories",
                 valueString(syntheticOptions.numCategories));
  writeConfigRow(out, "synthetic", "descriptor_dim",
                 valueString(syntheticOptions.descriptorDim));
  writeConfigRow(out, "synthetic", "semantic_accuracy",
                 valueString(syntheticOptions.semanticAccuracy));
  writeConfigRow(out, "synthetic", "descriptor_noise",
                 valueString(syntheticOptions.descriptorNoise));
  writeConfigRow(out, "synthetic", "shape_noise",
                 valueString(syntheticOptions.shapeNoise));
  writeConfigRow(out, "synthetic", "repeated_object_ratio",
                 valueString(syntheticOptions.repeatedObjectRatio));
  writeConfigRow(out, "synthetic", "symmetry_yaw_ambiguity_rate",
                 valueString(syntheticOptions.symmetryYawAmbiguityRate));
  writeConfigRow(out, "backend", "max_iterations",
                 valueString(backendOptions.maxIterations));
  writeConfigRow(out, "backend", "beta", valueString(backendOptions.beta));
  writeConfigRow(out, "backend", "local_max_iterations",
                 valueString(backendOptions.localTrustRegionIterations));
  writeConfigRow(out, "backend", "local_max_tcg_iterations",
                 valueString(
                     backendOptions.localTrustRegionMaxInnerIterations));
  writeConfigRow(out, "backend", "association_temperature",
                 valueString(backendOptions.associationTemperature));
  writeConfigRow(out, "backend", "association_sinkhorn_iterations",
                 valueString(backendOptions.associationSinkhornIterations));
  writeConfigRow(out, "backend", "association_max_candidates_per_edge",
                 valueString(backendOptions.associationMaxCandidatesPerEdge));
  writeConfigRow(out, "backend", "association_sinkhorn_max_nonzeros",
                 valueString(backendOptions.associationSinkhornMaxNonzeros));
  writeConfigRow(out, "backend", "association_policy",
                 associationPolicyName(backendOptions.associationPolicy));
  writeConfigRow(out, "backend", "association_update_period",
                 valueString(backendOptions.associationUpdatePeriod));
  writeConfigRow(out, "backend", "association_row_capacity",
                 valueString(backendOptions.associationRowCapacity));
  writeConfigRow(out, "backend", "association_column_capacity",
                 valueString(backendOptions.associationColumnCapacity));
  writeConfigRow(out, "backend", "association_semantic_weight",
                 valueString(backendOptions.associationSemanticWeight));
  writeConfigRow(out, "backend", "association_shape_weight",
                 valueString(backendOptions.associationShapeWeight));
  writeConfigRow(out, "backend", "association_descriptor_weight",
                 valueString(backendOptions.associationDescriptorWeight));
  writeConfigRow(out, "backend", "association_require_observed_objects",
                 boolString(backendOptions.associationRequireObservedObjects));
  writeConfigRow(out, "backend", "association_use_quotient_metric",
                 boolString(backendOptions.associationUseQuotientMetric));
  writeConfigRow(out, "backend", "association_cycle_objective_weight",
                 valueString(backendOptions.associationCycleObjectiveWeight));
  writeConfigRow(out, "backend", "association_cycle_objective_min_edge_mass",
                 valueString(
                     backendOptions.associationCycleObjectiveMinEdgeMass));
  writeConfigRow(out, "backend", "association_adaptive_upper_enabled",
                 boolString(backendOptions.associationAdaptiveUpperEnabled));
  writeConfigRow(out, "backend", "association_adaptive_upper_initial_step",
                 valueString(
                     backendOptions.associationAdaptiveUpperInitialStep));
  writeConfigRow(out, "backend", "association_adaptive_upper_epsilon",
                 valueString(backendOptions.associationAdaptiveUpperEpsilon));
  writeConfigRow(out, "backend", "association_adaptive_upper_max_step",
                 valueString(backendOptions.associationAdaptiveUpperMaxStep));
  writeConfigRow(out, "backend", "association_backend_feedback_gradient_scale",
                 valueString(
                     backendOptions.associationBackendFeedbackGradientScale));
  writeConfigRow(out, "backend", "association_backend_feedback_min_trust",
                 valueString(backendOptions.associationBackendFeedbackMinTrust));
  writeConfigRow(out, "backend", "association_backend_response_gain",
                 valueString(backendOptions.associationBackendResponseGain));
  writeConfigRow(out, "backend", "association_backend_response_damping",
                 valueString(
                     backendOptions.associationBackendResponseDamping));
  writeConfigRow(
      out, "backend",
      "association_backend_response_max_reduction_fraction",
      valueString(
          backendOptions.associationBackendResponseMaxReductionFraction));
  writeConfigRow(out, "backend", "association_backend_adjoint_gain",
                 valueString(backendOptions.associationBackendAdjointGain));
  writeConfigRow(out, "backend", "association_backend_adjoint_damping",
                 valueString(
                     backendOptions.associationBackendAdjointDamping));
  writeConfigRow(
      out, "backend",
      "association_backend_adjoint_max_reduction_fraction",
      valueString(
          backendOptions.associationBackendAdjointMaxReductionFraction));
  writeConfigRow(out, "backend",
                 "association_backend_adjoint_max_increase_fraction",
                 valueString(
                     backendOptions
                         .associationBackendAdjointMaxIncreaseFraction));
  writeConfigRow(
      out, "backend", "association_backend_coupled_adjoint_gain",
      valueString(backendOptions.associationBackendCoupledAdjointGain));
  writeConfigRow(
      out, "backend", "association_backend_coupled_adjoint_damping",
      valueString(backendOptions.associationBackendCoupledAdjointDamping));
  writeConfigRow(
      out, "backend",
      "association_backend_coupled_adjoint_max_reduction_fraction",
      valueString(backendOptions
                      .associationBackendCoupledAdjointMaxReductionFraction));
  writeConfigRow(
      out, "backend",
      "association_backend_coupled_adjoint_max_increase_fraction",
      valueString(backendOptions
                      .associationBackendCoupledAdjointMaxIncreaseFraction));
  writeConfigRow(out, "backend",
                 "association_backend_adjoint_reduction_metadata_margin",
                 valueString(backendOptions
                                 .associationBackendAdjointReductionMetadataMargin));
  writeConfigRow(out, "backend",
                 "association_backend_adjoint_allow_cost_reduction",
                 boolString(backendOptions
                                .associationBackendAdjointAllowCostReduction));
  writeConfigRow(out, "backend", "association_backend_snapshot_period",
                 valueString(backendOptions.associationBackendSnapshotPeriod));
  writeConfigRow(out, "backend", "association_backend_max_effective_ratio",
                 valueString(
                     backendOptions.associationBackendMaxEffectiveRatio));
  writeConfigRow(
      out, "backend",
      "association_backend_stationarity_trust_max_gradient_norm",
      valueString(backendOptions
                      .associationBackendStationarityTrustMaxGradientNorm));
  writeConfigRow(out, "backend", "association_temporal_blend",
                 valueString(backendOptions.associationTemporalBlend));
  writeConfigRow(out, "backend", "association_temporal_kl_weight",
                 valueString(backendOptions.associationTemporalKlWeight));
  writeConfigRow(out, "backend", "association_temporal_kl_epsilon",
                 valueString(backendOptions.associationTemporalKlEpsilon));
  writeConfigRow(
      out, "backend", "association_entropy_regularization_weight",
      valueString(backendOptions.associationEntropyRegularizationWeight));
  writeConfigRow(out, "backend", "association_renormalize_sparse_weights",
                 boolString(
                     backendOptions.associationRenormalizeSparseWeights));
  writeConfigRow(out, "backend", "association_residual_warm_start_use_geometry",
                 boolString(
                     backendOptions.associationResidualWarmStartUseGeometry));
  writeConfigRow(out, "backend",
                 "association_residual_use_transform_alignment",
                 boolString(
                     backendOptions.associationResidualUseTransformAlignment));
  writeConfigRow(out, "backend", "association_relational_geometry_weight",
                 valueString(
                     backendOptions.associationRelationalGeometryWeight));
  writeConfigRow(
      out, "backend", "association_relational_geometry_distance_scale",
      valueString(
          backendOptions.associationRelationalGeometryDistanceScale));
  writeConfigRow(out, "backend", "association_relational_geometry_center_rows",
                 boolString(
                     backendOptions.associationRelationalGeometryCenterRows));
  writeConfigRow(
      out, "backend", "association_roto_translation_alignment_weight",
      valueString(
          backendOptions.associationRotoTranslationAlignmentWeight));
  writeConfigRow(
      out, "backend",
      "association_roto_translation_alignment_distance_scale",
      valueString(
          backendOptions
              .associationRotoTranslationAlignmentDistanceScale));
  writeConfigRow(
      out, "backend", "association_roto_translation_hypothesis_top_k",
      std::to_string(
          backendOptions.associationRotoTranslationHypothesisTopK));
  writeConfigRow(
      out, "backend", "association_roto_translation_hypothesis_angle_bins",
      std::to_string(
          backendOptions.associationRotoTranslationHypothesisAngleBins));
  writeConfigRow(
      out, "backend",
      "association_roto_translation_hypothesis_softmin_temperature",
      valueString(
          backendOptions
              .associationRotoTranslationHypothesisSoftminTemperature));
  writeConfigRow(out, "backend", "association_coverage_prior_weight",
                 valueString(backendOptions.associationCoveragePriorWeight));
  writeConfigRow(out, "backend", "association_reference_trust_weight",
                 valueString(backendOptions.associationReferenceTrustWeight));
  writeConfigRow(
      out, "backend", "association_reference_trust_scale_with_upper_cost",
      boolString(backendOptions.associationReferenceTrustScaleWithUpperCost));
  writeConfigRow(
      out, "backend",
      "association_reference_trust_minimum_cost_rms_scale",
      valueString(
          backendOptions.associationReferenceTrustMinimumCostRmsScale));
  writeConfigRow(out, "backend", "association_reference_mass_trust_weight",
                 valueString(
                     backendOptions.associationReferenceMassTrustWeight));
  writeConfigRow(
      out, "backend",
      "association_reference_mass_trust_scale_with_upper_cost",
      boolString(
          backendOptions.associationReferenceMassTrustScaleWithUpperCost));
  writeConfigRow(
      out, "backend",
      "association_reference_mass_trust_minimum_cost_rms_scale",
      valueString(
          backendOptions.associationReferenceMassTrustMinimumCostRmsScale));
  writeConfigRow(out, "backend", "association_normalize_upper_costs",
                 boolString(backendOptions.associationNormalizeUpperCosts));
  writeConfigRow(
      out, "backend", "association_upper_cost_normalization_target_rms",
      valueString(backendOptions.associationUpperCostNormalizationTargetRms));
  writeConfigRow(out, "backend", "association_min_consensus_weight",
                 valueString(backendOptions.associationMinConsensusWeight));
  writeConfigRow(out, "backend", "association_min_consensus_margin",
                 valueString(backendOptions.associationMinConsensusMargin));
  writeConfigRow(out, "backend", "association_consensus_weight_power",
                 valueString(backendOptions.associationConsensusWeightPower));
  writeConfigRow(out, "backend", "association_active_min_stable_updates",
                 valueString(
                     backendOptions.associationActiveMinStableUpdates));
  writeConfigRow(out, "backend", "association_activation_trust_ramp_updates",
                 valueString(
                     backendOptions.associationActivationTrustRampUpdates));
  writeConfigRow(out, "backend", "association_tentative_weight_scale",
                 valueString(backendOptions.associationTentativeWeightScale));
  writeConfigRow(out, "backend", "association_reject_residual_threshold",
                 valueString(
                     backendOptions.associationRejectResidualThreshold));
  writeConfigRow(out, "backend", "association_reject_residual_best_margin",
                 valueString(
                     backendOptions.associationRejectResidualBestMargin));
  writeConfigRow(out, "backend", "association_reject_residual_quantile",
                 valueString(
                     backendOptions.associationRejectResidualQuantile));
  writeConfigRow(out, "backend", "association_reject_residual_quantile_scale",
                 valueString(
                     backendOptions.associationRejectResidualQuantileScale));
  writeConfigRow(out, "backend", "association_reject_min_bad_updates",
                 valueString(backendOptions.associationRejectMinBadUpdates));
}

void writeEffectiveMethodConfig(
    const std::filesystem::path &outputDir, const std::filesystem::path &dataDir,
    const std::vector<std::string> &methods,
    const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
    const DPGO::ManualObjectDranOptions &backendOptions) {
  if (methods.empty()) {
    return;
  }
  std::ofstream out(outputDir / "effective_method_config.csv");
  if (!out) {
    throw std::runtime_error(
        "Could not open effective_method_config.csv under " +
        outputDir.string());
  }
  out << "method,association_mode,association_policy,association_file,beta,"
         "association_reject_applies,"
         "association_use_synthetic_metadata,association_geometry_scale,"
         "association_semantic_weight,association_shape_weight,"
         "association_descriptor_weight,association_require_observed_objects,"
         "association_use_quotient_metric,"
         "association_relational_geometry_weight,"
         "association_relational_geometry_distance_scale,"
         "association_relational_geometry_center_rows,"
         "association_roto_translation_alignment_weight,"
         "association_roto_translation_alignment_distance_scale,"
         "association_roto_translation_hypothesis_top_k,"
         "association_roto_translation_hypothesis_angle_bins,"
         "association_roto_translation_hypothesis_softmin_temperature,"
         "association_coverage_prior_weight,"
         "association_reference_trust_weight,"
         "association_reference_trust_scale_with_upper_cost,"
         "association_reference_trust_minimum_cost_rms_scale,"
         "association_reference_mass_trust_weight,"
         "association_reference_mass_trust_scale_with_upper_cost,"
         "association_reference_mass_trust_minimum_cost_rms_scale,"
         "association_adaptive_upper_enabled,"
         "association_adaptive_upper_initial_step,"
         "association_adaptive_upper_epsilon,"
         "association_adaptive_upper_max_step,"
         "association_temporal_kl_weight,"
         "association_temporal_kl_epsilon,"
         "association_entropy_regularization_weight,"
         "association_backend_feedback_gradient_scale,"
         "association_backend_feedback_min_trust,"
         "association_backend_response_gain,"
         "association_backend_response_damping,"
         "association_backend_response_max_reduction_fraction,"
         "association_backend_adjoint_gain,"
         "association_backend_adjoint_damping,"
         "association_backend_adjoint_max_reduction_fraction,"
         "association_backend_adjoint_max_increase_fraction,"
         "association_backend_coupled_adjoint_gain,"
         "association_backend_coupled_adjoint_damping,"
         "association_backend_coupled_adjoint_max_reduction_fraction,"
         "association_backend_coupled_adjoint_max_increase_fraction,"
         "association_backend_adjoint_reduction_metadata_margin,"
         "association_backend_adjoint_allow_cost_reduction,"
         "association_backend_snapshot_period,"
         "association_backend_max_effective_ratio,"
         "association_backend_stationarity_trust_max_gradient_norm,"
         "association_min_consensus_weight,association_min_consensus_margin,"
         "association_consensus_weight_power,"
         "association_active_min_stable_updates,"
         "association_tentative_weight_scale,"
         "association_reject_residual_threshold,"
         "association_reject_residual_best_margin,"
         "association_reject_residual_quantile,"
         "association_reject_residual_quantile_scale,"
         "association_reject_min_bad_updates\n";
  for (const std::string &method : methods) {
    const DPGO::ManualObjectDranOptions options =
        effectiveOptionsForMethod(dataDir, method, syntheticOptions,
                                  backendOptions);
    out << csvQuote(canonicalMethodName(method)) << ","
        << csvQuote(associationModeName(options.associationMode)) << ","
        << csvQuote(associationPolicyName(options.associationPolicy)) << ","
        << csvQuote(options.associationFile) << ","
        << valueString(options.beta) << ","
        << boolString(associationRejectApplies(options)) << ","
        << boolString(options.associationUseSyntheticMetadata) << ","
        << valueString(options.associationGeometryScale) << ","
        << valueString(options.associationSemanticWeight) << ","
        << valueString(options.associationShapeWeight) << ","
        << valueString(options.associationDescriptorWeight) << ","
        << boolString(options.associationRequireObservedObjects) << ","
        << boolString(options.associationUseQuotientMetric) << ","
        << valueString(options.associationRelationalGeometryWeight) << ","
        << valueString(options.associationRelationalGeometryDistanceScale)
        << ","
        << boolString(options.associationRelationalGeometryCenterRows) << ","
        << valueString(options.associationRotoTranslationAlignmentWeight)
        << ","
        << valueString(
               options.associationRotoTranslationAlignmentDistanceScale)
        << ","
        << valueString(options.associationRotoTranslationHypothesisTopK)
        << ","
        << valueString(options.associationRotoTranslationHypothesisAngleBins)
        << ","
        << valueString(
               options.associationRotoTranslationHypothesisSoftminTemperature)
        << ","
        << valueString(options.associationCoveragePriorWeight) << ","
        << valueString(options.associationReferenceTrustWeight) << ","
        << boolString(options.associationReferenceTrustScaleWithUpperCost)
        << ","
        << valueString(options.associationReferenceTrustMinimumCostRmsScale)
        << ","
        << valueString(options.associationReferenceMassTrustWeight) << ","
        << boolString(options.associationReferenceMassTrustScaleWithUpperCost)
        << ","
        << valueString(
               options.associationReferenceMassTrustMinimumCostRmsScale)
        << ","
        << boolString(options.associationAdaptiveUpperEnabled) << ","
        << valueString(options.associationAdaptiveUpperInitialStep) << ","
        << valueString(options.associationAdaptiveUpperEpsilon) << ","
        << valueString(options.associationAdaptiveUpperMaxStep) << ","
        << valueString(options.associationTemporalKlWeight) << ","
        << valueString(options.associationTemporalKlEpsilon) << ","
        << valueString(options.associationEntropyRegularizationWeight) << ","
        << valueString(options.associationBackendFeedbackGradientScale) << ","
        << valueString(options.associationBackendFeedbackMinTrust) << ","
        << valueString(options.associationBackendResponseGain) << ","
        << valueString(options.associationBackendResponseDamping) << ","
        << valueString(options.associationBackendResponseMaxReductionFraction)
        << ","
        << valueString(options.associationBackendAdjointGain) << ","
        << valueString(options.associationBackendAdjointDamping) << ","
        << valueString(options.associationBackendAdjointMaxReductionFraction)
        << ","
        << valueString(options.associationBackendAdjointMaxIncreaseFraction)
        << ","
        << valueString(options.associationBackendCoupledAdjointGain) << ","
        << valueString(options.associationBackendCoupledAdjointDamping)
        << ","
        << valueString(
               options.associationBackendCoupledAdjointMaxReductionFraction)
        << ","
        << valueString(
               options.associationBackendCoupledAdjointMaxIncreaseFraction)
        << ","
        << valueString(
               options.associationBackendAdjointReductionMetadataMargin)
        << ","
        << boolString(options.associationBackendAdjointAllowCostReduction)
        << ","
        << valueString(options.associationBackendSnapshotPeriod) << ","
        << valueString(options.associationBackendMaxEffectiveRatio) << ","
        << valueString(
               options.associationBackendStationarityTrustMaxGradientNorm)
        << ","
        << valueString(options.associationMinConsensusWeight) << ","
        << valueString(options.associationMinConsensusMargin) << ","
        << valueString(options.associationConsensusWeightPower) << ","
        << valueString(options.associationActiveMinStableUpdates) << ","
        << valueString(options.associationTentativeWeightScale) << ","
        << valueString(options.associationRejectResidualThreshold) << ","
        << valueString(options.associationRejectResidualBestMargin) << ","
        << valueString(options.associationRejectResidualQuantile) << ","
        << valueString(options.associationRejectResidualQuantileScale) << ","
        << valueString(options.associationRejectMinBadUpdates) << "\n";
  }
}

struct MethodMetrics {
  std::string method;
  DPGO::ManualObjectDranRunResult result;
  DPGO::SyntheticObjectPGOEvaluation evaluation;
};

const DPGO::ManualObjectDranIterationSummary &lastRow(
    const MethodMetrics &metrics) {
  if (metrics.result.iterations.empty()) {
    throw std::runtime_error("ManualObjectDran returned no iterations");
  }
  return metrics.result.iterations.back();
}

MethodMetrics runMethod(const std::filesystem::path &dataDir,
                        const std::string &method,
                        const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
                        const DPGO::ManualObjectDranOptions &backendOptions) {
  const DPGO::ManualObjectDranOptions options =
      effectiveOptionsForMethod(dataDir, method, syntheticOptions,
                                backendOptions);

  MethodMetrics metrics;
  metrics.method = canonicalMethodName(method);
  metrics.result = DPGO::runManualObjectDran(dataDir.string(), options);
  metrics.evaluation = DPGO::evaluateSyntheticObjectPGOEstimates(
      syntheticOptions, metrics.result.finalEstimates);
  return metrics;
}

void printUsage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "Options:\n"
      << "  --output_dir path\n"
      << "  --methods no_cross_object,oracle,class_only,metadata_hard,geometry_hard,metadata_geometry_hard,graph_consistency_hard,geometry_soft,transform_soft,residual_soft,backend_feedback_soft,backend_response_soft,backend_adjoint_soft,backend_coupled_adjoint_soft,backend_penalty_coupled_adjoint_soft,backend_guarded_coupled_adjoint_soft,backend_hybrid_soft\n"
      << "  --failure_search true|false\n"
      << "  --generate_only true|false\n"
      << "  --instance_seed N\n"
      << "  --dimension 2|3\n"
      << "  --num_robots N\n"
      << "  --num_objects N\n"
      << "  --poses_per_robot N\n"
      << "  --iters N\n"
      << "  --beta value\n"
      << "  --pose_spacing value\n"
      << "  --robot_spacing value\n"
      << "  --object_spacing value\n"
      << "  --object_y_offset value\n"
      << "  --world_length value\n"
      << "  --world_width value\n"
      << "  --world_height value\n"
      << "  --detection_range value\n"
      << "  --horizontal_fov_deg value\n"
      << "  --vertical_fov_deg value\n"
      << "  --miss_detection_rate value\n"
      << "  --observation_translation_noise value\n"
      << "  --observation_rotation_noise value\n"
      << "  --object_observations_per_object N\n"
      << "  --false_positive_objects_per_robot N\n"
      << "  --duplicate_object_ratio value\n"
      << "  --permute_local_object_ids true|false\n"
      << "  --observe_all_objects true|false\n"
      << "  --robot_drift_per_robot value\n"
      << "  --object_initial_drift_per_robot value\n"
      << "  --num_categories N\n"
      << "  --descriptor_dim N\n"
      << "  --semantic_accuracy value\n"
      << "  --descriptor_noise value\n"
      << "  --shape_noise value\n"
      << "  --repeated_object_ratio value\n"
      << "  --symmetry_yaw_ambiguity_rate value\n"
      << "  --association_temperature value\n"
      << "  --association_sinkhorn_iterations N\n"
      << "  --association_max_candidates_per_edge N\n"
      << "  --association_sinkhorn_max_nonzeros N\n"
      << "  --association_policy manual|two_time_scale_bilevel_v1|single_loop_bilevel_v2|single_loop_roto_bilevel_v3|final_dba_p4_203|shared_edge_bilevel_v1|shared_edge_joint_bilevel_v2|decentralized_feedback_uot_bilevel_v1\n"
      << "  --association_update_period N\n"
      << "  --association_row_capacity value\n"
      << "  --association_column_capacity value\n"
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
      << "  --association_backend_snapshot_period N\n"
      << "  --association_backend_max_effective_ratio value\n"
      << "  --association_backend_stationarity_trust_max_gradient_norm value\n"
      << "  --association_temporal_blend value\n"
      << "  --association_temporal_kl_weight value\n"
      << "  --association_temporal_kl_epsilon value\n"
      << "  --association_entropy_regularization_weight value\n"
      << "  --association_renormalize_sparse_weights true|false\n"
      << "  --association_residual_warm_start_use_geometry true|false\n"
      << "  --association_residual_use_transform_alignment true|false\n"
      << "  --association_relational_geometry_weight value\n"
      << "  --association_relational_geometry_distance_scale value\n"
      << "  --association_relational_geometry_center_rows true|false\n"
      << "  --association_roto_translation_alignment_weight value\n"
      << "  --association_roto_translation_alignment_distance_scale value\n"
      << "  --association_roto_translation_hypothesis_top_k N\n"
      << "  --association_roto_translation_hypothesis_angle_bins N\n"
      << "  --association_roto_translation_hypothesis_softmin_temperature value\n"
      << "  --association_coverage_prior_weight value\n"
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
      << "  --local_max_iterations N\n"
      << "  --local_max_tcg_iterations N\n";
}

void writeSummaryHeader(std::ofstream &out) {
  out << "method,iterations,trajectory_translation_rmse,"
         "trajectory_rotation_frobenius_rmse,object_translation_rmse,"
         "object_rotation_frobenius_rmse,"
         "object_quotient_rotation_frobenius_rmse,"
         "observed_object_copy_count,"
         "observed_object_translation_rmse,"
         "observed_object_rotation_frobenius_rmse,"
         "observed_object_quotient_rotation_frobenius_rmse,"
         "global_trajectory_translation_rmse,"
         "global_trajectory_rotation_frobenius_rmse,"
         "global_object_translation_rmse,"
         "global_object_rotation_frobenius_rmse,"
         "global_object_quotient_rotation_frobenius_rmse,"
         "observed_global_object_translation_rmse,"
         "observed_global_object_rotation_frobenius_rmse,"
         "observed_global_object_quotient_rotation_frobenius_rmse,"
         "object_copy_translation_spread_rmse,"
         "object_copy_spread_count,"
         "measurement_cost,consensus_cost,"
         "augmented_cost,"
         "gradient,association_candidate_count,association_weight_mass,"
         "association_correct_mass,association_false_merge_mass,"
         "association_entropy,association_precision,association_recall,"
         "association_f1,association_false_merge_rate,"
         "association_factor_weight_mass,"
         "association_factor_correct_mass,"
         "association_factor_false_merge_mass,"
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
         "association_coverage_prior_gradient_norm,"
         "association_coverage_prior_to_local_gradient_ratio,"
         "association_coverage_prior_objective_cost,"
         "association_actual_coverage_prior_alpha,"
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
         "active_consensus_pairs,"
         "comm_pose_count,total_comm_mb,elapsed_sec\n";
}

void writeSummaryRow(std::ofstream &out, const std::string &method,
                     const DPGO::ManualObjectDranRunResult &result,
                     const DPGO::SyntheticObjectPGOEvaluation &evaluation) {
  if (result.iterations.empty()) {
    throw std::runtime_error("ManualObjectDran returned no iterations");
  }
  const DPGO::ManualObjectDranIterationSummary &row =
      result.iterations.back();
  out << method << "," << row.iter << ","
      << evaluation.trajectoryTranslationRmse << ","
      << evaluation.trajectoryRotationFrobeniusRmse << ","
      << evaluation.objectTranslationRmse << ","
      << evaluation.objectRotationFrobeniusRmse << ","
      << evaluation.objectQuotientRotationFrobeniusRmse << ","
      << evaluation.observedObjectCopyCount << ","
      << evaluation.observedObjectTranslationRmse << ","
      << evaluation.observedObjectRotationFrobeniusRmse << ","
      << evaluation.observedObjectQuotientRotationFrobeniusRmse << ","
      << evaluation.globalTrajectoryTranslationRmse << ","
      << evaluation.globalTrajectoryRotationFrobeniusRmse << ","
      << evaluation.globalObjectTranslationRmse << ","
      << evaluation.globalObjectRotationFrobeniusRmse << ","
      << evaluation.globalObjectQuotientRotationFrobeniusRmse << ","
      << evaluation.observedGlobalObjectTranslationRmse << ","
      << evaluation.observedGlobalObjectRotationFrobeniusRmse << ","
      << evaluation.observedGlobalObjectQuotientRotationFrobeniusRmse << ","
      << evaluation.objectCopyTranslationSpreadRmse << ","
      << evaluation.objectCopySpreadCount << ","
      << row.measurementCost << ","
      << row.consensusCost << "," << row.measurementCost + row.consensusCost
      << "," << row.gradient << "," << row.associationCandidateCount << ","
      << row.associationWeightMass << "," << row.associationCorrectMass
      << "," << row.associationFalseMergeMass << ","
      << row.associationEntropy << "," << row.associationPrecision << ","
      << row.associationRecall << "," << row.associationF1 << ","
      << row.associationFalseMergeRate << ","
      << row.associationFactorWeightMass << ","
      << row.associationFactorCorrectMass << ","
      << row.associationFactorFalseMergeMass << ","
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
      << row.associationCoveragePriorGradientNorm << ","
      << row.associationCoveragePriorToLocalGradientRatio << ","
      << row.associationCoveragePriorObjectiveCost << ","
      << row.associationActualCoveragePriorAlpha << ","
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
      << row.activeConsensusPairs
      << "," << row.cumulativeCommPoseCount << "," << row.cumulativeCommMb
      << "," << result.elapsedSeconds << "\n";
}

void writeSweepHeader(std::ofstream &out) {
  out << "scenario_id,method,num_robots,num_objects,poses_per_robot,"
         "num_categories,semantic_accuracy,descriptor_noise,shape_noise,"
         "repeated_object_ratio,trajectory_translation_rmse,"
         "object_translation_rmse,global_trajectory_translation_rmse,"
         "global_object_translation_rmse,object_copy_translation_spread_rmse,"
         "measurement_cost,consensus_cost,gradient,"
         "association_f1,association_false_merge_rate,"
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
         "association_coverage_prior_gradient_norm,"
         "association_coverage_prior_to_local_gradient_ratio,"
         "association_coverage_prior_objective_cost,"
         "association_actual_coverage_prior_alpha,"
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
         "total_comm_mb,"
         "elapsed_sec\n";
}

void writeSweepRow(std::ofstream &out, std::size_t scenarioId,
                   const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
                   const MethodMetrics &metrics) {
  const DPGO::ManualObjectDranIterationSummary &row = lastRow(metrics);
  out << scenarioId << "," << metrics.method << ","
      << syntheticOptions.numRobots << "," << syntheticOptions.numObjects
      << "," << syntheticOptions.posesPerRobot << ","
      << syntheticOptions.numCategories << ","
      << syntheticOptions.semanticAccuracy << ","
      << syntheticOptions.descriptorNoise << ","
      << syntheticOptions.shapeNoise << ","
      << syntheticOptions.repeatedObjectRatio << ","
      << metrics.evaluation.trajectoryTranslationRmse << ","
      << metrics.evaluation.objectTranslationRmse << ","
      << metrics.evaluation.globalTrajectoryTranslationRmse << ","
      << metrics.evaluation.globalObjectTranslationRmse << ","
      << metrics.evaluation.objectCopyTranslationSpreadRmse << ","
      << row.measurementCost << "," << row.consensusCost << ","
      << row.gradient << "," << row.associationF1 << ","
      << row.associationFalseMergeRate << ","
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
      << row.associationCoveragePriorGradientNorm << ","
      << row.associationCoveragePriorToLocalGradientRatio << ","
      << row.associationCoveragePriorObjectiveCost << ","
      << row.associationActualCoveragePriorAlpha << ","
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
      << row.cumulativeCommMb
      << "," << metrics.result.elapsedSeconds << "\n";
}

void writeFailureHeader(std::ofstream &out) {
  out << "scenario_id,method,reference_method,num_robots,num_objects,"
         "poses_per_robot,num_categories,semantic_accuracy,descriptor_noise,"
         "shape_noise,repeated_object_ratio,method_object_rmse,"
         "reference_object_rmse,object_rmse_gap,method_trajectory_rmse,"
         "reference_trajectory_rmse,method_association_f1,"
         "reference_association_f1,f1_gap,method_false_merge_rate,"
         "reference_false_merge_rate\n";
}

void writeFailureCandidate(
    std::ofstream &out, std::size_t scenarioId,
    const DPGO::SyntheticObjectPGOOptions &syntheticOptions,
    const MethodMetrics &method, const MethodMetrics &reference) {
  const DPGO::ManualObjectDranIterationSummary &methodRow = lastRow(method);
  const DPGO::ManualObjectDranIterationSummary &referenceRow =
      lastRow(reference);
  const double objectGap = method.evaluation.objectTranslationRmse -
                           reference.evaluation.objectTranslationRmse;
  const double f1Gap = referenceRow.associationF1 - methodRow.associationF1;
  out << scenarioId << "," << method.method << "," << reference.method << ","
      << syntheticOptions.numRobots << "," << syntheticOptions.numObjects
      << "," << syntheticOptions.posesPerRobot << ","
      << syntheticOptions.numCategories << ","
      << syntheticOptions.semanticAccuracy << ","
      << syntheticOptions.descriptorNoise << ","
      << syntheticOptions.shapeNoise << ","
      << syntheticOptions.repeatedObjectRatio << ","
      << method.evaluation.objectTranslationRmse << ","
      << reference.evaluation.objectTranslationRmse << "," << objectGap
      << "," << method.evaluation.trajectoryTranslationRmse << ","
      << reference.evaluation.trajectoryTranslationRmse << ","
      << methodRow.associationF1 << "," << referenceRow.associationF1
      << "," << f1Gap << "," << methodRow.associationFalseMergeRate << ","
      << referenceRow.associationFalseMergeRate << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::filesystem::path outputDir = "results/bilevel_synthetic_object";
  std::vector<std::string> methods = {"no_cross_object", "oracle", "class_only",
                                      "metadata_hard", "geometry_hard",
                                      "metadata_geometry_hard",
                                      "graph_consistency_hard",
                                      "residual_soft"};
  bool failureSearch = false;
  bool generateOnly = false;

  DPGO::SyntheticObjectPGOOptions syntheticOptions;
  DPGO::ManualObjectDranOptions backendOptions;
  backendOptions.maxIterations = 5;
  backendOptions.topology = DPGO::ManualObjectDranTopology::Complete;
  backendOptions.objectInitialization =
      DPGO::ManualObjectDranObjectInitialization::NeighborAverage;
  backendOptions.localTrustRegionIterations = 1;
  backendOptions.localTrustRegionMaxInnerIterations = 2;
  backendOptions.associationTemperature = 0.5;
  backendOptions.associationSinkhornIterations = 30;
  backendOptions.associationMaxCandidatesPerEdge = 50;
  backendOptions.associationUpdatePeriod = 1;
  backendOptions.associationUseSyntheticMetadata = true;
  backendOptions.associationSemanticWeight = 1.0;
  backendOptions.associationShapeWeight = 1.0;
  backendOptions.associationDescriptorWeight = 1.0;
  backendOptions.associationRequireObservedObjects = true;
  backendOptions.associationTemporalBlend = 0.25;

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
      } else if (arg == "--output_dir") {
        outputDir = requireValue(arg);
      } else if (arg == "--methods") {
        methods = splitCommaList(requireValue(arg));
      } else if (arg == "--failure_search") {
        failureSearch = parseBool(requireValue(arg));
      } else if (arg == "--generate_only") {
        generateOnly = parseBool(requireValue(arg));
      } else if (arg == "--instance_seed") {
        syntheticOptions.instanceSeed =
            static_cast<std::size_t>(std::stoull(requireValue(arg)));
      } else if (arg == "--dimension") {
        syntheticOptions.dimension =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--num_robots") {
        syntheticOptions.numRobots =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--num_objects") {
        syntheticOptions.numObjects =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--poses_per_robot") {
        syntheticOptions.posesPerRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--iters") {
        backendOptions.maxIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--beta") {
        backendOptions.beta = std::stod(requireValue(arg));
      } else if (arg == "--pose_spacing") {
        syntheticOptions.poseSpacing = std::stod(requireValue(arg));
      } else if (arg == "--robot_spacing") {
        syntheticOptions.robotSpacing = std::stod(requireValue(arg));
      } else if (arg == "--object_spacing") {
        syntheticOptions.objectSpacing = std::stod(requireValue(arg));
      } else if (arg == "--object_y_offset") {
        syntheticOptions.objectYOffset = std::stod(requireValue(arg));
      } else if (arg == "--world_length") {
        syntheticOptions.worldLength = std::stod(requireValue(arg));
      } else if (arg == "--world_width") {
        syntheticOptions.worldWidth = std::stod(requireValue(arg));
      } else if (arg == "--world_height") {
        syntheticOptions.worldHeight = std::stod(requireValue(arg));
      } else if (arg == "--detection_range") {
        syntheticOptions.detectionRange = std::stod(requireValue(arg));
      } else if (arg == "--horizontal_fov_deg") {
        syntheticOptions.horizontalFovDeg = std::stod(requireValue(arg));
      } else if (arg == "--vertical_fov_deg") {
        syntheticOptions.verticalFovDeg = std::stod(requireValue(arg));
      } else if (arg == "--miss_detection_rate") {
        syntheticOptions.missDetectionRate = std::stod(requireValue(arg));
      } else if (arg == "--observation_translation_noise") {
        syntheticOptions.observationTranslationNoise =
            std::stod(requireValue(arg));
      } else if (arg == "--observation_rotation_noise") {
        syntheticOptions.observationRotationNoise =
            std::stod(requireValue(arg));
      } else if (arg == "--object_observations_per_object") {
        syntheticOptions.objectObservationsPerObject =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--false_positive_objects_per_robot") {
        syntheticOptions.falsePositiveObjectsPerRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--duplicate_object_ratio") {
        syntheticOptions.duplicateObjectRatio =
            std::stod(requireValue(arg));
      } else if (arg == "--permute_local_object_ids") {
        syntheticOptions.permuteLocalObjectIds =
            parseBool(requireValue(arg));
      } else if (arg == "--observe_all_objects") {
        syntheticOptions.observeAllObjects = parseBool(requireValue(arg));
      } else if (arg == "--robot_drift_per_robot") {
        syntheticOptions.robotDriftPerRobot = std::stod(requireValue(arg));
      } else if (arg == "--object_initial_drift_per_robot") {
        syntheticOptions.objectInitialDriftPerRobot =
            std::stod(requireValue(arg));
      } else if (arg == "--num_categories") {
        syntheticOptions.numCategories =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--descriptor_dim") {
        syntheticOptions.descriptorDim =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--semantic_accuracy") {
        syntheticOptions.semanticAccuracy = std::stod(requireValue(arg));
      } else if (arg == "--descriptor_noise") {
        syntheticOptions.descriptorNoise = std::stod(requireValue(arg));
      } else if (arg == "--shape_noise") {
        syntheticOptions.shapeNoise = std::stod(requireValue(arg));
      } else if (arg == "--repeated_object_ratio") {
        syntheticOptions.repeatedObjectRatio = std::stod(requireValue(arg));
      } else if (arg == "--symmetry_yaw_ambiguity_rate") {
        syntheticOptions.symmetryYawAmbiguityRate =
            std::stod(requireValue(arg));
      } else if (arg == "--association_temperature") {
        backendOptions.associationTemperature = std::stod(requireValue(arg));
      } else if (arg == "--association_sinkhorn_iterations") {
        backendOptions.associationSinkhornIterations =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_max_candidates_per_edge") {
        backendOptions.associationMaxCandidatesPerEdge =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_sinkhorn_max_nonzeros") {
        backendOptions.associationSinkhornMaxNonzeros =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_policy") {
        backendOptions.associationPolicy =
            parseAssociationPolicy(requireValue(arg));
      } else if (arg == "--association_update_period") {
        backendOptions.associationUpdatePeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_row_capacity") {
        backendOptions.associationRowCapacity = std::stod(requireValue(arg));
      } else if (arg == "--association_column_capacity") {
        backendOptions.associationColumnCapacity =
            std::stod(requireValue(arg));
      } else if (arg == "--association_semantic_weight") {
        backendOptions.associationSemanticWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_shape_weight") {
        backendOptions.associationShapeWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_descriptor_weight") {
        backendOptions.associationDescriptorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_require_observed_objects") {
        backendOptions.associationRequireObservedObjects =
            parseBool(requireValue(arg));
      } else if (arg == "--association_use_quotient_metric") {
        backendOptions.associationUseQuotientMetric =
            parseBool(requireValue(arg));
      } else if (arg == "--association_transform_min_inliers") {
        backendOptions.associationTransformMinInliers =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_transform_max_residual") {
        backendOptions.associationTransformMaxResidual =
            std::stod(requireValue(arg));
      } else if (arg == "--association_transform_min_score") {
        backendOptions.associationTransformMinScore =
            std::stod(requireValue(arg));
      } else if (arg == "--association_transform_weight_scale") {
        backendOptions.associationTransformWeightScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_transform_max_seed_candidates") {
        backendOptions.associationTransformMaxSeedCandidates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_transform_plane_axes") {
        backendOptions.associationTransformPlaneAxes = requireValue(arg);
      } else if (arg == "--association_cycle_consistency_enabled") {
        backendOptions.associationCycleConsistencyEnabled =
            parseBool(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_direct_weight") {
        backendOptions.associationCycleConsistencyMinDirectWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_composed_weight") {
        backendOptions.associationCycleConsistencyMinComposedWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_cycle_consistency_min_two_hop_support") {
        backendOptions.associationCycleConsistencyMinTwoHopSupport =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg ==
                 "--association_cycle_consistency_keep_without_evidence") {
        backendOptions.associationCycleConsistencyKeepWithoutEvidence =
            parseBool(requireValue(arg));
      } else if (arg == "--association_cycle_objective_weight") {
        backendOptions.associationCycleObjectiveWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_cycle_objective_min_edge_mass") {
        backendOptions.associationCycleObjectiveMinEdgeMass =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_enabled") {
        backendOptions.associationAdaptiveUpperEnabled =
            parseBool(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_initial_step") {
        backendOptions.associationAdaptiveUpperInitialStep =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_epsilon") {
        backendOptions.associationAdaptiveUpperEpsilon =
            std::stod(requireValue(arg));
      } else if (arg == "--association_adaptive_upper_max_step") {
        backendOptions.associationAdaptiveUpperMaxStep =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_feedback_gradient_scale") {
        backendOptions.associationBackendFeedbackGradientScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_feedback_min_trust") {
        backendOptions.associationBackendFeedbackMinTrust =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_response_gain") {
        backendOptions.associationBackendResponseGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_response_damping") {
        backendOptions.associationBackendResponseDamping =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_response_max_reduction_fraction") {
        backendOptions.associationBackendResponseMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_adjoint_gain") {
        backendOptions.associationBackendAdjointGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_adjoint_damping") {
        backendOptions.associationBackendAdjointDamping =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_adjoint_max_reduction_fraction") {
        backendOptions.associationBackendAdjointMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_backend_adjoint_max_increase_fraction") {
        backendOptions.associationBackendAdjointMaxIncreaseFraction =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_coupled_adjoint_gain") {
        backendOptions.associationBackendCoupledAdjointGain =
            std::stod(requireValue(arg));
      } else if (arg == "--association_backend_coupled_adjoint_damping") {
        backendOptions.associationBackendCoupledAdjointDamping =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_coupled_adjoint_max_reduction_fraction") {
        backendOptions
            .associationBackendCoupledAdjointMaxReductionFraction =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_coupled_adjoint_max_increase_fraction") {
        backendOptions.associationBackendCoupledAdjointMaxIncreaseFraction =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_adjoint_reduction_metadata_margin") {
        backendOptions.associationBackendAdjointReductionMetadataMargin =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_backend_adjoint_allow_cost_reduction") {
        backendOptions.associationBackendAdjointAllowCostReduction =
            parseBool(requireValue(arg));
      } else if (arg == "--association_backend_snapshot_period") {
        backendOptions.associationBackendSnapshotPeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_backend_max_effective_ratio") {
        backendOptions.associationBackendMaxEffectiveRatio =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_backend_stationarity_trust_max_gradient_norm") {
        backendOptions.associationBackendStationarityTrustMaxGradientNorm =
            std::stod(requireValue(arg));
      } else if (arg == "--association_temporal_blend") {
        backendOptions.associationTemporalBlend =
            std::stod(requireValue(arg));
      } else if (arg == "--association_temporal_kl_weight") {
        backendOptions.associationTemporalKlWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_temporal_kl_epsilon") {
        backendOptions.associationTemporalKlEpsilon =
            std::stod(requireValue(arg));
      } else if (arg == "--association_entropy_regularization_weight") {
        backendOptions.associationEntropyRegularizationWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_renormalize_sparse_weights") {
        backendOptions.associationRenormalizeSparseWeights =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_warm_start_use_geometry") {
        backendOptions.associationResidualWarmStartUseGeometry =
            parseBool(requireValue(arg));
      } else if (arg == "--association_residual_use_transform_alignment") {
        backendOptions.associationResidualUseTransformAlignment =
            parseBool(requireValue(arg));
      } else if (arg == "--association_relational_geometry_weight") {
        backendOptions.associationRelationalGeometryWeight =
            std::stod(requireValue(arg));
      } else if (arg ==
                 "--association_relational_geometry_distance_scale") {
        backendOptions.associationRelationalGeometryDistanceScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_relational_geometry_center_rows") {
        backendOptions.associationRelationalGeometryCenterRows =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_roto_translation_alignment_weight") {
        backendOptions.associationRotoTranslationAlignmentWeight =
            std::stod(requireValue(arg));
      } else if (
          arg ==
          "--association_roto_translation_alignment_distance_scale") {
        backendOptions.associationRotoTranslationAlignmentDistanceScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_roto_translation_hypothesis_top_k") {
        backendOptions.associationRotoTranslationHypothesisTopK =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg == "--association_roto_translation_hypothesis_angle_bins") {
        backendOptions.associationRotoTranslationHypothesisAngleBins =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (
          arg ==
          "--association_roto_translation_hypothesis_softmin_temperature") {
        backendOptions.associationRotoTranslationHypothesisSoftminTemperature =
            std::stod(requireValue(arg));
      } else if (arg == "--association_coverage_prior_weight") {
        backendOptions.associationCoveragePriorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reference_trust_weight") {
        backendOptions.associationReferenceTrustWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reference_trust_scale_with_upper_cost") {
        backendOptions.associationReferenceTrustScaleWithUpperCost =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_reference_trust_minimum_cost_rms_scale") {
        backendOptions.associationReferenceTrustMinimumCostRmsScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reference_mass_trust_weight") {
        backendOptions.associationReferenceMassTrustWeight =
            std::stod(requireValue(arg));
      } else if (
          arg == "--association_reference_mass_trust_scale_with_upper_cost") {
        backendOptions.associationReferenceMassTrustScaleWithUpperCost =
            parseBool(requireValue(arg));
      } else if (
          arg ==
          "--association_reference_mass_trust_minimum_cost_rms_scale") {
        backendOptions.associationReferenceMassTrustMinimumCostRmsScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_normalize_upper_costs") {
        backendOptions.associationNormalizeUpperCosts =
            parseBool(requireValue(arg));
      } else if (
          arg == "--association_upper_cost_normalization_target_rms" ||
          arg == "--association_upper_cost_normalization_target_norm") {
        backendOptions.associationUpperCostNormalizationTargetRms =
            std::stod(requireValue(arg));
      } else if (arg == "--association_min_consensus_weight") {
        backendOptions.associationMinConsensusWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--association_min_consensus_margin") {
        backendOptions.associationMinConsensusMargin =
            std::stod(requireValue(arg));
      } else if (arg == "--association_consensus_weight_power") {
        backendOptions.associationConsensusWeightPower =
            std::stod(requireValue(arg));
      } else if (arg == "--association_active_min_stable_updates") {
        backendOptions.associationActiveMinStableUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_activation_trust_ramp_updates") {
        backendOptions.associationActivationTrustRampUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--association_tentative_weight_scale") {
        backendOptions.associationTentativeWeightScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_threshold") {
        backendOptions.associationRejectResidualThreshold =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_best_margin") {
        backendOptions.associationRejectResidualBestMargin =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_quantile") {
        backendOptions.associationRejectResidualQuantile =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_residual_quantile_scale") {
        backendOptions.associationRejectResidualQuantileScale =
            std::stod(requireValue(arg));
      } else if (arg == "--association_reject_min_bad_updates") {
        backendOptions.associationRejectMinBadUpdates =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_iterations") {
        backendOptions.localTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_tcg_iterations") {
        backendOptions.localTrustRegionMaxInnerIterations =
            std::stoi(requireValue(arg));
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    if (methods.empty() && !generateOnly) {
      throw std::invalid_argument("At least one method must be requested");
    }

    std::filesystem::create_directories(outputDir);
    std::cout << std::setprecision(10);
    writeExperimentConfig(outputDir, argc, argv, methods, syntheticOptions,
                          backendOptions, failureSearch, generateOnly);

    if (failureSearch) {
      DPGO::SyntheticObjectPGOOptions baseSyntheticOptions = syntheticOptions;
      baseSyntheticOptions.numObjects =
          std::max<std::size_t>(baseSyntheticOptions.numObjects, 12);
      baseSyntheticOptions.posesPerRobot =
          std::max<std::size_t>(baseSyntheticOptions.posesPerRobot, 8);
      baseSyntheticOptions.numCategories =
          std::max<std::size_t>(baseSyntheticOptions.numCategories, 3);
      baseSyntheticOptions.descriptorDim =
          std::max<std::size_t>(baseSyntheticOptions.descriptorDim, 6);

      const std::vector<double> semanticAccuracies = {0.0, 0.35, 0.6};
      const std::vector<double> descriptorNoises = {0.0, 0.08};
      const std::vector<double> shapeNoises = {0.0, 0.04};
      const std::vector<double> repeatedRatios = {0.5, 0.8};

      std::ofstream sweep(outputDir / "sweep_summary.csv");
      std::ofstream failures(outputDir / "failure_candidates.csv");
      if (!sweep || !failures) {
        throw std::runtime_error("Could not open failure-search CSV outputs");
      }
      sweep << std::setprecision(20);
      failures << std::setprecision(20);
      writeSweepHeader(sweep);
      writeFailureHeader(failures);

      std::size_t scenarioId = 0;
      std::size_t candidateCount = 0;
      double bestObjectGap = -std::numeric_limits<double>::infinity();
      std::string bestCandidate;
      for (double semanticAccuracy : semanticAccuracies) {
        for (double descriptorNoise : descriptorNoises) {
          for (double shapeNoise : shapeNoises) {
            for (double repeatedRatio : repeatedRatios) {
              DPGO::SyntheticObjectPGOOptions scenario = baseSyntheticOptions;
              scenario.semanticAccuracy = semanticAccuracy;
              scenario.descriptorNoise = descriptorNoise;
              scenario.shapeNoise = shapeNoise;
              scenario.repeatedObjectRatio = repeatedRatio;
              const std::filesystem::path scenarioDir =
                  outputDir / ("scenario_" + std::to_string(scenarioId));
              const std::filesystem::path dataDir =
                  scenarioDir / "synthetic_data";
              std::filesystem::create_directories(scenarioDir);
              std::filesystem::remove_all(dataDir);
              DPGO::writeSyntheticObjectPGODirectory(dataDir.string(),
                                                     scenario);

              std::vector<MethodMetrics> metrics;
              metrics.reserve(methods.size());
              for (const std::string &method : methods) {
                MethodMetrics row =
                    runMethod(dataDir, method, scenario, backendOptions);
                writeSweepRow(sweep, scenarioId, scenario, row);
                metrics.push_back(std::move(row));
              }

              const MethodMetrics *reference = nullptr;
              for (const MethodMetrics &row : metrics) {
                if (row.method == "residual_soft") {
                  reference = &row;
                  break;
                }
              }
              if (!reference) {
                for (const MethodMetrics &row : metrics) {
                  if (row.method == "oracle") {
                    reference = &row;
                    break;
                  }
                }
              }
              if (reference) {
                for (const MethodMetrics &row : metrics) {
                  if (row.method == reference->method ||
                      row.method == "oracle" ||
                      row.method == "geometry_soft") {
                    continue;
                  }
                  const double objectGap =
                      row.evaluation.objectTranslationRmse -
                      reference->evaluation.objectTranslationRmse;
                  const double f1Gap = lastRow(*reference).associationF1 -
                                       lastRow(row).associationF1;
                  if (objectGap > 0.05 || f1Gap > 0.1) {
                    writeFailureCandidate(failures, scenarioId, scenario, row,
                                          *reference);
                    ++candidateCount;
                    if (objectGap > bestObjectGap) {
                      bestObjectGap = objectGap;
                      bestCandidate = row.method + "@scenario_" +
                                      std::to_string(scenarioId);
                    }
                    std::cout << "FAILURE_CANDIDATE scenario=" << scenarioId
                              << " method=" << row.method
                              << " reference=" << reference->method
                              << " object_gap=" << objectGap
                              << " f1_gap=" << f1Gap
                              << " false_merge_rate="
                              << lastRow(row).associationFalseMergeRate
                              << "\n";
                  }
                }
              }
              ++scenarioId;
            }
          }
        }
      }
      std::cout << "FAILURE_SEARCH scenarios=" << scenarioId
                << " candidates=" << candidateCount
                << " best=" << bestCandidate
                << " best_object_gap=" << bestObjectGap << "\n";
      return 0;
    }

    const std::filesystem::path dataDir = outputDir / "synthetic_data";
    std::filesystem::remove_all(dataDir);
    DPGO::writeSyntheticObjectPGODirectory(dataDir.string(),
                                           syntheticOptions);
    writeEffectiveMethodConfig(outputDir, dataDir, methods, syntheticOptions,
                               backendOptions);
    if (generateOnly) {
      std::cout << "SYNTHETIC_DATASET output_dir=" << outputDir.string()
                << " data_dir=" << dataDir.string()
                << " dimension=" << syntheticOptions.dimension
                << " robots=" << syntheticOptions.numRobots
                << " objects=" << syntheticOptions.numObjects
                << " local_slots_per_robot="
                << DPGO::syntheticLocalObjectSlotCount(syntheticOptions)
                << " poses_per_robot=" << syntheticOptions.posesPerRobot
                << " categories=" << syntheticOptions.numCategories
                << " observe_all_objects="
                << (syntheticOptions.observeAllObjects ? "true" : "false")
                << "\n";
      return 0;
    }

    std::ofstream summary(outputDir / "summary.csv");
    if (!summary) {
      throw std::runtime_error("Could not open summary.csv under " +
                               outputDir.string());
    }
    summary << std::setprecision(20);
    writeSummaryHeader(summary);

    for (const std::string &method : methods) {
      const MethodMetrics metrics =
          runMethod(dataDir, method, syntheticOptions, backendOptions);
      writeSummaryRow(summary, metrics.method, metrics.result,
                      metrics.evaluation);
      const DPGO::ManualObjectDranIterationSummary &row = lastRow(metrics);
      std::cout << "SYNTHETIC_DBA method=" << metrics.method
                << " traj_rmse="
                << metrics.evaluation.trajectoryTranslationRmse
                << " object_rmse=" << metrics.evaluation.objectTranslationRmse
                << " observed_object_rmse="
                << metrics.evaluation.observedObjectTranslationRmse
                << " global_traj_rmse="
                << metrics.evaluation.globalTrajectoryTranslationRmse
                << " global_object_rmse="
                << metrics.evaluation.globalObjectTranslationRmse
                << " observed_global_object_rmse="
                << metrics.evaluation.observedGlobalObjectTranslationRmse
                << " object_copy_spread_rmse="
                << metrics.evaluation.objectCopyTranslationSpreadRmse
                << " measurement_cost=" << row.measurementCost
                << " consensus_cost=" << row.consensusCost
                << " gradient=" << row.gradient
                << " association_f1=" << row.associationF1
                << " false_merge_rate=" << row.associationFalseMergeRate
                << " high_confidence_association_f1="
                << row.highConfidenceAssociationF1
                << " high_confidence_false_merge_rate="
                << row.highConfidenceAssociationFalseMergeRate
                << " activated_association_f1="
                << row.activatedAssociationF1
                << " activated_false_merge_rate="
                << row.activatedAssociationFalseMergeRate
                << " active_association_f1="
                << row.activeAssociationF1
                << " active_false_merge_rate="
                << row.activeAssociationFalseMergeRate
                << " active_association_factor_weight_mass="
                << row.activeAssociationFactorWeightMass
                << " active_association_factor_correct_mass="
                << row.activeAssociationFactorCorrectMass
                << " active_association_factor_false_merge_mass="
                << row.activeAssociationFactorFalseMergeMass
                << " rejected_association_candidates="
                << row.associationRejectedCandidateCount
                << " rejected_association_weight_mass="
                << row.associationRejectedWeightMass
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
                << " association_coverage_prior_gradient_norm="
                << row.associationCoveragePriorGradientNorm
                << " association_coverage_prior_to_local_gradient_ratio="
                << row.associationCoveragePriorToLocalGradientRatio
                << " association_coverage_prior_objective_cost="
                << row.associationCoveragePriorObjectiveCost
                << " association_actual_coverage_prior_alpha="
                << row.associationActualCoveragePriorAlpha
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
                << " total_comm_mb=" << row.cumulativeCommMb << "\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "bilevel synthetic object runner failed: " << e.what()
              << std::endl;
    return 1;
  }
}
