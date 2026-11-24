/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/DecentralizedBilevelAssociation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace DPGO {

SharedEdgeCommand reciprocalSharedEdgeCommand(
    const SharedEdgeCommand &command) {
  SharedEdgeCommand reverse;
  reverse.firstRobotId = command.secondRobotId;
  reverse.secondRobotId = command.firstRobotId;
  reverse.association = command.association.transpose();
  const int dimension = command.alignment.rows();
  if (dimension > 0 && command.alignment.cols() == dimension + 1) {
    reverse.alignment = Matrix::Zero(dimension, dimension + 1);
    const Matrix rotation = command.alignment.leftCols(dimension);
    const Matrix inverseRotation = rotation.transpose();
    reverse.alignment.leftCols(dimension) = inverseRotation;
    reverse.alignment.col(dimension).noalias() =
        -inverseRotation * command.alignment.col(dimension);
  }
  return reverse;
}

FixedCommandLowerContract sharedEdgeInexactBilevelLowerContract() {
  return {};
}

LocalInexactnessCertificate evaluateLocalInexactnessCertificate(
    double quotientStationarityResidual,
    double robustLatentFixedPointResidual, double stationarityScale,
    double robustLatentScale) {
  LocalInexactnessCertificate certificate;
  certificate.quotientStationarityResidual =
      std::isfinite(quotientStationarityResidual)
          ? std::max(0.0, quotientStationarityResidual)
          : 0.0;
  certificate.robustLatentFixedPointResidual =
      std::isfinite(robustLatentFixedPointResidual)
          ? std::max(0.0, robustLatentFixedPointResidual)
          : 0.0;
  const double scaleFloor =
      std::sqrt(std::numeric_limits<double>::epsilon());
  const double stationarityRatio =
      std::isfinite(stationarityScale) && stationarityScale > scaleFloor
          ? certificate.quotientStationarityResidual / stationarityScale
          : 0.0;
  const double latentRatio =
      std::isfinite(robustLatentScale) && robustLatentScale > scaleFloor
          ? certificate.robustLatentFixedPointResidual / robustLatentScale
          : 0.0;
  certificate.normalizedResidual =
      std::hypot(stationarityRatio, latentRatio);
  return certificate;
}

bool localLowerContinuationRequired(
    const LocalInexactnessCertificate &certificate, double tolerance) {
  if (!std::isfinite(tolerance) || tolerance < 0.0 ||
      !std::isfinite(certificate.normalizedResidual)) {
    return true;
  }
  return certificate.normalizedResidual > tolerance;
}

LocalInexactAdmmCertificate evaluateLocalInexactAdmmCertificate(
    const LocalInexactnessCertificate &local,
    double stationarityScale, double robustLatentScale,
    double incidentPrimalResidual, double incidentDualResidual,
    double absoluteTolerance, double relativeTolerance) {
  LocalInexactAdmmCertificate certificate;
  certificate.local = local;
  const auto residualRatio = [&](double residual, double scale) {
    if (!std::isfinite(residual) || residual < 0.0 ||
        !std::isfinite(scale) || scale < 0.0 ||
        !std::isfinite(absoluteTolerance) || absoluteTolerance <= 0.0 ||
        !std::isfinite(relativeTolerance) || relativeTolerance < 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    const double tolerance = absoluteTolerance + relativeTolerance * scale;
    return tolerance > 0.0
               ? residual / tolerance
               : std::numeric_limits<double>::infinity();
  };
  certificate.stationarityRatio = residualRatio(
      local.quotientStationarityResidual, stationarityScale);
  certificate.robustLatentRatio = residualRatio(
      local.robustLatentFixedPointResidual, robustLatentScale);
  certificate.incidentPrimalResidual =
      std::isfinite(incidentPrimalResidual)
          ? std::max(0.0, incidentPrimalResidual)
          : std::numeric_limits<double>::infinity();
  certificate.incidentDualResidual =
      std::isfinite(incidentDualResidual)
          ? std::max(0.0, incidentDualResidual)
          : std::numeric_limits<double>::infinity();
  certificate.normalizedResidual = std::max(
      {certificate.stationarityRatio, certificate.robustLatentRatio,
       certificate.incidentPrimalResidual,
       certificate.incidentDualResidual});
  return certificate;
}

bool localInexactAdmmContinuationRequired(
    const LocalInexactAdmmCertificate &certificate) {
  return !std::isfinite(certificate.normalizedResidual) ||
         certificate.normalizedResidual > 1.0;
}

LocalCommandForcingCertificate evaluateLocalCommandForcingCertificate(
    const Matrix &currentKktResidual,
    const Matrix &previousCommandKktResidual,
    const Matrix &currentCommandKktResidual) {
  if (currentKktResidual.rows() != previousCommandKktResidual.rows() ||
      currentKktResidual.cols() != previousCommandKktResidual.cols() ||
      currentKktResidual.rows() != currentCommandKktResidual.rows() ||
      currentKktResidual.cols() != currentCommandKktResidual.cols() ||
      !currentKktResidual.allFinite() ||
      !previousCommandKktResidual.allFinite() ||
      !currentCommandKktResidual.allFinite()) {
    throw std::invalid_argument(
        "local command forcing residuals must be finite and aligned");
  }

  LocalCommandForcingCertificate certificate;
  certificate.stationarityResidual = currentKktResidual.norm();
  certificate.commandInducedResidual =
      (currentCommandKktResidual - previousCommandKktResidual).norm();
  if (certificate.commandInducedResidual > 0.0) {
    certificate.forcingRatio =
        certificate.stationarityResidual /
        certificate.commandInducedResidual;
  } else if (certificate.stationarityResidual > 0.0) {
    certificate.forcingRatio = std::numeric_limits<double>::infinity();
  }
  return certificate;
}

bool localCommandForcingContinuationRequired(
    const LocalCommandForcingCertificate &certificate) {
  return certificate.stationarityResidual >
         certificate.commandInducedResidual;
}

SharedObjectEdgePenaltyBalanceUpdate updateSharedObjectEdgePenaltyBalance(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectEdgePenaltyBalanceState &state,
    const SharedObjectEdgePenaltyBalanceParameters &parameters,
    const SharedObjectConsensusPenalty &information, double edgeWeight,
    std::size_t iteration, bool advanceState, std::size_t commandEpoch) {
  if (firstEndpoint.rows() == 0 || firstEndpoint.cols() == 0 ||
      firstEndpoint.rows() != secondEndpoint.rows() ||
      firstEndpoint.cols() != secondEndpoint.cols() ||
      !firstEndpoint.allFinite() || !secondEndpoint.allFinite()) {
    throw std::invalid_argument(
        "edge penalty-balance endpoints must be finite and aligned");
  }
  if (!std::isfinite(parameters.initialBeta) ||
      parameters.initialBeta <= 0.0 ||
      !std::isfinite(parameters.minimumBeta) ||
      parameters.minimumBeta <= 0.0 ||
      !std::isfinite(parameters.maximumBeta) ||
      parameters.maximumBeta < parameters.minimumBeta ||
      !std::isfinite(parameters.imbalanceRatio) ||
      parameters.imbalanceRatio <= 1.0 ||
      !std::isfinite(parameters.updateFactor) ||
      parameters.updateFactor <= 1.0 ||
      !std::isfinite(parameters.emaDecay) || parameters.emaDecay < 0.0 ||
      parameters.emaDecay >= 1.0 || parameters.updatePeriod == 0 ||
      !std::isfinite(parameters.absoluteTolerance) ||
      parameters.absoluteTolerance <= 0.0 ||
      !std::isfinite(parameters.relativeTolerance) ||
      parameters.relativeTolerance < 0.0 || !std::isfinite(edgeWeight) ||
      edgeWeight <= 0.0) {
    throw std::invalid_argument("edge penalty-balance parameters are invalid");
  }

  const bool rotationEnabled =
      std::isfinite(information.rotation) && information.rotation > 0.0;
  const bool translationEnabled = firstEndpoint.cols() > 1 &&
                                  std::isfinite(information.translation) &&
                                  information.translation > 0.0;
  const Eigen::Index rotationCols =
      std::max<Eigen::Index>(1, firstEndpoint.cols() - 1);
  const auto weightedNorm = [&](const Matrix &value) {
    double squaredNorm = 0.0;
    if (rotationEnabled) {
      squaredNorm += information.rotation *
                     value.leftCols(rotationCols).squaredNorm();
    }
    if (translationEnabled) {
      squaredNorm += information.translation *
                     value.rightCols(1).squaredNorm();
    }
    return std::sqrt(std::max(0.0, squaredNorm));
  };

  if (!state.initialized) {
    state.beta = parameters.enabled
                     ? std::clamp(parameters.initialBeta,
                                  parameters.minimumBeta,
                                  parameters.maximumBeta)
                     : parameters.initialBeta;
    state.commandEpoch = commandEpoch;
    state.initialized = true;
  }
  const bool commandChanged = state.commandEpoch != commandEpoch;
  if (commandChanged && advanceState) {
    state.midpointPrimed = false;
    state.residualRatioPrimed = false;
    state.logResidualRatioEma = 0.0;
    state.commandEpoch = commandEpoch;
  }

  const Matrix midpoint = 0.5 * (firstEndpoint + secondEndpoint);
  const double weightedEndpointScale =
      edgeWeight * std::max(weightedNorm(firstEndpoint),
                            weightedNorm(secondEndpoint));
  double weightedEntryCount = 0.0;
  if (rotationEnabled) {
    weightedEntryCount += information.rotation * firstEndpoint.rows() *
                          rotationCols;
  }
  if (translationEnabled) {
    weightedEntryCount +=
        information.translation * firstEndpoint.rows();
  }
  const double normalization =
      parameters.absoluteTolerance * edgeWeight *
          std::sqrt(weightedEntryCount) +
      parameters.relativeTolerance * weightedEndpointScale;
  const double safeNormalization =
      normalization > 0.0 ? normalization : 1.0;

  SharedObjectEdgePenaltyBalanceUpdate update;
  update.beta = state.beta;
  update.normalizedPrimalResidual =
      edgeWeight * weightedNorm(firstEndpoint - secondEndpoint) /
      safeNormalization;
  const bool midpointCurrent =
      state.midpointPrimed && !commandChanged &&
      state.previousMidpoint.rows() == midpoint.rows() &&
      state.previousMidpoint.cols() == midpoint.cols() &&
      state.previousMidpoint.allFinite();
  if (midpointCurrent) {
    update.normalizedMidpointMotionResidual =
        state.beta * edgeWeight * edgeWeight *
        weightedNorm(midpoint - state.previousMidpoint) /
        safeNormalization;
  }

  const bool scheduledUpdate =
      parameters.enabled && advanceState && midpointCurrent &&
      iteration > 0 && iteration % parameters.updatePeriod == 0;
  if (scheduledUpdate) {
    const double logRatio = std::log(
        (update.normalizedPrimalResidual + 1.0) /
        (update.normalizedMidpointMotionResidual + 1.0));
    if (state.residualRatioPrimed) {
      state.logResidualRatioEma =
          parameters.emaDecay * state.logResidualRatioEma +
          (1.0 - parameters.emaDecay) * logRatio;
    } else {
      state.logResidualRatioEma = logRatio;
      state.residualRatioPrimed = true;
    }
    ++state.updateCount;

    const double logImbalance = std::log(parameters.imbalanceRatio);
    const bool residualActive =
        std::max(update.normalizedPrimalResidual,
                 update.normalizedMidpointMotionResidual) > 1.0;
    const double previousBeta = state.beta;
    if (residualActive && state.logResidualRatioEma > logImbalance) {
      state.beta = std::min(parameters.maximumBeta,
                            state.beta * parameters.updateFactor);
      update.penaltyIncreased = state.beta > previousBeta;
      state.increaseCount += update.penaltyIncreased ? 1u : 0u;
    } else if (residualActive &&
               state.logResidualRatioEma < -logImbalance) {
      state.beta = std::max(parameters.minimumBeta,
                            state.beta / parameters.updateFactor);
      update.penaltyDecreased = state.beta < previousBeta;
      state.decreaseCount += update.penaltyDecreased ? 1u : 0u;
    }
    update.penaltyChanged = state.beta != previousBeta;
    update.beta = state.beta;
  }

  if (advanceState) {
    state.previousMidpoint = midpoint;
    state.midpointPrimed = true;
  }
  return update;
}

SharedObjectConsensusPenalty eliminateSharedObjectConsensusLatentPenalty(
    const SharedObjectConsensusPenalty &splitPenalty) {
  const auto eliminate = [](double value) {
    if (std::isfinite(value) && value == 0.0) {
      return 0.0;
    }
    return 0.5 * (std::isfinite(value) && value > 0.0 ? value : 1.0);
  };
  return SharedObjectConsensusPenalty{eliminate(splitPenalty.rotation),
                                      eliminate(splitPenalty.translation)};
}

Matrix buildSharedObjectDirectEdgeDualProximalTarget(
    const Matrix &currentEndpoint, const Matrix &biasedNeighborTarget) {
  return buildSharedObjectDirectEdgeDualProximalTarget(
      currentEndpoint, biasedNeighborTarget, 1.0);
}

double reduceSharedObjectDirectEdgeDualSpectralStep(
    const std::vector<SharedObjectDirectEdgeDualSpectralContribution>
        &contributions,
    double fallbackStep, double previousStep, double minimumStep,
    double maximumStep, double emaDecay,
    std::size_t *validContributions) {
  if (!std::isfinite(fallbackStep) || fallbackStep <= 0.0 ||
      !std::isfinite(previousStep) || previousStep <= 0.0 ||
      !std::isfinite(minimumStep) || minimumStep <= 0.0 ||
      !std::isfinite(maximumStep) || maximumStep < minimumStep ||
      !std::isfinite(emaDecay) || emaDecay < 0.0 || emaDecay >= 1.0) {
    throw std::invalid_argument(
        "invalid global direct edge-dual spectral step input");
  }
  double numerator = 0.0;
  double denominator = 0.0;
  std::size_t count = 0u;
  for (const auto &contribution : contributions) {
    if (!contribution.valid || !std::isfinite(contribution.numerator) ||
        !std::isfinite(contribution.denominator) ||
        contribution.numerator <= 0.0 || contribution.denominator <= 0.0) {
      continue;
    }
    numerator += contribution.numerator;
    denominator += contribution.denominator;
    ++count;
  }
  if (validContributions != nullptr) {
    *validContributions = count;
  }
  if (count == 0u || !std::isfinite(numerator) ||
      !std::isfinite(denominator) || numerator <= 0.0 ||
      denominator <= 0.0) {
    return std::clamp(fallbackStep, minimumStep, maximumStep);
  }
  const double rawStep = numerator / denominator;
  if (!std::isfinite(rawStep) || rawStep <= 0.0) {
    return std::clamp(fallbackStep, minimumStep, maximumStep);
  }
  const double clippedRaw = std::clamp(rawStep, minimumStep, maximumStep);
  return std::clamp(emaDecay * previousStep +
                        (1.0 - emaDecay) * clippedRaw,
                    minimumStep, maximumStep);
}

SharedObjectDirectEdgeMomentumContribution
sharedObjectDirectEdgeEndpointMomentumContribution(
    const Matrix &previousStep, const Matrix &currentStep) {
  SharedObjectDirectEdgeMomentumContribution contribution;
  if (previousStep.rows() == 0 || previousStep.cols() == 0 ||
      previousStep.rows() != currentStep.rows() ||
      previousStep.cols() != currentStep.cols() ||
      !previousStep.allFinite() || !currentStep.allFinite()) {
    return contribution;
  }
  const Matrix stepSecant = currentStep - previousStep;
  contribution.numerator =
      -currentStep.cwiseProduct(stepSecant).sum();
  contribution.denominator = stepSecant.squaredNorm();
  const double scale =
      std::max(1.0, currentStep.norm() * stepSecant.norm());
  contribution.valid =
      std::isfinite(contribution.numerator) &&
      std::isfinite(contribution.denominator) &&
      contribution.numerator > 1e-12 * scale &&
      contribution.denominator > 1e-12 * scale;
  return contribution;
}

SharedObjectDirectEdgeMomentumContribution
updateSharedObjectDirectEdgeEndpointMomentumHistory(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state) {
  SharedObjectDirectEdgeMomentumContribution contribution;
  const bool endpointHistoryValid =
      state.inertialHistoryPrimed &&
      state.previousFirstEndpoint.rows() == firstEndpoint.rows() &&
      state.previousFirstEndpoint.cols() == firstEndpoint.cols() &&
      state.previousSecondEndpoint.rows() == secondEndpoint.rows() &&
      state.previousSecondEndpoint.cols() == secondEndpoint.cols() &&
      firstEndpoint.rows() == secondEndpoint.rows() &&
      firstEndpoint.cols() == secondEndpoint.cols() &&
      firstEndpoint.cols() > 0 && firstEndpoint.allFinite() &&
      secondEndpoint.allFinite() &&
      state.previousFirstEndpoint.allFinite() &&
      state.previousSecondEndpoint.allFinite();
  if (!endpointHistoryValid) {
    state.momentumStepHistoryPrimed = false;
    state.previousFirstEndpointStep.resize(0, 0);
    state.previousSecondEndpointStep.resize(0, 0);
    return contribution;
  }

  const Matrix currentFirstStep =
      firstEndpoint.rightCols(1) -
      state.previousFirstEndpoint.rightCols(1);
  const Matrix currentSecondStep =
      secondEndpoint.rightCols(1) -
      state.previousSecondEndpoint.rightCols(1);
  const bool stepHistoryValid =
      state.momentumStepHistoryPrimed &&
      state.previousFirstEndpointStep.rows() ==
          currentFirstStep.rows() &&
      state.previousFirstEndpointStep.cols() ==
          currentFirstStep.cols() &&
      state.previousSecondEndpointStep.rows() ==
          currentSecondStep.rows() &&
      state.previousSecondEndpointStep.cols() ==
          currentSecondStep.cols() &&
      state.previousFirstEndpointStep.allFinite() &&
      state.previousSecondEndpointStep.allFinite();
  if (stepHistoryValid) {
    const Matrix firstSecant =
        currentFirstStep - state.previousFirstEndpointStep;
    const Matrix secondSecant =
        currentSecondStep - state.previousSecondEndpointStep;
    contribution.numerator =
        -currentFirstStep.cwiseProduct(firstSecant).sum() -
        currentSecondStep.cwiseProduct(secondSecant).sum();
    contribution.denominator =
        firstSecant.squaredNorm() + secondSecant.squaredNorm();
    const double currentNorm = std::sqrt(
        currentFirstStep.squaredNorm() + currentSecondStep.squaredNorm());
    const double secantNorm = std::sqrt(contribution.denominator);
    const double scale = std::max(1.0, currentNorm * secantNorm);
    contribution.valid =
        std::isfinite(contribution.numerator) &&
        std::isfinite(contribution.denominator) &&
        contribution.numerator > 1e-12 * scale &&
        contribution.denominator > 1e-12 * scale;
  }
  state.previousFirstEndpointStep = currentFirstStep;
  state.previousSecondEndpointStep = currentSecondStep;
  state.momentumStepHistoryPrimed = true;
  return contribution;
}

double reduceSharedObjectDirectEdgeMomentum(
    const std::vector<SharedObjectDirectEdgeMomentumContribution>
        &contributions,
    double fallbackMomentum, double previousMomentum,
    double minimumMomentum, double maximumMomentum, double emaDecay,
    std::size_t *validContributions) {
  if (!std::isfinite(fallbackMomentum) || fallbackMomentum < 0.0 ||
      !std::isfinite(previousMomentum) || previousMomentum < 0.0 ||
      !std::isfinite(minimumMomentum) || minimumMomentum < 0.0 ||
      !std::isfinite(maximumMomentum) ||
      maximumMomentum < minimumMomentum ||
      !std::isfinite(emaDecay) || emaDecay < 0.0 || emaDecay >= 1.0) {
    throw std::invalid_argument(
        "invalid direct edge endpoint momentum parameters");
  }
  double numerator = 0.0;
  double denominator = 0.0;
  std::size_t count = 0u;
  for (const SharedObjectDirectEdgeMomentumContribution &contribution :
       contributions) {
    if (!contribution.valid ||
        !std::isfinite(contribution.numerator) ||
        !std::isfinite(contribution.denominator) ||
        contribution.numerator <= 0.0 ||
        contribution.denominator <= 0.0) {
      continue;
    }
    numerator += contribution.numerator;
    denominator += contribution.denominator;
    ++count;
  }
  if (validContributions != nullptr) {
    *validContributions = count;
  }
  if (count == 0u || !std::isfinite(numerator) ||
      !std::isfinite(denominator) || numerator <= 0.0 ||
      denominator <= 0.0) {
    return std::clamp(fallbackMomentum, minimumMomentum,
                      maximumMomentum);
  }
  const double rawMomentum = numerator / denominator;
  if (!std::isfinite(rawMomentum) || rawMomentum < 0.0) {
    return std::clamp(fallbackMomentum, minimumMomentum,
                      maximumMomentum);
  }
  const double clippedRaw =
      std::clamp(rawMomentum, minimumMomentum, maximumMomentum);
  return std::clamp(
      emaDecay * previousMomentum + (1.0 - emaDecay) * clippedRaw,
      minimumMomentum, maximumMomentum);
}

SharedObjectDirectEdgeDualSpectralContribution
updateSharedObjectDirectEdgeDualSpectralHistory(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty) {
  if (firstEndpoint.rows() == 0 || firstEndpoint.cols() == 0 ||
      firstEndpoint.rows() != secondEndpoint.rows() ||
      firstEndpoint.cols() != secondEndpoint.cols() ||
      !firstEndpoint.allFinite() || !secondEndpoint.allFinite()) {
    throw std::invalid_argument(
        "invalid direct edge-dual spectral history input");
  }
  const bool validMultiplier =
      state.initialized &&
      state.physicalMultiplier.rows() == firstEndpoint.rows() &&
      state.physicalMultiplier.cols() == firstEndpoint.cols() &&
      state.physicalMultiplier.allFinite();
  const Matrix currentMultiplier =
      validMultiplier
          ? state.physicalMultiplier
          : Matrix::Zero(firstEndpoint.rows(), firstEndpoint.cols());
  if (!validMultiplier) {
    state.spectralStepPrimed = false;
  }
  const Eigen::Index rotationCols =
      std::max<Eigen::Index>(1, firstEndpoint.cols() - 1);
  Matrix whitenedResidual = Matrix::Zero(firstEndpoint.rows(),
                                         firstEndpoint.cols());
  Matrix scaledMultiplier = whitenedResidual;
  const Matrix residual = firstEndpoint - secondEndpoint;
  const auto fillChannel = [&](bool translation, double information) {
    if (!std::isfinite(information) || information <= 0.0 ||
        (translation && firstEndpoint.cols() <= 1)) {
      return;
    }
    const double root = std::sqrt(information);
    if (translation) {
      whitenedResidual.rightCols(1) = root * residual.rightCols(1);
      scaledMultiplier.rightCols(1) =
          currentMultiplier.rightCols(1) / root;
    } else {
      whitenedResidual.leftCols(rotationCols) =
          root * residual.leftCols(rotationCols);
      scaledMultiplier.leftCols(rotationCols) =
          currentMultiplier.leftCols(rotationCols) / root;
    }
  };
  fillChannel(false, penalty.rotation);
  fillChannel(true, penalty.translation);

  SharedObjectDirectEdgeDualSpectralContribution contribution;
  const bool validHistory =
      state.spectralStepPrimed &&
      state.previousSpectralScaledMultiplier.rows() ==
          scaledMultiplier.rows() &&
      state.previousSpectralScaledMultiplier.cols() ==
          scaledMultiplier.cols() &&
      state.previousSpectralWhitenedResidual.rows() ==
          whitenedResidual.rows() &&
      state.previousSpectralWhitenedResidual.cols() ==
          whitenedResidual.cols() &&
      state.previousSpectralScaledMultiplier.allFinite() &&
      state.previousSpectralWhitenedResidual.allFinite();
  if (validHistory) {
    const Matrix multiplierSecant =
        scaledMultiplier - state.previousSpectralScaledMultiplier;
    const Matrix residualSecant =
        whitenedResidual - state.previousSpectralWhitenedResidual;
    const auto fillContribution =
        [&](bool translation, double &numerator, double &denominator,
            bool &valid) {
          const Matrix multiplierChannel =
              translation ? Matrix(multiplierSecant.rightCols(1))
                          : Matrix(multiplierSecant.leftCols(rotationCols));
          const Matrix residualChannel =
              translation ? Matrix(residualSecant.rightCols(1))
                          : Matrix(residualSecant.leftCols(rotationCols));
          numerator = multiplierChannel.squaredNorm();
          denominator =
              -multiplierChannel.cwiseProduct(residualChannel).sum();
          const double scale = std::max(
              1.0, multiplierChannel.norm() * residualChannel.norm());
          valid = std::isfinite(numerator) && std::isfinite(denominator) &&
                  numerator > 0.0 && denominator > 1e-12 * scale;
        };
    fillContribution(false, contribution.rotationNumerator,
                     contribution.rotationDenominator,
                     contribution.rotationValid);
    fillContribution(true, contribution.translationNumerator,
                     contribution.translationDenominator,
                     contribution.translationValid);
    contribution.numerator = multiplierSecant.squaredNorm();
    contribution.denominator =
        -multiplierSecant.cwiseProduct(residualSecant).sum();
    const double scale =
        std::max(1.0, multiplierSecant.norm() * residualSecant.norm());
    contribution.valid =
        std::isfinite(contribution.numerator) &&
        std::isfinite(contribution.denominator) &&
        contribution.numerator > 0.0 &&
        contribution.denominator > 1e-12 * scale;
  }
  state.previousSpectralScaledMultiplier = std::move(scaledMultiplier);
  state.previousSpectralWhitenedResidual = std::move(whitenedResidual);
  state.spectralStepPrimed = true;
  return contribution;
}

double estimateSharedObjectDirectEdgeDualSpectralStep(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double fallbackStep,
    double minimumStep, double maximumStep, double emaDecay) {
  if (firstEndpoint.rows() == 0 || firstEndpoint.cols() == 0 ||
      firstEndpoint.rows() != secondEndpoint.rows() ||
      firstEndpoint.cols() != secondEndpoint.cols() ||
      !firstEndpoint.allFinite() || !secondEndpoint.allFinite() ||
      !std::isfinite(fallbackStep) || fallbackStep <= 0.0 ||
      !std::isfinite(minimumStep) || minimumStep <= 0.0 ||
      !std::isfinite(maximumStep) || maximumStep < minimumStep ||
      !std::isfinite(emaDecay) || emaDecay < 0.0 || emaDecay >= 1.0) {
    throw std::invalid_argument("invalid direct edge-dual spectral step input");
  }
  const auto contribution =
      updateSharedObjectDirectEdgeDualSpectralHistory(
          firstEndpoint, secondEndpoint, state, penalty);
  const double previousStep =
      std::isfinite(state.spectralStep) && state.spectralStep > 0.0
          ? state.spectralStep
          : fallbackStep;
  const double step = reduceSharedObjectDirectEdgeDualSpectralStep(
      {contribution}, fallbackStep, previousStep, minimumStep, maximumStep,
      emaDecay);
  state.spectralStep = step;
  return step;
}

Matrix buildSharedObjectDirectEdgeDualProximalTarget(
    const Matrix &currentEndpoint, const Matrix &biasedNeighborTarget,
    double curvatureScale) {
  if (currentEndpoint.rows() == 0 || currentEndpoint.cols() == 0 ||
      currentEndpoint.rows() != biasedNeighborTarget.rows() ||
      currentEndpoint.cols() != biasedNeighborTarget.cols() ||
      !currentEndpoint.allFinite() || !biasedNeighborTarget.allFinite()) {
    throw std::invalid_argument(
        "direct edge-dual proximal endpoints must be finite and aligned");
  }
  if (!std::isfinite(curvatureScale) || curvatureScale < 0.5 ||
      curvatureScale > 1.0) {
    throw std::invalid_argument(
        "direct edge-dual curvature scale must be in [0.5, 1]");
  }
  const double targetStep = 0.5 / curvatureScale;
  return currentEndpoint +
         targetStep * (biasedNeighborTarget - currentEndpoint);
}

SharedObjectConsensusUpdate updateSharedObjectConsensusState(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectConsensusState &state,
    const SharedObjectConsensusPenalty &penalty,
    double absoluteTolerance, double relativeTolerance,
    std::size_t commandEpoch) {
  if (firstEndpoint.rows() == 0 || firstEndpoint.cols() == 0 ||
      firstEndpoint.rows() != secondEndpoint.rows() ||
      firstEndpoint.cols() != secondEndpoint.cols() ||
      !firstEndpoint.allFinite() || !secondEndpoint.allFinite()) {
    throw std::invalid_argument(
        "shared-object consensus endpoints must be finite and aligned");
  }
  if (!std::isfinite(absoluteTolerance) || absoluteTolerance <= 0.0 ||
      !std::isfinite(relativeTolerance) || relativeTolerance < 0.0) {
    throw std::invalid_argument(
        "shared-object consensus tolerances must be finite and positive");
  }
  const Eigen::Index rotationCols =
      std::max<Eigen::Index>(1, firstEndpoint.cols() - 1);
  const bool hasTranslation = firstEndpoint.cols() > 1;
  const bool rotationEnabled =
      std::isfinite(penalty.rotation) && penalty.rotation > 0.0;
  const bool translationEnabled =
      hasTranslation && std::isfinite(penalty.translation) &&
      penalty.translation > 0.0;
  const auto channelPenalty = [&](bool translation) {
    const double value = translation ? penalty.translation : penalty.rotation;
    const bool enabled = translation ? translationEnabled : rotationEnabled;
    return enabled ? value : 0.0;
  };
  const auto channelEntries = [&](bool translation) {
    return static_cast<double>(
        firstEndpoint.rows() * (translation ? 1 : rotationCols));
  };
  const auto twoEndpointChannelNorm =
      [&](const Matrix &firstResidual, const Matrix &secondResidual,
          bool translation) {
        if (translation) {
          if (!translationEnabled) {
            return 0.0;
          }
          return std::sqrt(
              firstResidual.rightCols(1).squaredNorm() +
              secondResidual.rightCols(1).squaredNorm());
        }
        return rotationEnabled ? std::sqrt(
            firstResidual.leftCols(rotationCols).squaredNorm() +
            secondResidual.leftCols(rotationCols).squaredNorm()) : 0.0;
      };
  const auto oneEndpointChannelNorm =
      [&](const Matrix &residual, bool translation) {
        if (translation) {
          return translationEnabled ? residual.rightCols(1).norm() : 0.0;
        }
        return rotationEnabled ? residual.leftCols(rotationCols).norm() : 0.0;
      };
  const auto normalizedPrimalResidual =
      [&](double residual, bool translation, const Matrix &latent) {
        if ((translation && !translationEnabled) ||
            (!translation && !rotationEnabled)) {
          return 0.0;
        }
        const double endpointNorm = twoEndpointChannelNorm(
            firstEndpoint, secondEndpoint, translation);
        const double latentNorm = std::sqrt(2.0) *
                                  oneEndpointChannelNorm(latent, translation);
        const double tolerance =
            std::sqrt(2.0 * channelEntries(translation)) *
                absoluteTolerance +
            relativeTolerance * std::max(endpointNorm, latentNorm);
        return tolerance > 0.0 ? residual / tolerance
                               : std::numeric_limits<double>::infinity();
      };
  const auto normalizedDualResidual =
      [&](double unscaledResidual, bool translation) {
        if ((translation && !translationEnabled) ||
            (!translation && !rotationEnabled)) {
          return 0.0;
        }
        const double rho = channelPenalty(translation);
        const double scaledResidual = rho * unscaledResidual;
        const double scaledDualNorm =
            rho * twoEndpointChannelNorm(state.firstScaledDual,
                                         state.secondScaledDual,
                                         translation);
        const double tolerance =
            std::sqrt(2.0 * channelEntries(translation)) *
                absoluteTolerance +
            relativeTolerance * scaledDualNorm;
        return tolerance > 0.0
                   ? scaledResidual / tolerance
                   : std::numeric_limits<double>::infinity();
  };

  SharedObjectConsensusUpdate update;
  const bool validState =
      state.initialized && state.latent.rows() == firstEndpoint.rows() &&
      state.latent.cols() == firstEndpoint.cols() &&
      state.firstScaledDual.rows() == firstEndpoint.rows() &&
      state.firstScaledDual.cols() == firstEndpoint.cols() &&
      state.secondScaledDual.rows() == firstEndpoint.rows() &&
      state.secondScaledDual.cols() == firstEndpoint.cols() &&
      state.latent.allFinite() && state.firstScaledDual.allFinite() &&
      state.secondScaledDual.allFinite();
  const bool dualCertificateCurrent =
      validState && state.dualCertificatePrimed &&
      state.commandEpoch == commandEpoch;

  if (!validState) {
    state.latent = 0.5 * (firstEndpoint + secondEndpoint);
    state.firstScaledDual = Matrix::Zero(firstEndpoint.rows(),
                                         firstEndpoint.cols());
    state.secondScaledDual = Matrix::Zero(firstEndpoint.rows(),
                                          firstEndpoint.cols());
    state.commandEpoch = commandEpoch;
    state.dualCertificatePrimed = true;
    state.initialized = true;
    update.firstTarget = state.latent;
    update.secondTarget = state.latent;
    const Matrix firstResidual = firstEndpoint - state.latent;
    const Matrix secondResidual = secondEndpoint - state.latent;
    const double rotationPrimal = twoEndpointChannelNorm(
        firstResidual, secondResidual, false);
    const double translationPrimal = twoEndpointChannelNorm(
        firstResidual, secondResidual, true);
    update.primalResidual = std::hypot(rotationPrimal, translationPrimal);
    update.normalizedPrimalResidual = std::max(
        normalizedPrimalResidual(rotationPrimal, false, state.latent),
        normalizedPrimalResidual(translationPrimal, true, state.latent));
    update.normalizedDualResidual =
        rotationEnabled || translationEnabled
            ? std::numeric_limits<double>::infinity()
            : 0.0;
    return update;
  }

  state.commandEpoch = commandEpoch;
  if (!rotationEnabled) {
    state.firstScaledDual.leftCols(rotationCols).setZero();
    state.secondScaledDual.leftCols(rotationCols).setZero();
  }
  if (hasTranslation && !translationEnabled) {
    state.firstScaledDual.rightCols(1).setZero();
    state.secondScaledDual.rightCols(1).setZero();
  }
  const Matrix previousLatent = state.latent;
  state.latent =
      0.5 * (firstEndpoint + state.firstScaledDual + secondEndpoint +
             state.secondScaledDual);
  state.firstScaledDual.noalias() += firstEndpoint - state.latent;
  state.secondScaledDual.noalias() += secondEndpoint - state.latent;
  if (!rotationEnabled) {
    state.firstScaledDual.leftCols(rotationCols).setZero();
    state.secondScaledDual.leftCols(rotationCols).setZero();
  }
  if (hasTranslation && !translationEnabled) {
    state.firstScaledDual.rightCols(1).setZero();
    state.secondScaledDual.rightCols(1).setZero();
  }

  update.firstTarget = state.latent - state.firstScaledDual;
  update.secondTarget = state.latent - state.secondScaledDual;
  const Matrix firstResidual = firstEndpoint - state.latent;
  const Matrix secondResidual = secondEndpoint - state.latent;
  const Matrix latentDisplacement = state.latent - previousLatent;
  const double rotationPrimal = twoEndpointChannelNorm(
      firstResidual, secondResidual, false);
  const double translationPrimal = twoEndpointChannelNorm(
      firstResidual, secondResidual, true);
  const double rotationDual =
      std::sqrt(2.0) * oneEndpointChannelNorm(latentDisplacement, false);
  const double translationDual =
      std::sqrt(2.0) * oneEndpointChannelNorm(latentDisplacement, true);
  update.primalResidual = std::hypot(rotationPrimal, translationPrimal);
  update.dualResidual = std::hypot(rotationDual, translationDual);
  update.rotationDualResidual = rotationDual;
  update.translationDualResidual = translationDual;
  update.rotationPhysicalMultiplierNorm =
      channelPenalty(false) * twoEndpointChannelNorm(
                                  state.firstScaledDual,
                                  state.secondScaledDual, false);
  update.translationPhysicalMultiplierNorm =
      channelPenalty(true) * twoEndpointChannelNorm(
                                 state.firstScaledDual,
                                 state.secondScaledDual, true);
  update.normalizedPrimalResidual = std::max(
      normalizedPrimalResidual(rotationPrimal, false, state.latent),
      normalizedPrimalResidual(translationPrimal, true, state.latent));
  update.normalizedRotationDualResidual =
      dualCertificateCurrent
          ? normalizedDualResidual(rotationDual, false)
          : std::numeric_limits<double>::infinity();
  update.normalizedTranslationDualResidual =
      dualCertificateCurrent
          ? normalizedDualResidual(translationDual, true)
          : std::numeric_limits<double>::infinity();
  update.normalizedDualResidual = std::max(
      update.normalizedRotationDualResidual,
      update.normalizedTranslationDualResidual);
  state.dualCertificatePrimed = true;
  return update;
}

SharedObjectConsensusUpdate
updateSharedObjectDirectEdgeDualStateWithChannelSteps(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double rotationStepSize,
    double translationStepSize,
    double absoluteTolerance, double relativeTolerance,
    std::size_t commandEpoch, double robustReliability,
    double associationProbability, bool advanceDual) {
  if (firstEndpoint.rows() == 0 || firstEndpoint.cols() == 0 ||
      firstEndpoint.rows() != secondEndpoint.rows() ||
      firstEndpoint.cols() != secondEndpoint.cols() ||
      !firstEndpoint.allFinite() || !secondEndpoint.allFinite()) {
    throw std::invalid_argument(
        "direct edge-dual endpoints must be finite and aligned");
  }
  if (!std::isfinite(rotationStepSize) || rotationStepSize <= 0.0 ||
      !std::isfinite(translationStepSize) || translationStepSize <= 0.0 ||
      !std::isfinite(absoluteTolerance) || absoluteTolerance <= 0.0 ||
      !std::isfinite(relativeTolerance) || relativeTolerance < 0.0 ||
      !std::isfinite(robustReliability) || robustReliability < 0.0 ||
      robustReliability > 1.0 ||
      !std::isfinite(associationProbability) ||
      associationProbability < 0.0 || associationProbability > 1.0) {
    throw std::invalid_argument(
        "direct edge-dual step, tolerances, probability, and reliability "
        "are invalid");
  }

  const Eigen::Index rotationCols =
      std::max<Eigen::Index>(1, firstEndpoint.cols() - 1);
  const bool hasTranslation = firstEndpoint.cols() > 1;
  const bool rotationEnabled =
      std::isfinite(penalty.rotation) && penalty.rotation > 0.0;
  const bool translationEnabled =
      hasTranslation && std::isfinite(penalty.translation) &&
      penalty.translation > 0.0;
  const auto channelPenalty = [&](bool translation) {
    const double value = translation ? penalty.translation : penalty.rotation;
    const bool enabled = translation ? translationEnabled : rotationEnabled;
    return enabled ? value : 0.0;
  };
  const auto channelNorm = [&](const Matrix &value, bool translation) {
    if (translation) {
      return translationEnabled ? value.rightCols(1).norm() : 0.0;
    }
    return rotationEnabled ? value.leftCols(rotationCols).norm() : 0.0;
  };
  const auto channelEntries = [&](bool translation) {
    return static_cast<double>(
        firstEndpoint.rows() * (translation ? 1 : rotationCols));
  };

  const bool validState =
      state.initialized &&
      state.physicalMultiplier.rows() == firstEndpoint.rows() &&
      state.physicalMultiplier.cols() == firstEndpoint.cols() &&
      state.physicalMultiplier.allFinite();
  // The cache key fixes the physical edge and endpoint identities. A new
  // upper command invalidates convergence certificates, but the multiplier
  // remains a valid owner-frame warm start for that same constraint.
  const bool commandChanged =
      validState && state.commandEpoch != commandEpoch;
  const bool dualCertificateCurrent =
      validState && !commandChanged && state.dualCertificatePrimed;
  if (!validState) {
    state.physicalMultiplier =
        Matrix::Zero(firstEndpoint.rows(), firstEndpoint.cols());
    state.initialized = true;
    state.dualCertificatePrimed = false;
    state.dualRelaxationSettled = false;
    state.spectralStepPrimed = false;
    state.momentumStepHistoryPrimed = false;
    state.previousFirstEndpointStep.resize(0, 0);
    state.previousSecondEndpointStep.resize(0, 0);
  }
  if (!rotationEnabled) {
    state.physicalMultiplier.leftCols(rotationCols).setZero();
  }
  if (hasTranslation && !translationEnabled) {
    state.physicalMultiplier.rightCols(1).setZero();
  }

  const Matrix residual = firstEndpoint - secondEndpoint;
  const Matrix previousMultiplier = state.physicalMultiplier;
  const double confidence = associationProbability * robustReliability;
  // Maximizing a confidence-regularized edge dual gives the update
  // lambda+ = (1-alpha(1-c)) lambda + alpha c rho r. It recovers standard
  // AL at c=1 while keeping fractional hypotheses continuously soft.
  if (validState && advanceDual && !commandChanged) {
    state.physicalMultiplier.leftCols(rotationCols) *=
        1.0 - rotationStepSize * (1.0 - confidence);
    if (hasTranslation) {
      state.physicalMultiplier.rightCols(1) *=
          1.0 - translationStepSize * (1.0 - confidence);
    }
  }
  // A changed upper command defines a new fixed-command primal subproblem.
  // Keep the transported multiplier as its warm start, but do not ascend it
  // until an endpoint update has been produced for that command.
  if (validState && advanceDual && !commandChanged) {
    if (rotationEnabled) {
      state.physicalMultiplier.leftCols(rotationCols).noalias() +=
          rotationStepSize * confidence * channelPenalty(false) *
          residual.leftCols(rotationCols);
    }
    if (translationEnabled) {
      state.physicalMultiplier.rightCols(1).noalias() +=
          translationStepSize * confidence * channelPenalty(true) *
          residual.rightCols(1);
    }
  }
  const Matrix multiplierIncrement =
      state.physicalMultiplier - previousMultiplier;
  state.commandEpoch = commandEpoch;

  Matrix scaledMultiplier = state.physicalMultiplier;
  if (rotationEnabled) {
    scaledMultiplier.leftCols(rotationCols) /= channelPenalty(false);
  } else {
    scaledMultiplier.leftCols(rotationCols).setZero();
  }
  if (translationEnabled) {
    scaledMultiplier.rightCols(1) /= channelPenalty(true);
  } else if (hasTranslation) {
    scaledMultiplier.rightCols(1).setZero();
  }

  SharedObjectConsensusUpdate update;
  update.firstTarget = secondEndpoint - scaledMultiplier;
  update.secondTarget = firstEndpoint + scaledMultiplier;
  // Association probability and the line-process latent multiply the
  // complete augmented edge term.
  const double rotationPrimal =
      confidence * channelNorm(residual, false);
  const double translationPrimal =
      confidence * channelNorm(residual, true);
  const double rotationDual = channelNorm(multiplierIncrement, false);
  const double translationDual = channelNorm(multiplierIncrement, true);
  update.primalResidual = std::hypot(rotationPrimal, translationPrimal);
  update.dualResidual = std::hypot(rotationDual, translationDual);
  update.rotationDualResidual = rotationDual;
  update.translationDualResidual = translationDual;
  update.rotationPhysicalMultiplierNorm =
      channelNorm(state.physicalMultiplier, false);
  update.translationPhysicalMultiplierNorm =
      channelNorm(state.physicalMultiplier, true);

  const auto normalizedPrimal = [&](double value, bool translation) {
    const bool enabled = translation ? translationEnabled : rotationEnabled;
    if (!enabled) {
      return 0.0;
    }
    const double endpointScale =
        std::max(channelNorm(firstEndpoint, translation),
                 channelNorm(secondEndpoint, translation));
    const double tolerance =
        std::sqrt(channelEntries(translation)) * absoluteTolerance +
        relativeTolerance * endpointScale;
    return tolerance > 0.0 ? value / tolerance
                           : std::numeric_limits<double>::infinity();
  };
  const auto normalizedDual = [&](double value, bool translation) {
    const bool enabled = translation ? translationEnabled : rotationEnabled;
    if (!enabled) {
      return 0.0;
    }
    const double tolerance =
        std::sqrt(channelEntries(translation)) * absoluteTolerance +
        relativeTolerance *
            channelNorm(state.physicalMultiplier, translation);
    return tolerance > 0.0 ? value / tolerance
                           : std::numeric_limits<double>::infinity();
  };
  update.normalizedPrimalResidual =
      std::max(normalizedPrimal(rotationPrimal, false),
               normalizedPrimal(translationPrimal, true));
  update.normalizedRotationDualResidual =
      dualCertificateCurrent
          ? normalizedDual(rotationDual, false)
          : std::numeric_limits<double>::infinity();
  update.normalizedTranslationDualResidual =
      dualCertificateCurrent
          ? normalizedDual(translationDual, true)
          : std::numeric_limits<double>::infinity();
  update.normalizedDualResidual =
      std::max(update.normalizedRotationDualResidual,
               update.normalizedTranslationDualResidual);
  state.dualCertificatePrimed = true;
  return update;
}

SharedObjectConsensusUpdate updateSharedObjectDirectEdgeDualState(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double dualStepSize,
    double absoluteTolerance, double relativeTolerance,
    std::size_t commandEpoch, double robustReliability,
    double associationProbability, bool advanceDual) {
  return updateSharedObjectDirectEdgeDualStateWithChannelSteps(
      firstEndpoint, secondEndpoint, state, penalty, dualStepSize,
      dualStepSize, absoluteTolerance, relativeTolerance, commandEpoch,
      robustReliability, associationProbability, advanceDual);
}

SharedObjectDirectEdgeDualInertialUpdate
updateSharedObjectDirectEdgeDualStateInertial(
    const Matrix &firstEndpoint, const Matrix &secondEndpoint,
    SharedObjectDirectEdgeDualState &state,
    const SharedObjectConsensusPenalty &penalty, double dualStepSize,
    double momentum, double absoluteTolerance, double relativeTolerance,
    std::size_t commandEpoch, double robustReliability,
    double associationProbability, bool advanceDual) {
  if (!std::isfinite(momentum) || momentum < 0.0 || momentum >= 1.0) {
    throw std::invalid_argument(
        "direct edge-dual inertial momentum must be in [0, 1)");
  }

  const SharedObjectDirectEdgeDualState originalState = state;
  SharedObjectDirectEdgeDualState baselineState = state;
  SharedObjectDirectEdgeDualInertialUpdate result;
  result.consensus = updateSharedObjectDirectEdgeDualState(
      firstEndpoint, secondEndpoint, baselineState, penalty, dualStepSize,
      absoluteTolerance, relativeTolerance, commandEpoch, robustReliability,
      associationProbability, advanceDual);

  const auto historyMatches = [&](const Matrix &value) {
    return value.rows() == firstEndpoint.rows() &&
           value.cols() == firstEndpoint.cols() && value.allFinite();
  };
  const bool originalStateValid =
      originalState.initialized &&
      historyMatches(originalState.physicalMultiplier);
  const bool commandCurrent =
      originalStateValid && originalState.commandEpoch == commandEpoch;
  const bool canAttempt =
      advanceDual && momentum > 0.0 && commandCurrent &&
      originalState.inertialHistoryPrimed &&
      historyMatches(originalState.previousFirstEndpoint) &&
      historyMatches(originalState.previousSecondEndpoint) &&
      historyMatches(originalState.previousPhysicalMultiplier);

  const Eigen::Index rotationCols =
      std::max<Eigen::Index>(1, firstEndpoint.cols() - 1);
  const bool hasTranslation = firstEndpoint.cols() > 1;
  const bool rotationEnabled =
      std::isfinite(penalty.rotation) && penalty.rotation > 0.0;
  const bool translationEnabled =
      hasTranslation && std::isfinite(penalty.translation) &&
      penalty.translation > 0.0;
  const double confidence = associationProbability * robustReliability;
  const auto merit = [&](const SharedObjectDirectEdgeDualState &candidate,
                         const SharedObjectConsensusUpdate &candidateUpdate) {
    const Matrix multiplierStep =
        candidate.physicalMultiplier - originalState.physicalMultiplier;
    const Matrix firstTargetResidual =
        firstEndpoint - candidateUpdate.firstTarget;
    const Matrix secondTargetResidual =
        secondEndpoint - candidateUpdate.secondTarget;
    double value = 0.0;
    if (rotationEnabled) {
      value += confidence * confidence * penalty.rotation *
               (firstTargetResidual.leftCols(rotationCols).squaredNorm() +
                secondTargetResidual.leftCols(rotationCols).squaredNorm());
      value += multiplierStep.leftCols(rotationCols).squaredNorm() /
               penalty.rotation;
    }
    if (translationEnabled) {
      value += confidence * confidence * penalty.translation *
               (firstTargetResidual.rightCols(1).squaredNorm() +
                secondTargetResidual.rightCols(1).squaredNorm());
      value += multiplierStep.rightCols(1).squaredNorm() /
               penalty.translation;
    }
    return value;
  };

  result.baselineMerit = originalStateValid
                             ? merit(baselineState, result.consensus)
                             : result.consensus.primalResidual *
                                   result.consensus.primalResidual;
  if (canAttempt) {
    result.attempted = true;
    const Matrix extrapolatedFirst =
        firstEndpoint +
        momentum * (firstEndpoint - originalState.previousFirstEndpoint);
    const Matrix extrapolatedSecond =
        secondEndpoint +
        momentum * (secondEndpoint - originalState.previousSecondEndpoint);
    SharedObjectDirectEdgeDualState acceleratedState = originalState;
    acceleratedState.physicalMultiplier =
        originalState.physicalMultiplier +
        momentum * (originalState.physicalMultiplier -
                    originalState.previousPhysicalMultiplier);
    SharedObjectConsensusUpdate acceleratedUpdate =
        updateSharedObjectDirectEdgeDualState(
            extrapolatedFirst, extrapolatedSecond, acceleratedState, penalty,
            dualStepSize, absoluteTolerance, relativeTolerance, commandEpoch,
            robustReliability, associationProbability, advanceDual);
    result.acceleratedMerit = merit(acceleratedState, acceleratedUpdate);
    const double comparisonScale =
        std::max(1.0, std::abs(result.baselineMerit));
    if (std::isfinite(result.acceleratedMerit) &&
        result.acceleratedMerit <
            result.baselineMerit -
                16.0 * std::numeric_limits<double>::epsilon() *
                    comparisonScale) {
      state = std::move(acceleratedState);
      result.consensus = std::move(acceleratedUpdate);
      result.accepted = true;
    } else {
      state = std::move(baselineState);
      result.restarted = true;
    }
  } else {
    state = std::move(baselineState);
    result.acceleratedMerit = result.baselineMerit;
  }

  // Stopping diagnostics remain tied to the current physical endpoints even
  // when the accepted proximal targets came from an extrapolated edge state.
  SharedObjectDirectEdgeDualState diagnosticState = state;
  const SharedObjectConsensusUpdate currentEndpointDiagnostics =
      updateSharedObjectDirectEdgeDualState(
          firstEndpoint, secondEndpoint, diagnosticState, penalty,
          dualStepSize, absoluteTolerance, relativeTolerance, commandEpoch,
          robustReliability, associationProbability, false);
  result.consensus.primalResidual =
      currentEndpointDiagnostics.primalResidual;
  result.consensus.normalizedPrimalResidual =
      currentEndpointDiagnostics.normalizedPrimalResidual;
  Matrix committedMultiplierStep =
      Matrix::Zero(firstEndpoint.rows(), firstEndpoint.cols());
  if (originalStateValid) {
    committedMultiplierStep =
        state.physicalMultiplier - originalState.physicalMultiplier;
  }
  const double rotationDual =
      rotationEnabled
          ? committedMultiplierStep.leftCols(rotationCols).norm()
          : 0.0;
  const double translationDual =
      translationEnabled ? committedMultiplierStep.rightCols(1).norm()
                         : 0.0;
  result.consensus.dualResidual =
      std::hypot(rotationDual, translationDual);
  result.consensus.rotationDualResidual = rotationDual;
  result.consensus.translationDualResidual = translationDual;
  result.consensus.rotationPhysicalMultiplierNorm =
      rotationEnabled
          ? state.physicalMultiplier.leftCols(rotationCols).norm()
          : 0.0;
  result.consensus.translationPhysicalMultiplierNorm =
      translationEnabled ? state.physicalMultiplier.rightCols(1).norm()
                         : 0.0;
  const bool dualCertificateCurrent =
      originalStateValid && commandCurrent &&
      originalState.dualCertificatePrimed;
  const auto normalizedCommittedDual =
      [&](double value, bool translation) {
        const bool enabled = translation ? translationEnabled
                                         : rotationEnabled;
        if (!enabled) {
          return 0.0;
        }
        if (!dualCertificateCurrent) {
          return std::numeric_limits<double>::infinity();
        }
        const double entries = static_cast<double>(
            firstEndpoint.rows() * (translation ? 1 : rotationCols));
        const double multiplierNorm =
            translation
                ? result.consensus.translationPhysicalMultiplierNorm
                : result.consensus.rotationPhysicalMultiplierNorm;
        const double tolerance =
            std::sqrt(entries) * absoluteTolerance +
            relativeTolerance * multiplierNorm;
        return tolerance > 0.0
                   ? value / tolerance
                   : std::numeric_limits<double>::infinity();
      };
  result.consensus.normalizedRotationDualResidual =
      normalizedCommittedDual(rotationDual, false);
  result.consensus.normalizedTranslationDualResidual =
      normalizedCommittedDual(translationDual, true);
  result.consensus.normalizedDualResidual = std::max(
      result.consensus.normalizedRotationDualResidual,
      result.consensus.normalizedTranslationDualResidual);

  if (advanceDual) {
    state.inertialHistoryPrimed = true;
    state.previousFirstEndpoint = firstEndpoint;
    state.previousSecondEndpoint = secondEndpoint;
    state.previousPhysicalMultiplier =
        result.restarted || !originalStateValid || !commandCurrent
            ? state.physicalMultiplier
            : originalState.physicalMultiplier;
  }
  return result;
}

void transportSharedObjectDirectEdgeDualState(
    SharedObjectDirectEdgeDualState &state,
    const Matrix &ownerGaugeRotation,
    const Matrix &ownerGaugeTranslation) {
  if (!state.initialized) {
    return;
  }
  if (ownerGaugeRotation.rows() == 0 ||
      ownerGaugeRotation.rows() != ownerGaugeRotation.cols() ||
      state.physicalMultiplier.rows() != ownerGaugeRotation.cols() ||
      !ownerGaugeRotation.allFinite() ||
      !state.physicalMultiplier.allFinite()) {
    throw std::invalid_argument(
        "direct edge-dual gauge transport must be finite and aligned");
  }
  if (ownerGaugeTranslation.size() > 0 &&
      (ownerGaugeTranslation.rows() != ownerGaugeRotation.rows() ||
       ownerGaugeTranslation.cols() != 1 ||
       !ownerGaugeTranslation.allFinite())) {
    throw std::invalid_argument(
        "direct edge-dual gauge translation must be finite and aligned");
  }
  state.physicalMultiplier =
      ownerGaugeRotation * state.physicalMultiplier;
  if (state.inertialHistoryPrimed) {
    const auto validHistory = [&](const Matrix &value) {
      return value.rows() == ownerGaugeRotation.cols() &&
             value.cols() == state.physicalMultiplier.cols() &&
             value.allFinite();
    };
    if (!validHistory(state.previousFirstEndpoint) ||
        !validHistory(state.previousSecondEndpoint) ||
        !validHistory(state.previousPhysicalMultiplier)) {
      throw std::invalid_argument(
          "direct edge-dual inertial history gauge transport is invalid");
    }
    state.previousFirstEndpoint =
        ownerGaugeRotation * state.previousFirstEndpoint;
    state.previousSecondEndpoint =
        ownerGaugeRotation * state.previousSecondEndpoint;
    state.previousPhysicalMultiplier =
        ownerGaugeRotation * state.previousPhysicalMultiplier;
    if (ownerGaugeTranslation.size() > 0) {
      state.previousFirstEndpoint.rightCols(1) += ownerGaugeTranslation;
      state.previousSecondEndpoint.rightCols(1) += ownerGaugeTranslation;
    }
  }
  if (state.momentumStepHistoryPrimed) {
    const auto validMomentumStep = [&](const Matrix &value) {
      return value.rows() == ownerGaugeRotation.cols() &&
             value.cols() == 1 && value.allFinite();
    };
    if (!validMomentumStep(state.previousFirstEndpointStep) ||
        !validMomentumStep(state.previousSecondEndpointStep)) {
      throw std::invalid_argument(
          "direct edge endpoint momentum history gauge transport is "
          "invalid");
    }
    state.previousFirstEndpointStep =
        ownerGaugeRotation * state.previousFirstEndpointStep;
    state.previousSecondEndpointStep =
        ownerGaugeRotation * state.previousSecondEndpointStep;
  }
  if (state.spectralStepPrimed) {
    if (state.previousSpectralScaledMultiplier.rows() !=
            ownerGaugeRotation.cols() ||
        state.previousSpectralWhitenedResidual.rows() !=
            ownerGaugeRotation.cols() ||
        state.previousSpectralScaledMultiplier.cols() !=
            state.physicalMultiplier.cols() ||
        state.previousSpectralWhitenedResidual.cols() !=
            state.physicalMultiplier.cols() ||
        !state.previousSpectralScaledMultiplier.allFinite() ||
        !state.previousSpectralWhitenedResidual.allFinite()) {
      throw std::invalid_argument(
          "direct edge-dual spectral history gauge transport is invalid");
    }
    state.previousSpectralScaledMultiplier =
        ownerGaugeRotation * state.previousSpectralScaledMultiplier;
    state.previousSpectralWhitenedResidual =
        ownerGaugeRotation * state.previousSpectralWhitenedResidual;
  }
  if (state.penaltyBalance.midpointPrimed) {
    if (state.penaltyBalance.previousMidpoint.rows() !=
            ownerGaugeRotation.cols() ||
        state.penaltyBalance.previousMidpoint.cols() < 1 ||
        !state.penaltyBalance.previousMidpoint.allFinite()) {
      throw std::invalid_argument(
          "edge penalty-balance midpoint gauge transport is invalid");
    }
    state.penaltyBalance.previousMidpoint =
        ownerGaugeRotation * state.penaltyBalance.previousMidpoint;
    if (ownerGaugeTranslation.size() > 0) {
      state.penaltyBalance.previousMidpoint.rightCols(1) +=
          ownerGaugeTranslation;
    }
  }
}

bool SharedObjectLowerMessageKey::operator==(
    const SharedObjectLowerMessageKey &other) const {
  return receiverRobotId == other.receiverRobotId &&
         senderRobotId == other.senderRobotId &&
         receiverObjectId == other.receiverObjectId &&
         senderObjectId == other.senderObjectId;
}

bool SharedObjectLowerMessageKey::operator<(
    const SharedObjectLowerMessageKey &other) const {
  return std::tie(receiverRobotId, senderRobotId, receiverObjectId,
                  senderObjectId) <
         std::tie(other.receiverRobotId, other.senderRobotId,
                  other.receiverObjectId, other.senderObjectId);
}

void SharedObjectLowerMessageQueue::enqueueAttempt(
    std::size_t messageSequence, std::size_t round, int deliveryDelay) {
  if (deliveryDelay < 0) {
    throw std::invalid_argument("shared-object lower delivery delay < 0");
  }
  const std::size_t delay = static_cast<std::size_t>(deliveryDelay);
  if (delay > std::numeric_limits<std::size_t>::max() - round) {
    throw std::overflow_error("shared-object lower delivery round overflow");
  }
  PendingAttempt attempt;
  attempt.deliveryRound = round + delay;
  attempt.attemptSequence = nextAttemptSequence_++;
  attempt.messageSequence = messageSequence;
  pending_.push_back(attempt);
}

SharedObjectLowerMessage SharedObjectLowerMessageQueue::send(
    std::size_t round, int deliveryDelay, bool dropAttempt,
    SharedObjectLowerMessage message) {
  if (deliveryDelay < 0) {
    throw std::invalid_argument("shared-object lower delivery delay < 0");
  }
  if (!dropAttempt &&
      static_cast<std::size_t>(deliveryDelay) >
          std::numeric_limits<std::size_t>::max() - round) {
    throw std::overflow_error("shared-object lower delivery round overflow");
  }
  for (auto existing = outstanding_.begin(); existing != outstanding_.end();) {
    if (existing->second.key == message.key) {
      const bool hasPendingAttempt =
          std::any_of(pending_.begin(), pending_.end(),
                      [&](const PendingAttempt &attempt) {
                        return attempt.messageSequence == existing->first;
                      });
      if (!hasPendingAttempt) {
        existing = outstanding_.erase(existing);
        continue;
      }
    }
    ++existing;
  }
  message.messageSequence = nextMessageSequence_++;
  outstanding_[message.messageSequence] = message;
  if (dropAttempt) {
    ++droppedAttempts_;
  } else {
    enqueueAttempt(message.messageSequence, round, deliveryDelay);
  }
  return message;
}

bool SharedObjectLowerMessageQueue::retry(
    std::size_t messageSequence, std::size_t round, int deliveryDelay) {
  if (deliveryDelay < 0) {
    throw std::invalid_argument("shared-object lower delivery delay < 0");
  }
  if (outstanding_.find(messageSequence) == outstanding_.end()) {
    return false;
  }
  const bool alreadyPending =
      std::any_of(pending_.begin(), pending_.end(),
                  [&](const PendingAttempt &attempt) {
                    return attempt.messageSequence == messageSequence;
                  });
  if (alreadyPending) {
    return false;
  }
  enqueueAttempt(messageSequence, round, deliveryDelay);
  return true;
}

std::vector<SharedObjectLowerMessage>
SharedObjectLowerMessageQueue::deliverDueMessages(
    std::size_t receiverRobotId, std::size_t round) {
  return deliverDueMessagesWithReachability(receiverRobotId, round, nullptr);
}

std::vector<SharedObjectLowerMessage>
SharedObjectLowerMessageQueue::deliverDueMessages(
    std::size_t receiverRobotId, std::size_t round,
    const std::vector<std::size_t> &reachableSenderRobotIds) {
  return deliverDueMessagesWithReachability(receiverRobotId, round,
                                           &reachableSenderRobotIds);
}

std::vector<SharedObjectLowerMessage>
SharedObjectLowerMessageQueue::deliverDueMessagesWithReachability(
    std::size_t receiverRobotId, std::size_t round,
    const std::vector<std::size_t> *reachableSenderRobotIds) {
  std::sort(pending_.begin(), pending_.end(),
            [](const PendingAttempt &lhs, const PendingAttempt &rhs) {
              return std::tie(lhs.deliveryRound, lhs.attemptSequence,
                              lhs.messageSequence) <
                     std::tie(rhs.deliveryRound, rhs.attemptSequence,
                              rhs.messageSequence);
            });

  std::vector<SharedObjectLowerMessage> delivered;
  std::vector<PendingAttempt> remaining;
  remaining.reserve(pending_.size());
  for (const PendingAttempt &attempt : pending_) {
    const auto outstanding = outstanding_.find(attempt.messageSequence);
    if (outstanding == outstanding_.end()) {
      continue;
    }
    const bool due = attempt.deliveryRound <= round;
    const bool intendedReceiver =
        outstanding->second.key.receiverRobotId == receiverRobotId;
    const bool linkActive =
        reachableSenderRobotIds == nullptr ||
        std::find(reachableSenderRobotIds->begin(),
                  reachableSenderRobotIds->end(),
                  outstanding->second.key.senderRobotId) !=
            reachableSenderRobotIds->end();
    if (due && intendedReceiver && linkActive) {
      delivered.push_back(outstanding->second);
    } else {
      remaining.push_back(attempt);
    }
  }
  pending_ = std::move(remaining);
  return delivered;
}

bool SharedObjectLowerMessageQueue::acknowledge(
    std::size_t receiverRobotId, std::size_t messageSequence) {
  const auto outstanding = outstanding_.find(messageSequence);
  if (outstanding == outstanding_.end() ||
      outstanding->second.key.receiverRobotId != receiverRobotId) {
    return false;
  }
  outstanding_.erase(outstanding);
  pending_.erase(
      std::remove_if(pending_.begin(), pending_.end(),
                     [&](const PendingAttempt &attempt) {
                       return attempt.messageSequence == messageSequence;
                     }),
      pending_.end());
  return true;
}

std::vector<SharedObjectLowerMessage>
SharedObjectLowerMessageQueue::retryableMessagesBySender(
    std::size_t senderRobotId) const {
  std::vector<SharedObjectLowerMessage> messages;
  for (const auto &entry : outstanding_) {
    const SharedObjectLowerMessage &message = entry.second;
    if (message.key.senderRobotId != senderRobotId) {
      continue;
    }
    const bool alreadyPending =
        std::any_of(pending_.begin(), pending_.end(),
                    [&](const PendingAttempt &attempt) {
                      return attempt.messageSequence == entry.first;
                    });
    if (!alreadyPending) {
      messages.push_back(message);
    }
  }
  return messages;
}

std::size_t SharedObjectLowerMessageQueue::outstandingCountBySender(
    std::size_t senderRobotId) const {
  return static_cast<std::size_t>(
      std::count_if(outstanding_.begin(), outstanding_.end(),
                    [&](const auto &entry) {
                      return entry.second.key.senderRobotId == senderRobotId;
                    }));
}

std::size_t SharedObjectLowerMessageQueue::pendingDeliveryAttempts() const {
  return pending_.size();
}

std::size_t SharedObjectLowerMessageQueue::droppedAttempts() const {
  return droppedAttempts_;
}

bool SharedObjectLowerMessageQueue::empty() const {
  return outstanding_.empty() && pending_.empty();
}

bool SharedObjectLowerMessageQueue::hasOutstandingMessages() const {
  return !outstanding_.empty();
}

SharedObjectLowerMessageStatus classifySharedObjectLowerMessage(
    const SharedObjectLowerMessage &message,
    const SharedObjectLowerMessageReceiverState &receiverState) {
  const auto validResidual = [](double value) {
    return !std::isnan(value) && value >= 0.0;
  };
  if (message.target.rows() == 0 || message.target.cols() == 0 ||
      !message.target.allFinite() ||
      !std::isfinite(message.robustReliability) ||
      message.robustReliability < 0.0 ||
      message.robustReliability > 1.0 ||
      !validResidual(message.normalizedPrimalResidual) ||
      !validResidual(message.normalizedDualResidual)) {
    return SharedObjectLowerMessageStatus::InvalidPayload;
  }
  if (receiverState.initialized &&
      !(message.key == receiverState.streamKey)) {
    return SharedObjectLowerMessageStatus::KeyMismatch;
  }
  if (!receiverState.streamActive) {
    return SharedObjectLowerMessageStatus::InactiveStream;
  }
  if (message.commandEpoch != receiverState.expectedCommandEpoch) {
    return SharedObjectLowerMessageStatus::StaleEpoch;
  }
  if (message.receiverEndpointSequence !=
      receiverState.currentEndpointSequence) {
    return SharedObjectLowerMessageStatus::StaleEndpoint;
  }
  if (message.messageSequence <= receiverState.lastAcceptedMessageSequence) {
    return SharedObjectLowerMessageStatus::StaleSequence;
  }
  return SharedObjectLowerMessageStatus::Accepted;
}

bool sharedObjectLowerLocallyQuiescent(
    const LocalSharedObjectLowerQuiescenceState &state) {
  return state.localLowerCertificateConverged &&
         state.currentIncidentCertificates ==
             state.expectedIncidentCertificates &&
         state.senderOutstandingMessages == 0u;
}

SharedObjectSplitLatentOperator::SharedObjectSplitLatentOperator(
    std::size_t numEdges, std::size_t blockDimension,
    std::vector<std::vector<std::size_t>> incidentEdgesByRobot,
    LocalResponse localResponse)
    : numEdges_(numEdges),
      blockDimension_(blockDimension),
      incidentEdgesByRobot_(std::move(incidentEdgesByRobot)),
      localResponse_(std::move(localResponse)) {
  if (numEdges_ == 0 || blockDimension_ == 0 || !localResponse_) {
    throw std::invalid_argument(
        "split-latent operator requires edges, block dimension, and response");
  }
  std::vector<std::size_t> incidenceCounts(numEdges_, 0u);
  for (const auto &robotEdges : incidentEdgesByRobot_) {
    std::vector<std::size_t> sorted = robotEdges;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      throw std::invalid_argument(
          "a split-latent edge may occur only once per robot");
    }
    for (const std::size_t edge : robotEdges) {
      if (edge >= numEdges_) {
        throw std::invalid_argument("split-latent edge index is out of range");
      }
      ++incidenceCounts[edge];
    }
  }
  if (std::any_of(incidenceCounts.begin(), incidenceCounts.end(),
                  [](std::size_t count) { return count != 2u; })) {
    throw std::invalid_argument(
        "every split-latent edge must have exactly two endpoint robots");
  }
}

void SharedObjectSplitLatentOperator::validateEdgeMatrix(
    const Matrix &values, const char *name) const {
  if (values.rows() != static_cast<Eigen::Index>(blockDimension_) ||
      values.cols() != static_cast<Eigen::Index>(numEdges_) ||
      !values.allFinite()) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite block-by-edge matrix");
  }
}

Matrix SharedObjectSplitLatentOperator::applyResponse(
    const Matrix &scaledEdgeDirections) const {
  validateEdgeMatrix(scaledEdgeDirections, "split-latent direction");
  Matrix response = Matrix::Zero(static_cast<Eigen::Index>(blockDimension_),
                                 static_cast<Eigen::Index>(numEdges_));
  for (std::size_t robotId = 0;
       robotId < incidentEdgesByRobot_.size(); ++robotId) {
    const auto &robotEdges = incidentEdgesByRobot_[robotId];
    if (robotEdges.empty()) {
      continue;
    }
    Matrix localDirections(
        static_cast<Eigen::Index>(blockDimension_),
        static_cast<Eigen::Index>(robotEdges.size()));
    for (std::size_t local = 0; local < robotEdges.size(); ++local) {
      localDirections.col(static_cast<Eigen::Index>(local)) =
          scaledEdgeDirections.col(
              static_cast<Eigen::Index>(robotEdges[local]));
    }
    const Matrix localResponse = localResponse_(robotId, localDirections);
    if (localResponse.rows() != localDirections.rows() ||
        localResponse.cols() != localDirections.cols() ||
        !localResponse.allFinite()) {
      throw std::runtime_error(
          "split-latent local response must preserve finite dimensions");
    }
    for (std::size_t local = 0; local < robotEdges.size(); ++local) {
      response.col(static_cast<Eigen::Index>(robotEdges[local])) +=
          0.5 * localResponse.col(static_cast<Eigen::Index>(local));
    }
  }
  return response;
}

Matrix SharedObjectSplitLatentOperator::applySystem(
    const Matrix &scaledEdgeLatents) const {
  validateEdgeMatrix(scaledEdgeLatents, "split-latent value");
  return scaledEdgeLatents - applyResponse(scaledEdgeLatents);
}

Matrix SharedObjectSplitLatentOperator::richardsonStep(
    const Matrix &scaledEdgeLatents, const Matrix &rhs) const {
  validateEdgeMatrix(scaledEdgeLatents, "split-latent value");
  validateEdgeMatrix(rhs, "split-latent right-hand side");
  return rhs + applyResponse(scaledEdgeLatents);
}


}  // namespace DPGO
