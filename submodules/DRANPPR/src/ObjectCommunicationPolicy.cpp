#include <DPGO/ObjectCommunicationPolicy.h>

#include <algorithm>
#include <stdexcept>

namespace DPGO {

ObjectCommunicationPolicy parseObjectCommunicationPolicy(
    const std::string &value) {
  if (value == "triggered" || value == "default") {
    return ObjectCommunicationPolicy::Triggered;
  }
  if (value == "innovation_gated" || value == "innovation-gated") {
    return ObjectCommunicationPolicy::InnovationGated;
  }
  if (value == "boundary_budget" || value == "boundary-budget" ||
      value == "receiver_boundary_budget" ||
      value == "receiver-boundary-budget") {
    return ObjectCommunicationPolicy::BoundaryBudget;
  }
  if (value == "boundary_predictive" || value == "boundary-predictive" ||
      value == "predictive_boundary_budget" ||
      value == "predictive-boundary-budget") {
    return ObjectCommunicationPolicy::BoundaryPredictive;
  }
  if (value == "full" || value == "full_communication") {
    return ObjectCommunicationPolicy::Full;
  }
  throw std::invalid_argument("unsupported object communication policy: " +
                              value);
}

std::string objectCommunicationPolicyName(ObjectCommunicationPolicy policy) {
  switch (policy) {
    case ObjectCommunicationPolicy::Triggered:
      return "triggered";
    case ObjectCommunicationPolicy::InnovationGated:
      return "innovation_gated";
    case ObjectCommunicationPolicy::BoundaryBudget:
      return "boundary_budget";
    case ObjectCommunicationPolicy::BoundaryPredictive:
      return "boundary_predictive";
    case ObjectCommunicationPolicy::Full:
      return "full";
  }
  throw std::invalid_argument("invalid object communication policy");
}

bool shouldSendObject(ObjectCommunicationPolicy policy,
                      const ObjectCommunicationInputs &inputs) {
  switch (policy) {
    case ObjectCommunicationPolicy::Full:
      return true;
    case ObjectCommunicationPolicy::Triggered:
      return inputs.baseTrigger;
    case ObjectCommunicationPolicy::InnovationGated:
      return inputs.baseTrigger && inputs.innovationAllows;
    case ObjectCommunicationPolicy::BoundaryBudget:
    case ObjectCommunicationPolicy::BoundaryPredictive:
      return inputs.baseTrigger && inputs.budgetAllows;
  }
  throw std::invalid_argument("invalid object communication policy");
}

std::size_t expectedFullObjectBlocks(
    const std::vector<DirectedObjectEligibility> &eligibility) {
  std::size_t result = 0;
  for (const DirectedObjectEligibility &directed : eligibility) {
    result += static_cast<std::size_t>(std::count(
        directed.eligibleObjects.begin(), directed.eligibleObjects.end(),
        true));
  }
  return result;
}

}  // namespace DPGO
