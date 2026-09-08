#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace DPGO {

enum class ObjectCommunicationPolicy {
  Triggered,
  InnovationGated,
  BoundaryBudget,
  BoundaryPredictive,
  Full,
};

struct ObjectCommunicationInputs {
  bool baseTrigger{false};
  bool innovationAllows{false};
  bool budgetAllows{false};
};

struct DirectedObjectEligibility {
  std::size_t senderId{0};
  std::size_t receiverId{0};
  std::vector<bool> eligibleObjects;
};

ObjectCommunicationPolicy parseObjectCommunicationPolicy(
    const std::string &value);

std::string objectCommunicationPolicyName(ObjectCommunicationPolicy policy);

bool shouldSendObject(ObjectCommunicationPolicy policy,
                      const ObjectCommunicationInputs &inputs);

std::size_t expectedFullObjectBlocks(
    const std::vector<DirectedObjectEligibility> &eligibility);

}  // namespace DPGO
