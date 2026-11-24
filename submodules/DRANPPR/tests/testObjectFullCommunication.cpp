#include <DPGO/ObjectCommunicationPolicy.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

std::vector<DPGO::DirectedObjectEligibility> tinyDirectedEligibility() {
  return {
      {0, 1, {true, true}},
      {1, 0, {true, true}},
  };
}

}  // namespace

TEST(ObjectFullCommunication, SendsEveryDirectedSharedBlockEveryRound) {
  const DPGO::ObjectCommunicationInputs suppressed{
      false, false, false};
  EXPECT_TRUE(DPGO::shouldSendObject(
      DPGO::ObjectCommunicationPolicy::Full, suppressed));
  EXPECT_EQ(DPGO::expectedFullObjectBlocks(tinyDirectedEligibility()), 4u);

  for (unsigned round = 0; round < 3; ++round) {
    static_cast<void>(round);
    EXPECT_TRUE(DPGO::shouldSendObject(
        DPGO::ObjectCommunicationPolicy::Full, suppressed));
    EXPECT_EQ(DPGO::expectedFullObjectBlocks(tinyDirectedEligibility()), 4u);
  }
}

TEST(ObjectFullCommunication, ParsesCanonicalFullAliases) {
  EXPECT_EQ(DPGO::parseObjectCommunicationPolicy("full"),
            DPGO::ObjectCommunicationPolicy::Full);
  EXPECT_EQ(DPGO::parseObjectCommunicationPolicy("full_communication"),
            DPGO::ObjectCommunicationPolicy::Full);
  EXPECT_EQ(DPGO::objectCommunicationPolicyName(
                DPGO::ObjectCommunicationPolicy::Full),
            "full");
  EXPECT_THROW(DPGO::parseObjectCommunicationPolicy("all"),
               std::invalid_argument);
}

TEST(ObjectFullCommunication, RetainsSuppressionForNonFullPolicies) {
  const DPGO::ObjectCommunicationInputs noTrigger{false, true, true};
  const DPGO::ObjectCommunicationInputs noInnovation{true, false, true};
  const DPGO::ObjectCommunicationInputs noBudget{true, true, false};
  EXPECT_FALSE(DPGO::shouldSendObject(
      DPGO::ObjectCommunicationPolicy::Triggered, noTrigger));
  EXPECT_FALSE(DPGO::shouldSendObject(
      DPGO::ObjectCommunicationPolicy::InnovationGated, noInnovation));
  EXPECT_FALSE(DPGO::shouldSendObject(
      DPGO::ObjectCommunicationPolicy::BoundaryBudget, noBudget));
  EXPECT_FALSE(DPGO::shouldSendObject(
      DPGO::ObjectCommunicationPolicy::BoundaryPredictive, noBudget));
}

TEST(ObjectFullCommunication, CountsOnlyEligibleDirectedBlocks) {
  std::vector<DPGO::DirectedObjectEligibility> eligibility = {
      {0, 1, {true, false, true}},
      {1, 0, {true, true, false}},
      {1, 2, {false, false, false}},
  };
  EXPECT_EQ(DPGO::expectedFullObjectBlocks(eligibility), 4u);
}
