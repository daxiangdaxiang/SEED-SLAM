#pragma once

#include <DPGO/NativeSchurModel.h>
#include <gtsam/geometry/Point2.h>
#include <string>

namespace DPGO {

struct NativeInterfaceEndpoint {
  std::uint32_t robot{0};
  std::shared_ptr<const NativeLocalResponse> response;
  std::map<gtsam::Key, gtsam::Point2> chartBases;
  // Optional complete injective local-key -> network-identity map for this epoch.
  // Empty retains the original identity mapping; responses keep stable local keys.
  std::map<gtsam::Key, gtsam::Key> interfaceKeys;
  // Optional complete Point2 displacement from each declared chart. The solver
  // returns corrections at these current points, using the immutable response.
  // Empty means zero. Shared identities require equal displacements as well.
  std::map<gtsam::Key, gtsam::Point2> chartDisplacements;
};

enum class NativeInterfaceMessageKind {
  Model, Direction, Product, Update, VerifyProduct, Reduce, Broadcast
};

struct NativeInterfaceMessage {
  std::uint32_t from, to, origin, destination;
  gtsam::Key key;
  std::uint64_t epoch, revision;
  NativeInterfaceMessageKind kind;
  gtsam::Vector payload;
  // Application header, not transport framing: 4 uint32 + 3 uint64.
  std::size_t bytes() const { return 40+sizeof(double)*payload.size(); }
};

struct NativeInterfaceOptions {
  std::size_t maxIterations{100};
  double relativeTolerance{1e-10};
  std::uint64_t epoch{0};
  bool rigidSubspace{false};
  bool fusedOneStep{false};
};

struct NativeInterfaceResult {
  std::string status;
  std::size_t iterations{0};
  std::size_t subspaceSteps{0}, subspaceRank{0};
  double relativeResidual{0.};
  std::map<std::uint32_t, gtsam::Vector> localSteps;
  std::map<std::uint32_t, std::size_t> operatorApplications;
  std::vector<NativeInterfaceMessage> messages;
  std::size_t payloadBytes{0}, applicationBytes{0};
};

struct NativeInterfaceReduction {
  gtsam::Vector sum;
  std::vector<NativeInterfaceMessage> messages;
  std::size_t payloadBytes{0}, applicationBytes{0};
};

// Small per-robot scalar vectors (cost, slope, etc.), using the same tree and
// logical packet ledger as PCG. No local response operator is applied.
NativeInterfaceReduction sumNativeInterfaceScalars(
    const std::vector<NativeInterfaceEndpoint>& endpoints,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& links,
    const std::map<std::uint32_t, gtsam::Vector>& values,
    const NativeInterfaceOptions& options = {});

// Deterministic routed-message simulation, not a socket/MPI transport. Interface
// identity roster, common Point2 charts and a connected topology are prerequisites;
// their discovery/initialization is not included in this solver's byte ledger.
// Each response is an immutable local snapshot, reused throughout this epoch.
NativeInterfaceResult solveNativeInterface(
    const std::vector<NativeInterfaceEndpoint>& endpoints,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& links,
    const NativeInterfaceOptions& options = {});

}  // namespace DPGO
