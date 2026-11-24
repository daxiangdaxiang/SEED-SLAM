#include <DPGO/DPGO_utils.h>
#include <DPGO/RIFTIF.h>
#include <DPGO/TEDCCI.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

Matrix rotation2(double theta) {
  Matrix R(2, 2);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  R << c, -s, s, c;
  return R;
}

Matrix rotation3(double yaw, double pitch, double roll) {
  const Eigen::Matrix3d R =
      (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  return R;
}

Matrix deterministicMatrix(int rows, int cols, double phase) {
  Matrix A(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      A(r, c) =
          std::sin(phase + 0.11 * static_cast<double>((r + 1) * (c + 2))) +
          0.2 * std::cos(0.17 * static_cast<double>(r + c + 1));
    }
  }
  return A;
}

Vector deterministicVector(int rows, double phase) {
  Vector b(rows);
  for (int r = 0; r < rows; ++r) {
    b(r) = std::cos(phase + 0.23 * static_cast<double>(r + 1)) +
           0.1 * std::sin(0.31 * static_cast<double>(r + 2));
  }
  return b;
}

CondensedFactor makeCondensed(int robot, const std::vector<PoseKey> &keys,
                              int rows, int blockDim, double phase) {
  CondensedFactor factor;
  factor.owner_robot_id = robot;
  factor.factor_type = FactorType::TRANSLATION;
  factor.boundary_keys = keys;
  factor.Abar =
      deterministicMatrix(rows, static_cast<int>(keys.size()) * blockDim,
                          phase);
  factor.bbar = deterministicVector(rows, phase);
  return factor;
}

LinearFactorBlock makeCross(const std::vector<PoseKey> &keys, int rows,
                            int blockDim, double phase) {
  LinearFactorBlock factor;
  factor.keys = keys;
  factor.A =
      deterministicMatrix(rows, static_cast<int>(keys.size()) * blockDim,
                          phase);
  factor.b = deterministicVector(rows, phase);
  return factor;
}

Vector vectorizeForTest(const Matrix &M) {
  return Eigen::Map<const Vector>(M.data(), M.rows() * M.cols());
}

Vector flattenMultiRhsResidualForVectorizedOrder(const Matrix &residual,
                                                 int dimension) {
  Vector flattened(residual.rows() * dimension);
  for (int rhs = 0; rhs < dimension; ++rhs) {
    for (int row = 0; row < residual.rows(); ++row) {
      flattened(rhs + row * dimension) = residual(row, rhs);
    }
  }
  return flattened;
}

std::size_t estimatedTreeBytes(const InterfaceCliqueTree &tree) {
  std::size_t bytes = 0;
  for (const auto &edge : tree.edges) {
    bytes += 2u * edge.estimated_message_bytes;
  }
  return bytes;
}

RIFTNetworkMessage makeNetworkMessage(int src, int dst, std::uint64_t seq,
                                      double sendTimeMs,
                                      std::size_t payloadBytes = 16) {
  RIFTNetworkMessage message;
  message.src_robot = src;
  message.dst_robot = dst;
  message.seq = seq;
  message.payload_bytes = payloadBytes;
  message.send_time_ms = sendTimeMs;
  message.rift_message.header.src = src;
  message.rift_message.header.dst = dst;
  message.rift_message.header.seq = seq;
  return message;
}

InterfaceCliqueTree makeChainMessageTree(int numCliques) {
  InterfaceCliqueTree tree;
  tree.cliques.resize(numCliques);
  for (int i = 0; i < numCliques; ++i) {
    tree.cliques[i].id = i;
    if (i > 0) {
      tree.cliques[i].neighbors.push_back(i - 1);
      InterfaceCliqueTreeEdge edge;
      edge.a = i - 1;
      edge.b = i;
      edge.estimated_message_bytes = 32;
      tree.edges.push_back(edge);
    }
    if (i + 1 < numCliques) {
      tree.cliques[i].neighbors.push_back(i + 1);
    }
  }
  return tree;
}

std::set<DirectedCliqueEdge> directedEdgeSet(
    const std::vector<DirectedCliqueEdge> &edges) {
  return std::set<DirectedCliqueEdge>(edges.begin(), edges.end());
}

void cacheAllDirectedMessages(const InterfaceCliqueTree &tree,
                              RIFTDirtyMessageTracker *tracker) {
  for (const auto &clique : tree.cliques) {
    for (const RIFTCliqueId neighbor : clique.neighbors) {
      tracker->CacheMessage(DirectedCliqueEdge{clique.id, neighbor});
    }
  }
}

InterfaceFactor makeScalarInterfaceFactor(RIFTFactorId id,
                                          const std::vector<PoseKey> &scope,
                                          int rows, double phase,
                                          int ownerRobot,
                                          std::uint64_t epoch = 0) {
  InterfaceFactor factor;
  factor.id = id;
  factor.stage = FactorType::TRANSLATION;
  factor.scope = scope;
  factor.A =
      deterministicMatrix(rows, static_cast<int>(scope.size()), phase);
  factor.B.resize(rows, 1);
  factor.B.col(0) = deterministicVector(rows, phase + 0.37);
  factor.owner_robot = ownerRobot;
  factor.graph_epoch = epoch;
  return factor;
}

InterfaceCliqueTree makeIncrementalRiftChainTree(bool includeLoopFactor) {
  const PoseKey a{0, 0};
  const PoseKey b{0, 1};
  const PoseKey c{1, 2};
  const PoseKey d{1, 3};

  InterfaceCliqueTree tree;
  tree.cliques.resize(3);
  tree.cliques[0].id = 0;
  tree.cliques[0].host_robot = 0;
  tree.cliques[0].variables = {a, b};
  tree.cliques[0].assigned_factors = {0, 1};
  tree.cliques[0].neighbors = {1};

  tree.cliques[1].id = 1;
  tree.cliques[1].host_robot = 1;
  tree.cliques[1].variables = {b, c};
  tree.cliques[1].assigned_factors = includeLoopFactor
                                         ? std::vector<RIFTFactorId>{2, 5}
                                         : std::vector<RIFTFactorId>{2};
  tree.cliques[1].neighbors = {0, 2};

  tree.cliques[2].id = 2;
  tree.cliques[2].host_robot = 1;
  tree.cliques[2].variables = {c, d};
  tree.cliques[2].assigned_factors = {3, 4};
  tree.cliques[2].neighbors = {1};

  InterfaceCliqueTreeEdge left;
  left.a = 0;
  left.b = 1;
  left.separator = {b};
  left.estimated_message_bytes = 32;
  InterfaceCliqueTreeEdge right;
  right.a = 1;
  right.b = 2;
  right.separator = {c};
  right.estimated_message_bytes = 32;
  tree.edges = {left, right};
  return tree;
}

std::vector<InterfaceFactor> makeIncrementalRiftFactors(bool includeLoopFactor) {
  const PoseKey a{0, 0};
  const PoseKey b{0, 1};
  const PoseKey c{1, 2};
  const PoseKey d{1, 3};
  std::vector<InterfaceFactor> factors = {
      makeScalarInterfaceFactor(0, {a}, 2, 0.11, 0),
      makeScalarInterfaceFactor(1, {a, b}, 4, 0.23, 0),
      makeScalarInterfaceFactor(2, {b, c}, 4, 0.41, 1),
      makeScalarInterfaceFactor(3, {c, d}, 4, 0.67, 1),
      makeScalarInterfaceFactor(4, {d}, 2, 0.89, 1)};
  if (includeLoopFactor) {
    factors.push_back(makeScalarInterfaceFactor(5, {b, c}, 3, 1.13, 1,
                                                /*epoch=*/7));
  }
  return factors;
}

std::vector<InterfaceFactor> makeAsyncSchurFactors(bool includeThirdRobot) {
  const PoseKey a{0, 0};
  const PoseKey b{1, 1};
  const PoseKey c{2, 2};
  std::vector<InterfaceFactor> factors = {
      makeScalarInterfaceFactor(0, {a}, 1, 0.0, 0),
      makeScalarInterfaceFactor(1, {b}, 1, 0.0, 1),
      makeScalarInterfaceFactor(2, {a, b}, 1, 0.0, 0)};
  factors[0].A.resize(1, 1);
  factors[0].A << 2.0;
  factors[0].B.resize(1, 1);
  factors[0].B << 2.0;
  factors[1].A.resize(1, 1);
  factors[1].A << 2.0;
  factors[1].B.resize(1, 1);
  factors[1].B << 4.0;
  factors[2].A.resize(1, 2);
  factors[2].A << 0.25, -0.25;
  factors[2].B.resize(1, 1);
  factors[2].B << -0.25;
  if (includeThirdRobot) {
    InterfaceFactor f3 = makeScalarInterfaceFactor(3, {c}, 1, 0.0, 2);
    f3.A.resize(1, 1);
    f3.A << 2.0;
    f3.B.resize(1, 1);
    f3.B << 6.0;
    InterfaceFactor f4 = makeScalarInterfaceFactor(4, {b, c}, 1, 0.0, 1);
    f4.A.resize(1, 2);
    f4.A << 0.20, -0.20;
    f4.B.resize(1, 1);
    f4.B << 0.80;
    factors.push_back(f3);
    factors.push_back(f4);
  }
  return factors;
}

Matrix explicitNormalMatrixForTest(const InterfaceProblem &problem) {
  const int cols =
      static_cast<int>(problem.variables.size()) * problem.block_dim;
  Matrix H = Matrix::Zero(cols, cols);
  for (const auto &factor : problem.factors) {
    Matrix expanded = Matrix::Zero(factor.A.rows(), cols);
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int dst = static_cast<int>(
                          std::distance(problem.variables.begin(),
                                        std::find(problem.variables.begin(),
                                                  problem.variables.end(),
                                                  factor.scope[k]))) *
                      problem.block_dim;
      const int src = static_cast<int>(k) * problem.block_dim;
      expanded.block(0, dst, factor.A.rows(), problem.block_dim) =
          factor.A.block(0, src, factor.A.rows(), problem.block_dim);
    }
    H.noalias() += expanded.transpose() * expanded;
  }
  return H;
}

Matrix explicitNormalRhsForTest(const InterfaceProblem &problem) {
  const int rows =
      static_cast<int>(problem.variables.size()) * problem.block_dim;
  Matrix rhs = Matrix::Zero(rows, problem.rhs_dim);
  for (const auto &factor : problem.factors) {
    const Matrix local = factor.A.transpose() * factor.B;
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int dst = static_cast<int>(
                          std::distance(problem.variables.begin(),
                                        std::find(problem.variables.begin(),
                                                  problem.variables.end(),
                                                  factor.scope[k]))) *
                      problem.block_dim;
      const int src = static_cast<int>(k) * problem.block_dim;
      rhs.block(dst, 0, problem.block_dim, problem.rhs_dim) +=
          local.block(src, 0, problem.block_dim, problem.rhs_dim);
    }
  }
  return rhs;
}

RelativeSEMeasurement makeMeasurement(size_t i, size_t j,
                                      const std::vector<int> &owner,
                                      const Matrix &Ri, const Vector &ti,
                                      const Matrix &Rj, const Vector &tj,
                                      double kappa, double tau) {
  return RelativeSEMeasurement(owner.at(i), owner.at(j), i, j,
                               Ri.transpose() * Rj,
                               Ri.transpose() * (tj - ti), kappa, tau);
}

std::vector<int> contiguousOwner(size_t numPoses, int robots) {
  std::vector<int> owner(numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    owner[pose] = std::min<int>(
        static_cast<int>((pose * static_cast<size_t>(robots)) / numPoses),
        robots - 1);
  }
  return owner;
}

std::vector<TEDCCIPartition> makePartitions(
    size_t numPoses, int robots,
    const std::vector<RelativeSEMeasurement> &measurements,
    const TEDCCIParams &params) {
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  std::vector<RelativeSEMeasurement> local;
  std::vector<RelativeSEMeasurement> shared;
  for (const auto &measurement : measurements) {
    if (measurement.r1 == measurement.r2) {
      local.push_back(measurement);
    } else {
      shared.push_back(measurement);
    }
  }

  std::vector<TEDCCIPartition> partitions;
  for (int robot = 0; robot < robots; ++robot) {
    std::vector<PoseKey> localPoses;
    for (size_t pose = 0; pose < numPoses; ++pose) {
      if (owner[pose] == robot) {
        localPoses.push_back(PoseKey{robot, static_cast<int>(pose)});
      }
    }
    std::vector<RelativeSEMeasurement> robotLocal;
    for (const auto &measurement : local) {
      if (static_cast<int>(measurement.r1) == robot) {
        robotLocal.push_back(measurement);
      }
    }
    partitions.push_back(BoundarySelector::SelectBoundaryVariables(
        robot, localPoses, robotLocal, shared, params));
  }
  return partitions;
}

void expectPoseEstimateClose(const Matrix &actual, const Matrix &expected,
                             size_t d, size_t numPoses,
                             double tolerance = 1e-8) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(
        maxDiff,
        (actual.block(0, pose * (d + 1), d, d + 1) -
         expected.block(0, pose * (d + 1), d, d + 1))
            .norm());
  }
  EXPECT_LT(maxDiff, tolerance);
}

}  // namespace

TEST(testDPGO, RIFTIFBuildsInterfaceProblemFromTEDFactors) {
  const int blockDim = 2;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const auto c0 = makeCondensed(0, {a, b}, 5, blockDim, 0.1);
  const auto c1 = makeCondensed(1, {b, c}, 6, blockDim, 0.7);
  const auto cross = makeCross({a, c}, 4, blockDim, 1.3);

  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {c0, c1}, {cross},
          /*use_rotation_multi_rhs=*/false);

  ASSERT_EQ(problem.factors.size(), 3u);
  EXPECT_EQ(problem.variables.size(), 3u);
  EXPECT_EQ(problem.block_dim, blockDim);
  EXPECT_EQ(problem.rhs_dim, 1);
  EXPECT_EQ(problem.factors[0].owner_robot, 0);
  EXPECT_EQ(problem.factors[2].A.rows(), cross.A.rows());
}

TEST(testDPGO, RIFTIFRotationMultiRHSFactorMatchesVectorizedResiduals) {
  for (const int d : {2, 3}) {
    TEDCCIParams params;
    params.anchor_pose_id = 99;
    params.use_measurement_weight = false;
    const std::vector<int> owner = {0, 0, 1};
    const Matrix R0 = Matrix::Identity(d, d);
    const Matrix R1 = d == 2 ? rotation2(0.31) : rotation3(0.31, -0.08, 0.14);
    const Matrix R2 = d == 2 ? rotation2(0.72) : rotation3(0.72, 0.16, -0.22);
    Vector t0 = Vector::Zero(d);
    Vector t1 = Vector::Zero(d);
    Vector t2 = Vector::Zero(d);
    for (int i = 0; i < d; ++i) {
      t1(i) = 0.2 + 0.1 * i;
      t2(i) = 0.6 - 0.05 * i;
    }
    const RelativeSEMeasurement measurement =
        makeMeasurement(1, 2, owner, R1, t1, R2, t2, 2.0, 3.0);
    const LinearFactorBlock vectorized =
        CCIFactorBuilder::BuildRotationFactors(d, {measurement}, params)
            .front();
    const InterfaceFactor multi =
        BuildRotationMultiRHSFactorFromVectorized(
            0, vectorized.keys, vectorized.A, vectorized.b, d, 0);

    Vector xVec(static_cast<int>(vectorized.keys.size()) * d * d);
    Matrix yMulti(static_cast<int>(vectorized.keys.size()) * d, d);
    for (std::size_t keyIndex = 0; keyIndex < vectorized.keys.size();
         ++keyIndex) {
      const Matrix X = deterministicMatrix(d, d, 0.25 + keyIndex);
      xVec.segment(static_cast<int>(keyIndex) * d * d, d * d) =
          vectorizeForTest(X);
      yMulti.block(static_cast<int>(keyIndex) * d, 0, d, d) =
          X.transpose();
    }

    const Vector residualVec = vectorized.A * xVec - vectorized.b;
    const Matrix residualMulti = multi.A * yMulti - multi.B;
    const Vector residualMultiVec =
        flattenMultiRhsResidualForVectorizedOrder(residualMulti, d);
    EXPECT_LT((residualVec - residualMultiVec).norm(), 1e-12);
    EXPECT_EQ(multi.A.cols(),
              static_cast<int>(vectorized.keys.size()) * d);
    EXPECT_EQ(multi.B.cols(), d);
  }
}

TEST(testDPGO, RIFTIFRotationMultiRHSRejectsOffRowCoupling) {
  const int d = 3;
  TEDCCIParams params;
  params.anchor_pose_id = 99;
  params.use_measurement_weight = false;
  const std::vector<int> owner = {0, 1};
  Vector t0 = Vector::Zero(d);
  Vector t1 = Vector::Zero(d);
  const RelativeSEMeasurement measurement =
      makeMeasurement(0, 1, owner, Matrix::Identity(d, d), t0,
                      rotation3(0.2, 0.1, -0.1), t1, 2.0, 3.0);
  LinearFactorBlock vectorized =
      CCIFactorBuilder::BuildRotationFactors(d, {measurement}, params)
          .front();
  vectorized.A(0, 1) = 0.5;
  EXPECT_THROW(BuildRotationMultiRHSFactorFromVectorized(
                   0, vectorized.keys, vectorized.A, vectorized.b, d, 0),
               std::invalid_argument);
}

TEST(testDPGO, RIFTIFRotationMultiRHSPayloadEstimateIsSmallerIn3D) {
  const int d = 3;
  TEDCCIParams params;
  params.anchor_pose_id = 99;
  params.use_measurement_weight = false;
  const std::vector<int> owner = {0, 0, 1, 1};
  std::vector<Matrix> rotations = {
      Matrix::Identity(d, d), rotation3(0.2, 0.1, -0.1),
      rotation3(0.45, -0.2, 0.05), rotation3(0.7, 0.15, 0.2)};
  std::vector<Vector> translations(4, Vector::Zero(d));
  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < rotations.size(); ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0, 3.0));
  }
  measurements.push_back(makeMeasurement(0, 2, owner, rotations[0],
                                         translations[0], rotations[2],
                                         translations[2], 2.0, 3.0));
  const auto vectorFactors =
      CCIFactorBuilder::BuildRotationFactors(d, measurements, params);

  const InterfaceProblem vectorProblem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::ROTATION, d, d * d, {}, vectorFactors,
          /*use_rotation_multi_rhs=*/false);
  std::vector<InterfaceFactor> multiFactors;
  RIFTFactorId nextId = 0;
  for (const auto &factor : vectorFactors) {
    multiFactors.push_back(BuildRotationMultiRHSFactorFromVectorized(
        nextId++, factor.keys, factor.A, factor.b, d,
        factor.keys.empty() ? -1 : factor.keys.front().robot_id));
  }
  const InterfaceProblem multiProblem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, d, d, d, multiFactors);
  RIFTParams riftParams;
  const InterfaceCliqueTree vectorTree =
      InterfaceCliqueTreeBuilder::Build(vectorProblem, riftParams);
  const InterfaceCliqueTree multiTree =
      InterfaceCliqueTreeBuilder::Build(multiProblem, riftParams);

  EXPECT_LT(estimatedTreeBytes(multiTree), estimatedTreeBytes(vectorTree));
}

TEST(testDPGO, RIFTIFMultiRHSCondensationAvoidsDenseChainQR) {
  const int d = 3;
  const int interiorBlocks = 400;
  const PoseKey leftBoundary{0, 0};
  const PoseKey rightBoundary{1, interiorBlocks + 1};
  std::vector<PoseKey> interior;
  interior.reserve(interiorBlocks);
  for (int i = 0; i < interiorBlocks; ++i) {
    interior.push_back(PoseKey{0, i + 1});
  }

  std::vector<InterfaceFactor> factors;
  factors.reserve(interiorBlocks + 1);
  for (int edge = 0; edge <= interiorBlocks; ++edge) {
    const PoseKey a = edge == 0 ? leftBoundary : interior[edge - 1];
    const PoseKey b = edge == interiorBlocks ? rightBoundary : interior[edge];
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(edge);
    factor.stage = FactorType::ROTATION;
    factor.scope = {a, b};
    factor.A = Matrix::Zero(d, 2 * d);
    factor.A.block(0, 0, d, d).setIdentity();
    factor.A.block(0, d, d, d) = -Matrix::Identity(d, d);
    factor.B = deterministicMatrix(d, d, 0.01 * static_cast<double>(edge));
    factor.owner_robot = 0;
    factors.push_back(std::move(factor));
  }

  const auto start = std::chrono::steady_clock::now();
  const InterfaceFactor condensed = CondenseMultiRHSFactors(
      99, FactorType::ROTATION, d, factors, interior,
      {leftBoundary, rightBoundary}, 0);
  const auto finish = std::chrono::steady_clock::now();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(finish - start).count();

  EXPECT_EQ(condensed.scope.size(), 2u);
  EXPECT_EQ(condensed.A.cols(), 2 * d);
  EXPECT_EQ(condensed.B.cols(), d);
  EXPECT_EQ(condensed.A.rows(), d);
  EXPECT_TRUE(condensed.A.allFinite());
  EXPECT_TRUE(condensed.B.allFinite());
  EXPECT_LT(elapsedMs, 120.0);
}

TEST(testDPGO, RIFTIFMultiRHSCondensationBackSubstitutesInteriorBlocks) {
  const int d = 3;
  const PoseKey leftBoundary{0, 0};
  const PoseKey interior0{0, 1};
  const PoseKey interior1{0, 2};
  const PoseKey rightBoundary{1, 3};
  const std::vector<PoseKey> interior = {interior0, interior1};
  const std::vector<PoseKey> boundary = {leftBoundary, rightBoundary};

  std::vector<InterfaceFactor> factors;
  const std::vector<std::pair<PoseKey, PoseKey>> edges = {
      {leftBoundary, interior0}, {interior0, interior1},
      {interior1, rightBoundary}};
  for (std::size_t edge = 0; edge < edges.size(); ++edge) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(edge);
    factor.stage = FactorType::ROTATION;
    factor.scope = {edges[edge].first, edges[edge].second};
    factor.A = Matrix::Zero(d, 2 * d);
    factor.A.block(0, 0, d, d) =
        (1.0 + 0.1 * static_cast<double>(edge)) * Matrix::Identity(d, d);
    factor.A.block(0, d, d, d) =
        (-0.7 - 0.05 * static_cast<double>(edge)) *
        Matrix::Identity(d, d);
    factor.B = deterministicMatrix(d, d, 0.3 + 0.2 * edge);
    factor.owner_robot = 0;
    factors.push_back(std::move(factor));
  }

  MultiRHSBackSubstitutionCache cache;
  (void)CondenseMultiRHSFactors(7, FactorType::ROTATION, d, factors,
                                interior, boundary, 0, 0, &cache);

  Matrix boundarySolution(2 * d, d);
  boundarySolution << deterministicMatrix(d, d, 1.1),
      deterministicMatrix(d, d, 1.7);
  const Matrix recovered = BackSubstituteMultiRHS(cache, boundarySolution);

  Matrix AI = Matrix::Zero(static_cast<int>(factors.size()) * d,
                           static_cast<int>(interior.size()) * d);
  Matrix AB = Matrix::Zero(static_cast<int>(factors.size()) * d,
                           static_cast<int>(boundary.size()) * d);
  Matrix B = Matrix::Zero(static_cast<int>(factors.size()) * d, d);
  for (std::size_t factorIndex = 0; factorIndex < factors.size();
       ++factorIndex) {
    const auto &factor = factors[factorIndex];
    const int rowOffset = static_cast<int>(factorIndex) * d;
    for (std::size_t keyIndex = 0; keyIndex < factor.scope.size();
         ++keyIndex) {
      const auto interiorIt =
          std::find(interior.begin(), interior.end(), factor.scope[keyIndex]);
      const auto boundaryIt =
          std::find(boundary.begin(), boundary.end(), factor.scope[keyIndex]);
      const int src = static_cast<int>(keyIndex) * d;
      if (interiorIt != interior.end()) {
        const int dst =
            static_cast<int>(std::distance(interior.begin(), interiorIt)) * d;
        AI.block(rowOffset, dst, d, d) = factor.A.block(0, src, d, d);
      } else {
        ASSERT_NE(boundaryIt, boundary.end());
        const int dst =
            static_cast<int>(std::distance(boundary.begin(), boundaryIt)) * d;
        AB.block(rowOffset, dst, d, d) = factor.A.block(0, src, d, d);
      }
    }
    B.block(rowOffset, 0, d, d) = factor.B;
  }
  const Matrix expected =
      AI.colPivHouseholderQr().solve(B - AB * boundarySolution);
  EXPECT_LT((recovered - expected).norm(), 1e-10);
}

TEST(testDPGO, RIFTIFCliqueTreeCoversFactorsAndRunningIntersection) {
  const int blockDim = 1;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const PoseKey d{3, 4};
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim,
          {makeCondensed(0, {a, b}, 3, blockDim, 0.1),
           makeCondensed(1, {b, c}, 3, blockDim, 0.3),
           makeCondensed(2, {c, d}, 3, blockDim, 0.5)},
          {makeCross({a, d}, 2, blockDim, 0.8)},
          /*use_rotation_multi_rhs=*/false);

  RIFTParams params;
  const InterfaceCliqueTree tree = InterfaceCliqueTreeBuilder::Build(
      problem, params);

  EXPECT_FALSE(tree.cliques.empty());
  EXPECT_TRUE(InterfaceCliqueTreeBuilder::VerifyRunningIntersection(tree));
  for (const auto &clique : tree.cliques) {
    EXPECT_GE(clique.host_robot, 0);
    std::map<int, int> counts;
    for (const auto &key : clique.variables) {
      ++counts[key.robot_id];
    }
    ASSERT_FALSE(counts.empty());
    int expectedHost = counts.begin()->first;
    int expectedCount = counts.begin()->second;
    for (const auto &entry : counts) {
      if (entry.second > expectedCount ||
          (entry.second == expectedCount && entry.first < expectedHost)) {
        expectedHost = entry.first;
        expectedCount = entry.second;
      }
    }
    EXPECT_EQ(clique.host_robot, expectedHost);
  }
  for (const auto &edge : tree.edges) {
    ASSERT_GE(edge.a, 0);
    ASSERT_GE(edge.b, 0);
    ASSERT_LT(static_cast<std::size_t>(edge.a), tree.cliques.size());
    ASSERT_LT(static_cast<std::size_t>(edge.b), tree.cliques.size());
    EXPECT_GE(tree.cliques[edge.a].host_robot, 0);
    EXPECT_GE(tree.cliques[edge.b].host_robot, 0);
    EXPECT_GT(edge.estimated_message_bytes, 0u);
  }
  for (const auto &factor : problem.factors) {
    bool covered = false;
    for (const auto &clique : tree.cliques) {
      bool contains = true;
      for (const auto &key : factor.scope) {
        contains = contains &&
                   std::binary_search(clique.variables.begin(),
                                      clique.variables.end(), key);
      }
      covered = covered || contains;
    }
    EXPECT_TRUE(covered);
  }
}

TEST(testDPGO, RIFTIFRootlessSchedulerCompletesAllDirectedMessages) {
  InterfaceCliqueTree tree;
  tree.cliques.resize(3);
  for (int i = 0; i < 3; ++i) {
    tree.cliques[i].id = i;
  }
  tree.cliques[0].neighbors = {1};
  tree.cliques[1].neighbors = {0, 2};
  tree.cliques[2].neighbors = {1};

  RIFTRootlessScheduler scheduler;
  scheduler.Initialize(tree);
  while (!scheduler.AllDirectedMessagesComplete()) {
    const auto ready = scheduler.ReadyOutgoingMessages();
    ASSERT_FALSE(ready.empty());
    for (const auto &edge : ready) {
      RIFTMessage msg;
      msg.header.src = edge.src;
      msg.header.dst = edge.dst;
      scheduler.MarkMessageSent(edge.src, edge.dst);
      scheduler.OnMessageReceived(msg);
    }
  }
  EXPECT_TRUE(scheduler.CliqueBeliefReady(0));
  EXPECT_TRUE(scheduler.CliqueBeliefReady(1));
  EXPECT_TRUE(scheduler.CliqueBeliefReady(2));
}

TEST(testDPGO, RIFTIFDirtyTrackerInvalidatesDirectedDependenciesOnly) {
  {
    const InterfaceCliqueTree tree = makeChainMessageTree(3);
    RIFTDirtyMessageTracker tracker(tree);
    cacheAllDirectedMessages(tree, &tracker);
    tracker.MarkCliqueDirty(1);
    const std::set<DirectedCliqueEdge> expected = {
        DirectedCliqueEdge{1, 0}, DirectedCliqueEdge{1, 2}};
    EXPECT_EQ(directedEdgeSet(tracker.InvalidatedMessages()), expected);
    EXPECT_EQ(tracker.FullDirectedMessageCount(), 4);
    EXPECT_EQ(static_cast<int>(tracker.InvalidatedMessages().size()), 2);
    EXPECT_EQ(tracker.ReusableCachedMessageCount(), 2);
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{0, 1}));
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{2, 1}));
  }

  {
    const InterfaceCliqueTree tree = makeChainMessageTree(4);
    RIFTDirtyMessageTracker tracker(tree);
    cacheAllDirectedMessages(tree, &tracker);
    tracker.MarkCliqueDirty(0);
    const std::set<DirectedCliqueEdge> expected = {
        DirectedCliqueEdge{0, 1}, DirectedCliqueEdge{1, 2},
        DirectedCliqueEdge{2, 3}};
    EXPECT_EQ(directedEdgeSet(tracker.InvalidatedMessages()), expected);
    EXPECT_EQ(tracker.FullDirectedMessageCount(), 6);
    EXPECT_EQ(tracker.ReusableCachedMessageCount(), 3);
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{1, 0}));
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{2, 1}));
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{3, 2}));
  }
}

TEST(testDPGO, RIFTIFDirtyTrackerMergesMultipleDirtyWaves) {
  {
    const InterfaceCliqueTree tree = makeChainMessageTree(3);
    RIFTDirtyMessageTracker tracker(tree);
    cacheAllDirectedMessages(tree, &tracker);
    tracker.MarkCliqueDirty(0);
    tracker.MarkCliqueDirty(2);
    EXPECT_EQ(static_cast<int>(tracker.InvalidatedMessages().size()),
              tracker.FullDirectedMessageCount());
    EXPECT_EQ(tracker.ReusableCachedMessageCount(), 0);
  }

  {
    const InterfaceCliqueTree tree = makeChainMessageTree(3);
    RIFTDirtyMessageTracker tracker(tree);
    cacheAllDirectedMessages(tree, &tracker);
    tracker.MarkCliqueDirty(0);
    tracker.MarkCliqueDirty(1);
    const std::set<DirectedCliqueEdge> expected = {
        DirectedCliqueEdge{0, 1}, DirectedCliqueEdge{1, 0},
        DirectedCliqueEdge{1, 2}};
    EXPECT_EQ(directedEdgeSet(tracker.InvalidatedMessages()), expected);
    EXPECT_EQ(tracker.ReusableCachedMessageCount(), 1);
    EXPECT_FALSE(tracker.IsMessageInvalidated(DirectedCliqueEdge{2, 1}));
  }
}

TEST(testDPGO, RIFTIFIncrementalDirtyMessagesReuseCacheAndMatchFullRebuild) {
  const InterfaceProblem baseProblem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/false));
  const InterfaceCliqueTree baseTree =
      makeIncrementalRiftChainTree(/*includeLoopFactor=*/false);
  ASSERT_TRUE(InterfaceCliqueTreeBuilder::VerifyRunningIntersection(baseTree));

  RIFTParams params;
  std::map<DirectedCliqueEdge, RIFTMessage> baseCache;
  RIFTStats baseStats;
  RIFTExactSolver::SolveMatrix(baseProblem, baseTree, params, &baseStats,
                               nullptr, nullptr, nullptr, &baseCache);
  ASSERT_EQ(baseStats.directed_messages_sent, 4);
  ASSERT_EQ(static_cast<int>(baseCache.size()), 4);

  const InterfaceProblem updatedProblem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const InterfaceCliqueTree updatedTree =
      makeIncrementalRiftChainTree(/*includeLoopFactor=*/true);
  ASSERT_TRUE(InterfaceCliqueTreeBuilder::VerifyRunningIntersection(
      updatedTree));

  RIFTStats fullStats;
  std::map<DirectedCliqueEdge, RIFTMessage> fullCache;
  const Matrix fullRebuild =
      RIFTExactSolver::SolveMatrix(updatedProblem, updatedTree, params,
                                   &fullStats, nullptr, nullptr, nullptr,
                                   &fullCache);
  ASSERT_EQ(fullStats.directed_messages_sent, 4);

  RIFTDirtyMessageTracker tracker(updatedTree);
  cacheAllDirectedMessages(updatedTree, &tracker);
  tracker.MarkFactorUpdated(5);
  ASSERT_EQ(static_cast<int>(tracker.InvalidatedMessages().size()), 2);
  ASSERT_LT(static_cast<int>(tracker.InvalidatedMessages().size()),
            tracker.FullDirectedMessageCount());

  RIFTStats incrementalStats;
  std::map<DirectedCliqueEdge, RIFTMessage> incrementalCache;
  const Matrix incremental =
      RIFTExactSolver::SolveMatrix(updatedProblem, updatedTree, params,
                                   &incrementalStats, nullptr, &baseCache,
                                   &tracker, &incrementalCache);

  EXPECT_LT((incremental - fullRebuild).norm(), 1e-10);
  EXPECT_EQ(incrementalStats.directed_messages_sent, 2);
  EXPECT_EQ(incrementalStats.directed_messages_reused, 2);
  EXPECT_EQ(incrementalStats.directed_messages_invalidated, 2);
  EXPECT_EQ(tracker.ReusableCachedMessageCount(), 2);
  ASSERT_EQ(incrementalCache.size(), fullCache.size());
  for (const auto &entry : fullCache) {
    const auto updated = incrementalCache.find(entry.first);
    ASSERT_TRUE(updated != incrementalCache.end());
    EXPECT_LT((updated->second.R - entry.second.R).norm(), 1e-10);
    EXPECT_LT((updated->second.D - entry.second.D).norm(), 1e-10);
  }
  EXPECT_EQ(incrementalCache.at(DirectedCliqueEdge{1, 0}).header.graph_epoch,
            7u);
  EXPECT_EQ(incrementalCache.at(DirectedCliqueEdge{1, 2}).header.graph_epoch,
            7u);
  EXPECT_EQ(incrementalCache.at(DirectedCliqueEdge{0, 1}).header.graph_epoch,
            0u);
}

TEST(testDPGO, RIFTIFCAKHApplyMatchesExplicitNormalProduct) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  Matrix X = deterministicMatrix(H.cols(), 1, 0.31);

  const Matrix applied = RIFTCAKSolver::ApplyNormalOperator(problem, X);

  EXPECT_LT((applied - H * X).norm(), 1e-12);
}

TEST(testDPGO, RIFTIFBlockSparseNormalMatchesFactorWiseHApply) {
  const int blockDim = 3;
  const int rhsDim = 2;
  std::vector<InterfaceFactor> factors;
  factors.reserve(12);
  for (int i = 0; i < 12; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{0, i % 4}, PoseKey{1, (i + 1) % 4},
                    PoseKey{2, (i + 2) % 4}};
    factor.A = deterministicMatrix(blockDim + 2, 3 * blockDim,
                                   0.1 + 0.07 * i);
    factor.B = deterministicMatrix(blockDim + 2, rhsDim, 0.3 + 0.05 * i);
    factor.owner_robot = i % 3;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);
  Matrix X(static_cast<int>(problem.variables.size()) * blockDim, rhsDim);
  X = deterministicMatrix(X.rows(), X.cols(), 0.9);

  const Matrix factorWise = RIFTCAKSolver::ApplyNormalOperator(problem, X);
  RIFTBlockSparseNormalOperator op(problem);
  const Matrix blockSparse = op.Apply(X);

  EXPECT_LT((blockSparse - factorWise).norm() /
                std::max(1.0, factorWise.norm()),
            1e-12);
  EXPECT_GT(op.numMergedBlockContributions(), 0);
}

TEST(testDPGO, RIFTIFCAKSolverBlockSparseOperatorMatchesDefaultSolve) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 28;
  std::vector<InterfaceFactor> factors;
  factors.reserve(2 * variables - 1);
  for (int i = 0; i < variables; ++i) {
    Matrix block = Matrix::Zero(blockDim, blockDim);
    block(0, 0) = 2.0 + 0.01 * i;
    block(1, 1) = 3.0 + 0.02 * (i % 5);
    block(2, 2) = 4.0 + 0.03 * (i % 7);
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 4, i}};
    factor.A = block;
    factor.B = block * deterministicMatrix(blockDim, rhsDim, 0.1 * i + 0.2);
    factor.owner_robot = i % 4;
    factors.push_back(std::move(factor));
  }
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 4, i}, PoseKey{(i + 1) % 4, i + 1}};
    factor.A = deterministicMatrix(blockDim + 1, 2 * blockDim,
                                   0.08 * i + 0.5);
    factor.B = deterministicMatrix(blockDim + 1, rhsDim, 0.06 * i + 1.0);
    factor.owner_robot = i % 4;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 4, blockDim, rhsDim, factors);

  RIFTParams baseParams;
  baseParams.cak_rel_tol = 1e-13;
  baseParams.cak_abs_tol = 1e-11;
  baseParams.forbid_global_interface_matrix = true;
  baseParams.forbid_direct_solver_in_deployment = true;
  baseParams.forbid_collectives = true;
  RIFTStats baseStats;
  DecentralizationGuard baseGuard;
  const Matrix base =
      RIFTCAKSolver::SolveMatrix(problem, baseParams, &baseStats, &baseGuard);

  RIFTParams sparseParams = baseParams;
  sparseParams.cak_use_block_sparse_operator = true;
  RIFTStats sparseStats;
  DecentralizationGuard sparseGuard;
  const Matrix sparse = RIFTCAKSolver::SolveMatrix(
      problem, sparseParams, &sparseStats, &sparseGuard);

  EXPECT_LT((sparse - base).norm() / std::max(1.0, base.norm()), 1e-10);
  EXPECT_LT(sparseStats.cak_final_residual, 1e-8);
  EXPECT_EQ(sparseStats.cak_iterations, baseStats.cak_iterations);
  EXPECT_EQ(sparseStats.cak_scalar_reductions,
            baseStats.cak_scalar_reductions);
  EXPECT_GT(sparseStats.cak_block_sparse_entries, 0);
  EXPECT_GT(sparseStats.cak_block_sparse_merged_contributions, 0);
  EXPECT_FALSE(sparseStats.used_global_matrix);
  EXPECT_FALSE(sparseStats.used_direct_solver);
  EXPECT_FALSE(sparseStats.used_collective);
}

TEST(testDPGO, RIFTIFCAKHApplyAvoidsRepeatedLinearKeyLookup) {
  const int blockDim = 3;
  const int variables = 12000;
  std::vector<InterfaceFactor> factors;
  factors.reserve(variables - 1);
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = Matrix::Zero(blockDim, 2 * blockDim);
    factor.A.block(0, 0, blockDim, blockDim).setIdentity();
    factor.A.block(0, blockDim, blockDim, blockDim) =
        -Matrix::Identity(blockDim, blockDim);
    factor.B = Matrix::Zero(blockDim, 1);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 3, blockDim, 1, factors);
  Matrix X(static_cast<int>(problem.variables.size()) * blockDim, 1);
  X.col(0) = deterministicVector(static_cast<int>(X.rows()), 0.42);

  const auto start = std::chrono::steady_clock::now();
  const Matrix applied = RIFTCAKSolver::ApplyNormalOperator(problem, X);
  const auto finish = std::chrono::steady_clock::now();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(finish - start).count();

  EXPECT_EQ(applied.rows(), X.rows());
  EXPECT_TRUE(applied.allFinite());
  EXPECT_LT(elapsedMs, 80.0);
}

TEST(testDPGO, RIFTIFCAKSolverMatchesDirectNormalEquationAndUsesTreeReduce) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  Eigen::ColPivHouseholderQR<Matrix> qr(H);
  const Matrix direct = qr.solve(rhs);

  RIFTParams params;
  params.forbid_global_interface_matrix = true;
  params.forbid_direct_solver_in_deployment = true;
  params.forbid_collectives = true;
  RIFTStats stats;
  DecentralizationGuard guard;
  const Matrix cak =
      RIFTCAKSolver::SolveMatrix(problem, params, &stats, &guard);

  EXPECT_LT((cak - direct).norm(), 1e-8);
  EXPECT_EQ(stats.selected_backend, RIFTInterfaceBackend::RIFT_CAK);
  EXPECT_GT(stats.cak_iterations, 0);
  EXPECT_GT(stats.cak_scalar_reductions, 0);
  EXPECT_GT(stats.cak_scalar_reduction_bytes, 0u);
  EXPECT_LT(stats.cak_final_residual, 1e-8);
  EXPECT_FALSE(stats.used_global_matrix);
  EXPECT_FALSE(stats.used_direct_solver);
  EXPECT_FALSE(stats.used_collective);
}

TEST(testDPGO, RIFTIFCAKMultiRHSSolveBatchesSharedHApplyTraversal) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 40000;
  std::vector<InterfaceFactor> factors;
  factors.reserve(variables);
  for (int i = 0; i < variables; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = 2.0 * Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.01 * i);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTParams params;
  RIFTStats stats;
  const auto start = std::chrono::steady_clock::now();
  const Matrix solution = RIFTCAKSolver::SolveMatrix(problem, params, &stats);
  const auto finish = std::chrono::steady_clock::now();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(finish - start).count();

  Matrix expected(static_cast<int>(problem.variables.size()) * blockDim,
                  rhsDim);
  std::map<PoseKey, Matrix> expectedByKey;
  for (const auto &factor : factors) {
    expectedByKey[factor.scope.front()] = 0.5 * factor.B;
  }
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    expected.block(static_cast<int>(i) * blockDim, 0, blockDim, rhsDim) =
        expectedByKey.at(problem.variables[i]);
  }
  EXPECT_LT((solution - expected).norm(), 1e-10);
  EXPECT_LT(elapsedMs, 80.0);
}

TEST(testDPGO, RIFTIFCAKBlockJacobiWarmStartSolvesIndependentBlockNormals) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 24;
  Matrix blockA(blockDim, blockDim);
  blockA << 1.4, 0.30, -0.10,
            0.2, 1.10, 0.35,
           -0.1, 0.25, 0.95;

  std::vector<InterfaceFactor> factors;
  factors.reserve(variables);
  std::map<PoseKey, Matrix> expectedByKey;
  for (int i = 0; i < variables; ++i) {
    const PoseKey key{i % 5, i};
    const Matrix expectedBlock =
        deterministicMatrix(blockDim, rhsDim, 0.17 + 0.03 * i);
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {key};
    factor.A = blockA;
    factor.B = blockA * expectedBlock;
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
    expectedByKey[key] = expectedBlock;
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(
      problem, RIFTParams(), &stats);

  Matrix expected(static_cast<int>(problem.variables.size()) * blockDim,
                  rhsDim);
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    expected.block(static_cast<int>(i) * blockDim, 0, blockDim, rhsDim) =
        expectedByKey.at(problem.variables[i]);
  }
  EXPECT_LT((solution - expected).norm() / expected.norm(), 1e-11);
  EXPECT_EQ(stats.cak_iterations, 0);
  EXPECT_LT(stats.cak_final_residual, 1e-10);
}

TEST(testDPGO, RIFTIFCAKSolveUsesPrecomputedSmallNormals) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 3000;
  std::vector<InterfaceFactor> factors;
  factors.reserve(2 * variables - 1);
  for (int i = 0; i < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = 0.2 * Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.01 * i);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = Matrix::Zero(blockDim, 2 * blockDim);
    factor.A.block(0, 0, blockDim, blockDim).setIdentity();
    factor.A.block(0, blockDim, blockDim, blockDim) =
        -Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.02 * i);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTStats stats;
  const auto start = std::chrono::steady_clock::now();
  const Matrix solution = RIFTCAKSolver::SolveMatrix(
      problem, RIFTParams(), &stats);
  const auto finish = std::chrono::steady_clock::now();
  const double elapsedMs =
      std::chrono::duration<double, std::milli>(finish - start).count();

  EXPECT_TRUE(solution.allFinite());
  EXPECT_LT(stats.cak_final_residual, 2e-8);
  EXPECT_LT(elapsedMs, 60.0);
}

TEST(testDPGO, RIFTIFCAKSolvePrecomputesMediumRepeatedNormals) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 32;
  std::vector<InterfaceFactor> factors;
  factors.reserve(1);
  InterfaceFactor factor;
  factor.id = 0;
  factor.stage = FactorType::ROTATION;
  for (int i = 0; i < variables; ++i) {
    factor.scope.push_back(PoseKey{i % 5, i});
  }
  factor.A = deterministicMatrix(192, variables * blockDim, 0.43);
  factor.B = deterministicMatrix(192, rhsDim, 0.91);
  factor.owner_robot = 0;
  factors.push_back(std::move(factor));
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(
      problem, RIFTParams(), &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_GT(stats.cak_precomputed_normal_factors, 0);
}

TEST(testDPGO, RIFTIFCAKSolveReusesHApplyWorkspaceWithinSolve) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 36;
  std::vector<InterfaceFactor> factors;
  factors.reserve(2 * variables - 1);
  for (int i = 0; i < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = deterministicMatrix(blockDim, blockDim, 0.04 * i + 0.2);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.07 * i + 0.3);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = deterministicMatrix(blockDim, 2 * blockDim, 0.05 * i + 0.9);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.11 * i + 1.4);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTParams params;
  params.cak_rel_tol = 1e-14;
  params.cak_abs_tol = 1e-12;
  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(problem, params, &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_GT(stats.cak_happly_calls, 2);
  EXPECT_EQ(stats.cak_happly_workspace_reuses, stats.cak_happly_calls - 1);
  EXPECT_LT(stats.cak_final_residual, 1e-7);
}

TEST(testDPGO, RIFTIFCAKSolveDetectsContiguousScopeFactors) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 16;
  std::vector<InterfaceFactor> factors;
  factors.reserve(variables - 1);
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{0, i}, PoseKey{0, i + 1}};
    factor.A = Matrix::Zero(blockDim, 2 * blockDim);
    factor.A.block(0, 0, blockDim, blockDim).setIdentity();
    factor.A.block(0, blockDim, blockDim, blockDim) =
        -Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.03 * i + 0.6);
    factor.owner_robot = 0;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(
      problem, RIFTParams(), &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_EQ(stats.cak_contiguous_scope_factors,
            static_cast<int>(factors.size()));
  EXPECT_EQ(stats.cak_single_block_factors, 0);
}

TEST(testDPGO, RIFTIFCAKSolveReportsRepeatedScopeGroups) {
  const int blockDim = 3;
  const int rhsDim = 3;
  std::vector<InterfaceFactor> factors;
  factors.reserve(4);
  for (int i = 0; i < 4; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{0, 0}, PoseKey{1, 1}};
    factor.A = deterministicMatrix(blockDim, 2 * blockDim, 0.13 * i + 0.2);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.17 * i + 0.3);
    factor.owner_robot = i % 2;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTStats stats;
  const Matrix solution =
      RIFTCAKSolver::SolveMatrix(problem, RIFTParams(), &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_EQ(stats.cak_scope_groups, 1);
  EXPECT_EQ(stats.cak_repeated_scope_factors, 3);
}

TEST(testDPGO, RIFTIFCAKUsesJacobiPreconditioningOnIllConditionedDiagonal) {
  const int variables = 32;
  std::vector<InterfaceFactor> factors;
  factors.reserve(variables);
  for (int i = 0; i < variables; ++i) {
    const double scale = std::pow(10.0, -3.0 + 6.0 * i / (variables - 1));
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = Matrix::Constant(1, 1, scale);
    factor.B = Matrix::Constant(1, 1, 1.0 + 0.01 * i);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 1, 1, 1, factors);

  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(
      problem, RIFTParams(), &stats);

  std::map<PoseKey, double> expectedByKey;
  for (int i = 0; i < variables; ++i) {
    expectedByKey[factors[i].scope.front()] =
        factors[i].B(0, 0) / factors[i].A(0, 0);
  }
  Matrix expected(static_cast<int>(problem.variables.size()), 1);
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    expected(static_cast<int>(i), 0) =
        expectedByKey.at(problem.variables[i]);
  }
  EXPECT_LT((solution - expected).norm() / expected.norm(), 1e-8);
  EXPECT_LT(stats.cak_iterations, 5);
  EXPECT_LT(stats.cak_final_residual, 1e-8);
}

TEST(testDPGO, RIFTIFCAKBlockJacobiPCGReducesCoupledChainIterations) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 36;
  std::vector<InterfaceFactor> factors;
  factors.reserve(2 * variables - 1);
  for (int i = 0; i < variables; ++i) {
    Matrix block = Matrix::Zero(blockDim, blockDim);
    block(0, 0) = 1.0 + 0.05 * (i % 5);
    block(1, 1) = 20.0 + 0.4 * (i % 7);
    block(2, 2) = 200.0 + 2.0 * (i % 3);
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = block;
    factor.B = block * deterministicMatrix(blockDim, rhsDim, 0.03 * i + 0.4);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = Matrix::Zero(blockDim, 2 * blockDim);
    factor.A.block(0, 0, blockDim, blockDim) =
        0.7 * Matrix::Identity(blockDim, blockDim);
    factor.A.block(0, blockDim, blockDim, blockDim) =
        -0.7 * Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.05 * i + 1.1);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTParams baseParams;
  baseParams.cak_rel_tol = 1e-12;
  baseParams.cak_abs_tol = 1e-10;
  RIFTStats baseStats;
  const Matrix base =
      RIFTCAKSolver::SolveMatrix(problem, baseParams, &baseStats);

  RIFTParams pcgParams = baseParams;
  pcgParams.cak_use_block_jacobi_pcg = true;
  RIFTStats pcgStats;
  const Matrix pcg = RIFTCAKSolver::SolveMatrix(problem, pcgParams, &pcgStats);

  EXPECT_LT((pcg - base).norm() / std::max(1.0, base.norm()), 1e-8)
      << "base_iters=" << baseStats.cak_iterations
      << " pcg_iters=" << pcgStats.cak_iterations
      << " pcg_residual=" << pcgStats.cak_final_residual;
  EXPECT_LT(pcgStats.cak_final_residual, 5e-8)
      << "base_iters=" << baseStats.cak_iterations
      << " pcg_iters=" << pcgStats.cak_iterations;
  EXPECT_GT(pcgStats.cak_preconditioner_applications, 1);
  EXPECT_LT(pcgStats.cak_iterations, baseStats.cak_iterations)
      << "base_residual=" << baseStats.cak_final_residual
      << " pcg_residual=" << pcgStats.cak_final_residual;
}

TEST(testDPGO, RIFTIFCAKCertifiedPCGRefreshesExactResidualBeforeStopping) {
  const int blockDim = 3;
  const int rhsDim = 3;
  const int variables = 42;
  std::vector<InterfaceFactor> factors;
  factors.reserve(2 * variables - 1);
  for (int i = 0; i < variables; ++i) {
    Matrix block = Matrix::Zero(blockDim, blockDim);
    block(0, 0) = 1.0 + 0.03 * (i % 5);
    block(1, 1) = 10.0 + 0.5 * (i % 7);
    block(2, 2) = 120.0 + 3.0 * (i % 3);
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}};
    factor.A = block;
    factor.B = block * deterministicMatrix(blockDim, rhsDim, 0.04 * i + 0.2);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = Matrix::Zero(blockDim, 2 * blockDim);
    factor.A.block(0, 0, blockDim, blockDim) =
        0.9 * Matrix::Identity(blockDim, blockDim);
    factor.A.block(0, blockDim, blockDim, blockDim) =
        -0.9 * Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.08 * i + 1.2);
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTParams params;
  params.cak_use_block_jacobi_pcg = true;
  params.cak_pcg_certify_residual = true;
  params.cak_rel_tol = 1e-10;
  params.cak_abs_tol = 1e-8;
  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(problem, params, &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_GT(stats.cak_pcg_residual_replacements, 0);
  EXPECT_LT(stats.cak_final_residual, 1e-8);
}

TEST(testDPGO, RIFTIFCAKRobotMeanCoarsePCGUsesLowFrequencyCorrection) {
  const int robots = 5;
  const int posesPerRobot = 16;
  std::vector<InterfaceFactor> factors;
  factors.reserve(static_cast<std::size_t>(robots * posesPerRobot * 2));

  auto addUnary = [&](int robot, int pose, double scale, double value) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{robot, robot * posesPerRobot + pose}};
    factor.A = Matrix::Constant(1, 1, scale);
    factor.B = Matrix::Constant(1, 1, scale * value);
    factor.owner_robot = robot;
    factors.push_back(std::move(factor));
  };
  auto addBinary = [&](int robotA, int poseA, int robotB, int poseB,
                       double scale, double delta) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{robotA, robotA * posesPerRobot + poseA},
                    PoseKey{robotB, robotB * posesPerRobot + poseB}};
    factor.A = Matrix::Zero(1, 2);
    factor.A(0, 0) = -scale;
    factor.A(0, 1) = scale;
    factor.B = Matrix::Constant(1, 1, scale * delta);
    factor.owner_robot = robotA;
    factors.push_back(std::move(factor));
  };

  addUnary(0, 0, 2.0, 0.0);
  for (int robot = 0; robot < robots; ++robot) {
    for (int pose = 0; pose + 1 < posesPerRobot; ++pose) {
      addBinary(robot, pose, robot, pose + 1, 4.0, 0.0);
    }
  }
  for (int robot = 0; robot + 1 < robots; ++robot) {
    addBinary(robot, posesPerRobot - 1, robot + 1, 0, 0.4, 1.0);
  }
  for (int robot = 0; robot < robots; ++robot) {
    addUnary(robot, posesPerRobot / 2, 0.02, static_cast<double>(robot));
  }

  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 1, 1, 1, factors);

  RIFTParams pcgParams;
  pcgParams.cak_use_block_jacobi_pcg = true;
  pcgParams.cak_rel_tol = 1e-12;
  pcgParams.cak_abs_tol = 1e-10;
  RIFTStats pcgStats;
  const Matrix pcg = RIFTCAKSolver::SolveMatrix(problem, pcgParams, &pcgStats);

  RIFTParams coarseParams = pcgParams;
  coarseParams.cak_use_robot_mean_coarse_pcg = true;
  RIFTStats coarseStats;
  const Matrix coarse =
      RIFTCAKSolver::SolveMatrix(problem, coarseParams, &coarseStats);

  EXPECT_LT((coarse - pcg).norm() / std::max(1.0, pcg.norm()), 1e-8);
  EXPECT_GT(coarseStats.cak_coarse_dim, 0);
  EXPECT_GT(coarseStats.cak_coarse_preconditioner_applications, 0);
  EXPECT_LE(coarseStats.cak_iterations, pcgStats.cak_iterations);
  EXPECT_LT(coarseStats.cak_final_residual, 1e-6);
}

TEST(testDPGO, RIFTIFCAKRobotMeanCoarseWarmStartIsOneShot) {
  const int robots = 5;
  const int posesPerRobot = 16;
  std::vector<InterfaceFactor> factors;
  factors.reserve(static_cast<std::size_t>(robots * posesPerRobot * 2));

  auto addUnary = [&](int robot, int pose, double scale, double value) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{robot, robot * posesPerRobot + pose}};
    factor.A = Matrix::Constant(1, 1, scale);
    factor.B = Matrix::Constant(1, 1, scale * value);
    factor.owner_robot = robot;
    factors.push_back(std::move(factor));
  };
  auto addBinary = [&](int robotA, int poseA, int robotB, int poseB,
                       double scale, double delta) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{robotA, robotA * posesPerRobot + poseA},
                    PoseKey{robotB, robotB * posesPerRobot + poseB}};
    factor.A = Matrix::Zero(1, 2);
    factor.A(0, 0) = -scale;
    factor.A(0, 1) = scale;
    factor.B = Matrix::Constant(1, 1, scale * delta);
    factor.owner_robot = robotA;
    factors.push_back(std::move(factor));
  };

  addUnary(0, 0, 2.0, 0.0);
  for (int robot = 0; robot < robots; ++robot) {
    for (int pose = 0; pose + 1 < posesPerRobot; ++pose) {
      addBinary(robot, pose, robot, pose + 1, 4.0, 0.0);
    }
  }
  for (int robot = 0; robot + 1 < robots; ++robot) {
    addBinary(robot, posesPerRobot - 1, robot + 1, 0, 0.4, 1.0);
  }
  for (int robot = 0; robot < robots; ++robot) {
    addUnary(robot, posesPerRobot / 2, 0.02, static_cast<double>(robot));
  }

  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 1, 1, 1, factors);

  RIFTParams baseParams;
  baseParams.cak_rel_tol = 1e-12;
  baseParams.cak_abs_tol = 1e-10;
  RIFTStats baseStats;
  const Matrix base =
      RIFTCAKSolver::SolveMatrix(problem, baseParams, &baseStats);

  RIFTParams warmParams = baseParams;
  warmParams.cak_use_robot_mean_coarse_warm_start = true;
  RIFTStats warmStats;
  const Matrix warm =
      RIFTCAKSolver::SolveMatrix(problem, warmParams, &warmStats);

  EXPECT_LT((warm - base).norm() / std::max(1.0, base.norm()), 1e-8);
  EXPECT_GT(warmStats.cak_coarse_dim, 0);
  EXPECT_EQ(warmStats.cak_coarse_warm_start_applications, 1);
  EXPECT_EQ(warmStats.cak_coarse_preconditioner_applications, 0);
  EXPECT_LE(warmStats.cak_iterations, baseStats.cak_iterations);
}

TEST(testDPGO, RIFTIFCAKDelayedPCGDoesNotPayPreconditionerBeforeSwitch) {
  const int blockDim = 3;
  const int rhsDim = 3;
  std::vector<InterfaceFactor> factors;
  factors.reserve(8);
  for (int i = 0; i < 8; ++i) {
    InterfaceFactor factor;
    factor.id = i;
    factor.stage = FactorType::ROTATION;
    factor.scope = {PoseKey{i % 2, i}};
    factor.A = (1.0 + 0.1 * i) * Matrix::Identity(blockDim, blockDim);
    factor.B = deterministicMatrix(blockDim, rhsDim, 0.2 + 0.03 * i);
    factor.owner_robot = i % 2;
    factors.push_back(std::move(factor));
  }
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, 3, blockDim, rhsDim, factors);

  RIFTParams params;
  params.cak_block_jacobi_pcg_switch_iter = 1000;
  RIFTStats stats;
  const Matrix solution = RIFTCAKSolver::SolveMatrix(problem, params, &stats);

  EXPECT_TRUE(solution.allFinite());
  EXPECT_EQ(stats.cak_iterations, 0);
  EXPECT_EQ(stats.cak_preconditioner_applications, 1);
}

TEST(testDPGO, RIFTIFCAKHonorsConfigurableResidualTolerance) {
  const int variables = 80;
  std::vector<InterfaceFactor> factors;
  factors.reserve(variables);

  InterfaceFactor anchor;
  anchor.id = static_cast<RIFTFactorId>(factors.size());
  anchor.stage = FactorType::TRANSLATION;
  anchor.scope = {PoseKey{0, 0}};
  anchor.A = Matrix::Constant(1, 1, 1.0);
  anchor.B = Matrix::Constant(1, 1, 0.0);
  anchor.owner_robot = 0;
  factors.push_back(anchor);

  for (int i = 0; i + 1 < variables; ++i) {
    InterfaceFactor factor;
    factor.id = static_cast<RIFTFactorId>(factors.size());
    factor.stage = FactorType::TRANSLATION;
    factor.scope = {PoseKey{i % 5, i}, PoseKey{(i + 1) % 5, i + 1}};
    factor.A = Matrix::Zero(1, 2);
    factor.A << 1.0, -1.0;
    factor.B = Matrix::Constant(1, 1, 0.05 * std::sin(0.2 * i));
    factor.owner_robot = i % 5;
    factors.push_back(std::move(factor));
  }

  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 1, 1, 1, factors);

  RIFTParams tightParams;
  tightParams.cak_rel_tol = 1e-14;
  tightParams.cak_abs_tol = 1e-12;
  RIFTStats tightStats;
  const Matrix tight =
      RIFTCAKSolver::SolveMatrix(problem, tightParams, &tightStats);

  RIFTParams looseParams;
  looseParams.cak_rel_tol = 1e-2;
  looseParams.cak_abs_tol = 1e-1;
  RIFTStats looseStats;
  const Matrix loose =
      RIFTCAKSolver::SolveMatrix(problem, looseParams, &looseStats);

  EXPECT_GT(tightStats.cak_iterations, looseStats.cak_iterations);
  EXPECT_LT(tightStats.cak_final_residual, 1e-10);
  EXPECT_LT(looseStats.cak_final_residual, 1.01e-1);
  EXPECT_LT((tight - loose).norm() / std::max(1.0, tight.norm()), 1.0);
}

TEST(testDPGO, RIFTIFCAKBackendRunsThroughTEDRoute) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const Matrix exact =
      RIFTExactSolver::SolveMatrix(problem, makeIncrementalRiftChainTree(true),
                                   RIFTParams());

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_CAK;
  params.rift_forbid_direct_interface_solver = true;
  params.rift_forbid_global_interface_matrix = true;
  params.rift_forbid_collectives = true;
  TEDCCIStats stats;
  const Matrix cak = SolveInterfaceProblemWithRIFTExact(problem, params,
                                                       &stats);

  EXPECT_LT((cak - exact).norm(), 1e-8);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_CAK);
  EXPECT_GT(stats.rift_cak_iterations, 0);
  EXPECT_GT(stats.rift_cak_precomputed_normal_factors, 0);
  EXPECT_GT(stats.rift_cak_scalar_reductions, 0);
  EXPECT_GT(stats.rift_cak_scalar_reduction_bytes, 0u);
  EXPECT_LT(stats.rift_cak_final_residual, 1e-8);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, RIFTIFAutoSelectsExactWhenSymbolicGateAllows) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/false));

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_AUTO;
  params.rift_forbid_direct_interface_solver = true;
  params.rift_forbid_global_interface_matrix = true;
  params.rift_forbid_collectives = true;
  TEDCCIStats stats;
  const Matrix solution = SolveInterfaceProblemWithRIFTExact(problem, params,
                                                            &stats);
  const Matrix exact =
      RIFTExactSolver::SolveMatrix(problem, makeIncrementalRiftChainTree(false),
                                   RIFTParams());

  EXPECT_LT((solution - exact).norm(), 1e-10);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_EXACT);
  EXPECT_GT(stats.rift_num_cliques, 0);
  EXPECT_GT(stats.rift_directed_messages_sent, 0);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, RIFTIFAutoFallsBackToCAKWhenExactGateRejects) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_AUTO;
  params.rift_exact_max_separator_blocks_2d = 0;
  params.rift_exact_max_message_bytes = 0;
  params.rift_forbid_direct_interface_solver = true;
  params.rift_forbid_global_interface_matrix = true;
  params.rift_forbid_collectives = true;
  TEDCCIStats stats;
  const Matrix autoSolution = SolveInterfaceProblemWithRIFTExact(problem,
                                                                params,
                                                                &stats);

  EXPECT_LT((autoSolution - direct).norm(), 1e-8);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_CAK);
  EXPECT_EQ(stats.rift_num_cliques, 0);
  EXPECT_EQ(stats.rift_num_tree_edges, 0);
  EXPECT_GT(stats.rift_cak_iterations, 0);
  EXPECT_GT(stats.rift_cak_scalar_reductions, 0);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, RIFTIFAutoSkipsExactTreeWhenPreflightRejectsBudget) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeIncrementalRiftFactors(/*includeLoopFactor=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_AUTO;
  params.rift_exact_max_separator_blocks_2d = 256;
  params.rift_exact_max_message_bytes = 1;
  params.rift_forbid_direct_interface_solver = true;
  params.rift_forbid_global_interface_matrix = true;
  params.rift_forbid_collectives = true;
  TEDCCIStats stats;
  const Matrix autoSolution = SolveInterfaceProblemWithRIFTExact(problem,
                                                                params,
                                                                &stats);

  EXPECT_LT((autoSolution - direct).norm(), 1e-8);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_CAK);
  EXPECT_EQ(stats.rift_num_cliques, 0);
  EXPECT_EQ(stats.rift_num_tree_edges, 0);
  EXPECT_GT(stats.rift_cak_iterations, 0);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, RIFTIFAsyncSchurResidualDecreasesAndMatchesDirectSmallSPD) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeAsyncSchurFactors(/*includeThirdRobot=*/false));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  RIFTParams params;
  params.async_schur_max_iters = 1000;
  params.async_schur_rel_tol = 1e-10;
  params.async_schur_relaxation = 0.5;
  RIFTStats stats;
  DecentralizationGuard guard;
  const Matrix async =
      RIFTAsyncSchurSolver::SolveMatrix(problem, params, &stats, &guard);

  EXPECT_EQ(stats.selected_backend, RIFTInterfaceBackend::RIFT_ASYNC_SCHUR);
  EXPECT_TRUE(stats.async_schur_converged);
  EXPECT_TRUE(stats.async_schur_global_consistent);
  EXPECT_GT(stats.async_schur_iterations, 0);
  EXPECT_LT(stats.async_schur_final_residual,
            stats.async_schur_initial_residual);
  EXPECT_LT((async - direct).norm(), 1e-7);
  EXPECT_FALSE(stats.used_direct_solver);
  EXPECT_FALSE(stats.used_global_matrix);
  EXPECT_FALSE(stats.used_collective);
}

TEST(testDPGO, RIFTIFAsyncSchurDelayedReorderedDeliveryConverges) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeAsyncSchurFactors(/*includeThirdRobot=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  RIFTParams params;
  params.async_schur_max_iters = 5000;
  params.async_schur_rel_tol = 1e-9;
  params.async_schur_relaxation = 0.25;
  RIFTAsyncSchurSolver solver(problem, params, 7);
  RIFTLinkModel delayed;
  delayed.latency_mean_ms = 2.0;
  delayed.latency_jitter_ms = 1.0;
  delayed.reorder_prob = 1.0;
  solver.SetLinkModel(0, 1, delayed);
  solver.SetLinkModel(1, 2, delayed);
  solver.SetLinkModel(0, 2, delayed);
  solver.Run(params.async_schur_max_iters);

  EXPECT_TRUE(solver.GlobalConverged())
      << "residual=" << solver.GlobalResidualNorm()
      << " diff=" << (solver.Solution() - direct).norm();
  EXPECT_LT((solver.Solution() - direct).norm(), 1e-7);
}

TEST(testDPGO, RIFTIFAsyncSchurPermanentSplitReportsComponentWiseOnly) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeAsyncSchurFactors(/*includeThirdRobot=*/true));
  RIFTParams params;
  params.async_schur_max_iters = 800;
  params.async_schur_rel_tol = 1e-10;
  params.async_schur_relaxation = 0.5;
  RIFTAsyncSchurSolver solver(problem, params, 11);
  solver.DropLink(0, 2);
  solver.DropLink(1, 2);
  solver.Run(params.async_schur_max_iters);

  const auto components = solver.ConnectedComponents();
  ASSERT_EQ(components.size(), 2u);
  EXPECT_TRUE(solver.ComponentConsistent());
  EXPECT_TRUE(solver.ComponentConverged({0, 1}));
  EXPECT_TRUE(solver.ComponentConverged({2}));
  EXPECT_FALSE(solver.GlobalConverged());
  EXPECT_GT(solver.GlobalResidualNorm(), 1e-6);
}

TEST(testDPGO, RIFTIFAsyncSchurReconnectWarmStartMergesComponents) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeAsyncSchurFactors(/*includeThirdRobot=*/true));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  RIFTParams params;
  params.async_schur_max_iters = 5000;
  params.async_schur_rel_tol = 1e-9;
  params.async_schur_relaxation = 0.25;
  RIFTAsyncSchurSolver solver(problem, params, 13);
  solver.DropLink(0, 2);
  solver.DropLink(1, 2);
  solver.Run(50);
  const Matrix splitSolution = solver.Solution();
  const double splitResidual = solver.GlobalResidualNorm();
  ASSERT_FALSE(solver.GlobalConverged());

  solver.RestoreLink(1, 2);
  solver.RestoreLink(0, 2);
  solver.Run(params.async_schur_max_iters);

  EXPECT_GE(solver.reconnect_merges(), 1);
  EXPECT_GT(splitSolution.norm(), 0.0);
  EXPECT_LT(solver.GlobalResidualNorm(), splitResidual);
  EXPECT_TRUE(solver.GlobalConverged())
      << "residual=" << solver.GlobalResidualNorm()
      << " diff=" << (solver.Solution() - direct).norm();
  EXPECT_LT((solver.Solution() - direct).norm(), 1e-7);
}

TEST(testDPGO, RIFTIFNetworkZeroDelayMatchesRootlessSchedulerOrdering) {
  InterfaceCliqueTree tree;
  tree.cliques.resize(3);
  for (int i = 0; i < 3; ++i) {
    tree.cliques[i].id = i;
  }
  tree.cliques[0].neighbors = {1};
  tree.cliques[1].neighbors = {0, 2};
  tree.cliques[2].neighbors = {1};

  std::vector<DirectedCliqueEdge> directOrder;
  RIFTRootlessScheduler directScheduler;
  directScheduler.Initialize(tree);
  while (!directScheduler.AllDirectedMessagesComplete()) {
    const auto ready = directScheduler.ReadyOutgoingMessages();
    ASSERT_FALSE(ready.empty());
    for (const auto &edge : ready) {
      RIFTMessage msg;
      msg.header.src = edge.src;
      msg.header.dst = edge.dst;
      directOrder.push_back(edge);
      directScheduler.MarkMessageSent(edge.src, edge.dst);
      directScheduler.OnMessageReceived(msg);
    }
  }

  RIFTP2PNetworkSimulator network;
  RIFTLinkModel zeroDelay;
  network.AddLink(0, 1, zeroDelay);
  network.AddLink(1, 2, zeroDelay);
  RIFTRootlessScheduler networkScheduler;
  networkScheduler.Initialize(tree);
  std::uint64_t seq = 0;
  std::vector<DirectedCliqueEdge> networkOrder;
  while (!networkScheduler.AllDirectedMessagesComplete()) {
    const auto ready = networkScheduler.ReadyOutgoingMessages();
    ASSERT_FALSE(ready.empty());
    for (const auto &edge : ready) {
      networkScheduler.MarkMessageSent(edge.src, edge.dst);
      network.Send(makeNetworkMessage(edge.src, edge.dst, seq++, 0.0));
    }
    const auto delivered = network.DeliverReady(0.0);
    ASSERT_FALSE(delivered.empty());
    for (const auto &event : delivered) {
      networkOrder.push_back(
          DirectedCliqueEdge{event.message.rift_message.header.src,
                             event.message.rift_message.header.dst});
      networkScheduler.OnMessageReceived(event.message.rift_message);
    }
  }

  ASSERT_EQ(networkOrder.size(), directOrder.size());
  for (std::size_t i = 0; i < directOrder.size(); ++i) {
    EXPECT_EQ(networkOrder[i].src, directOrder[i].src);
    EXPECT_EQ(networkOrder[i].dst, directOrder[i].dst);
  }
}

TEST(testDPGO, RIFTIFNetworkDelayedReorderedDeliveryCompletesScheduler) {
  InterfaceCliqueTree tree;
  tree.cliques.resize(3);
  for (int i = 0; i < 3; ++i) {
    tree.cliques[i].id = i;
  }
  tree.cliques[0].neighbors = {1};
  tree.cliques[1].neighbors = {0, 2};
  tree.cliques[2].neighbors = {1};

  RIFTLinkModel model;
  model.latency_mean_ms = 5.0;
  model.latency_jitter_ms = 2.0;
  model.reorder_prob = 1.0;
  RIFTP2PNetworkSimulator network(7);
  network.AddLink(0, 1, model);
  network.AddLink(1, 2, model);

  RIFTRootlessScheduler scheduler;
  scheduler.Initialize(tree);
  std::uint64_t seq = 0;
  double now = 0.0;
  std::set<std::pair<int, int>> receivedEdges;
  for (int step = 0; step < 100 && !scheduler.AllDirectedMessagesComplete();
       ++step) {
    const auto ready = scheduler.ReadyOutgoingMessages();
    for (const auto &edge : ready) {
      scheduler.MarkMessageSent(edge.src, edge.dst);
      network.Send(makeNetworkMessage(edge.src, edge.dst, seq++, now, 32));
    }
    now += 2.0;
    for (const auto &event : network.DeliverReady(now)) {
      receivedEdges.insert({event.message.rift_message.header.src,
                            event.message.rift_message.header.dst});
      scheduler.OnMessageReceived(event.message.rift_message);
    }
  }

  EXPECT_TRUE(scheduler.AllDirectedMessagesComplete());
  EXPECT_EQ(receivedEdges.size(), 4u);
}

TEST(testDPGO, RIFTIFNetworkSerializesBandwidthPerDirectedLink) {
  RIFTP2PNetworkSimulator network;
  RIFTLinkModel model;
  model.bandwidth_bytes_per_sec = 1000.0;
  network.AddLink(0, 1, model);

  network.Send(makeNetworkMessage(0, 1, 1, 0.0, 1000));
  network.Send(makeNetworkMessage(0, 1, 2, 0.0, 500));
  EXPECT_TRUE(network.DeliverReady(999.0).empty());

  const auto first = network.DeliverReady(1000.0);
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first[0].message.seq, 1u);
  EXPECT_DOUBLE_EQ(first[0].delivery_time_ms, 1000.0);

  const auto second = network.DeliverReady(1500.0);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(second[0].message.seq, 2u);
  EXPECT_DOUBLE_EQ(second[0].delivery_time_ms, 1500.0);
}

TEST(testDPGO, RIFTIFNetworkDisconnectHoldAndDropPolicies) {
  RIFTP2PNetworkSimulator holdingNetwork;
  RIFTLinkModel holdModel;
  holdModel.latency_mean_ms = 10.0;
  holdingNetwork.AddLink(0, 1, holdModel);
  holdingNetwork.DropLink(0, 1);
  holdingNetwork.Send(makeNetworkMessage(0, 1, 1, 0.0));
  EXPECT_TRUE(holdingNetwork.DeliverReady(100.0).empty());
  EXPECT_EQ(holdingNetwork.PendingCount(), 1u);
  holdingNetwork.RestoreLink(0, 1);
  const auto deliveredAfterRestore = holdingNetwork.DeliverReady(100.0);
  ASSERT_EQ(deliveredAfterRestore.size(), 1u);
  EXPECT_EQ(deliveredAfterRestore[0].message.seq, 1u);

  RIFTP2PNetworkSimulator droppingNetwork;
  RIFTLinkModel dropWhileDisconnected;
  dropWhileDisconnected.disconnect_policy =
      RIFTDisconnectPolicy::DROP_WHILE_DISCONNECTED;
  droppingNetwork.AddLink(0, 1, dropWhileDisconnected);
  droppingNetwork.DropLink(0, 1);
  droppingNetwork.Send(makeNetworkMessage(0, 1, 2, 0.0));
  EXPECT_EQ(droppingNetwork.PendingCount(), 0u);
  EXPECT_EQ(droppingNetwork.DroppedCount(), 1u);
}

TEST(testDPGO, RIFTIFNetworkDropProbabilityCanDropDeterministically) {
  RIFTP2PNetworkSimulator network;
  RIFTLinkModel model;
  model.drop_prob = 1.0;
  network.AddLink(0, 1, model);
  network.Send(makeNetworkMessage(0, 1, 1, 0.0));
  EXPECT_EQ(network.PendingCount(), 0u);
  EXPECT_EQ(network.DroppedCount(), 1u);
}

TEST(testDPGO, RIFTIFExactMatchesDirectOracleOnSmallInterface) {
  const int blockDim = 2;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const auto c0 = makeCondensed(0, {a, b}, 8, blockDim, 0.1);
  const auto c1 = makeCondensed(1, {b, c}, 8, blockDim, 0.7);
  const auto cross = makeCross({c, a}, 5, blockDim, 1.3);

  TEDCCIParams tedParams;
  const Vector direct =
      InterfaceDirectSolver::Solve({c0, c1}, {cross}, blockDim, tedParams);

  RIFTParams riftParams;
  riftParams.forbid_global_interface_matrix = false;
  riftParams.forbid_direct_solver_in_deployment = false;
  riftParams.forbid_collectives = false;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {c0, c1}, {cross},
          /*use_rotation_multi_rhs=*/false);
  const InterfaceCliqueTree tree =
      InterfaceCliqueTreeBuilder::Build(problem, riftParams);
  DecentralizationGuard guard;
  RIFTStats stats;
  const Vector rift =
      RIFTExactSolver::Solve(problem, tree, riftParams, &stats, &guard);

  EXPECT_LT((rift - direct).norm(), 1e-8);
  EXPECT_EQ(stats.selected_backend, RIFTInterfaceBackend::RIFT_EXACT);
  EXPECT_FALSE(stats.used_direct_solver);
  EXPECT_FALSE(stats.used_global_matrix);
  EXPECT_FALSE(stats.used_collective);
  EXPECT_EQ(stats.directed_messages_sent,
            2 * static_cast<int>(tree.edges.size()));
  EXPECT_GT(stats.num_host_robots, 0);
  EXPECT_GT(stats.max_host_clique_load, 0);
  EXPECT_GE(stats.cross_host_tree_edges, 0);
  EXPECT_EQ(stats.estimated_route_hops,
            2 * stats.cross_host_tree_edges);
  EXPECT_LE(stats.estimated_routed_message_bytes,
            stats.estimated_message_bytes);
}

TEST(testDPGO, RIFTIFExactPreservesRankDeficientEliminationResidual) {
  const int blockDim = 1;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};

  CondensedFactor f0;
  f0.owner_robot_id = 0;
  f0.factor_type = FactorType::TRANSLATION;
  f0.boundary_keys = {a, b};
  f0.Abar.resize(1, 2);
  f0.Abar << 0.0, 1.0;
  f0.bbar.resize(1);
  f0.bbar << 1.0;

  CondensedFactor f1;
  f1.owner_robot_id = 1;
  f1.factor_type = FactorType::TRANSLATION;
  f1.boundary_keys = {b, c};
  f1.Abar.resize(2, 2);
  f1.Abar << 1.0, 0.0,
              0.0, 1.0;
  f1.bbar.resize(2);
  f1.bbar << 1.0, 2.0;

  TEDCCIParams tedParams;
  const Vector direct =
      InterfaceDirectSolver::Solve({f0, f1}, {}, blockDim, tedParams);

  RIFTParams riftParams;
  riftParams.forbid_global_interface_matrix = false;
  riftParams.forbid_direct_solver_in_deployment = false;
  riftParams.forbid_collectives = false;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {f0, f1}, {},
          /*use_rotation_multi_rhs=*/false);
  const InterfaceCliqueTree tree =
      InterfaceCliqueTreeBuilder::Build(problem, riftParams);
  RIFTStats stats;
  const Vector rift =
      RIFTExactSolver::Solve(problem, tree, riftParams, &stats);

  EXPECT_LT((rift - direct).norm(), 1e-10);
  EXPECT_NEAR(rift(1), 1.0, 1e-10);
  EXPECT_NEAR(rift(2), 2.0, 1e-10);
}

TEST(testDPGO, RIFTIFGuardDetectsDirectSolverUse) {
  DecentralizationGuard guard;
  guard.RecordDirectInterfaceSolverCall();
  EXPECT_THROW(guard.AssertNoDirectSolverCalled(), std::runtime_error);
}

TEST(testDPGO, RIFTIFGuardDetectsGlobalMatrixCollectiveAndAllFactorReceiver) {
  DecentralizationGuard matrixGuard;
  matrixGuard.RecordGlobalInterfaceMatrixConstruction();
  EXPECT_THROW(matrixGuard.AssertNoGlobalInterfaceMatrixConstructed(),
               std::runtime_error);

  DecentralizationGuard collectiveGuard;
  collectiveGuard.RecordCollectiveCall("MPI_Allreduce");
  EXPECT_THROW(collectiveGuard.AssertNoCollectiveCommunication(),
               std::runtime_error);

  DecentralizationGuard receiverGuard;
  receiverGuard.RecordFactorTransfer(0, 7, 0);
  receiverGuard.RecordFactorTransfer(1, 7, 1);
  receiverGuard.RecordFactorTransfer(2, 7, 2);
  EXPECT_THROW(receiverGuard.AssertNoNodeReceivedAllFactors(3),
               std::runtime_error);
}

TEST(testDPGO, RIFTIFDirectBackendRejectedInDeploymentPath) {
  const int blockDim = 2;
  const PoseKey a{0, 1};
  CondensedFactor factor;
  factor.owner_robot_id = 0;
  factor.factor_type = FactorType::TRANSLATION;
  factor.boundary_keys = {a};
  factor.Abar = Matrix::Identity(blockDim, blockDim);
  factor.bbar = Vector::Ones(blockDim);

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::DIRECT_ORACLE;
  params.rift_forbid_direct_interface_solver = true;
  TEDCCIStats stats;
  EXPECT_THROW(SolveTEDInterfaceWithRIFTExact({factor}, {}, blockDim, {},
                                             params, &stats),
               std::invalid_argument);
}

TEST(testDPGO, RIFTIFAsyncSchurBackendRunsThroughTEDRoute) {
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::TRANSLATION, 2, 1, 1,
          makeAsyncSchurFactors(/*includeThirdRobot=*/false));
  const Matrix H = explicitNormalMatrixForTest(problem);
  const Matrix rhs = explicitNormalRhsForTest(problem);
  const Matrix direct = H.colPivHouseholderQr().solve(rhs);

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_ASYNC_SCHUR;
  params.rift_forbid_direct_interface_solver = true;
  params.rift_forbid_global_interface_matrix = true;
  params.rift_forbid_collectives = true;
  params.async_dd_max_iters = 1000;
  params.async_dd_rel_tol = 1e-10;
  TEDCCIStats stats;
  const Matrix async = SolveInterfaceProblemWithRIFTExact(problem, params,
                                                         &stats);

  EXPECT_LT((async - direct).norm(), 1e-7);
  EXPECT_EQ(stats.rift_selected_backend,
            RIFTInterfaceBackend::RIFT_ASYNC_SCHUR);
  EXPECT_TRUE(stats.async_dd_converged);
  EXPECT_GT(stats.async_dd_iterations, 0);
  EXPECT_LT(stats.async_dd_final_residual,
            stats.async_dd_initial_residual);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, TEDCCIRIFTIFMatchesCentralized2DChainSplit) {
  const size_t d = 2;
  const size_t numPoses = 5;
  const int robots = 2;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.15), rotation2(0.35), rotation2(0.62),
      rotation2(0.88)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0;
  translations[1] << 0.9, 0.1;
  translations[2] << 1.8, 0.4;
  translations[3] << 2.6, 0.9;
  translations[4] << 3.5, 1.1;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.5 * pose));
  }
  measurements.push_back(makeMeasurement(0, 3, owner, rotations[0],
                                         translations[0], rotations[3],
                                         translations[3], 4.0, 2.5));

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_EXACT;
  params.use_measurement_weight = false;
  EXPECT_TRUE(params.rift_use_rotation_multi_rhs);
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         params);
  TEDCCIStats stats;
  const Matrix rift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, params, &stats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  expectPoseEstimateClose(rift, centralized, d, numPoses, 1e-8);
  EXPECT_EQ(stats.effective_rotation_mode, CCIInitMode::TED_CCI_RIFT_IF);
  EXPECT_EQ(stats.effective_translation_mode, CCIInitMode::TED_CCI_RIFT_IF);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_EXACT);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

TEST(testDPGO, TEDCCIRIFTIFMultiRHSMatchesVectorizedRIFT3DChainSplit) {
  const size_t d = 3;
  const size_t numPoses = 5;
  const int robots = 2;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      Matrix::Identity(d, d), rotation3(0.18, 0.04, -0.05),
      rotation3(0.39, -0.06, 0.08), rotation3(0.61, 0.09, -0.12),
      rotation3(0.83, -0.11, 0.16)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[1] << 0.6, 0.1, -0.1;
  translations[2] << 1.3, 0.3, 0.2;
  translations[3] << 2.1, 0.5, 0.4;
  translations[4] << 2.8, 0.9, 0.5;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.5 + pose,
        3.5 + 0.25 * pose));
  }
  measurements.push_back(makeMeasurement(0, 3, owner, rotations[0],
                                         translations[0], rotations[3],
                                         translations[3], 4.0, 2.5));
  measurements.push_back(makeMeasurement(1, 4, owner, rotations[1],
                                         translations[1], rotations[4],
                                         translations[4], 3.0, 2.0));

  TEDCCIParams vectorParams;
  vectorParams.mode = CCIInitMode::TED_CCI_RIFT_IF;
  vectorParams.rift_interface_backend = RIFTInterfaceBackend::RIFT_EXACT;
  vectorParams.rift_use_rotation_multi_rhs = false;
  vectorParams.use_measurement_weight = false;
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         vectorParams);
  TEDCCIStats vectorStats;
  const Matrix vectorizedRift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, vectorParams, &vectorStats);

  TEDCCIParams multiParams = vectorParams;
  multiParams.rift_use_rotation_multi_rhs = true;
  TEDCCIStats multiStats;
  const Matrix multiRift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, multiParams, &multiStats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  expectPoseEstimateClose(multiRift, vectorizedRift, d, numPoses, 1e-8);
  expectPoseEstimateClose(multiRift, centralized, d, numPoses, 1e-8);
  EXPECT_LT(multiStats.rift_estimated_message_bytes,
            vectorStats.rift_estimated_message_bytes);
  EXPECT_FALSE(multiStats.rift_used_direct_solver);
  EXPECT_FALSE(multiStats.rift_used_global_matrix);
  EXPECT_FALSE(multiStats.rift_used_collective);
}
