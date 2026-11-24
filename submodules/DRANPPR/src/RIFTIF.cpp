#include <DPGO/RIFTIF.h>

#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SPQRSupport>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace DPGO {
namespace {

using ColMajorSparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

template <typename Fn>
double elapsedMilliseconds(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::vector<InterfaceKey> sortedUnique(std::vector<InterfaceKey> keys) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

bool containsKey(const std::vector<InterfaceKey> &keys,
                 const InterfaceKey &key) {
  return std::binary_search(keys.begin(), keys.end(), key);
}

bool containsScope(const std::vector<InterfaceKey> &clique,
                   const std::vector<InterfaceKey> &scope) {
  for (const auto &key : scope) {
    if (!containsKey(clique, key)) {
      return false;
    }
  }
  return true;
}

std::vector<InterfaceKey> intersectKeys(const std::vector<InterfaceKey> &a,
                                        const std::vector<InterfaceKey> &b) {
  std::vector<InterfaceKey> result;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<InterfaceKey> subtractKeys(const std::vector<InterfaceKey> &keys,
                                       const std::vector<InterfaceKey> &drop) {
  std::vector<InterfaceKey> result;
  std::set_difference(keys.begin(), keys.end(), drop.begin(), drop.end(),
                      std::back_inserter(result));
  return result;
}

int keyOffset(const std::vector<InterfaceKey> &keys, const InterfaceKey &key,
              int blockDim) {
  const auto it = std::find(keys.begin(), keys.end(), key);
  if (it == keys.end() || !(*it == key)) {
    throw std::invalid_argument("RIFT-IF key not present in variable list");
  }
  return static_cast<int>(std::distance(keys.begin(), it)) * blockDim;
}

std::map<InterfaceKey, int> buildColumnOffsets(
    const std::vector<InterfaceKey> &keys, int blockDim) {
  std::map<InterfaceKey, int> offsets;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    offsets[keys[i]] = static_cast<int>(i) * blockDim;
  }
  return offsets;
}

Matrix vectorAsRhsMatrix(const Vector &b) {
  Matrix B(b.rows(), 1);
  B.col(0) = b;
  return B;
}

void appendExpandedRows(const std::vector<InterfaceKey> &targetKeys,
                        int blockDim, int rhsDim,
                        const std::vector<InterfaceKey> &factorKeys,
                        const Matrix &factorA, const Matrix &factorB,
                        Matrix *A, Matrix *B, int *rowOffset) {
  if (factorB.cols() != rhsDim) {
    throw std::invalid_argument("RIFT-IF factor RHS dimension mismatch");
  }
  if (factorA.rows() != factorB.rows()) {
    throw std::invalid_argument("RIFT-IF factor A/B row mismatch");
  }
  if (factorA.cols() != static_cast<int>(factorKeys.size()) * blockDim) {
    throw std::invalid_argument("RIFT-IF factor A column mismatch");
  }
  const int rows = factorA.rows();
  for (std::size_t k = 0; k < factorKeys.size(); ++k) {
    const int dst = keyOffset(targetKeys, factorKeys[k], blockDim);
    const int src = static_cast<int>(k) * blockDim;
    A->block(*rowOffset, dst, rows, blockDim) =
        factorA.block(0, src, rows, blockDim);
  }
  B->block(*rowOffset, 0, rows, rhsDim) = factorB;
  *rowOffset += rows;
}

std::pair<Matrix, Matrix> stackFactorsForClique(
    const std::vector<InterfaceKey> &cliqueKeys, int blockDim, int rhsDim,
    const std::vector<const InterfaceFactor *> &factors,
    const std::vector<const RIFTMessage *> &messages) {
  int rows = 0;
  for (const auto *factor : factors) {
    rows += factor->A.rows();
  }
  for (const auto *message : messages) {
    rows += message->R.rows();
  }

  Matrix A = Matrix::Zero(rows,
                          static_cast<int>(cliqueKeys.size()) * blockDim);
  Matrix B = Matrix::Zero(rows, rhsDim);
  int rowOffset = 0;
  for (const auto *factor : factors) {
    appendExpandedRows(cliqueKeys, blockDim, rhsDim, factor->scope, factor->A,
                       factor->B, &A, &B, &rowOffset);
  }
  for (const auto *message : messages) {
    appendExpandedRows(cliqueKeys, blockDim, rhsDim, message->separator_keys,
                       message->R, message->D, &A, &B, &rowOffset);
  }
  return {A, B};
}

void stackMultiRhsFactorsSparseSplit(
    const std::vector<const InterfaceFactor *> &factors,
    const std::vector<InterfaceKey> &interiorKeys,
    const std::vector<InterfaceKey> &boundaryKeys, int blockDim, int rhsDim,
    ColMajorSparseMatrix *AI, Matrix *boundaryAndRhs) {
  if (blockDim <= 0 || rhsDim <= 0) {
    throw std::invalid_argument(
        "RIFT-IF sparse multi-RHS stack dimensions must be positive");
  }
  const auto interiorOffsets = buildColumnOffsets(interiorKeys, blockDim);
  const auto boundaryOffsets = buildColumnOffsets(boundaryKeys, blockDim);
  int rows = 0;
  std::size_t tripletReserve = 0;
  for (const auto *factor : factors) {
    if (factor->A.rows() != factor->B.rows()) {
      throw std::invalid_argument("RIFT-IF factor A/B row mismatch");
    }
    if (factor->B.cols() != rhsDim) {
      throw std::invalid_argument("RIFT-IF factor RHS dimension mismatch");
    }
    if (factor->A.cols() !=
        static_cast<int>(factor->scope.size()) * blockDim) {
      throw std::invalid_argument("RIFT-IF factor column mismatch");
    }
    rows += factor->A.rows();
    tripletReserve += static_cast<std::size_t>(factor->A.rows()) *
                      static_cast<std::size_t>(factor->A.cols());
  }

  const int boundaryDim = static_cast<int>(boundaryKeys.size()) * blockDim;
  boundaryAndRhs->setZero(rows, boundaryDim + rhsDim);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(tripletReserve);

  int rowOffset = 0;
  for (const auto *factor : factors) {
    for (std::size_t keyIndex = 0; keyIndex < factor->scope.size();
         ++keyIndex) {
      const InterfaceKey key = factor->scope[keyIndex];
      const auto interiorOffset = interiorOffsets.find(key);
      const auto boundaryOffset = boundaryOffsets.find(key);
      if (interiorOffset == interiorOffsets.end() &&
          boundaryOffset == boundaryOffsets.end()) {
        throw std::invalid_argument(
            "RIFT-IF factor references a key outside condensation split");
      }
      for (int r = 0; r < factor->A.rows(); ++r) {
        for (int c = 0; c < blockDim; ++c) {
          const double value =
              factor->A(r, static_cast<int>(keyIndex) * blockDim + c);
          if (value == 0.0) {
            continue;
          }
          if (interiorOffset != interiorOffsets.end()) {
            triplets.emplace_back(rowOffset + r, interiorOffset->second + c,
                                  value);
          } else {
            (*boundaryAndRhs)(rowOffset + r, boundaryOffset->second + c) +=
                value;
          }
        }
      }
    }
    boundaryAndRhs->block(rowOffset, boundaryDim, factor->B.rows(), rhsDim) =
        factor->B;
    rowOffset += factor->A.rows();
  }

  AI->resize(rows, static_cast<int>(interiorKeys.size()) * blockDim);
  AI->setFromTriplets(triplets.begin(), triplets.end());
  AI->makeCompressed();
}

std::pair<Matrix, Matrix> compressRowsByQR(const Matrix &A,
                                           const Matrix &B) {
  if (A.cols() == 0) {
    return {Matrix::Zero(0, 0), Matrix::Zero(0, B.cols())};
  }
  if (A.rows() == 0) {
    return {Matrix::Zero(0, A.cols()), Matrix::Zero(0, B.cols())};
  }
  Eigen::HouseholderQR<Matrix> qr(A);
  const Matrix transformedB = qr.householderQ().transpose() * B;
  const Matrix Rfull = qr.matrixQR()
                           .topLeftCorner(A.rows(), A.cols())
                           .template triangularView<Eigen::Upper>();
  const int rows = std::min(A.rows(), A.cols());
  return {Rfull.topRows(rows), transformedB.topRows(rows)};
}

RIFTMessage computeDirectedMessage(
    const InterfaceProblem &problem, const InterfaceCliqueTree &tree,
    const std::map<RIFTFactorId, InterfaceFactor> &factorById,
    RIFTCliqueId src, RIFTCliqueId dst,
    const std::map<DirectedCliqueEdge, RIFTMessage> &messageByEdge) {
  const InterfaceClique &srcClique = tree.cliques.at(src);
  const InterfaceClique &dstClique = tree.cliques.at(dst);
  const std::vector<InterfaceKey> separator =
      intersectKeys(srcClique.variables, dstClique.variables);
  const std::vector<InterfaceKey> eliminated =
      subtractKeys(srcClique.variables, separator);
  std::vector<InterfaceKey> orderedKeys = eliminated;
  orderedKeys.insert(orderedKeys.end(), separator.begin(), separator.end());

  std::vector<const InterfaceFactor *> factors;
  factors.reserve(srcClique.assigned_factors.size());
  for (const RIFTFactorId id : srcClique.assigned_factors) {
    factors.push_back(&factorById.at(id));
  }

  std::vector<const RIFTMessage *> incoming;
  for (const RIFTCliqueId neighbor : srcClique.neighbors) {
    if (neighbor == dst) {
      continue;
    }
    const auto it = messageByEdge.find(DirectedCliqueEdge{neighbor, src});
    if (it == messageByEdge.end()) {
      throw std::runtime_error("RIFT-IF directed message missing dependency");
    }
    incoming.push_back(&it->second);
  }

  const auto stacked = stackFactorsForClique(orderedKeys, problem.block_dim,
                                             problem.rhs_dim, factors,
                                             incoming);
  const Matrix &A = stacked.first;
  const Matrix &B = stacked.second;
  const int elimDim =
      static_cast<int>(eliminated.size()) * problem.block_dim;
  const int sepDim = static_cast<int>(separator.size()) * problem.block_dim;

  Matrix residualA;
  Matrix residualB;
  if (elimDim > 0 && A.rows() > 0) {
    const Matrix AE = A.leftCols(elimDim);
    Matrix rest(A.rows(), sepDim + problem.rhs_dim);
    if (sepDim > 0) {
      rest.leftCols(sepDim) = A.middleCols(elimDim, sepDim);
    }
    rest.rightCols(problem.rhs_dim) = B;
    Eigen::ColPivHouseholderQR<Matrix> qr(AE);
    const Matrix transformed = qr.householderQ().transpose() * rest;
    const int totalRows = static_cast<int>(A.rows());
    const int start = std::min(static_cast<int>(qr.rank()), totalRows);
    if (sepDim == 0) {
      residualA = Matrix::Zero(totalRows - start, 0);
    } else {
      residualA = transformed.block(start, 0, totalRows - start, sepDim);
    }
    residualB = transformed.block(start, sepDim, totalRows - start,
                                  problem.rhs_dim);
  } else {
    if (sepDim == 0) {
      residualA = Matrix::Zero(A.rows(), 0);
    } else {
      residualA = A.rightCols(sepDim);
    }
    residualB = B;
  }

  const auto compressed = compressRowsByQR(residualA, residualB);

  RIFTMessage msg;
  msg.header.stage = problem.stage;
  msg.header.src = src;
  msg.header.dst = dst;
  for (const auto *factor : factors) {
    msg.header.graph_epoch =
        std::max(msg.header.graph_epoch, factor->graph_epoch);
  }
  for (const auto *incoming_message : incoming) {
    msg.header.graph_epoch =
        std::max(msg.header.graph_epoch,
                 incoming_message->header.graph_epoch);
  }
  msg.separator_keys = separator;
  msg.R = compressed.first;
  msg.D = compressed.second;
  return msg;
}

std::size_t messageBytes(const RIFTMessage &message) {
  return sizeof(int) * 4u +
         message.separator_keys.size() * 2u * sizeof(int) +
         static_cast<std::size_t>(message.R.rows()) *
             static_cast<std::size_t>(message.R.cols()) * sizeof(double) +
         static_cast<std::size_t>(message.D.rows()) *
             static_cast<std::size_t>(message.D.cols()) * sizeof(double);
}

double interfaceResidualNorm(const InterfaceProblem &problem,
                             const Matrix &solution) {
  if (solution.rows() !=
          static_cast<int>(problem.variables.size()) * problem.block_dim ||
      solution.cols() != problem.rhs_dim) {
    throw std::invalid_argument("RIFT-IF solution matrix dimension mismatch");
  }
  double squared = 0.0;
  for (const auto &factor : problem.factors) {
    Matrix X(factor.A.cols(), problem.rhs_dim);
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int globalOffset =
          keyOffset(problem.variables, factor.scope[k], problem.block_dim);
      const int localOffset = static_cast<int>(k) * problem.block_dim;
      X.block(localOffset, 0, problem.block_dim, problem.rhs_dim) =
          solution.block(globalOffset, 0, problem.block_dim, problem.rhs_dim);
    }
    const Matrix residual = factor.A * X - factor.B;
    squared += residual.squaredNorm();
  }
  return std::sqrt(squared);
}

struct CompiledCAKFactor {
  const InterfaceFactor *factor = nullptr;
  std::vector<int> global_offsets;
  std::vector<int> local_offsets;
  bool use_normal = false;
  bool contiguous_scope = false;
  int contiguous_global_offset = -1;
  Matrix normal;
};

struct CAKNormalApplyScratch {
  Matrix local_x;
  Matrix factor_times_x;
  Matrix contribution;
  bool has_workspace = false;
};

std::vector<CompiledCAKFactor> compileCAKFactors(
    const InterfaceProblem &problem, bool precomputeSmallNormals = false) {
  const auto offsets = buildColumnOffsets(problem.variables, problem.block_dim);
  std::vector<CompiledCAKFactor> compiled;
  compiled.reserve(problem.factors.size());
  std::vector<std::size_t> normalJobs;
  std::size_t normalWorkEstimate = 0;
  for (const auto &factor : problem.factors) {
    CompiledCAKFactor item;
    item.factor = &factor;
    item.global_offsets.reserve(factor.scope.size());
    item.local_offsets.reserve(factor.scope.size());
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const auto global = offsets.find(factor.scope[k]);
      if (global == offsets.end()) {
        throw std::invalid_argument(
            "RIFT-CAK factor references a key outside problem variables");
      }
      item.global_offsets.push_back(global->second);
      item.local_offsets.push_back(static_cast<int>(k) * problem.block_dim);
    }
    if (!item.global_offsets.empty()) {
      item.contiguous_scope = true;
      item.contiguous_global_offset = item.global_offsets.front();
      for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
        if (item.global_offsets[k] !=
            item.contiguous_global_offset + item.local_offsets[k]) {
          item.contiguous_scope = false;
          item.contiguous_global_offset = -1;
          break;
        }
      }
    }
    const int cols = factor.A.cols();
    const int rows = factor.A.rows();
    const std::size_t normalEntries =
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols);
    constexpr int kPrecomputedNormalMaxCols = 1024;
    constexpr std::size_t kPrecomputedNormalMaxEntries = 1024u * 1024u;
    if (precomputeSmallNormals && cols <= kPrecomputedNormalMaxCols &&
        normalEntries <= kPrecomputedNormalMaxEntries &&
        cols <= 2 * rows) {
      item.use_normal = true;
      normalJobs.push_back(compiled.size());
      normalWorkEstimate +=
          static_cast<std::size_t>(rows) * normalEntries;
    }
    compiled.push_back(std::move(item));
  }
  auto computeNormal = [](const InterfaceFactor *factor) {
    Matrix normal(factor->A.cols(), factor->A.cols());
    normal.setZero();
    normal.selfadjointView<Eigen::Lower>().rankUpdate(factor->A.transpose());
    normal.triangularView<Eigen::StrictlyUpper>() =
        normal.transpose().triangularView<Eigen::StrictlyUpper>();
    return normal;
  };
  constexpr std::size_t kParallelNormalWorkThreshold = 32ull * 1024ull * 1024ull;
  if (normalJobs.size() > 1 &&
      normalWorkEstimate >= kParallelNormalWorkThreshold) {
    std::vector<std::future<std::pair<std::size_t, Matrix>>> futures;
    futures.reserve(normalJobs.size());
    for (const std::size_t index : normalJobs) {
      const InterfaceFactor *factor = compiled[index].factor;
      futures.push_back(std::async(std::launch::async,
                                   [index, factor, computeNormal]() {
        return std::make_pair(index, computeNormal(factor));
      }));
    }
    for (auto &future : futures) {
      auto result = future.get();
      compiled[result.first].normal = std::move(result.second);
    }
  } else {
    for (const std::size_t index : normalJobs) {
      compiled[index].normal = computeNormal(compiled[index].factor);
    }
  }
  return compiled;
}

void applyNormalOperatorCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled, const Matrix &X,
    Matrix *Hx, CAKNormalApplyScratch *scratch,
    RIFTStats *stats = nullptr) {
  if (X.rows() !=
      static_cast<int>(problem.variables.size()) * problem.block_dim) {
    throw std::invalid_argument("RIFT-CAK HApply input row mismatch");
  }
  if (Hx == nullptr || scratch == nullptr) {
    throw std::invalid_argument("RIFT-CAK HApply requires output and scratch");
  }
  if (stats != nullptr) {
    ++stats->cak_happly_calls;
    if (scratch->has_workspace) {
      ++stats->cak_happly_workspace_reuses;
    }
  }
  scratch->has_workspace = true;
  Hx->resize(X.rows(), X.cols());
  Hx->setZero();
  for (const auto &item : compiled) {
    const InterfaceFactor &factor = *item.factor;
    const bool useDirectBlockFastPath =
        item.contiguous_scope && factor.A.cols() <= 2 * problem.block_dim;
    if (useDirectBlockFastPath) {
      const auto localXBlock =
          X.middleRows(item.contiguous_global_offset, factor.A.cols());
      scratch->contribution.resize(factor.A.cols(), X.cols());
      if (item.use_normal) {
        scratch->contribution.noalias() = item.normal * localXBlock;
      } else {
        scratch->factor_times_x.resize(factor.A.rows(), X.cols());
        scratch->factor_times_x.noalias() = factor.A * localXBlock;
        scratch->contribution.noalias() =
            factor.A.transpose() * scratch->factor_times_x;
      }
      Hx->middleRows(item.contiguous_global_offset, factor.A.cols()) +=
          scratch->contribution;
      continue;
    }
    scratch->local_x.resize(factor.A.cols(), X.cols());
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      scratch->local_x.block(item.local_offsets[k], 0, problem.block_dim,
                             X.cols()) =
          X.block(item.global_offsets[k], 0, problem.block_dim, X.cols());
    }
    scratch->contribution.resize(factor.A.cols(), X.cols());
    if (item.use_normal) {
      scratch->contribution.noalias() = item.normal * scratch->local_x;
    } else {
      scratch->factor_times_x.resize(factor.A.rows(), X.cols());
      scratch->factor_times_x.noalias() = factor.A * scratch->local_x;
      scratch->contribution.noalias() =
          factor.A.transpose() * scratch->factor_times_x;
    }
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      Hx->block(item.global_offsets[k], 0, problem.block_dim, X.cols()) +=
          scratch->contribution.block(item.local_offsets[k], 0,
                                      problem.block_dim, X.cols());
    }
  }
}

Matrix applyNormalOperatorCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled, const Matrix &X) {
  Matrix Hx;
  CAKNormalApplyScratch scratch;
  applyNormalOperatorCompiled(problem, compiled, X, &Hx, &scratch);
  return Hx;
}

Matrix normalEquationRhsCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled) {
  Matrix rhs = Matrix::Zero(
      static_cast<int>(problem.variables.size()) * problem.block_dim,
      problem.rhs_dim);
  for (const auto &item : compiled) {
    const InterfaceFactor &factor = *item.factor;
    Matrix contribution = factor.A.transpose() * factor.B;
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      rhs.block(item.global_offsets[k], 0, problem.block_dim,
                problem.rhs_dim) +=
          contribution.block(item.local_offsets[k], 0, problem.block_dim,
                             problem.rhs_dim);
    }
  }
  return rhs;
}

Vector normalEquationDiagonalCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled) {
  Vector diagonal = Vector::Zero(
      static_cast<int>(problem.variables.size()) * problem.block_dim);
  for (const auto &item : compiled) {
    const InterfaceFactor &factor = *item.factor;
    Vector localDiagonal;
    if (item.use_normal) {
      localDiagonal = item.normal.diagonal();
    } else {
      localDiagonal = factor.A.array().square().colwise().sum().transpose();
    }
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      for (int c = 0; c < problem.block_dim; ++c) {
        diagonal(item.global_offsets[k] + c) +=
            localDiagonal(item.local_offsets[k] + c);
      }
    }
  }
  return diagonal;
}

std::vector<Matrix> normalEquationBlockDiagonalCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled) {
  if (problem.block_dim <= 0) {
    throw std::invalid_argument(
        "RIFT-CAK block diagonal requires positive block dimension");
  }
  std::vector<Matrix> blocks(
      problem.variables.size(),
      Matrix::Zero(problem.block_dim, problem.block_dim));
  for (const auto &item : compiled) {
    const InterfaceFactor &factor = *item.factor;
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      const int blockIndex = item.global_offsets[k] / problem.block_dim;
      if (blockIndex < 0 ||
          blockIndex >= static_cast<int>(blocks.size())) {
        throw std::invalid_argument(
            "RIFT-CAK block diagonal offset outside variables");
      }
      if (item.use_normal) {
        blocks[static_cast<std::size_t>(blockIndex)] +=
            item.normal.block(item.local_offsets[k], item.local_offsets[k],
                              problem.block_dim, problem.block_dim);
      } else {
        const Matrix localBlock =
            factor.A.block(0, item.local_offsets[k], factor.A.rows(),
                           problem.block_dim);
        blocks[static_cast<std::size_t>(blockIndex)].noalias() +=
            localBlock.transpose() * localBlock;
      }
    }
  }
  return blocks;
}

Matrix applyJacobiPreconditioner(const Vector &diagonal, const Matrix &R) {
  if (R.rows() != diagonal.rows()) {
    throw std::invalid_argument("RIFT-CAK Jacobi preconditioner row mismatch");
  }
  Matrix Z = R;
  const double maxAbsDiagonal =
      diagonal.size() == 0 ? 0.0 : diagonal.cwiseAbs().maxCoeff();
  const double floor = std::max(1e-18, 1e-15 * maxAbsDiagonal);
  for (int r = 0; r < R.rows(); ++r) {
    const double value = diagonal(r);
    if (std::abs(value) <= floor) {
      continue;
    }
    Z.row(r) /= value;
  }
  return Z;
}

Matrix applyBlockJacobiPreconditioner(
    const std::vector<Matrix> &blocks, const Vector &diagonal,
    const Matrix &R, int blockDim) {
  if (blockDim <= 0) {
    throw std::invalid_argument(
        "RIFT-CAK block Jacobi requires positive block dimension");
  }
  if (R.rows() != diagonal.rows() ||
      R.rows() != static_cast<int>(blocks.size()) * blockDim) {
    throw std::invalid_argument(
        "RIFT-CAK block Jacobi preconditioner row mismatch");
  }
  Matrix Z = Matrix::Zero(R.rows(), R.cols());
  const double maxAbsDiagonal =
      diagonal.size() == 0 ? 0.0 : diagonal.cwiseAbs().maxCoeff();
  const double scalarFloor = std::max(1e-18, 1e-15 * maxAbsDiagonal);
  for (std::size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
    const int row = static_cast<int>(blockIndex) * blockDim;
    const Matrix symmetricBlock =
        0.5 * (blocks[blockIndex] + blocks[blockIndex].transpose());
    const double blockScale = symmetricBlock.size() == 0
                                  ? 0.0
                                  : symmetricBlock.cwiseAbs().maxCoeff();
    const double blockFloor = std::max(1e-18, 1e-15 * blockScale);
    bool solvedBlock = false;
    if (blockScale > blockFloor) {
      Eigen::LDLT<Matrix> ldlt(symmetricBlock);
      bool positive = ldlt.info() == Eigen::Success;
      if (positive) {
        const auto d = ldlt.vectorD();
        for (int i = 0; i < d.size(); ++i) {
          if (!(d(i) > blockFloor) || !std::isfinite(d(i))) {
            positive = false;
            break;
          }
        }
      }
      if (positive) {
        const Matrix solved =
            ldlt.solve(R.block(row, 0, blockDim, R.cols()));
        if (solved.allFinite()) {
          Z.block(row, 0, blockDim, R.cols()) = solved;
          solvedBlock = true;
        }
      }
    }
    if (!solvedBlock) {
      for (int c = 0; c < blockDim; ++c) {
        const int scalarRow = row + c;
        Z.row(scalarRow) = R.row(scalarRow);
        const double value = diagonal(scalarRow);
        if (std::abs(value) > scalarFloor) {
          Z.row(scalarRow) /= value;
        }
      }
    }
  }
  return Z;
}

double interfaceResidualNormCompiled(
    const InterfaceProblem &problem,
    const std::vector<CompiledCAKFactor> &compiled, const Matrix &solution) {
  if (solution.rows() !=
          static_cast<int>(problem.variables.size()) * problem.block_dim ||
      solution.cols() != problem.rhs_dim) {
    throw std::invalid_argument("RIFT-IF solution matrix dimension mismatch");
  }
  double squared = 0.0;
  for (const auto &item : compiled) {
    const InterfaceFactor &factor = *item.factor;
    Matrix X(factor.A.cols(), problem.rhs_dim);
    for (std::size_t k = 0; k < item.global_offsets.size(); ++k) {
      X.block(item.local_offsets[k], 0, problem.block_dim, problem.rhs_dim) =
          solution.block(item.global_offsets[k], 0, problem.block_dim,
                         problem.rhs_dim);
    }
    const Matrix residual = factor.A * X - factor.B;
    squared += residual.squaredNorm();
  }
  return std::sqrt(squared);
}

std::vector<int> interfaceOwnerRobots(const InterfaceProblem &problem) {
  std::set<int> owners;
  for (const auto &factor : problem.factors) {
    if (factor.owner_robot >= 0) {
      owners.insert(factor.owner_robot);
    }
  }
  if (owners.empty()) {
    owners.insert(0);
  }
  return std::vector<int>(owners.begin(), owners.end());
}

void recordTreeScalarReduction(const std::vector<int> &robots,
                               RIFTStats *stats,
                               DecentralizationGuard *guard) {
  if (stats != nullptr) {
    ++stats->cak_scalar_reductions;
  }
  if (robots.size() < 2) {
    return;
  }
  const std::size_t bytes_per_message = sizeof(double);
  for (std::size_t i = 0; i + 1 < robots.size(); ++i) {
    const int src = robots[i + 1];
    const int dst = robots[i];
    if (stats != nullptr) {
      stats->cak_scalar_reduction_bytes += bytes_per_message;
      stats->actual_message_bytes += bytes_per_message;
      ++stats->directed_messages_sent;
    }
    if (guard != nullptr) {
      guard->RecordMessage(src, dst, bytes_per_message);
    }
  }
  for (std::size_t i = 0; i + 1 < robots.size(); ++i) {
    const int src = robots[i];
    const int dst = robots[i + 1];
    if (stats != nullptr) {
      stats->cak_scalar_reduction_bytes += bytes_per_message;
      stats->actual_message_bytes += bytes_per_message;
      ++stats->directed_messages_sent;
    }
    if (guard != nullptr) {
      guard->RecordMessage(src, dst, bytes_per_message);
    }
  }
}

template <typename DerivedA, typename DerivedB>
double treeReducedDot(const Eigen::MatrixBase<DerivedA> &A,
                      const Eigen::MatrixBase<DerivedB> &B,
                      const std::vector<int> &robots, RIFTStats *stats,
                      DecentralizationGuard *guard) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::invalid_argument("RIFT-CAK dot dimensions mismatch");
  }
  recordTreeScalarReduction(robots, stats, guard);
  return A.cwiseProduct(B).sum();
}

std::vector<int> asyncRobotIds(const InterfaceProblem &problem) {
  std::set<int> robots;
  for (const auto &key : problem.variables) {
    if (key.robot_id >= 0) {
      robots.insert(key.robot_id);
    }
  }
  for (const auto &factor : problem.factors) {
    if (factor.owner_robot >= 0) {
      robots.insert(factor.owner_robot);
    }
  }
  if (robots.empty()) {
    robots.insert(0);
  }
  return std::vector<int>(robots.begin(), robots.end());
}

bool factorInsideRobots(const InterfaceFactor &factor,
                        const std::set<int> &robots) {
  if (factor.scope.empty()) {
    return factor.owner_robot < 0 || robots.count(factor.owner_robot) != 0;
  }
  for (const auto &key : factor.scope) {
    if (robots.count(key.robot_id) == 0) {
      return false;
    }
  }
  return true;
}

std::size_t asyncBlockMessageBytes(std::size_t num_keys, int block_dim,
                                   int rhs_dim, bool include_diagonal) {
  std::size_t bytes = sizeof(int) * 4u + sizeof(std::uint64_t) +
                      num_keys * 2u * sizeof(int) +
                      num_keys * static_cast<std::size_t>(block_dim) *
                          static_cast<std::size_t>(rhs_dim) * sizeof(double);
  if (include_diagonal) {
    bytes += num_keys * static_cast<std::size_t>(block_dim) *
             static_cast<std::size_t>(block_dim) * sizeof(double);
  }
  return bytes;
}

std::size_t estimatedMessageBytesForSeparator(std::size_t separatorBlocks,
                                              int blockDim, int rhsDim) {
  const std::size_t sepDim = separatorBlocks * static_cast<std::size_t>(blockDim);
  return sizeof(int) * 4u + separatorBlocks * 2u * sizeof(int) +
         sepDim * sepDim * sizeof(double) +
         sepDim * static_cast<std::size_t>(rhsDim) * sizeof(double);
}

void PopulateCliqueTreeStats(const InterfaceCliqueTree &tree,
                             RIFTStats *stats) {
  if (stats == nullptr) {
    return;
  }
  stats->num_cliques = static_cast<int>(tree.cliques.size());
  stats->num_tree_edges = static_cast<int>(tree.edges.size());
  std::map<int, int> hostedCliqueCounts;
  for (const auto &clique : tree.cliques) {
    stats->max_clique_blocks =
        std::max(stats->max_clique_blocks,
                 static_cast<int>(clique.variables.size()));
    if (clique.host_robot >= 0) {
      stats->max_host_clique_load =
          std::max(stats->max_host_clique_load,
                   ++hostedCliqueCounts[clique.host_robot]);
    }
  }
  stats->num_host_robots = static_cast<int>(hostedCliqueCounts.size());
  for (const auto &edge : tree.edges) {
    stats->max_separator_blocks =
        std::max(stats->max_separator_blocks,
                 static_cast<int>(edge.separator.size()));
    stats->estimated_message_bytes += 2u * edge.estimated_message_bytes;
    const int hostA = tree.cliques.at(edge.a).host_robot;
    const int hostB = tree.cliques.at(edge.b).host_robot;
    const int routeHops = hostA == hostB ? 0 : 1;
    if (routeHops > 0) {
      ++stats->cross_host_tree_edges;
      stats->estimated_route_hops += 2 * routeHops;
      stats->estimated_routed_message_bytes +=
          2u * edge.estimated_message_bytes *
          static_cast<std::size_t>(routeHops);
    }
  }
}

bool ExactSymbolicGateAllows(const InterfaceProblem &problem,
                             const RIFTParams &params,
                             const RIFTStats &stats) {
  const int maxSeparatorAllowed =
      problem.dimension == 2 ? params.exact_max_separator_blocks_2d
                             : params.exact_max_separator_blocks_3d;
  return stats.max_separator_blocks <= maxSeparatorAllowed &&
         stats.estimated_message_bytes <= params.exact_max_message_bytes;
}

bool AutoExactPreflightAllows(const InterfaceProblem &problem,
                              const RIFTParams &params) {
  const int n = static_cast<int>(problem.variables.size());
  if (n == 0) {
    return true;
  }
  const int maxSeparatorAllowed =
      problem.dimension == 2 ? params.exact_max_separator_blocks_2d
                             : params.exact_max_separator_blocks_3d;
  for (const auto &factor : problem.factors) {
    const int factorSeparatorBlocks =
        std::max(0, static_cast<int>(factor.scope.size()) - 1);
    if (factorSeparatorBlocks > maxSeparatorAllowed) {
      return false;
    }
    if (factorSeparatorBlocks > 0 &&
        2u * estimatedMessageBytesForSeparator(
                 factorSeparatorBlocks, problem.block_dim, problem.rhs_dim) >
            params.exact_max_message_bytes) {
      return false;
    }
  }

  std::map<InterfaceKey, int> indexOf;
  for (int i = 0; i < n; ++i) {
    indexOf[problem.variables[i]] = i;
  }

  std::vector<std::set<int>> adjacency(n);
  for (const auto &factor : problem.factors) {
    std::vector<int> scope;
    scope.reserve(factor.scope.size());
    for (const auto &key : factor.scope) {
      scope.push_back(indexOf.at(key));
    }
    for (std::size_t i = 0; i < scope.size(); ++i) {
      for (std::size_t j = i + 1; j < scope.size(); ++j) {
        adjacency[scope[i]].insert(scope[j]);
        adjacency[scope[j]].insert(scope[i]);
      }
    }
  }

  std::vector<bool> active(n, true);
  RIFTStats preflightStats;
  for (int step = 0; step < n; ++step) {
    int best = -1;
    int bestDegree = std::numeric_limits<int>::max();
    for (int v = 0; v < n; ++v) {
      if (!active[v]) {
        continue;
      }
      int degree = 0;
      for (const int nb : adjacency[v]) {
        if (active[nb]) {
          ++degree;
        }
      }
      if (degree < bestDegree || (degree == bestDegree && v < best)) {
        best = v;
        bestDegree = degree;
      }
    }
    if (best < 0) {
      break;
    }

    std::vector<int> clique = {best};
    for (const int nb : adjacency[best]) {
      if (active[nb]) {
        clique.push_back(nb);
      }
    }
    preflightStats.max_clique_blocks = std::max(
        preflightStats.max_clique_blocks, static_cast<int>(clique.size()));
    const int separatorBlocks = std::max(0, static_cast<int>(clique.size()) - 1);
    preflightStats.max_separator_blocks =
        std::max(preflightStats.max_separator_blocks, separatorBlocks);
    if (separatorBlocks > 0) {
      preflightStats.estimated_message_bytes +=
          2u * estimatedMessageBytesForSeparator(
                   separatorBlocks, problem.block_dim, problem.rhs_dim);
    }
    if (!ExactSymbolicGateAllows(problem, params, preflightStats)) {
      return false;
    }

    for (std::size_t i = 1; i < clique.size(); ++i) {
      for (std::size_t j = i + 1; j < clique.size(); ++j) {
        adjacency[clique[i]].insert(clique[j]);
        adjacency[clique[j]].insert(clique[i]);
      }
    }
    active[best] = false;
  }
  return true;
}

void CopySymbolicStatsInto(const RIFTStats &symbolicStats,
                           RIFTStats *target) {
  if (target == nullptr) {
    return;
  }
  target->num_cliques = symbolicStats.num_cliques;
  target->num_tree_edges = symbolicStats.num_tree_edges;
  target->num_host_robots = symbolicStats.num_host_robots;
  target->max_host_clique_load = symbolicStats.max_host_clique_load;
  target->cross_host_tree_edges = symbolicStats.cross_host_tree_edges;
  target->estimated_route_hops = symbolicStats.estimated_route_hops;
  target->max_clique_blocks = symbolicStats.max_clique_blocks;
  target->max_separator_blocks = symbolicStats.max_separator_blocks;
  target->estimated_message_bytes = symbolicStats.estimated_message_bytes;
  target->estimated_routed_message_bytes =
      symbolicStats.estimated_routed_message_bytes;
}

}  // namespace

void DecentralizationGuard::RecordGlobalInterfaceMatrixConstruction() {
  used_global_matrix_ = true;
}

void DecentralizationGuard::RecordDirectInterfaceSolverCall() {
  used_direct_solver_ = true;
}

void DecentralizationGuard::RecordCollectiveCall(const std::string &) {
  used_collective_ = true;
}

void DecentralizationGuard::RecordMessage(int, int, std::size_t bytes) {
  ++message_count_;
  message_bytes_ += bytes;
}

void DecentralizationGuard::RecordFactorTransfer(int, int dst,
                                                 RIFTFactorId id) {
  factors_by_receiver_[dst].insert(id);
}

void DecentralizationGuard::AssertNoGlobalInterfaceMatrixConstructed() const {
  if (used_global_matrix_) {
    throw std::runtime_error("RIFT-IF guard: global interface matrix used");
  }
}

void DecentralizationGuard::AssertNoDirectSolverCalled() const {
  if (used_direct_solver_) {
    throw std::runtime_error("RIFT-IF guard: direct interface solver used");
  }
}

void DecentralizationGuard::AssertNoCollectiveCommunication() const {
  if (used_collective_) {
    throw std::runtime_error("RIFT-IF guard: collective communication used");
  }
}

void DecentralizationGuard::AssertNoNodeReceivedAllFactors(
    int num_total_factors) const {
  if (num_total_factors <= 0) {
    return;
  }
  for (const auto &entry : factors_by_receiver_) {
    if (static_cast<int>(entry.second.size()) >= num_total_factors &&
        num_total_factors > 1) {
      throw std::runtime_error(
          "RIFT-IF guard: one node received all interface factors");
    }
  }
}

RIFTBlockSparseNormalOperator::RIFTBlockSparseNormalOperator(
    const InterfaceProblem &problem)
    : block_dim_(problem.block_dim),
      num_variables_(static_cast<int>(problem.variables.size())) {
  if (problem.block_dim <= 0) {
    throw std::invalid_argument(
        "RIFT block-sparse normal operator requires positive block dimension");
  }
  std::map<InterfaceKey, int> indexOf;
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    indexOf[problem.variables[i]] = static_cast<int>(i);
  }

  std::size_t expectedBlockContributions = 0;
  for (const auto &factor : problem.factors) {
    expectedBlockContributions +=
        factor.scope.size() * factor.scope.size();
  }
  entries_.reserve(expectedBlockContributions);
  std::unordered_map<std::uint64_t, std::size_t> entryByPair;
  entryByPair.reserve(expectedBlockContributions);
  auto packedBlockKey = [](int row, int col) {
    return (static_cast<std::uint64_t>(static_cast<unsigned int>(row)) << 32) |
           static_cast<unsigned int>(col);
  };
  for (const auto &factor : problem.factors) {
    if (factor.A.cols() !=
        static_cast<int>(factor.scope.size()) * problem.block_dim) {
      throw std::invalid_argument(
          "RIFT block-sparse normal operator factor column mismatch");
    }
    if (factor.A.rows() != factor.B.rows()) {
      throw std::invalid_argument(
          "RIFT block-sparse normal operator factor A/B row mismatch");
    }
    Matrix normal(factor.A.cols(), factor.A.cols());
    normal.setZero();
    normal.selfadjointView<Eigen::Lower>().rankUpdate(factor.A.transpose());
    normal.triangularView<Eigen::StrictlyUpper>() =
        normal.transpose().triangularView<Eigen::StrictlyUpper>();
    for (std::size_t rowLocal = 0; rowLocal < factor.scope.size();
         ++rowLocal) {
      const auto rowIt = indexOf.find(factor.scope[rowLocal]);
      if (rowIt == indexOf.end()) {
        throw std::invalid_argument(
            "RIFT block-sparse normal operator missing row key");
      }
      for (std::size_t colLocal = 0; colLocal < factor.scope.size();
           ++colLocal) {
        const auto colIt = indexOf.find(factor.scope[colLocal]);
        if (colIt == indexOf.end()) {
          throw std::invalid_argument(
              "RIFT block-sparse normal operator missing column key");
        }
        const std::uint64_t key = packedBlockKey(rowIt->second, colIt->second);
        auto indexIt = entryByPair.find(key);
        if (indexIt == entryByPair.end()) {
          BlockEntry blockEntry;
          blockEntry.row_block = rowIt->second;
          blockEntry.col_block = colIt->second;
          blockEntry.value = Matrix::Zero(block_dim_, block_dim_);
          indexIt =
              entryByPair.emplace(key, entries_.size()).first;
          entries_.push_back(std::move(blockEntry));
        } else {
          ++merged_block_contributions_;
        }
        entries_[indexIt->second].value += normal.block(
            static_cast<int>(rowLocal) * block_dim_,
            static_cast<int>(colLocal) * block_dim_, block_dim_, block_dim_);
      }
    }
  }
  std::sort(entries_.begin(), entries_.end(),
            [](const BlockEntry &lhs, const BlockEntry &rhs) {
              if (lhs.row_block != rhs.row_block) {
                return lhs.row_block < rhs.row_block;
              }
              return lhs.col_block < rhs.col_block;
            });
}

Matrix RIFTBlockSparseNormalOperator::Apply(const Matrix &X) const {
  if (X.rows() != num_variables_ * block_dim_) {
    throw std::invalid_argument(
        "RIFT block-sparse normal operator input row mismatch");
  }
  Matrix Hx = Matrix::Zero(X.rows(), X.cols());
  for (const auto &entry : entries_) {
    Hx.block(entry.row_block * block_dim_, 0, block_dim_, X.cols())
        .noalias() +=
        entry.value * X.block(entry.col_block * block_dim_, 0, block_dim_,
                              X.cols());
  }
  return Hx;
}

int RIFTBlockSparseNormalOperator::numBlockEntries() const {
  return static_cast<int>(entries_.size());
}

int RIFTBlockSparseNormalOperator::numMergedBlockContributions() const {
  return merged_block_contributions_;
}

InterfaceProblem InterfaceProblemBuilder::BuildFromInterfaceFactors(
    FactorType stage, int dimension, int block_dim, int rhs_dim,
    const std::vector<InterfaceFactor> &factors) {
  if (dimension <= 0 || block_dim <= 0 || rhs_dim <= 0) {
    throw std::invalid_argument("RIFT-IF problem dimensions must be positive");
  }
  InterfaceProblem problem;
  problem.stage = stage;
  problem.dimension = dimension;
  problem.block_dim = block_dim;
  problem.rhs_dim = rhs_dim;
  std::vector<InterfaceKey> variables;
  for (const auto &factor : factors) {
    if (factor.stage != stage) {
      throw std::invalid_argument("RIFT-IF factor stage mismatch");
    }
    if (factor.A.rows() != factor.B.rows()) {
      throw std::invalid_argument("RIFT-IF factor A/B row mismatch");
    }
    if (factor.B.cols() != rhs_dim) {
      throw std::invalid_argument("RIFT-IF factor RHS dimension mismatch");
    }
    if (factor.A.cols() !=
        static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(),
                     factor.scope.end());
    problem.factors.push_back(factor);
  }
  problem.variables = sortedUnique(std::move(variables));
  return problem;
}

InterfaceProblem InterfaceProblemBuilder::BuildFromTEDFactors(
    FactorType stage, int dimension, int block_dim,
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors,
    bool use_rotation_multi_rhs) {
  if (block_dim <= 0) {
    throw std::invalid_argument("RIFT-IF block dimension must be positive");
  }

  InterfaceProblem problem;
  problem.stage = stage;
  problem.dimension = dimension;
  problem.block_dim = block_dim;
  problem.rhs_dim = 1;
  if (stage == FactorType::ROTATION && use_rotation_multi_rhs) {
    throw std::invalid_argument(
        "RIFT-IF rotation multi-RHS must be built from row-wise factors");
  }

  RIFTFactorId nextId = 0;
  std::vector<InterfaceKey> variables;
  for (const auto &condensed : condensed_factors) {
    InterfaceFactor factor;
    factor.id = nextId++;
    factor.stage = stage;
    factor.scope = condensed.boundary_keys;
    factor.A = condensed.Abar;
    factor.B = vectorAsRhsMatrix(condensed.bbar);
    factor.owner_robot = condensed.owner_robot_id;
    factor.graph_epoch = condensed.epoch;
    if (factor.A.cols() != static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF condensed factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(), factor.scope.end());
    problem.factors.push_back(std::move(factor));
  }
  for (const auto &cross : cross_factors) {
    InterfaceFactor factor;
    factor.id = nextId++;
    factor.stage = stage;
    factor.scope = cross.keys;
    factor.A = cross.A;
    factor.B = vectorAsRhsMatrix(cross.b);
    factor.owner_robot =
        factor.scope.empty() ? -1 : factor.scope.front().robot_id;
    if (factor.A.cols() != static_cast<int>(factor.scope.size()) * block_dim) {
      throw std::invalid_argument("RIFT-IF cross factor column mismatch");
    }
    variables.insert(variables.end(), factor.scope.begin(), factor.scope.end());
    problem.factors.push_back(std::move(factor));
  }
  problem.variables = sortedUnique(std::move(variables));
  return problem;
}

InterfaceFactor BuildRotationMultiRHSFactorFromVectorized(
    RIFTFactorId id, const std::vector<PoseKey> &scope,
    const Matrix &vectorized_A, const Vector &vectorized_b, int dimension,
    int owner_robot, std::uint64_t graph_epoch) {
  if (dimension <= 0) {
    throw std::invalid_argument("RIFT-IF multi-RHS dimension must be positive");
  }
  if (vectorized_A.rows() != vectorized_b.rows()) {
    throw std::invalid_argument("RIFT-IF vectorized factor A/b row mismatch");
  }
  if (vectorized_A.rows() % dimension != 0) {
    throw std::invalid_argument(
        "RIFT-IF vectorized rotation rows are not divisible by dimension");
  }
  if (vectorized_A.cols() !=
      static_cast<int>(scope.size()) * dimension * dimension) {
    throw std::invalid_argument(
        "RIFT-IF vectorized rotation factor column mismatch");
  }

  const int rows = vectorized_A.rows() / dimension;
  InterfaceFactor factor;
  factor.id = id;
  factor.stage = FactorType::ROTATION;
  factor.scope = scope;
  factor.A =
      Matrix::Zero(rows, static_cast<int>(scope.size()) * dimension);
  factor.B = Matrix::Zero(rows, dimension);
  factor.owner_robot = owner_robot;
  factor.graph_epoch = graph_epoch;

  constexpr double kCoeffTol = 1e-8;
  for (int rhs = 0; rhs < dimension; ++rhs) {
    for (int row = 0; row < rows; ++row) {
      const int vecRow = rhs + row * dimension;
      factor.B(row, rhs) = vectorized_b(vecRow);
      for (std::size_t keyIndex = 0; keyIndex < scope.size(); ++keyIndex) {
        for (int col = 0; col < dimension; ++col) {
          const int multiCol =
              static_cast<int>(keyIndex) * dimension + col;
          const int vecCol = static_cast<int>(keyIndex) * dimension *
                                 dimension +
                             rhs + col * dimension;
          const double coeff = vectorized_A(vecRow, vecCol);
          for (int otherRhs = 0; otherRhs < dimension; ++otherRhs) {
            if (otherRhs == rhs) {
              continue;
            }
            const int offRhsVecCol = static_cast<int>(keyIndex) * dimension *
                                         dimension +
                                     otherRhs + col * dimension;
            const double offCoeff = vectorized_A(vecRow, offRhsVecCol);
            if (std::abs(offCoeff) > kCoeffTol) {
              throw std::invalid_argument(
                  "RIFT-IF vectorized rotation factor has off-row coupling");
            }
          }
          if (rhs == 0) {
            factor.A(row, multiCol) = coeff;
          } else if (std::abs(factor.A(row, multiCol) - coeff) >
                     kCoeffTol *
                         std::max(1.0, std::abs(factor.A(row, multiCol)))) {
            throw std::invalid_argument(
                "RIFT-IF vectorized rotation factor is not row-separable");
          }
        }
      }
    }
  }
  return factor;
}

InterfaceFactor CondenseMultiRHSFactors(
    RIFTFactorId id, FactorType stage, int dimension,
    const std::vector<InterfaceFactor> &factors,
    const std::vector<PoseKey> &interior_keys,
    const std::vector<PoseKey> &boundary_keys, int owner_robot,
    std::uint64_t graph_epoch, MultiRHSBackSubstitutionCache *cache) {
  if (dimension <= 0) {
    throw std::invalid_argument("RIFT-IF multi-RHS dimension must be positive");
  }
  const std::vector<InterfaceKey> interior = sortedUnique(interior_keys);
  const std::vector<InterfaceKey> boundary = sortedUnique(boundary_keys);

  std::vector<const InterfaceFactor *> factorPtrs;
  factorPtrs.reserve(factors.size());
  for (const auto &factor : factors) {
    if (factor.stage != stage) {
      throw std::invalid_argument("RIFT-IF multi-RHS factor stage mismatch");
    }
    if (factor.B.cols() != dimension) {
      throw std::invalid_argument("RIFT-IF multi-RHS RHS dimension mismatch");
    }
    factorPtrs.push_back(&factor);
  }
  const int interiorDim = static_cast<int>(interior.size()) * dimension;
  const int boundaryDim = static_cast<int>(boundary.size()) * dimension;
  ColMajorSparseMatrix AI;
  Matrix boundaryAndRhs;
  stackMultiRhsFactorsSparseSplit(factorPtrs, interior, boundary, dimension,
                                  dimension, &AI, &boundaryAndRhs);
  if (cache != nullptr) {
    cache->stage = stage;
    cache->interior_keys = interior;
    cache->boundary_keys = boundary;
    cache->interior_column_permutation.clear();
  }

  InterfaceFactor condensed;
  condensed.id = id;
  condensed.stage = stage;
  condensed.scope = boundary;
  condensed.owner_robot = owner_robot;
  condensed.graph_epoch = graph_epoch;

  if (interiorDim == 0) {
    if (boundaryDim == 0) {
      condensed.A = Matrix::Zero(boundaryAndRhs.rows(), 0);
    } else {
      condensed.A = boundaryAndRhs.leftCols(boundaryDim);
    }
    condensed.B = boundaryAndRhs.rightCols(dimension);
    if (cache != nullptr) {
      cache->R_II.resize(0, 0);
      cache->R_IB.resize(0, boundaryDim);
      cache->D_I.resize(0, dimension);
    }
    return condensed;
  }
  if (AI.rows() < interiorDim) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS condensation has underdetermined interior block");
  }

  Eigen::SPQR<ColMajorSparseMatrix> qr(AI);
  if (qr.info() != Eigen::Success ||
      static_cast<int>(qr.rank()) < interiorDim) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS condensation requires full-rank interior block");
  }
  const Matrix transformed = qr.matrixQ().transpose() * boundaryAndRhs;
  if (transformed.rows() < AI.rows()) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS SPQR returned inconsistent transformed rows");
  }
  if (cache != nullptr) {
    const ColMajorSparseMatrix sparseR = qr.matrixR();
    const Matrix denseR = Matrix(sparseR);
    if (denseR.rows() < interiorDim || denseR.cols() < interiorDim) {
      throw std::invalid_argument(
          "RIFT-IF multi-RHS SPQR returned inconsistent R factor");
    }
    cache->R_II = denseR.topLeftCorner(interiorDim, interiorDim);
    cache->R_IB = transformed.block(0, 0, interiorDim, boundaryDim);
    cache->D_I = transformed.block(0, boundaryDim, interiorDim, dimension);
    const auto permutation = qr.colsPermutation();
    cache->interior_column_permutation.resize(interiorDim);
    for (int i = 0; i < interiorDim; ++i) {
      cache->interior_column_permutation[i] =
          static_cast<int>(permutation.indices()[i]);
    }
  }
  const int residualRows = AI.rows() - interiorDim;
  if (boundaryDim == 0) {
    condensed.A = Matrix::Zero(residualRows, 0);
  } else {
    condensed.A =
        transformed.block(interiorDim, 0, residualRows, boundaryDim);
  }
  condensed.B =
      transformed.block(interiorDim, boundaryDim, residualRows, dimension);
  return condensed;
}

Matrix BackSubstituteMultiRHS(const MultiRHSBackSubstitutionCache &cache,
                              const Matrix &boundary_solution) {
  if (cache.R_II.rows() == 0) {
    return Matrix::Zero(0, boundary_solution.cols());
  }
  if (cache.R_IB.cols() != boundary_solution.rows()) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS boundary solution has wrong row count");
  }
  if (cache.D_I.cols() != boundary_solution.cols()) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS boundary solution has wrong RHS count");
  }
  const Matrix rhs = cache.D_I - cache.R_IB * boundary_solution;
  const Matrix pivoted =
      cache.R_II.triangularView<Eigen::Upper>().solve(rhs);
  if (cache.interior_column_permutation.empty()) {
    return pivoted;
  }
  if (static_cast<int>(cache.interior_column_permutation.size()) !=
      pivoted.rows()) {
    throw std::invalid_argument(
        "RIFT-IF multi-RHS cache has invalid column permutation");
  }
  Matrix original = Matrix::Zero(pivoted.rows(), pivoted.cols());
  for (int i = 0; i < pivoted.rows(); ++i) {
    const int originalIndex = cache.interior_column_permutation[i];
    if (originalIndex < 0 || originalIndex >= pivoted.rows()) {
      throw std::invalid_argument(
          "RIFT-IF multi-RHS permutation index out of range");
    }
    original.row(originalIndex) = pivoted.row(i);
  }
  return original;
}

InterfaceCliqueTree InterfaceCliqueTreeBuilder::Build(
    const InterfaceProblem &problem, const RIFTParams &params) {
  (void)params;
  InterfaceCliqueTree tree;
  const int n = static_cast<int>(problem.variables.size());
  if (n == 0) {
    return tree;
  }

  std::map<InterfaceKey, int> indexOf;
  for (int i = 0; i < n; ++i) {
    indexOf[problem.variables[i]] = i;
  }

  std::vector<std::set<int>> adjacency(n);
  for (const auto &factor : problem.factors) {
    std::vector<int> scope;
    for (const auto &key : factor.scope) {
      scope.push_back(indexOf.at(key));
    }
    for (std::size_t i = 0; i < scope.size(); ++i) {
      for (std::size_t j = i + 1; j < scope.size(); ++j) {
        adjacency[scope[i]].insert(scope[j]);
        adjacency[scope[j]].insert(scope[i]);
      }
    }
  }

  std::vector<bool> active(n, true);
  std::vector<std::vector<int>> rawCliques;
  for (int step = 0; step < n; ++step) {
    int best = -1;
    int bestDegree = std::numeric_limits<int>::max();
    for (int v = 0; v < n; ++v) {
      if (!active[v]) {
        continue;
      }
      int degree = 0;
      for (const int nb : adjacency[v]) {
        if (active[nb]) {
          ++degree;
        }
      }
      if (degree < bestDegree || (degree == bestDegree && v < best)) {
        best = v;
        bestDegree = degree;
      }
    }
    if (best < 0) {
      break;
    }

    std::vector<int> clique = {best};
    for (const int nb : adjacency[best]) {
      if (active[nb]) {
        clique.push_back(nb);
      }
    }
    std::sort(clique.begin(), clique.end());
    rawCliques.push_back(clique);

    for (std::size_t i = 1; i < clique.size(); ++i) {
      for (std::size_t j = i + 1; j < clique.size(); ++j) {
        adjacency[clique[i]].insert(clique[j]);
        adjacency[clique[j]].insert(clique[i]);
      }
    }
    active[best] = false;
  }

  std::vector<std::vector<int>> maximal;
  for (const auto &candidate : rawCliques) {
    bool subset = false;
    for (const auto &other : rawCliques) {
      if (&candidate == &other || candidate.size() > other.size()) {
        continue;
      }
      if (std::includes(other.begin(), other.end(), candidate.begin(),
                        candidate.end()) &&
          candidate != other) {
        subset = true;
        break;
      }
    }
    if (!subset &&
        std::find(maximal.begin(), maximal.end(), candidate) == maximal.end()) {
      maximal.push_back(candidate);
    }
  }
  if (maximal.empty()) {
    maximal.push_back({0});
  }

  for (std::size_t c = 0; c < maximal.size(); ++c) {
    InterfaceClique clique;
    clique.id = static_cast<RIFTCliqueId>(c);
    for (const int variableIndex : maximal[c]) {
      clique.variables.push_back(problem.variables[variableIndex]);
    }
    std::map<int, int> robotCounts;
    for (const auto &key : clique.variables) {
      ++robotCounts[key.robot_id];
    }
    int bestRobot = clique.variables.empty() ? -1 : clique.variables.front().robot_id;
    int bestCount = -1;
    for (const auto &entry : robotCounts) {
      if (entry.second > bestCount ||
          (entry.second == bestCount && entry.first < bestRobot)) {
        bestRobot = entry.first;
        bestCount = entry.second;
      }
    }
    clique.host_robot = bestRobot;
    tree.cliques.push_back(std::move(clique));
  }

  struct WeightedEdge {
    int weight = 0;
    int a = -1;
    int b = -1;
    std::vector<InterfaceKey> separator;
  };
  std::vector<WeightedEdge> weightedEdges;
  for (std::size_t a = 0; a < tree.cliques.size(); ++a) {
    for (std::size_t b = a + 1; b < tree.cliques.size(); ++b) {
      WeightedEdge edge;
      edge.a = static_cast<int>(a);
      edge.b = static_cast<int>(b);
      edge.separator =
          intersectKeys(tree.cliques[a].variables, tree.cliques[b].variables);
      edge.weight = static_cast<int>(edge.separator.size());
      weightedEdges.push_back(std::move(edge));
    }
  }
  std::sort(weightedEdges.begin(), weightedEdges.end(),
            [](const WeightedEdge &lhs, const WeightedEdge &rhs) {
              if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
              }
              return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
            });

  std::vector<int> parent(tree.cliques.size());
  std::iota(parent.begin(), parent.end(), 0);
  const auto findRoot = [&](int x) {
    int root = x;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[x] != x) {
      const int next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  };
  auto unite = [&](int a, int b) {
    const int ra = findRoot(a);
    const int rb = findRoot(b);
    if (ra == rb) {
      return false;
    }
    parent[rb] = ra;
    return true;
  };

  for (const auto &weighted : weightedEdges) {
    if (!unite(weighted.a, weighted.b)) {
      continue;
    }
    InterfaceCliqueTreeEdge edge;
    edge.a = weighted.a;
    edge.b = weighted.b;
    edge.separator = weighted.separator;
    edge.estimated_message_bytes = estimatedMessageBytesForSeparator(
        edge.separator.size(), problem.block_dim, problem.rhs_dim);
    tree.edges.push_back(edge);
    tree.cliques[weighted.a].neighbors.push_back(weighted.b);
    tree.cliques[weighted.b].neighbors.push_back(weighted.a);
  }

  for (auto &clique : tree.cliques) {
    std::sort(clique.neighbors.begin(), clique.neighbors.end());
  }

  for (const auto &factor : problem.factors) {
    if (factor.scope.empty()) {
      continue;
    }
    bool assigned = false;
    for (auto &clique : tree.cliques) {
      if (containsScope(clique.variables, factor.scope)) {
        clique.assigned_factors.push_back(factor.id);
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      throw std::runtime_error("RIFT-IF clique tree does not cover factor");
    }
  }

  return tree;
}

bool InterfaceCliqueTreeBuilder::VerifyRunningIntersection(
    const InterfaceCliqueTree &tree) {
  std::map<InterfaceKey, std::vector<RIFTCliqueId>> containing;
  for (const auto &clique : tree.cliques) {
    for (const auto &key : clique.variables) {
      containing[key].push_back(clique.id);
    }
  }

  for (const auto &entry : containing) {
    const std::set<RIFTCliqueId> wanted(entry.second.begin(),
                                        entry.second.end());
    std::queue<RIFTCliqueId> queue;
    std::set<RIFTCliqueId> visited;
    queue.push(*wanted.begin());
    visited.insert(*wanted.begin());
    while (!queue.empty()) {
      const RIFTCliqueId current = queue.front();
      queue.pop();
      for (const RIFTCliqueId neighbor : tree.cliques.at(current).neighbors) {
        if (wanted.count(neighbor) == 0 || visited.count(neighbor) != 0) {
          continue;
        }
        visited.insert(neighbor);
        queue.push(neighbor);
      }
    }
    if (visited != wanted) {
      return false;
    }
  }
  return true;
}

void RIFTRootlessScheduler::Initialize(const InterfaceCliqueTree &tree) {
  neighbors_.clear();
  sent_.clear();
  received_.clear();
  for (const auto &clique : tree.cliques) {
    neighbors_[clique.id] =
        std::set<RIFTCliqueId>(clique.neighbors.begin(),
                               clique.neighbors.end());
  }
}

void RIFTRootlessScheduler::OnMessageReceived(const RIFTMessage &msg) {
  received_.insert(DirectedCliqueEdge{msg.header.src, msg.header.dst});
}

std::vector<DirectedCliqueEdge> RIFTRootlessScheduler::ReadyOutgoingMessages()
    const {
  std::vector<DirectedCliqueEdge> ready;
  for (const auto &entry : neighbors_) {
    const RIFTCliqueId src = entry.first;
    for (const RIFTCliqueId dst : entry.second) {
      const DirectedCliqueEdge edge{src, dst};
      if (sent_.count(edge) != 0) {
        continue;
      }
      bool canSend = true;
      for (const RIFTCliqueId other : entry.second) {
        if (other == dst) {
          continue;
        }
        if (received_.count(DirectedCliqueEdge{other, src}) == 0) {
          canSend = false;
          break;
        }
      }
      if (canSend) {
        ready.push_back(edge);
      }
    }
  }
  std::sort(ready.begin(), ready.end());
  return ready;
}

void RIFTRootlessScheduler::MarkMessageSent(RIFTCliqueId src,
                                            RIFTCliqueId dst) {
  sent_.insert(DirectedCliqueEdge{src, dst});
}

bool RIFTRootlessScheduler::AllDirectedMessagesComplete() const {
  std::size_t expected = 0;
  for (const auto &entry : neighbors_) {
    expected += entry.second.size();
  }
  return sent_.size() == expected && received_.size() == expected;
}

bool RIFTRootlessScheduler::CliqueBeliefReady(RIFTCliqueId alpha) const {
  const auto it = neighbors_.find(alpha);
  if (it == neighbors_.end()) {
    return false;
  }
  for (const RIFTCliqueId neighbor : it->second) {
    if (received_.count(DirectedCliqueEdge{neighbor, alpha}) == 0) {
      return false;
    }
  }
  return true;
}

RIFTDirtyMessageTracker::RIFTDirtyMessageTracker(
    const InterfaceCliqueTree &tree) {
  for (const auto &clique : tree.cliques) {
    if (clique.id < 0) {
      throw std::invalid_argument("RIFT dirty tracker clique id is invalid");
    }
    if (neighbors_.count(clique.id) != 0) {
      throw std::invalid_argument("RIFT dirty tracker duplicate clique id");
    }
    neighbors_[clique.id] =
        std::set<RIFTCliqueId>(clique.neighbors.begin(),
                               clique.neighbors.end());
    for (const RIFTFactorId factor_id : clique.assigned_factors) {
      if (factor_owner_clique_.count(factor_id) != 0) {
        throw std::invalid_argument(
            "RIFT dirty tracker duplicate factor assignment");
      }
      factor_owner_clique_[factor_id] = clique.id;
    }
  }

  for (const auto &entry : neighbors_) {
    const RIFTCliqueId clique_id = entry.first;
    for (const RIFTCliqueId neighbor : entry.second) {
      ValidateClique(neighbor);
      const auto reverse = neighbors_.find(neighbor);
      if (reverse == neighbors_.end() ||
          reverse->second.count(clique_id) == 0) {
        throw std::invalid_argument(
            "RIFT dirty tracker clique neighbors must be reciprocal");
      }
    }
  }
}

void RIFTDirtyMessageTracker::CacheMessage(const DirectedCliqueEdge &edge) {
  ValidateDirectedEdge(edge);
  cached_messages_.insert(edge);
}

void RIFTDirtyMessageTracker::MarkFactorUpdated(RIFTFactorId factor_id) {
  const auto owner = factor_owner_clique_.find(factor_id);
  if (owner == factor_owner_clique_.end()) {
    throw std::invalid_argument("RIFT dirty tracker factor is unknown");
  }
  MarkCliqueDirty(owner->second);
}

void RIFTDirtyMessageTracker::MarkFactorUpdated(
    RIFTFactorId factor_id, RIFTCliqueId assigned_clique) {
  ValidateClique(assigned_clique);
  factor_owner_clique_[factor_id] = assigned_clique;
  MarkCliqueDirty(assigned_clique);
}

void RIFTDirtyMessageTracker::MarkCliqueDirty(RIFTCliqueId clique_id) {
  ValidateClique(clique_id);
  dirty_cliques_.insert(clique_id);

  std::queue<DirectedCliqueEdge> queue;
  for (const RIFTCliqueId neighbor : neighbors_.at(clique_id)) {
    InvalidateDirectedMessage(DirectedCliqueEdge{clique_id, neighbor},
                              &queue);
  }

  while (!queue.empty()) {
    const DirectedCliqueEdge dirty_edge = queue.front();
    queue.pop();
    for (const RIFTCliqueId next : neighbors_.at(dirty_edge.dst)) {
      if (next == dirty_edge.src) {
        continue;
      }
      InvalidateDirectedMessage(DirectedCliqueEdge{dirty_edge.dst, next},
                                &queue);
    }
  }
}

bool RIFTDirtyMessageTracker::IsMessageCached(
    const DirectedCliqueEdge &edge) const {
  return cached_messages_.count(edge) != 0;
}

bool RIFTDirtyMessageTracker::IsMessageInvalidated(
    const DirectedCliqueEdge &edge) const {
  return invalidated_messages_.count(edge) != 0;
}

std::vector<DirectedCliqueEdge>
RIFTDirtyMessageTracker::InvalidatedMessages() const {
  return std::vector<DirectedCliqueEdge>(invalidated_messages_.begin(),
                                         invalidated_messages_.end());
}

std::vector<RIFTCliqueId> RIFTDirtyMessageTracker::DirtyCliques() const {
  return std::vector<RIFTCliqueId>(dirty_cliques_.begin(),
                                   dirty_cliques_.end());
}

int RIFTDirtyMessageTracker::CachedMessageCount() const {
  return static_cast<int>(cached_messages_.size());
}

int RIFTDirtyMessageTracker::ReusableCachedMessageCount() const {
  int count = 0;
  for (const auto &edge : cached_messages_) {
    if (invalidated_messages_.count(edge) == 0) {
      ++count;
    }
  }
  return count;
}

int RIFTDirtyMessageTracker::FullDirectedMessageCount() const {
  int count = 0;
  for (const auto &entry : neighbors_) {
    count += static_cast<int>(entry.second.size());
  }
  return count;
}

void RIFTDirtyMessageTracker::ValidateClique(RIFTCliqueId clique_id) const {
  if (neighbors_.count(clique_id) == 0) {
    throw std::invalid_argument("RIFT dirty tracker clique is unknown");
  }
}

void RIFTDirtyMessageTracker::ValidateDirectedEdge(
    const DirectedCliqueEdge &edge) const {
  ValidateClique(edge.src);
  ValidateClique(edge.dst);
  if (neighbors_.at(edge.src).count(edge.dst) == 0) {
    throw std::invalid_argument("RIFT dirty tracker edge is not adjacent");
  }
}

void RIFTDirtyMessageTracker::InvalidateDirectedMessage(
    const DirectedCliqueEdge &edge, std::queue<DirectedCliqueEdge> *queue) {
  ValidateDirectedEdge(edge);
  const bool inserted = invalidated_messages_.insert(edge).second;
  if (inserted && queue != nullptr) {
    queue->push(edge);
  }
}

RIFTP2PNetworkSimulator::RIFTP2PNetworkSimulator(std::uint64_t seed)
    : rng_(seed) {}

void RIFTP2PNetworkSimulator::SetRandomSeed(std::uint64_t seed) {
  rng_.seed(seed);
}

void RIFTP2PNetworkSimulator::AddRobot(int robot_id) {
  if (robot_id < 0) {
    throw std::invalid_argument("RIFT network robot id must be nonnegative");
  }
  robots_.insert(robot_id);
}

void RIFTP2PNetworkSimulator::AddLink(int a, int b,
                                      const RIFTLinkModel &model) {
  if (a < 0 || b < 0 || a == b) {
    throw std::invalid_argument("RIFT network link endpoints are invalid");
  }
  if (model.latency_mean_ms < 0.0 || model.latency_jitter_ms < 0.0 ||
      model.drop_prob < 0.0 || model.drop_prob > 1.0 ||
      model.reorder_prob < 0.0 || model.reorder_prob > 1.0 ||
      model.bandwidth_bytes_per_sec < 0.0) {
    throw std::invalid_argument("RIFT network link model is invalid");
  }
  AddRobot(a);
  AddRobot(b);
  links_[{a, b}] = LinkState{model, true, 0.0};
  if (model.bidirectional) {
    RIFTLinkModel reverse = model;
    reverse.bidirectional = false;
    links_[{b, a}] = LinkState{reverse, true, 0.0};
  }
}

void RIFTP2PNetworkSimulator::DropLink(int a, int b) {
  bool changed = false;
  for (auto &entry : links_) {
    const bool forward = entry.first.first == a && entry.first.second == b;
    const bool reverse = entry.first.first == b && entry.first.second == a;
    if (forward || reverse) {
      entry.second.up = false;
      changed = true;
    }
  }
  if (!changed) {
    throw std::invalid_argument("RIFT network link does not exist");
  }
}

void RIFTP2PNetworkSimulator::RestoreLink(int a, int b) {
  bool changed = false;
  for (auto &entry : links_) {
    const bool forward = entry.first.first == a && entry.first.second == b;
    const bool reverse = entry.first.first == b && entry.first.second == a;
    if (forward || reverse) {
      entry.second.up = true;
      changed = true;
    }
  }
  if (!changed) {
    throw std::invalid_argument("RIFT network link does not exist");
  }
}

bool RIFTP2PNetworkSimulator::HasLink(int a, int b) const {
  return links_.count({a, b}) != 0;
}

bool RIFTP2PNetworkSimulator::LinkUp(int a, int b) const {
  const auto it = links_.find({a, b});
  return it != links_.end() && it->second.up;
}

void RIFTP2PNetworkSimulator::Send(RIFTNetworkMessage message) {
  if (robots_.count(message.src_robot) == 0 ||
      robots_.count(message.dst_robot) == 0) {
    throw std::invalid_argument("RIFT network message endpoint is unknown");
  }
  auto link = links_.find({message.src_robot, message.dst_robot});
  if (link == links_.end()) {
    throw std::invalid_argument("RIFT network message has no route");
  }
  const RIFTLinkModel &model = link->second.model;
  if (!link->second.up &&
      model.disconnect_policy == RIFTDisconnectPolicy::DROP_WHILE_DISCONNECTED) {
    ++dropped_count_;
    return;
  }

  std::uniform_real_distribution<double> unit(0.0, 1.0);
  if (unit(rng_) < model.drop_prob) {
    ++dropped_count_;
    return;
  }

  double latency = model.latency_mean_ms;
  if (model.latency_jitter_ms > 0.0) {
    std::uniform_real_distribution<double> jitter(-model.latency_jitter_ms,
                                                  model.latency_jitter_ms);
    latency += jitter(rng_);
  }
  latency = std::max(0.0, latency);
  if (unit(rng_) < model.reorder_prob) {
    std::uniform_real_distribution<double> advance(
        0.0, latency + model.latency_jitter_ms + 1.0);
    latency = std::max(0.0, latency - advance(rng_));
  }

  const double txMs =
      model.bandwidth_bytes_per_sec > 0.0
          ? 1000.0 * static_cast<double>(message.payload_bytes) /
                model.bandwidth_bytes_per_sec
          : 0.0;
  const double txStartMs =
      std::max(message.send_time_ms, link->second.next_available_ms);
  const double txFinishMs = txStartMs + txMs;
  const double deliveryMs = txFinishMs + latency;
  link->second.next_available_ms = txFinishMs;

  PendingEvent event;
  event.delivered.message = std::move(message);
  event.delivered.delivery_time_ms = deliveryMs;
  event.insertion_order = next_insertion_order_++;
  pending_.push_back(std::move(event));
}

std::vector<RIFTDeliveredNetworkMessage>
RIFTP2PNetworkSimulator::DeliverReady(double now_ms) {
  std::vector<PendingEvent> remaining;
  std::vector<RIFTDeliveredNetworkMessage> delivered;
  remaining.reserve(pending_.size());
  for (const auto &event : pending_) {
    const auto &message = event.delivered.message;
    if (event.delivered.delivery_time_ms <= now_ms &&
        LinkUp(message.src_robot, message.dst_robot)) {
      delivered.push_back(event.delivered);
    } else {
      remaining.push_back(event);
    }
  }
  pending_ = std::move(remaining);
  std::sort(delivered.begin(), delivered.end(),
            [](const RIFTDeliveredNetworkMessage &lhs,
               const RIFTDeliveredNetworkMessage &rhs) {
              if (lhs.delivery_time_ms != rhs.delivery_time_ms) {
                return lhs.delivery_time_ms < rhs.delivery_time_ms;
              }
              return lhs.message.seq < rhs.message.seq;
            });
  return delivered;
}

std::size_t RIFTP2PNetworkSimulator::PendingCount() const {
  return pending_.size();
}

std::size_t RIFTP2PNetworkSimulator::DroppedCount() const {
  return dropped_count_;
}

Matrix RIFTExactSolver::SolveMatrix(const InterfaceProblem &problem,
                                    const InterfaceCliqueTree &tree,
                                    const RIFTParams &params,
                                    RIFTStats *stats,
                                    DecentralizationGuard *guard,
                                    const std::map<DirectedCliqueEdge,
                                                   RIFTMessage>
                                        *reusable_messages,
                                    const RIFTDirtyMessageTracker
                                        *dirty_tracker,
                                    std::map<DirectedCliqueEdge, RIFTMessage>
                                        *updated_messages) {
  if (problem.block_dim <= 0 || problem.rhs_dim <= 0) {
    throw std::invalid_argument("RIFT-IF problem dimensions must be positive");
  }
  if (!InterfaceCliqueTreeBuilder::VerifyRunningIntersection(tree)) {
    throw std::runtime_error("RIFT-IF clique tree violates running intersection");
  }

  RIFTStats localStats;
  localStats.selected_backend = RIFTInterfaceBackend::RIFT_EXACT;
  PopulateCliqueTreeStats(tree, &localStats);
  if (!ExactSymbolicGateAllows(problem, params, localStats)) {
    throw std::runtime_error("RIFT-IF exact symbolic gate rejected interface");
  }

  std::map<RIFTFactorId, InterfaceFactor> factorById;
  for (const auto &factor : problem.factors) {
    factorById[factor.id] = factor;
  }

  if (guard != nullptr) {
    for (const auto &clique : tree.cliques) {
      for (const RIFTFactorId factorId : clique.assigned_factors) {
        const auto &factor = factorById.at(factorId);
        guard->RecordFactorTransfer(factor.owner_robot, clique.host_robot,
                                    factorId);
      }
    }
  }

  std::map<DirectedCliqueEdge, RIFTMessage> messages;
  RIFTRootlessScheduler scheduler;
  scheduler.Initialize(tree);
  localStats.message_qr_ms += elapsedMilliseconds([&]() {
    while (!scheduler.AllDirectedMessagesComplete()) {
      const auto ready = scheduler.ReadyOutgoingMessages();
      if (ready.empty()) {
        throw std::runtime_error("RIFT-IF scheduler deadlock");
      }
      for (const auto &edge : ready) {
        bool reused = false;
        RIFTMessage msg;
        if (reusable_messages != nullptr) {
          const auto cached = reusable_messages->find(edge);
          const bool invalidated =
              dirty_tracker != nullptr &&
              dirty_tracker->IsMessageInvalidated(edge);
          if (cached != reusable_messages->end() && !invalidated) {
            msg = cached->second;
            reused = true;
            ++localStats.directed_messages_reused;
          }
        }
        if (!reused) {
          if (dirty_tracker != nullptr &&
              dirty_tracker->IsMessageInvalidated(edge)) {
            ++localStats.directed_messages_invalidated;
          }
          msg = computeDirectedMessage(problem, tree, factorById, edge.src,
                                       edge.dst, messages);
          const std::size_t bytes = messageBytes(msg);
          localStats.actual_message_bytes += bytes;
          ++localStats.directed_messages_sent;
          if (guard != nullptr) {
            guard->RecordMessage(tree.cliques.at(edge.src).host_robot,
                                 tree.cliques.at(edge.dst).host_robot, bytes);
          }
        }
        scheduler.MarkMessageSent(edge.src, edge.dst);
        scheduler.OnMessageReceived(msg);
        messages[edge] = std::move(msg);
      }
    }
  });

  std::map<InterfaceKey, Matrix> blockByKey;
  localStats.belief_solve_ms += elapsedMilliseconds([&]() {
    for (const auto &clique : tree.cliques) {
      std::vector<const InterfaceFactor *> factors;
      for (const RIFTFactorId id : clique.assigned_factors) {
        factors.push_back(&factorById.at(id));
      }
      std::vector<const RIFTMessage *> incoming;
      for (const RIFTCliqueId neighbor : clique.neighbors) {
        incoming.push_back(
            &messages.at(DirectedCliqueEdge{neighbor, clique.id}));
      }
      const auto stacked = stackFactorsForClique(
          clique.variables, problem.block_dim, problem.rhs_dim, factors,
          incoming);
      Matrix X = Matrix::Zero(
          static_cast<int>(clique.variables.size()) * problem.block_dim,
          problem.rhs_dim);
      if (stacked.first.cols() > 0 && stacked.first.rows() > 0) {
        Eigen::ColPivHouseholderQR<Matrix> qr(stacked.first);
        for (int rhs = 0; rhs < problem.rhs_dim; ++rhs) {
          X.col(rhs) = qr.solve(stacked.second.col(rhs));
        }
      }
      for (std::size_t i = 0; i < clique.variables.size(); ++i) {
        const InterfaceKey &key = clique.variables[i];
        if (blockByKey.count(key) != 0) {
          continue;
        }
        blockByKey[key] = X.block(static_cast<int>(i) * problem.block_dim, 0,
                                  problem.block_dim, problem.rhs_dim);
      }
    }
  });

  Matrix solution = Matrix::Zero(
      static_cast<int>(problem.variables.size()) * problem.block_dim,
      problem.rhs_dim);
  for (std::size_t i = 0; i < problem.variables.size(); ++i) {
    const auto block = blockByKey.find(problem.variables[i]);
    if (block == blockByKey.end()) {
      throw std::runtime_error("RIFT-IF failed to extract interface variable");
    }
    solution.block(static_cast<int>(i) * problem.block_dim, 0,
                   problem.block_dim, problem.rhs_dim) = block->second;
  }

  localStats.final_interface_residual =
      interfaceResidualNorm(problem, solution);
  if (guard != nullptr) {
    if (params.forbid_global_interface_matrix) {
      guard->AssertNoGlobalInterfaceMatrixConstructed();
    }
    if (params.forbid_direct_solver_in_deployment) {
      guard->AssertNoDirectSolverCalled();
    }
    if (params.forbid_collectives) {
      guard->AssertNoCollectiveCommunication();
    }
    if (params.forbid_global_interface_matrix ||
        params.forbid_direct_solver_in_deployment ||
        params.forbid_collectives) {
      guard->AssertNoNodeReceivedAllFactors(
          static_cast<int>(problem.factors.size()));
    }
    localStats.used_global_matrix = guard->used_global_matrix();
    localStats.used_direct_solver = guard->used_direct_solver();
    localStats.used_collective = guard->used_collective();
  }
  if (stats != nullptr) {
    *stats = localStats;
  }
  if (updated_messages != nullptr) {
    *updated_messages = messages;
  }
  return solution;
}

Vector RIFTExactSolver::Solve(const InterfaceProblem &problem,
                              const InterfaceCliqueTree &tree,
                              const RIFTParams &params, RIFTStats *stats,
                              DecentralizationGuard *guard) {
  if (problem.rhs_dim != 1) {
    throw std::invalid_argument(
        "RIFT-IF vector Solve requires a single RHS; use SolveMatrix");
  }
  return SolveMatrix(problem, tree, params, stats, guard).col(0);
}

Matrix RIFTCAKSolver::ApplyNormalOperator(const InterfaceProblem &problem,
                                          const Matrix &X) {
  const std::vector<CompiledCAKFactor> compiled =
      compileCAKFactors(problem);
  return applyNormalOperatorCompiled(problem, compiled, X);
}

Matrix RIFTCAKSolver::SolveMatrix(const InterfaceProblem &problem,
                                  const RIFTParams &params,
                                  RIFTStats *stats,
                                  DecentralizationGuard *guard) {
  if (problem.block_dim <= 0 || problem.rhs_dim <= 0) {
    throw std::invalid_argument("RIFT-CAK problem dimensions must be positive");
  }
  if (!std::isfinite(params.cak_rel_tol) || params.cak_rel_tol < 0.0 ||
      !std::isfinite(params.cak_abs_tol) || params.cak_abs_tol < 0.0) {
    throw std::invalid_argument(
        "RIFT-CAK residual tolerances must be finite and nonnegative");
  }
  if (params.cak_block_jacobi_pcg_switch_iter < -1) {
    throw std::invalid_argument(
        "RIFT-CAK delayed PCG switch iteration must be -1 or nonnegative");
  }

  RIFTStats localStats;
  localStats.selected_backend = RIFTInterfaceBackend::RIFT_CAK;
  const std::vector<int> robots = interfaceOwnerRobots(problem);
  localStats.num_host_robots = static_cast<int>(robots.size());
  for (const auto &factor : problem.factors) {
    localStats.max_clique_blocks =
        std::max(localStats.max_clique_blocks,
                 static_cast<int>(factor.scope.size()));
  }

  const std::vector<CompiledCAKFactor> compiled =
      compileCAKFactors(problem, /*precomputeSmallNormals=*/true);
  localStats.cak_precomputed_normal_factors =
      static_cast<int>(std::count_if(
          compiled.begin(), compiled.end(),
          [](const CompiledCAKFactor &factor) { return factor.use_normal; }));
  localStats.cak_single_block_factors =
      static_cast<int>(std::count_if(
          compiled.begin(), compiled.end(), [](const CompiledCAKFactor &factor) {
            return factor.global_offsets.size() == 1u;
          }));
  localStats.cak_contiguous_scope_factors =
      static_cast<int>(std::count_if(
          compiled.begin(), compiled.end(), [](const CompiledCAKFactor &factor) {
            return factor.contiguous_scope;
          }));
  std::map<std::vector<int>, int> scopeGroupCounts;
  for (const auto &factor : compiled) {
    ++scopeGroupCounts[factor.global_offsets];
  }
  localStats.cak_scope_groups = static_cast<int>(scopeGroupCounts.size());
  for (const auto &entry : scopeGroupCounts) {
    if (entry.second > 1) {
      localStats.cak_repeated_scope_factors += entry.second - 1;
    }
  }
  std::unique_ptr<RIFTBlockSparseNormalOperator> blockSparseOperator;
  if (params.cak_use_block_sparse_operator) {
    blockSparseOperator.reset(new RIFTBlockSparseNormalOperator(problem));
    localStats.cak_block_sparse_entries =
        blockSparseOperator->numBlockEntries();
    localStats.cak_block_sparse_merged_contributions =
        blockSparseOperator->numMergedBlockContributions();
  }
  const Matrix rhs = normalEquationRhsCompiled(problem, compiled);
  const Vector diagonal = normalEquationDiagonalCompiled(problem, compiled);
  const std::vector<Matrix> blockDiagonal =
      problem.block_dim > 1
          ? normalEquationBlockDiagonalCompiled(problem, compiled)
          : std::vector<Matrix>();
  Matrix X = Matrix::Zero(rhs.rows(), rhs.cols());
  const int maxIterations =
      std::max(20, 4 * static_cast<int>(rhs.rows()) + 20);
  const double tolerance = params.cak_rel_tol;
  const double absoluteResidualTolerance = params.cak_abs_tol;
  bool useBlockJacobiPCG = params.cak_use_block_jacobi_pcg;
  const int blockJacobiPCGSwitchIter =
      params.cak_block_jacobi_pcg_switch_iter;
  const bool certifyPcgResidual = params.cak_pcg_certify_residual;
  double maxFinalResidual = 0.0;

  localStats.message_qr_ms += elapsedMilliseconds([&]() {
    auto maxColumnResidual = [](const Matrix &residual) {
      double maxResidual = 0.0;
      for (int col = 0; col < residual.cols(); ++col) {
        maxResidual = std::max(maxResidual, residual.col(col).norm());
      }
      return maxResidual;
    };

    auto applyNormal = [&](const Matrix &input, Matrix *output,
                           CAKNormalApplyScratch *scratch) {
      if (blockSparseOperator) {
        ++localStats.cak_happly_calls;
        if (scratch->has_workspace) {
          ++localStats.cak_happly_workspace_reuses;
        }
        scratch->has_workspace = true;
        *output = blockSparseOperator->Apply(input);
        return;
      }
      applyNormalOperatorCompiled(problem, compiled, input, output, scratch,
                                  &localStats);
    };

    CAKNormalApplyScratch hApplyScratch;
    Matrix coarseBasis;
    Matrix coarseH;
    Eigen::LDLT<Matrix> coarseLdlt;
    bool hasCoarsePreconditioner = false;
    if ((params.cak_use_robot_mean_coarse_pcg ||
         params.cak_use_robot_mean_coarse_warm_start) &&
        !robots.empty()) {
      std::map<int, int> robotIndex;
      for (std::size_t i = 0; i < robots.size(); ++i) {
        robotIndex[robots[i]] = static_cast<int>(i);
      }
      std::vector<int> robotCounts(robots.size(), 0);
      for (const auto &key : problem.variables) {
        const auto it = robotIndex.find(key.robot_id);
        if (it != robotIndex.end()) {
          ++robotCounts[static_cast<std::size_t>(it->second)];
        }
      }
      const int coarseDim = static_cast<int>(robots.size()) * problem.block_dim;
      coarseBasis = Matrix::Zero(rhs.rows(), coarseDim);
      for (std::size_t var = 0; var < problem.variables.size(); ++var) {
        const auto it = robotIndex.find(problem.variables[var].robot_id);
        if (it == robotIndex.end()) {
          continue;
        }
        const int rIndex = it->second;
        const int count =
            std::max(1, robotCounts[static_cast<std::size_t>(rIndex)]);
        const double scale = 1.0 / std::sqrt(static_cast<double>(count));
        for (int c = 0; c < problem.block_dim; ++c) {
          coarseBasis(static_cast<int>(var) * problem.block_dim + c,
                      rIndex * problem.block_dim + c) = scale;
        }
      }
      applyNormal(coarseBasis, &coarseH, &hApplyScratch);
      Matrix coarseNormal = coarseBasis.transpose() * coarseH;
      const double maxDiag =
          coarseNormal.size() == 0
              ? 0.0
              : coarseNormal.diagonal().cwiseAbs().maxCoeff();
      const double damping = std::max(1e-12, 1e-12 * maxDiag);
      coarseNormal.diagonal().array() += damping;
      coarseLdlt.compute(coarseNormal);
      if (coarseLdlt.info() == Eigen::Success) {
        hasCoarsePreconditioner = true;
        localStats.cak_coarse_dim = coarseDim;
      }
    }

    auto applyPreconditioner = [&](const Matrix &residual) {
      ++localStats.cak_preconditioner_applications;
      if (params.cak_use_robot_mean_coarse_pcg && hasCoarsePreconditioner) {
        ++localStats.cak_coarse_preconditioner_applications;
        const Matrix coarseRhs = coarseBasis.transpose() * residual;
        const Matrix coarseStep = coarseLdlt.solve(coarseRhs);
        if (coarseStep.allFinite()) {
          const Matrix fineResidual = residual - coarseH * coarseStep;
          Matrix preconditioned =
              problem.block_dim > 1
                  ? applyBlockJacobiPreconditioner(blockDiagonal, diagonal,
                                                   fineResidual,
                                                   problem.block_dim)
                  : applyJacobiPreconditioner(diagonal, fineResidual);
          preconditioned.noalias() += coarseBasis * coarseStep;
          return preconditioned;
        }
      }
      Matrix preconditioned =
          problem.block_dim > 1
              ? applyBlockJacobiPreconditioner(blockDiagonal, diagonal,
                                               residual, problem.block_dim)
              : applyJacobiPreconditioner(diagonal, residual);
      return preconditioned;
    };

    X = applyPreconditioner(rhs);
    Matrix normalApplied;
    applyNormal(X, &normalApplied, &hApplyScratch);
    Matrix R = rhs - normalApplied;
    if (params.cak_use_robot_mean_coarse_warm_start &&
        hasCoarsePreconditioner) {
      const Matrix coarseRhs = coarseBasis.transpose() * R;
      const Matrix coarseStep = coarseLdlt.solve(coarseRhs);
      if (coarseStep.allFinite()) {
        X.noalias() += coarseBasis * coarseStep;
        ++localStats.cak_coarse_warm_start_applications;
        applyNormal(X, &normalApplied, &hApplyScratch);
        R = rhs - normalApplied;
      }
    }
    Matrix Z;
    Matrix P;
    if (useBlockJacobiPCG) {
      Z = applyPreconditioner(R);
      P = Z;
    } else {
      P = R;
    }
    Matrix HP;
    Vector rr = Vector::Zero(rhs.cols());
    Vector rz = Vector::Zero(rhs.cols());
    Vector initialResidual = Vector::Zero(rhs.cols());
    std::vector<bool> active(static_cast<std::size_t>(rhs.cols()), true);
    int activeCount = rhs.cols();
    for (int col = 0; col < rhs.cols(); ++col) {
      rr(col) = treeReducedDot(R.col(col), R.col(col), robots, &localStats,
                               guard);
      initialResidual(col) = std::sqrt(std::max(0.0, rr(col)));
      const double residualGate =
          std::min(tolerance * (1.0 + initialResidual(col)),
                   absoluteResidualTolerance);
      if (initialResidual(col) <= residualGate) {
        active[static_cast<std::size_t>(col)] = false;
        P.col(col).setZero();
        --activeCount;
        continue;
      }
      if (useBlockJacobiPCG) {
        rz(col) = treeReducedDot(R.col(col), Z.col(col), robots, &localStats,
                                 guard);
        if (rz(col) <= std::numeric_limits<double>::epsilon() ||
            !std::isfinite(rz(col))) {
          active[static_cast<std::size_t>(col)] = false;
          P.col(col).setZero();
          --activeCount;
        }
      }
    }
    for (int iter = 0; iter < maxIterations && activeCount > 0; ++iter) {
      if (!useBlockJacobiPCG && blockJacobiPCGSwitchIter >= 0 &&
          iter >= blockJacobiPCGSwitchIter) {
        useBlockJacobiPCG = true;
        Z = applyPreconditioner(R);
        P = Z;
        for (int col = 0; col < rhs.cols(); ++col) {
          if (!active[static_cast<std::size_t>(col)]) {
            continue;
          }
          rz(col) = treeReducedDot(R.col(col), Z.col(col), robots,
                                   &localStats, guard);
          if (rz(col) <= std::numeric_limits<double>::epsilon() ||
              !std::isfinite(rz(col))) {
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
          }
        }
        if (activeCount <= 0) {
          break;
        }
      }
      applyNormal(P, &HP, &hApplyScratch);
      if (useBlockJacobiPCG) {
        bool needsExactResidualCertification = false;
        for (int col = 0; col < rhs.cols(); ++col) {
          if (!active[static_cast<std::size_t>(col)]) {
            continue;
          }
          const double pHp = treeReducedDot(P.col(col), HP.col(col), robots,
                                            &localStats, guard);
          if (pHp <= std::numeric_limits<double>::epsilon() ||
              !std::isfinite(pHp)) {
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          const double alpha = rz(col) / pHp;
          X.col(col) += alpha * P.col(col);
          R.col(col) -= alpha * HP.col(col);
          const double nextRr = treeReducedDot(R.col(col), R.col(col), robots,
                                               &localStats, guard);
          ++localStats.cak_iterations;
          const double residual = std::sqrt(std::max(0.0, nextRr));
          const double residualGate =
              std::min(tolerance * (1.0 + initialResidual(col)),
                       absoluteResidualTolerance);
          if (residual <= residualGate) {
            rr(col) = nextRr;
            if (certifyPcgResidual) {
              needsExactResidualCertification = true;
              continue;
            }
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          rr(col) = nextRr;
        }
        if (needsExactResidualCertification) {
          applyNormal(X, &normalApplied, &hApplyScratch);
          R = rhs - normalApplied;
          ++localStats.cak_pcg_residual_replacements;
          activeCount = 0;
          for (int col = 0; col < rhs.cols(); ++col) {
            if (!active[static_cast<std::size_t>(col)]) {
              P.col(col).setZero();
              continue;
            }
            rr(col) = treeReducedDot(R.col(col), R.col(col), robots,
                                     &localStats, guard);
            const double exactResidual = std::sqrt(std::max(0.0, rr(col)));
            const double residualGate =
                std::min(tolerance * (1.0 + initialResidual(col)),
                         absoluteResidualTolerance);
            if (exactResidual <= residualGate) {
              active[static_cast<std::size_t>(col)] = false;
              P.col(col).setZero();
              continue;
            }
            ++activeCount;
          }
          if (activeCount <= 0) {
            break;
          }
          Z = applyPreconditioner(R);
          P = Z;
          for (int col = 0; col < rhs.cols(); ++col) {
            if (!active[static_cast<std::size_t>(col)]) {
              P.col(col).setZero();
              continue;
            }
            rz(col) = treeReducedDot(R.col(col), Z.col(col), robots,
                                     &localStats, guard);
            if (rz(col) <= std::numeric_limits<double>::epsilon() ||
                !std::isfinite(rz(col))) {
              active[static_cast<std::size_t>(col)] = false;
              P.col(col).setZero();
              --activeCount;
            }
          }
          if (activeCount <= 0) {
            break;
          }
          continue;
        }
        if (activeCount > 0) {
          Z = applyPreconditioner(R);
          for (int col = 0; col < rhs.cols(); ++col) {
            if (!active[static_cast<std::size_t>(col)]) {
              continue;
            }
            const double nextRz =
                treeReducedDot(R.col(col), Z.col(col), robots, &localStats,
                               guard);
            if (nextRz <= std::numeric_limits<double>::epsilon() ||
                !std::isfinite(nextRz)) {
              active[static_cast<std::size_t>(col)] = false;
              P.col(col).setZero();
              --activeCount;
              continue;
            }
            const double beta = nextRz / rz(col);
            P.col(col) = Z.col(col) + beta * P.col(col);
            rz(col) = nextRz;
          }
        }
      } else {
        for (int col = 0; col < rhs.cols(); ++col) {
          if (!active[static_cast<std::size_t>(col)]) {
            continue;
          }
          const double pHp = treeReducedDot(P.col(col), HP.col(col), robots,
                                            &localStats, guard);
          if (pHp <= std::numeric_limits<double>::epsilon() ||
              !std::isfinite(pHp)) {
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          const double alpha = rr(col) / pHp;
          X.col(col) += alpha * P.col(col);
          R.col(col) -= alpha * HP.col(col);
          const double nextRr = treeReducedDot(R.col(col), R.col(col), robots,
                                               &localStats, guard);
          ++localStats.cak_iterations;
          const double residual = std::sqrt(std::max(0.0, nextRr));
          const double residualGate =
              std::min(tolerance * (1.0 + initialResidual(col)),
                       absoluteResidualTolerance);
          if (residual <= residualGate) {
            rr(col) = nextRr;
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          const double beta = nextRr / rr(col);
          P.col(col) = R.col(col) + beta * P.col(col);
          rr(col) = nextRr;
        }
      }
    }

    applyNormal(X, &normalApplied, &hApplyScratch);
    Matrix normalResidual = rhs - normalApplied;
    auto residualGateForColumn = [&](int col) {
      return std::min(tolerance * (1.0 + initialResidual(col)),
                      absoluteResidualTolerance);
    };
    auto residualNeedsPolish = [&]() {
      for (int col = 0; col < rhs.cols(); ++col) {
        if (normalResidual.col(col).norm() > residualGateForColumn(col)) {
          return true;
        }
      }
      return false;
    };
    if (certifyPcgResidual && useBlockJacobiPCG && residualNeedsPolish()) {
      R = normalResidual;
      P = R;
      ++localStats.cak_pcg_residual_replacements;
      active.assign(static_cast<std::size_t>(rhs.cols()), true);
      activeCount = rhs.cols();
      for (int col = 0; col < rhs.cols(); ++col) {
        rr(col) = treeReducedDot(R.col(col), R.col(col), robots, &localStats,
                                 guard);
        if (std::sqrt(std::max(0.0, rr(col))) <= residualGateForColumn(col)) {
          active[static_cast<std::size_t>(col)] = false;
          P.col(col).setZero();
          --activeCount;
        }
      }
      for (int polishIter = 0; polishIter < maxIterations && activeCount > 0;
           ++polishIter) {
        applyNormal(P, &HP, &hApplyScratch);
        for (int col = 0; col < rhs.cols(); ++col) {
          if (!active[static_cast<std::size_t>(col)]) {
            continue;
          }
          const double pHp = treeReducedDot(P.col(col), HP.col(col), robots,
                                            &localStats, guard);
          if (pHp <= std::numeric_limits<double>::epsilon() ||
              !std::isfinite(pHp)) {
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          const double alpha = rr(col) / pHp;
          X.col(col) += alpha * P.col(col);
          R.col(col) -= alpha * HP.col(col);
          const double nextRr = treeReducedDot(R.col(col), R.col(col), robots,
                                               &localStats, guard);
          ++localStats.cak_iterations;
          const double residual = std::sqrt(std::max(0.0, nextRr));
          if (residual <= residualGateForColumn(col)) {
            rr(col) = nextRr;
            active[static_cast<std::size_t>(col)] = false;
            P.col(col).setZero();
            --activeCount;
            continue;
          }
          const double beta = nextRr / rr(col);
          P.col(col) = R.col(col) + beta * P.col(col);
          rr(col) = nextRr;
        }
      }
      applyNormal(X, &normalApplied, &hApplyScratch);
      normalResidual = rhs - normalApplied;
    }
    maxFinalResidual = maxColumnResidual(normalResidual);
  });

  localStats.cak_final_residual = maxFinalResidual;
  localStats.final_interface_residual =
      interfaceResidualNormCompiled(problem, compiled, X);
  if (guard != nullptr) {
    if (params.forbid_global_interface_matrix) {
      guard->AssertNoGlobalInterfaceMatrixConstructed();
    }
    if (params.forbid_direct_solver_in_deployment) {
      guard->AssertNoDirectSolverCalled();
    }
    if (params.forbid_collectives) {
      guard->AssertNoCollectiveCommunication();
    }
    localStats.used_global_matrix = guard->used_global_matrix();
    localStats.used_direct_solver = guard->used_direct_solver();
    localStats.used_collective = guard->used_collective();
  }
  if (stats != nullptr) {
    *stats = localStats;
  }
  return X;
}

Vector RIFTCAKSolver::Solve(const InterfaceProblem &problem,
                            const RIFTParams &params, RIFTStats *stats,
                            DecentralizationGuard *guard) {
  if (problem.rhs_dim != 1) {
    throw std::invalid_argument(
        "RIFT-CAK vector Solve requires a single RHS; use SolveMatrix");
  }
  return SolveMatrix(problem, params, stats, guard).col(0);
}

RIFTAsyncSchurSolver::RIFTAsyncSchurSolver(const InterfaceProblem &problem,
                                           const RIFTParams &params,
                                           std::uint64_t seed)
    : problem_(problem), params_(params), network_(seed) {
  if (problem_.block_dim <= 0 || problem_.rhs_dim <= 0) {
    throw std::invalid_argument(
        "RIFT async Schur problem dimensions must be positive");
  }
  if (params_.async_schur_max_iters < 0) {
    throw std::invalid_argument(
        "RIFT async Schur max iterations must be nonnegative");
  }
  if (!std::isfinite(params_.async_schur_rel_tol) ||
      params_.async_schur_rel_tol < 0.0) {
    throw std::invalid_argument(
        "RIFT async Schur tolerance must be finite and nonnegative");
  }
  if (!std::isfinite(params_.async_schur_relaxation) ||
      params_.async_schur_relaxation <= 0.0 ||
      params_.async_schur_relaxation > 1.0) {
    throw std::invalid_argument(
        "RIFT async Schur relaxation must be in (0, 1]");
  }
  if (!std::isfinite(params_.async_schur_damping) ||
      params_.async_schur_damping < 0.0) {
    throw std::invalid_argument(
        "RIFT async Schur damping must be finite and nonnegative");
  }

  robots_ = asyncRobotIds(problem_);
  for (const int robot : robots_) {
    network_.AddRobot(robot);
    value_cache_by_robot_[robot] =
        std::vector<Matrix>(problem_.variables.size(),
                            Matrix::Zero(problem_.block_dim,
                                         problem_.rhs_dim));
  }
  for (std::size_t i = 0; i < robots_.size(); ++i) {
    for (std::size_t j = i + 1; j < robots_.size(); ++j) {
      RIFTLinkModel link;
      network_.AddLink(robots_[i], robots_[j], link);
    }
  }

  for (std::size_t key_index = 0; key_index < problem_.variables.size();
       ++key_index) {
    const int owner = problem_.variables[key_index].robot_id;
    owned_key_indices_[owner].push_back(static_cast<int>(key_index));
  }
  values_ = Matrix::Zero(
      static_cast<int>(problem_.variables.size()) * problem_.block_dim,
      problem_.rhs_dim);
  for (const int robot : robots_) {
    for (const int key_index : owned_key_indices_[robot]) {
      value_cache_by_robot_[robot][key_index] =
          values_.block(key_index * problem_.block_dim, 0,
                        problem_.block_dim, problem_.rhs_dim);
    }
  }
  initial_residual_ = GlobalResidualNorm();
}

void RIFTAsyncSchurSolver::SetLinkModel(int a, int b,
                                        const RIFTLinkModel &model) {
  if (network_.HasLink(a, b)) {
    if (!network_.LinkUp(a, b)) {
      network_.RestoreLink(a, b);
    }
  }
  network_.AddLink(a, b, model);
}

void RIFTAsyncSchurSolver::DropLink(int a, int b) {
  network_.DropLink(a, b);
}

void RIFTAsyncSchurSolver::RestoreLink(int a, int b) {
  const int before = static_cast<int>(ConnectedComponents().size());
  network_.RestoreLink(a, b);
  const int after = static_cast<int>(ConnectedComponents().size());
  if (after < before) {
    ++reconnect_merges_;
  }
}

std::vector<int> RIFTAsyncSchurSolver::ReachableComponent(
    int robot_id) const {
  std::set<int> visited;
  std::queue<int> queue;
  visited.insert(robot_id);
  queue.push(robot_id);
  while (!queue.empty()) {
    const int current = queue.front();
    queue.pop();
    for (const int other : robots_) {
      if (visited.count(other) != 0 || current == other) {
        continue;
      }
      if (network_.HasLink(current, other) && network_.LinkUp(current, other)) {
        visited.insert(other);
        queue.push(other);
      }
    }
  }
  return std::vector<int>(visited.begin(), visited.end());
}

std::vector<std::vector<int>> RIFTAsyncSchurSolver::ConnectedComponents()
    const {
  std::set<int> remaining(robots_.begin(), robots_.end());
  std::vector<std::vector<int>> components;
  while (!remaining.empty()) {
    const int root = *remaining.begin();
    std::vector<int> component = ReachableComponent(root);
    for (const int robot : component) {
      remaining.erase(robot);
    }
    components.push_back(std::move(component));
  }
  return components;
}

void RIFTAsyncSchurSolver::DeliverReadyMessages() {
  const auto delivered = network_.DeliverReady(current_time_ms_);
  for (const auto &event : delivered) {
    const RIFTNetworkMessage &network_message = event.message;
    const RIFTMessage &message = network_message.rift_message;
    const int sender = network_message.src_robot;
    const int receiver = network_message.dst_robot;
    const bool is_gradient = message.R.rows() > 0;
    const int message_type = is_gradient ? 1 : 0;
    const std::uint64_t seq =
        std::max(network_message.seq, message.header.seq);
    for (std::size_t local = 0; local < message.separator_keys.size();
         ++local) {
      const InterfaceKey &key = message.separator_keys[local];
      const auto key_it = std::find(problem_.variables.begin(),
                                    problem_.variables.end(), key);
      if (key_it == problem_.variables.end()) {
        continue;
      }
      const int key_index =
          static_cast<int>(std::distance(problem_.variables.begin(), key_it));
      const auto seq_key =
          std::make_tuple(message_type, receiver, sender, key_index);
      if (last_message_seq_[seq_key] >= seq) {
        ++stale_messages_rejected_;
        continue;
      }
      last_message_seq_[seq_key] = seq;
      const int row = static_cast<int>(local) * problem_.block_dim;
      if (is_gradient) {
        gradient_cache_[{sender, key_index}] =
            message.D.block(row, 0, problem_.block_dim, problem_.rhs_dim);
        diagonal_cache_[{sender, key_index}] =
            message.R.block(row, 0, problem_.block_dim, problem_.block_dim);
      } else {
        value_cache_by_robot_[receiver][key_index] =
            message.D.block(row, 0, problem_.block_dim, problem_.rhs_dim);
      }
    }
  }
}

void RIFTAsyncSchurSolver::BroadcastState(int robot_id) {
  const auto owned = owned_key_indices_.find(robot_id);
  if (owned == owned_key_indices_.end() || owned->second.empty()) {
    return;
  }
  RIFTMessage message;
  message.header.stage = problem_.stage;
  message.header.src = robot_id;
  message.header.seq = next_seq_;
  message.separator_keys.reserve(owned->second.size());
  message.R = Matrix::Zero(0, 0);
  message.D = Matrix::Zero(
      static_cast<int>(owned->second.size()) * problem_.block_dim,
      problem_.rhs_dim);
  for (std::size_t local = 0; local < owned->second.size(); ++local) {
    const int key_index = owned->second[local];
    message.separator_keys.push_back(problem_.variables[key_index]);
    message.D.block(static_cast<int>(local) * problem_.block_dim, 0,
                    problem_.block_dim, problem_.rhs_dim) =
        values_.block(key_index * problem_.block_dim, 0, problem_.block_dim,
                      problem_.rhs_dim);
  }
  for (const int receiver : robots_) {
    if (receiver == robot_id || !network_.HasLink(robot_id, receiver) ||
        !network_.LinkUp(robot_id, receiver)) {
      continue;
    }
    RIFTNetworkMessage network_message;
    network_message.src_robot = robot_id;
    network_message.dst_robot = receiver;
    network_message.seq = next_seq_++;
    network_message.payload_bytes = asyncBlockMessageBytes(
        owned->second.size(), problem_.block_dim, problem_.rhs_dim,
        /*include_diagonal=*/false);
    message_bytes_ += network_message.payload_bytes;
    ++message_count_;
    if (guard_ != nullptr) {
      guard_->RecordMessage(robot_id, receiver,
                            network_message.payload_bytes);
    }
    network_message.send_time_ms = current_time_ms_;
    network_message.rift_message = message;
    network_message.rift_message.header.dst = receiver;
    network_message.rift_message.header.seq = network_message.seq;
    network_.Send(std::move(network_message));
  }
}

void RIFTAsyncSchurSolver::BroadcastFactorGradients(
    int robot_id, const std::vector<int> &component) {
  const std::set<int> component_set(component.begin(), component.end());
  std::map<int, std::vector<int>> key_indices_by_owner;
  std::map<int, Matrix> gradient_by_key;
  std::map<int, Matrix> diagonal_by_key;

  for (const auto &factor : problem_.factors) {
    if (factor.owner_robot != robot_id ||
        !factorInsideRobots(factor, component_set)) {
      continue;
    }
    Matrix Xscope(factor.A.cols(), problem_.rhs_dim);
    for (std::size_t local = 0; local < factor.scope.size(); ++local) {
      const int key_index = keyOffset(problem_.variables, factor.scope[local],
                                      problem_.block_dim) /
                            problem_.block_dim;
      const int row = static_cast<int>(local) * problem_.block_dim;
      Xscope.block(row, 0, problem_.block_dim, problem_.rhs_dim) =
          factor.scope[local].robot_id == robot_id
              ? values_.block(key_index * problem_.block_dim, 0,
                              problem_.block_dim, problem_.rhs_dim)
              : value_cache_by_robot_[robot_id][key_index];
    }
    const Matrix residual = factor.A * Xscope - factor.B;
    for (std::size_t local = 0; local < factor.scope.size(); ++local) {
      const int key_index = keyOffset(problem_.variables, factor.scope[local],
                                      problem_.block_dim) /
                            problem_.block_dim;
      const int col = static_cast<int>(local) * problem_.block_dim;
      const Matrix Ablock = factor.A.block(0, col, factor.A.rows(),
                                           problem_.block_dim);
      if (gradient_by_key.count(key_index) == 0) {
        gradient_by_key[key_index] =
            Matrix::Zero(problem_.block_dim, problem_.rhs_dim);
        diagonal_by_key[key_index] =
            Matrix::Zero(problem_.block_dim, problem_.block_dim);
      }
      gradient_by_key[key_index] += Ablock.transpose() * residual;
      diagonal_by_key[key_index] += Ablock.transpose() * Ablock;
    }
  }

  for (const auto &entry : gradient_by_key) {
    const int key_index = entry.first;
    const int owner = problem_.variables[key_index].robot_id;
    key_indices_by_owner[owner].push_back(key_index);
  }

  for (const auto &entry : key_indices_by_owner) {
    const int receiver = entry.first;
    const auto &keys = entry.second;
    if (keys.empty()) {
      continue;
    }
    RIFTMessage message;
    message.header.stage = problem_.stage;
    message.header.src = robot_id;
    message.header.dst = receiver;
    message.header.seq = next_seq_;
    message.separator_keys.reserve(keys.size());
    message.R = Matrix::Zero(static_cast<int>(keys.size()) *
                                 problem_.block_dim,
                             problem_.block_dim);
    message.D = Matrix::Zero(static_cast<int>(keys.size()) *
                                 problem_.block_dim,
                             problem_.rhs_dim);
    for (std::size_t local = 0; local < keys.size(); ++local) {
      const int key_index = keys[local];
      const int row = static_cast<int>(local) * problem_.block_dim;
      message.separator_keys.push_back(problem_.variables[key_index]);
      message.R.block(row, 0, problem_.block_dim, problem_.block_dim) =
          diagonal_by_key[key_index];
      message.D.block(row, 0, problem_.block_dim, problem_.rhs_dim) =
          gradient_by_key[key_index];
    }

    if (receiver == robot_id) {
      for (std::size_t local = 0; local < keys.size(); ++local) {
        const int key_index = keys[local];
        const int row = static_cast<int>(local) * problem_.block_dim;
        gradient_cache_[{robot_id, key_index}] =
            message.D.block(row, 0, problem_.block_dim, problem_.rhs_dim);
        diagonal_cache_[{robot_id, key_index}] =
            message.R.block(row, 0, problem_.block_dim, problem_.block_dim);
      }
      continue;
    }
    if (!network_.HasLink(robot_id, receiver) ||
        !network_.LinkUp(robot_id, receiver)) {
      continue;
    }
    RIFTNetworkMessage network_message;
    network_message.src_robot = robot_id;
    network_message.dst_robot = receiver;
    network_message.seq = next_seq_++;
    network_message.payload_bytes = asyncBlockMessageBytes(
        keys.size(), problem_.block_dim, problem_.rhs_dim,
        /*include_diagonal=*/true);
    message_bytes_ += network_message.payload_bytes;
    ++message_count_;
    if (guard_ != nullptr) {
      guard_->RecordMessage(robot_id, receiver,
                            network_message.payload_bytes);
    }
    network_message.send_time_ms = current_time_ms_;
    network_message.rift_message = std::move(message);
    network_message.rift_message.header.seq = network_message.seq;
    network_.Send(std::move(network_message));
  }
}

void RIFTAsyncSchurSolver::StepRobot(int robot_id) {
  DeliverReadyMessages();
  const std::vector<int> component = ReachableComponent(robot_id);
  const std::set<int> component_set(component.begin(), component.end());
  BroadcastFactorGradients(robot_id, component);
  DeliverReadyMessages();

  const auto owned = owned_key_indices_.find(robot_id);
  if (owned != owned_key_indices_.end()) {
    for (const int key_index : owned->second) {
      Matrix gradient = Matrix::Zero(problem_.block_dim, problem_.rhs_dim);
      Matrix diagonal = params_.async_schur_damping *
                        Matrix::Identity(problem_.block_dim,
                                         problem_.block_dim);
      for (const int factor_robot : component) {
        const auto grad = gradient_cache_.find({factor_robot, key_index});
        const auto diag = diagonal_cache_.find({factor_robot, key_index});
        if (grad != gradient_cache_.end()) {
          gradient += grad->second;
        }
        if (diag != diagonal_cache_.end()) {
          diagonal += diag->second;
        }
      }
      if (diagonal.norm() == 0.0) {
        continue;
      }
      Eigen::ColPivHouseholderQR<Matrix> qr(diagonal);
      const Matrix step = qr.solve(-gradient);
      values_.block(key_index * problem_.block_dim, 0, problem_.block_dim,
                    problem_.rhs_dim) += params_.async_schur_relaxation * step;
      value_cache_by_robot_[robot_id][key_index] =
          values_.block(key_index * problem_.block_dim, 0, problem_.block_dim,
                        problem_.rhs_dim);
    }
  }
  BroadcastState(robot_id);
  current_time_ms_ += 1.0;
}

void RIFTAsyncSchurSolver::Run(int max_iterations) {
  for (int iter = 0; iter < max_iterations && !GlobalConverged(); ++iter) {
    for (const int robot : robots_) {
      StepRobot(robot);
    }
    ++iterations_;
  }
  current_time_ms_ += 1000.0;
  DeliverReadyMessages();
}

Matrix RIFTAsyncSchurSolver::Solution() const {
  return values_;
}

double RIFTAsyncSchurSolver::GlobalResidualNorm() const {
  std::set<int> all_robots(robots_.begin(), robots_.end());
  std::vector<int> all_component(all_robots.begin(), all_robots.end());
  return ComponentResidualNorm(all_component);
}

double RIFTAsyncSchurSolver::ComponentResidualNorm(
    const std::vector<int> &component) const {
  const std::set<int> component_set(component.begin(), component.end());
  Matrix gradient = Matrix::Zero(values_.rows(), values_.cols());
  for (const auto &factor : problem_.factors) {
    if (!factorInsideRobots(factor, component_set)) {
      continue;
    }
    Matrix X(factor.A.cols(), problem_.rhs_dim);
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int globalOffset =
          keyOffset(problem_.variables, factor.scope[k], problem_.block_dim);
      const int localOffset = static_cast<int>(k) * problem_.block_dim;
      X.block(localOffset, 0, problem_.block_dim, problem_.rhs_dim) =
          values_.block(globalOffset, 0, problem_.block_dim,
                        problem_.rhs_dim);
    }
    const Matrix localGradient = factor.A.transpose() * (factor.A * X -
                                                         factor.B);
    for (std::size_t k = 0; k < factor.scope.size(); ++k) {
      const int globalOffset =
          keyOffset(problem_.variables, factor.scope[k], problem_.block_dim);
      const int localOffset = static_cast<int>(k) * problem_.block_dim;
      gradient.block(globalOffset, 0, problem_.block_dim, problem_.rhs_dim) +=
          localGradient.block(localOffset, 0, problem_.block_dim,
                              problem_.rhs_dim);
    }
  }
  double squared = 0.0;
  for (std::size_t key_index = 0; key_index < problem_.variables.size();
       ++key_index) {
    if (component_set.count(problem_.variables[key_index].robot_id) == 0) {
      continue;
    }
    squared += gradient
                   .block(static_cast<int>(key_index) * problem_.block_dim, 0,
                          problem_.block_dim, problem_.rhs_dim)
                   .squaredNorm();
  }
  return std::sqrt(squared);
}

bool RIFTAsyncSchurSolver::ComponentConverged(
    const std::vector<int> &component) const {
  const std::set<int> component_set(component.begin(), component.end());
  Matrix gradient = Matrix::Zero(values_.rows(), values_.cols());
  for (const auto &factor : problem_.factors) {
    if (factorInsideRobots(factor, component_set)) {
      const Matrix localGradient = -factor.A.transpose() * factor.B;
      for (std::size_t k = 0; k < factor.scope.size(); ++k) {
        const int globalOffset =
            keyOffset(problem_.variables, factor.scope[k], problem_.block_dim);
        const int localOffset = static_cast<int>(k) * problem_.block_dim;
        gradient.block(globalOffset, 0, problem_.block_dim, problem_.rhs_dim) +=
            localGradient.block(localOffset, 0, problem_.block_dim,
                                problem_.rhs_dim);
      }
    }
  }
  double initial_squared = 0.0;
  for (std::size_t key_index = 0; key_index < problem_.variables.size();
       ++key_index) {
    if (component_set.count(problem_.variables[key_index].robot_id) == 0) {
      continue;
    }
    initial_squared +=
        gradient
            .block(static_cast<int>(key_index) * problem_.block_dim, 0,
                   problem_.block_dim, problem_.rhs_dim)
            .squaredNorm();
  }
  const double initial = std::sqrt(initial_squared);
  const double gate = params_.async_schur_rel_tol * std::max(1.0, initial);
  return ComponentResidualNorm(component) <= gate;
}

bool RIFTAsyncSchurSolver::GlobalConverged() const {
  const double gate =
      params_.async_schur_rel_tol * std::max(1.0, initial_residual_);
  return GlobalResidualNorm() <= gate;
}

bool RIFTAsyncSchurSolver::ComponentConsistent() const {
  const auto components = ConnectedComponents();
  if (components.empty()) {
    return true;
  }
  for (const auto &component : components) {
    if (!ComponentConverged(component)) {
      return false;
    }
  }
  return true;
}

Matrix RIFTAsyncSchurSolver::SolveMatrix(const InterfaceProblem &problem,
                                         const RIFTParams &params,
                                         RIFTStats *stats,
                                         DecentralizationGuard *guard) {
  RIFTAsyncSchurSolver solver(problem, params);
  solver.guard_ = guard;
  RIFTStats localStats;
  localStats.selected_backend = RIFTInterfaceBackend::RIFT_ASYNC_SCHUR;
  localStats.num_host_robots = static_cast<int>(solver.robots_.size());
  localStats.async_schur_initial_residual = solver.GlobalResidualNorm();

  localStats.message_qr_ms += elapsedMilliseconds([&]() {
    solver.Run(params.async_schur_max_iters);
  });

  localStats.async_schur_iterations = solver.iterations();
  localStats.async_schur_final_residual = solver.GlobalResidualNorm();
  localStats.async_schur_converged = solver.GlobalConverged();
  localStats.async_schur_components =
      static_cast<int>(solver.ConnectedComponents().size());
  localStats.async_schur_component_consistent = solver.ComponentConsistent();
  localStats.async_schur_global_consistent =
      localStats.async_schur_components == 1 &&
      localStats.async_schur_converged;
  localStats.async_schur_stale_messages_rejected =
      solver.stale_messages_rejected();
  localStats.async_schur_reconnect_merges = solver.reconnect_merges();
  localStats.final_interface_residual =
      interfaceResidualNorm(problem, solver.Solution());
  localStats.actual_message_bytes = solver.message_bytes_;
  localStats.directed_messages_sent = solver.message_count_;
  if (guard != nullptr) {
    if (params.forbid_global_interface_matrix) {
      guard->AssertNoGlobalInterfaceMatrixConstructed();
    }
    if (params.forbid_direct_solver_in_deployment) {
      guard->AssertNoDirectSolverCalled();
    }
    if (params.forbid_collectives) {
      guard->AssertNoCollectiveCommunication();
    }
    localStats.used_global_matrix = guard->used_global_matrix();
    localStats.used_direct_solver = guard->used_direct_solver();
    localStats.used_collective = guard->used_collective();
  }
  if (stats != nullptr) {
    *stats = localStats;
  }
  return solver.Solution();
}

Vector RIFTAsyncSchurSolver::Solve(const InterfaceProblem &problem,
                                   const RIFTParams &params,
                                   RIFTStats *stats,
                                   DecentralizationGuard *guard) {
  if (problem.rhs_dim != 1) {
    throw std::invalid_argument(
        "RIFT async Schur vector Solve requires a single RHS; use SolveMatrix");
  }
  return SolveMatrix(problem, params, stats, guard).col(0);
}

Matrix SolveInterfaceProblemWithRIFTExact(
    const InterfaceProblem &problem, const TEDCCIParams &params,
    TEDCCIStats *stats) {
  RIFTParams riftParams;
  riftParams.backend = params.rift_interface_backend;
  if (riftParams.backend == RIFTInterfaceBackend::DIRECT_ORACLE) {
    throw std::invalid_argument(
        "RIFT-IF deployment mode requires rift_exact or rift_auto backend");
  }
  riftParams.exact_max_separator_blocks_2d =
      params.rift_exact_max_separator_blocks_2d;
  riftParams.exact_max_separator_blocks_3d =
      params.rift_exact_max_separator_blocks_3d;
  riftParams.exact_max_message_bytes = params.rift_exact_max_message_bytes;
  riftParams.use_rotation_multi_rhs = params.rift_use_rotation_multi_rhs;
  riftParams.forbid_direct_solver_in_deployment =
      params.rift_forbid_direct_interface_solver;
  riftParams.forbid_global_interface_matrix =
      params.rift_forbid_global_interface_matrix;
  riftParams.forbid_collectives = params.rift_forbid_collectives;
  riftParams.cak_rel_tol = params.rift_cak_rel_tol;
  riftParams.cak_abs_tol = params.rift_cak_abs_tol;
  riftParams.cak_use_block_jacobi_pcg =
      params.rift_cak_use_block_jacobi_pcg;
  riftParams.cak_block_jacobi_pcg_switch_iter =
      params.rift_cak_block_jacobi_pcg_switch_iter;
  riftParams.cak_pcg_certify_residual =
      params.rift_cak_pcg_certify_residual;
  riftParams.cak_use_robot_mean_coarse_pcg =
      params.rift_cak_use_robot_mean_coarse_pcg;
  riftParams.cak_use_robot_mean_coarse_warm_start =
      params.rift_cak_use_robot_mean_coarse_warm_start;
  riftParams.cak_use_block_sparse_operator =
      params.rift_cak_use_block_sparse_operator;
  riftParams.async_schur_max_iters =
      std::max(riftParams.async_schur_max_iters, params.async_dd_max_iters);
  riftParams.async_schur_rel_tol = params.async_dd_rel_tol;

  RIFTStats riftStats;
  double symbolicMs = 0.0;
  DecentralizationGuard guard;
  Matrix solution;
  if (riftParams.backend == RIFTInterfaceBackend::RIFT_CAK) {
    solution = RIFTCAKSolver::SolveMatrix(problem, riftParams, &riftStats,
                                          &guard);
  } else if (riftParams.backend == RIFTInterfaceBackend::RIFT_ASYNC_SCHUR) {
    solution = RIFTAsyncSchurSolver::SolveMatrix(problem, riftParams,
                                                 &riftStats, &guard);
  } else if (riftParams.backend == RIFTInterfaceBackend::RIFT_AUTO &&
             !AutoExactPreflightAllows(problem, riftParams)) {
    solution = RIFTCAKSolver::SolveMatrix(problem, riftParams, &riftStats,
                                          &guard);
  } else {
    InterfaceCliqueTree tree;
    symbolicMs = elapsedMilliseconds([&]() {
      tree = InterfaceCliqueTreeBuilder::Build(problem, riftParams);
    });
    if (riftParams.backend == RIFTInterfaceBackend::RIFT_AUTO) {
      RIFTStats symbolicStats;
      PopulateCliqueTreeStats(tree, &symbolicStats);
      const bool exactReady =
          InterfaceCliqueTreeBuilder::VerifyRunningIntersection(tree) &&
          ExactSymbolicGateAllows(problem, riftParams, symbolicStats);
      if (exactReady) {
        solution = RIFTExactSolver::SolveMatrix(problem, tree, riftParams,
                                                &riftStats, &guard);
      } else {
        solution = RIFTCAKSolver::SolveMatrix(problem, riftParams, &riftStats,
                                              &guard);
        CopySymbolicStatsInto(symbolicStats, &riftStats);
      }
    } else {
      solution = RIFTExactSolver::SolveMatrix(problem, tree, riftParams,
                                              &riftStats, &guard);
    }
  }
  riftStats.symbolic_ms += symbolicMs;

  if (stats != nullptr) {
    stats->effective_mode = CCIInitMode::TED_CCI_RIFT_IF;
    stats->effective_interface_backend = params.interface_backend;
    stats->num_interface_vars = static_cast<int>(problem.variables.size());
    stats->num_condensed_rows = 0;
    for (const auto &factor : problem.factors) {
      stats->num_condensed_rows += factor.A.rows();
    }
    stats->max_separator_size = riftStats.max_separator_blocks;
    stats->num_factor_messages = static_cast<int>(problem.factors.size());
    stats->num_solution_messages = riftStats.directed_messages_sent;
    stats->bytes_sent_upward = riftStats.actual_message_bytes;
    stats->bytes_sent_downward = 0;
    stats->async_dd_iterations = riftStats.async_schur_iterations;
    stats->async_dd_converged = riftStats.async_schur_converged;
    stats->async_dd_initial_residual =
        riftStats.async_schur_initial_residual;
    stats->async_dd_final_residual = riftStats.async_schur_final_residual;
    stats->rift_selected_backend = riftStats.selected_backend;
    stats->rift_num_cliques = riftStats.num_cliques;
    stats->rift_num_tree_edges = riftStats.num_tree_edges;
    stats->rift_num_host_robots = riftStats.num_host_robots;
    stats->rift_max_host_clique_load = riftStats.max_host_clique_load;
    stats->rift_cross_host_tree_edges = riftStats.cross_host_tree_edges;
    stats->rift_estimated_route_hops = riftStats.estimated_route_hops;
    stats->rift_max_clique_blocks = riftStats.max_clique_blocks;
    stats->rift_max_separator_blocks = riftStats.max_separator_blocks;
    stats->rift_estimated_message_bytes = riftStats.estimated_message_bytes;
    stats->rift_estimated_routed_message_bytes =
        riftStats.estimated_routed_message_bytes;
    stats->rift_actual_message_bytes = riftStats.actual_message_bytes;
    stats->rift_directed_messages_sent = riftStats.directed_messages_sent;
    stats->rift_cak_iterations = riftStats.cak_iterations;
    stats->rift_cak_precomputed_normal_factors =
        riftStats.cak_precomputed_normal_factors;
    stats->rift_cak_single_block_factors =
        riftStats.cak_single_block_factors;
    stats->rift_cak_contiguous_scope_factors =
        riftStats.cak_contiguous_scope_factors;
    stats->rift_cak_scope_groups = riftStats.cak_scope_groups;
    stats->rift_cak_repeated_scope_factors =
        riftStats.cak_repeated_scope_factors;
    stats->rift_cak_block_sparse_entries =
        riftStats.cak_block_sparse_entries;
    stats->rift_cak_block_sparse_merged_contributions =
        riftStats.cak_block_sparse_merged_contributions;
    stats->rift_cak_happly_calls = riftStats.cak_happly_calls;
    stats->rift_cak_happly_workspace_reuses =
        riftStats.cak_happly_workspace_reuses;
    stats->rift_cak_preconditioner_applications =
        riftStats.cak_preconditioner_applications;
    stats->rift_cak_pcg_residual_replacements =
        riftStats.cak_pcg_residual_replacements;
    stats->rift_cak_coarse_dim = riftStats.cak_coarse_dim;
    stats->rift_cak_coarse_preconditioner_applications =
        riftStats.cak_coarse_preconditioner_applications;
    stats->rift_cak_coarse_warm_start_applications =
        riftStats.cak_coarse_warm_start_applications;
    stats->rift_cak_scalar_reductions = riftStats.cak_scalar_reductions;
    stats->rift_cak_scalar_reduction_bytes =
        riftStats.cak_scalar_reduction_bytes;
    stats->rift_cak_final_residual = riftStats.cak_final_residual;
    stats->rift_symbolic_ms = riftStats.symbolic_ms;
    stats->rift_message_qr_ms = riftStats.message_qr_ms;
    stats->rift_belief_solve_ms = riftStats.belief_solve_ms;
    stats->rift_final_interface_residual =
        riftStats.final_interface_residual;
    stats->rift_used_global_matrix = riftStats.used_global_matrix;
    stats->rift_used_direct_solver = riftStats.used_direct_solver;
    stats->rift_used_collective = riftStats.used_collective;
  }
  return solution;
}

Vector SolveTEDInterfaceWithRIFTExact(
    const std::vector<CondensedFactor> &condensed_factors,
    const std::vector<LinearFactorBlock> &cross_factors, int block_dim,
    const std::vector<TEDCCIPartition> &partitions,
    const TEDCCIParams &params, TEDCCIStats *stats) {
  if (block_dim <= 0) {
    throw std::invalid_argument("RIFT-IF TED interface block dimension invalid");
  }
  (void)partitions;
  const FactorType stage =
      !condensed_factors.empty()
          ? condensed_factors.front().factor_type
          : (!cross_factors.empty() ? FactorType::ROTATION
                                    : FactorType::TRANSLATION);
  const int dimension =
      stage == FactorType::ROTATION
          ? static_cast<int>(std::round(std::sqrt(static_cast<double>(block_dim))))
          : block_dim;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          stage, dimension, block_dim, condensed_factors, cross_factors,
          /*use_rotation_multi_rhs=*/false);
  const Matrix solution =
      SolveInterfaceProblemWithRIFTExact(problem, params, stats);
  if (solution.cols() != 1) {
    throw std::invalid_argument(
        "RIFT-IF TED vector interface solve produced multiple RHS columns");
  }
  return solution.col(0);
}

std::string RIFTInterfaceBackendName(RIFTInterfaceBackend backend) {
  switch (backend) {
    case RIFTInterfaceBackend::DIRECT_ORACLE:
      return "direct_oracle";
    case RIFTInterfaceBackend::RIFT_EXACT:
      return "rift_exact";
    case RIFTInterfaceBackend::RIFT_AUTO:
      return "rift_auto";
    case RIFTInterfaceBackend::RIFT_CAK:
      return "rift_cak";
    case RIFTInterfaceBackend::RIFT_ASYNC_SCHUR:
      return "rift_async_schur";
  }
  return "unknown";
}

}  // namespace DPGO
