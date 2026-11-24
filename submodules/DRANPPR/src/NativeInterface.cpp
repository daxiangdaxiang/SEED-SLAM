#include <DPGO/NativeInterface.h>
#include <DPGO/NativeSubspace.h>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace DPGO {
namespace {
using Robot = std::uint32_t;
using Kind = NativeInterfaceMessageKind;

struct OwnedBlock {
  gtsam::Matrix2 diagonal{gtsam::Matrix2::Zero()}, inverse;
  gtsam::Vector2 gradient{gtsam::Vector2::Zero()};
  gtsam::Vector2 step{gtsam::Vector2::Zero()}, residual, direction, preconditioned;
  gtsam::Vector2 product{gtsam::Vector2::Zero()};
};

class RoutedSolve {
 public:
  RoutedSolve(const std::vector<NativeInterfaceEndpoint>& endpoints,
      const std::vector<std::pair<Robot,Robot>>& links, const NativeInterfaceOptions& options)
      : options_(options) {
    if (endpoints.empty() || options.maxIterations == 0 ||
        ((options.rigidSubspace || options.fusedOneStep) && options.maxIterations != 1) ||
        (options.rigidSubspace && options.fusedOneStep) ||
        !std::isfinite(options.relativeTolerance) || options.relativeTolerance <= 0.) {
      throw std::invalid_argument("invalid native interface configuration");
    }
    std::map<gtsam::Key, gtsam::Point2> charts, displacements;
    for (const auto& endpoint : endpoints) {
      if (!endpoint.response || !endpoints_.emplace(endpoint.robot, &endpoint).second) {
        throw std::invalid_argument("null or repeated interface endpoint");
      }
      adjacency_[endpoint.robot];
      owned_[endpoint.robot];
      result_.operatorApplications[endpoint.robot] = 0;
      const auto& keys = endpoint.response->boundaryKeys();
      if (endpoint.response->gradient().size() != 2*static_cast<Eigen::Index>(keys.size()) ||
          endpoint.chartBases.size() != keys.size() ||
          (!endpoint.interfaceKeys.empty() && endpoint.interfaceKeys.size() != keys.size()) ||
          (!endpoint.chartDisplacements.empty() && endpoint.chartDisplacements.size() != keys.size())) {
        throw std::invalid_argument("native interface requires Point2 blocks and matching charts");
      }
      std::set<gtsam::Key> identities;
      for (const auto key : keys) {
        if (!endpoint.interfaceKeys.empty() && !endpoint.interfaceKeys.count(key)) {
          throw std::invalid_argument("incomplete interface identity map");
        }
        const auto identity = wireKey(endpoint.robot, key);
        if (!identities.insert(identity).second) throw std::invalid_argument("duplicate local interface identity");
        if (endpoint.response->boundaryDimensions().at(key) != 2) {
          throw std::invalid_argument("native interface requires every boundary block to be Point2");
        }
        const auto found = endpoint.chartBases.find(key);
        if (found == endpoint.chartBases.end() || !found->second.allFinite()) {
          throw std::invalid_argument("missing or invalid interface chart");
        }
        const auto inserted = charts.emplace(identity, found->second);
        if (!inserted.second && (inserted.first->second-found->second).norm() > 1e-12) {
          throw std::invalid_argument("inconsistent shared interface charts");
        }
        if (!endpoint.chartDisplacements.empty() && !endpoint.chartDisplacements.count(key)) {
          throw std::invalid_argument("incomplete interface chart displacement");
        }
        const gtsam::Point2 displacement = endpoint.chartDisplacements.empty() ?
            gtsam::Point2::Zero().eval() : endpoint.chartDisplacements.at(key);
        const auto current = displacements.emplace(identity, displacement);
        if (!displacement.allFinite() || (!current.second &&
            (current.first->second-displacement).norm() > 1e-12)) {
          throw std::invalid_argument("invalid or inconsistent shared interface displacement");
        }
        observers_[identity].insert(endpoint.robot);
      }
    }
    if (observers_.empty()) throw std::invalid_argument("empty native interface");
    for (const auto& link : links) {
      if (link.first == link.second || !endpoints_.count(link.first) || !endpoints_.count(link.second) ||
          !adjacency_[link.first].insert(link.second).second) {
        throw std::invalid_argument("invalid or repeated communication link");
      }
      adjacency_[link.second].insert(link.first);
    }
    root_ = endpoints_.begin()->first;
    std::queue<Robot> queue;
    queue.push(root_);
    parent_[root_] = root_;
    while (!queue.empty()) {
      const auto robot = queue.front();
      queue.pop();
      order_.push_back(robot);
      for (const auto next : adjacency_.at(robot)) {
        if (parent_.emplace(next, robot).second) queue.push(next);
      }
    }
    if (order_.size() != endpoints_.size()) throw std::invalid_argument("disconnected communication topology");
    for (const auto& entry : observers_) {
      const auto owner = *entry.second.begin();
      owner_[entry.first] = owner;
      owned_[owner][entry.first] = OwnedBlock();
    }
  }

  NativeInterfaceResult solve() {
    // Only block diagonals are exported; the local response stays at its robot.
    for (const auto& entry : endpoints_) {
      const auto robot = entry.first;
      const auto& response = *entry.second->response;
      const auto& keys = response.boundaryKeys();
      gtsam::Vector gradient = response.gradient();
      if (!entry.second->chartDisplacements.empty()) {
        gtsam::Vector displacement(gradient.size());
        for (std::size_t i=0; i<keys.size(); ++i) {
          displacement.segment<2>(2*i) = entry.second->chartDisplacements.at(keys[i]);
        }
        gradient += apply(robot, displacement);
      }
      gradients_[robot] = gradient;
      for (std::size_t i=0; i<keys.size(); ++i) {
        gtsam::Matrix probe = gtsam::Matrix::Zero(response.gradient().size(), 2);
        probe.block<2,2>(2*i,0).setIdentity();
        const gtsam::Matrix product = apply(robot, probe);
        gtsam::Vector payload(6);
        payload.head<2>() = gradient.segment<2>(2*i);
        const gtsam::Matrix2 diagonal = product.block<2,2>(2*i,0);
        payload.tail<4>() = Eigen::Map<const gtsam::Vector4>(diagonal.data());
        const auto identity = wireKey(robot, keys[i]);
        const auto received = send(robot, owner_.at(identity), Kind::Model, identity, payload);
        auto& block = owned_.at(owner_.at(identity)).at(identity);
        block.gradient += received.head<2>();
        block.diagonal += Eigen::Map<const gtsam::Matrix2>(received.data()+2);
      }
    }
    for (auto& robot : owned_) for (auto& entry : robot.second) {
      auto& block = entry.second;
      Eigen::LDLT<gtsam::Matrix2> factor(block.diagonal);
      const double scale = block.diagonal.diagonal().cwiseAbs().maxCoeff();
      if (factor.info() != Eigen::Success || !factor.vectorD().allFinite() || scale <= 0. ||
          (factor.vectorD().array() <= 2*std::numeric_limits<double>::epsilon()*scale).any()) {
        throw std::runtime_error("native interface block preconditioner is not positive definite");
      }
      block.inverse = factor.solve(gtsam::Matrix2::Identity());
      block.residual = -block.gradient;
      block.preconditioned = block.inverse*block.residual;
      block.direction = block.preconditioned;
    }
    if (options_.rigidSubspace) return solveRigidSubspace();
    auto moments = residualMoments();
    const double denominatorScale = std::max(1., std::sqrt(moments[1]));
    double rho = moments[0];
    double lastAlpha = 0.;
    result_.status = "iteration_limit";
    while (result_.iterations < options_.maxIterations &&
           std::sqrt(moments[1]) > options_.relativeTolerance*denominatorScale) {
      multiply(false);
      std::map<Robot,gtsam::Vector> contributions;
      for (const auto& robot : owned_) {
        double value = 0.;
        for (const auto& entry : robot.second) value += entry.second.direction.dot(entry.second.product);
        contributions[robot.first] = gtsam::Vector::Constant(1,value);
      }
      const double curvature = reduce(contributions)[0];
      if (!std::isfinite(curvature) || curvature <= 0. || !std::isfinite(rho) || rho <= 0.) {
        throw std::runtime_error("native interface CG curvature is not positive");
      }
      const double alpha = rho/curvature;
      lastAlpha = alpha;
      for (auto& robot : owned_) for (auto& entry : robot.second) {
        auto& block = entry.second;
        block.step += alpha*block.direction;
        block.residual -= alpha*block.product;
        block.preconditioned = block.inverse*block.residual;
      }
      moments = residualMoments();
      const double beta = moments[0]/rho;
      for (auto& robot : owned_) for (auto& entry : robot.second) {
        auto& block = entry.second;
        block.direction = block.preconditioned+beta*block.direction;
      }
      rho = moments[0];
      ++result_.iterations;
    }
    if (options_.fusedOneStep) {
      // Every observer already has p and the reduced scalars defining alpha.
      // This is exact linear reconstruction, not another physical observation.
      for (const auto& endpoint : endpoints_) {
        result_.localSteps[endpoint.first] = result_.iterations ?
            (lastAlpha*lastDirections_.at(endpoint.first)).eval() :
            gtsam::Vector::Zero(endpoint.second->response->gradient().size()).eval();
      }
      for (auto& robot : owned_) for (auto& entry : robot.second) entry.second.product *= lastAlpha;
    } else {
      // Default path independently reapplies the operator to the final step.
      multiply(true);
    }
    std::map<Robot,gtsam::Vector> residuals;
    for (const auto& robot : owned_) {
      double norm = 0.;
      for (const auto& entry : robot.second) norm += (entry.second.gradient+entry.second.product).squaredNorm();
      residuals[robot.first] = gtsam::Vector::Constant(1,norm);
    }
    result_.relativeResidual = std::sqrt(reduce(residuals)[0])/denominatorScale;
    if (!std::isfinite(result_.relativeResidual)) throw std::runtime_error("nonfinite interface residual");
    if (result_.relativeResidual <= options_.relativeTolerance) result_.status = "linear_residual_small";
    return std::move(result_);
  }

  NativeInterfaceReduction sumScalars(const std::map<Robot,gtsam::Vector>& values) {
    if (values.size() != endpoints_.size()) throw std::invalid_argument("missing scalar contributors");
    const auto size = values.begin()->second.size();
    if (size == 0) throw std::invalid_argument("empty scalar contribution");
    for (const auto& entry : values) {
      if (!endpoints_.count(entry.first) || entry.second.size() != size || !entry.second.allFinite()) {
        throw std::invalid_argument("invalid scalar contribution");
      }
    }
    NativeInterfaceReduction result;
    result.sum = reduce(values);
    if (!result.sum.allFinite()) throw std::runtime_error("nonfinite scalar sum");
    result.messages = std::move(result_.messages);
    result.payloadBytes = result_.payloadBytes;
    result.applicationBytes = result_.applicationBytes;
    return result;
  }

 private:
  NativeInterfaceResult solveRigidSubspace() {
    std::map<Robot,gtsam::Matrix> bases, products;
    std::map<Robot,gtsam::Vector> contributions;
    for (const auto& entry : endpoints_) {
      const auto robot = entry.first;
      const auto& endpoint = *entry.second;
      const auto& keys = endpoint.response->boundaryKeys();
      auto& basis = bases[robot];
      basis.resize(2*keys.size(),4);
      for (std::size_t i=0; i<keys.size(); ++i) {
        const auto key = keys[i], identity = wireKey(robot,key);
        gtsam::Point2 point = endpoint.chartBases.at(key);
        if (!endpoint.chartDisplacements.empty()) point += endpoint.chartDisplacements.at(key);
        basis.block(2*i,0,2,3) = nativeRigidPointBasis(point);
        const auto owner = owner_.at(identity);
        basis.block<2,1>(2*i,3) = send(owner,robot,Kind::Direction,identity,
            owned_.at(owner).at(identity).preconditioned);
      }
      products[robot] = apply(robot,basis);
      const gtsam::Matrix4 h = basis.transpose()*products.at(robot);
      auto& value = contributions[robot];
      value = gtsam::Vector::Zero(21);
      value.head<16>() = Eigen::Map<const gtsam::Vector>(h.data(),16);
      value.segment<4>(16) = basis.transpose()*gradients_.at(robot);
      for (const auto& block : owned_.at(robot)) value[20] += block.second.gradient.squaredNorm();
    }
    const auto totals = reduce(contributions);
    const auto solution = solveNativeSubspace(
        Eigen::Map<const gtsam::Matrix4>(totals.data()),totals.segment<4>(16));
    result_.subspaceSteps = 1;
    result_.subspaceRank = solution.rank;
    for (const auto& entry : endpoints_) {
      const auto robot = entry.first;
      const auto& keys = entry.second->response->boundaryKeys();
      result_.localSteps[robot] = bases.at(robot)*solution.coefficients;
      const gtsam::Vector product = products.at(robot)*solution.coefficients;
      for (std::size_t i=0; i<keys.size(); ++i) {
        const auto identity = wireKey(robot,keys[i]);
        const auto owner = owner_.at(identity);
        owned_.at(owner).at(identity).product += send(robot,owner,Kind::VerifyProduct,
            identity,product.segment<2>(2*i));
      }
    }
    std::map<Robot,gtsam::Vector> residuals;
    for (const auto& robot : owned_) {
      double norm=0.;
      for (const auto& block : robot.second) norm += (block.second.gradient+block.second.product).squaredNorm();
      residuals[robot.first] = gtsam::Vector::Constant(1,norm);
    }
    result_.relativeResidual = std::sqrt(reduce(residuals)[0])/std::max(1.,std::sqrt(totals[20]));
    result_.status = result_.relativeResidual<=options_.relativeTolerance ?
        "linear_residual_small" : "subspace_step";
    return std::move(result_);
  }

  gtsam::Key wireKey(Robot robot, gtsam::Key local) const {
    const auto& mapping = endpoints_.at(robot)->interfaceKeys;
    return mapping.empty() ? local : mapping.at(local);
  }

  gtsam::Matrix apply(Robot robot, const gtsam::Matrix& input) {
    ++result_.operatorApplications.at(robot);
    return endpoints_.at(robot)->response->apply(input);
  }

  gtsam::Vector send(Robot origin, Robot destination, Kind kind,
      gtsam::Key key, const gtsam::Vector& payload) {
    if (!payload.allFinite()) throw std::runtime_error("nonfinite interface message");
    if (origin == destination) return payload;
    std::map<Robot,Robot> previous{{origin,origin}};
    std::queue<Robot> queue;
    queue.push(origin);
    while (!queue.empty() && !previous.count(destination)) {
      const auto robot = queue.front();
      queue.pop();
      for (const auto next : adjacency_.at(robot)) {
        if (previous.emplace(next,robot).second) queue.push(next);
      }
    }
    std::vector<Robot> path{destination};
    while (path.back() != origin) path.push_back(previous.at(path.back()));
    std::reverse(path.begin(),path.end());
    gtsam::Vector received = payload;
    for (std::size_t i=1; i<path.size(); ++i) {
      NativeInterfaceMessage message{path[i-1],path[i],origin,destination,key,options_.epoch,
          endpoints_.at(origin)->response->revision(),kind,received};
      result_.payloadBytes += sizeof(double)*message.payload.size();
      result_.applicationBytes += message.bytes();
      result_.messages.push_back(std::move(message));
      received = result_.messages.back().payload;
    }
    return received;
  }

  gtsam::Vector reduce(std::map<Robot,gtsam::Vector> values) {
    for (auto it=order_.rbegin(); it!=order_.rend(); ++it) {
      if (*it == root_) continue;
      const auto parent = parent_.at(*it);
      values.at(parent) += send(*it,parent,Kind::Reduce,0,values.at(*it));
    }
    for (const auto robot : order_) {
      if (robot == root_) continue;
      const auto parent = parent_.at(robot);
      values.at(robot) = send(parent,robot,Kind::Broadcast,0,values.at(parent));
    }
    return values.at(root_);
  }

  gtsam::Vector residualMoments() {
    std::map<Robot,gtsam::Vector> values;
    for (const auto& robot : owned_) {
      gtsam::Vector2 pair = gtsam::Vector2::Zero();
      for (const auto& entry : robot.second) {
        pair[0] += entry.second.residual.dot(entry.second.preconditioned);
        pair[1] += entry.second.residual.squaredNorm();
      }
      values[robot.first] = pair;
    }
    const auto result = reduce(values);
    if (!result.allFinite() || (result.array() < 0.).any()) throw std::runtime_error("invalid interface residual moments");
    return result;
  }

  void multiply(bool final) {
    for (auto& robot : owned_) for (auto& entry : robot.second) entry.second.product.setZero();
    for (const auto& entry : endpoints_) {
      const auto robot = entry.first;
      const auto& response = *entry.second->response;
      const auto& keys = response.boundaryKeys();
      gtsam::Vector local(response.gradient().size());
      for (std::size_t i=0; i<keys.size(); ++i) {
        const auto identity = wireKey(robot, keys[i]);
        const auto owner = owner_.at(identity);
        const auto& block = owned_.at(owner).at(identity);
        local.segment<2>(2*i) = send(owner,robot,final ? Kind::Update : Kind::Direction,
            identity,final ? block.step : block.direction);
      }
      const gtsam::Vector product = apply(robot,local);
      if (options_.fusedOneStep && !final) lastDirections_[robot] = local;
      for (std::size_t i=0; i<keys.size(); ++i) {
        const auto identity = wireKey(robot, keys[i]);
        const auto owner = owner_.at(identity);
        owned_.at(owner).at(identity).product += send(robot,owner,
            final ? Kind::VerifyProduct : Kind::Product,identity,product.segment<2>(2*i));
      }
      if (final) result_.localSteps[robot] = std::move(local);
    }
  }

  NativeInterfaceOptions options_;
  NativeInterfaceResult result_;
  std::map<Robot,const NativeInterfaceEndpoint*> endpoints_;
  std::map<Robot,gtsam::Vector> gradients_;
  std::map<Robot,gtsam::Vector> lastDirections_;
  std::map<Robot,std::set<Robot>> adjacency_;
  std::map<gtsam::Key,std::set<Robot>> observers_;
  std::map<gtsam::Key,Robot> owner_;
  std::map<Robot,std::map<gtsam::Key,OwnedBlock>> owned_;
  Robot root_;
  std::map<Robot,Robot> parent_;
  std::vector<Robot> order_;
};
}  // namespace

NativeInterfaceResult solveNativeInterface(
    const std::vector<NativeInterfaceEndpoint>& endpoints,
    const std::vector<std::pair<std::uint32_t,std::uint32_t>>& links,
    const NativeInterfaceOptions& options) {
  return RoutedSolve(endpoints,links,options).solve();
}

NativeInterfaceReduction sumNativeInterfaceScalars(
    const std::vector<NativeInterfaceEndpoint>& endpoints,
    const std::vector<std::pair<std::uint32_t,std::uint32_t>>& links,
    const std::map<std::uint32_t,gtsam::Vector>& values,
    const NativeInterfaceOptions& options) {
  return RoutedSolve(endpoints,links,options).sumScalars(values);
}

}  // namespace DPGO
