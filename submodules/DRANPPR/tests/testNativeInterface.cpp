#include <DPGO/NativeInterface.h>
#include <DPGO/NativeBetweenPose2Factor.h>
#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <Eigen/Cholesky>

#include <iomanip>
#include <sstream>

namespace {
using namespace DPGO;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

NativeInterfaceEndpoint endpoint(std::uint32_t robot,
    const std::vector<gtsam::Key>& keys, const gtsam::Matrix& h, const gtsam::Vector& g) {
  NativeLinearSystem system;
  system.revision = 10+robot;
  system.keys = keys;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    system.offsets[keys[i]] = 2*i;
    system.dimensions[keys[i]] = 2;
  }
  system.hessian = h.sparseView();
  system.gradient = g;
  NativeInterfaceEndpoint result;
  result.robot = robot;
  result.response = std::make_shared<NativeSchurModel>(system, std::vector<gtsam::Key>{});
  for (const auto key : keys) result.chartBases[key] = gtsam::Point2(double(gtsam::Symbol(key).index()), 1.);
  return result;
}

std::vector<NativeInterfaceEndpoint> fixture() {
  std::vector<NativeInterfaceEndpoint> nodes;
  const std::vector<std::vector<gtsam::Key>> keys{{L(0),L(1)}, {L(1),L(2)}, {L(0),L(2)}};
  for (std::uint32_t r = 0; r < 3; ++r) {
    gtsam::Matrix4 a;
    a << 2.+r, .2, .4, .1, .2, 1.5+r, -.1, .3, .4, -.1, 2.5, .2, .1, .3, .2, 3.;
    nodes.push_back(endpoint(r, keys[r], a, gtsam::Vector4(.3+r, -.2, .5, -.7)));
  }
  return nodes;
}

std::pair<gtsam::Matrix, gtsam::Vector> dense(const std::vector<NativeInterfaceEndpoint>& nodes) {
  gtsam::Matrix h = gtsam::Matrix::Zero(6,6);
  gtsam::Vector g = gtsam::Vector::Zero(6);
  for (const auto& node : nodes) {
    const auto& keys = node.response->boundaryKeys();
    const auto local = node.response->apply(gtsam::Matrix::Identity(2*keys.size(), 2*keys.size()));
    for (std::size_t i=0; i<keys.size(); ++i) {
      const auto gi = 2*gtsam::Symbol(keys[i]).index();
      g.segment<2>(gi) += node.response->gradient().segment<2>(2*i);
      for (std::size_t j=0; j<keys.size(); ++j) {
        h.block<2,2>(gi, 2*gtsam::Symbol(keys[j]).index()) += local.block<2,2>(2*i,2*j);
      }
    }
  }
  return {h,g};
}

gtsam::Vector gather(const std::vector<NativeInterfaceEndpoint>& nodes, const NativeInterfaceResult& result) {
  gtsam::Vector value = gtsam::Vector::Zero(6);
  std::set<gtsam::Key> seen;
  for (const auto& node : nodes) {
    for (std::size_t i=0; i<node.response->boundaryKeys().size(); ++i) {
      const auto key = node.response->boundaryKeys()[i];
      if (!seen.insert(key).second) {
        EXPECT_TRUE(value.segment<2>(2*gtsam::Symbol(key).index()).isApprox(
            result.localSteps.at(node.robot).segment<2>(2*i), 0.));
      }
      value.segment<2>(2*gtsam::Symbol(key).index()) = result.localSteps.at(node.robot).segment<2>(2*i);
    }
  }
  return value;
}

TEST(NativeInterface, RoutedPcgMatchesDenseAndCountsEveryHop) {
  const auto nodes = fixture();
  NativeInterfaceOptions options;
  options.relativeTolerance = 1e-12;
  options.epoch = 42;
  const auto result = solveNativeInterface(nodes, {{0,1},{1,2}}, options);
  const auto hg = dense(nodes);
  const gtsam::Vector exact = hg.first.ldlt().solve(-hg.second);
  EXPECT_LE((gather(nodes,result)-exact).norm(), 1e-10);
  EXPECT_LE(result.relativeResidual, 1e-12);
  EXPECT_EQ(result.status, "linear_residual_small");
  EXPECT_GT(result.iterations, 0u);
  std::size_t bytes = 0, payload = 0, modelHops = 0;
  for (const auto& message : result.messages) {
    EXPECT_EQ(std::max(message.from,message.to)-std::min(message.from,message.to), 1u);
    EXPECT_EQ(message.epoch, 42u);
    EXPECT_EQ(message.revision, 10u+message.origin);
    EXPECT_TRUE(message.payload.allFinite());
    bytes += message.bytes();
    payload += message.payload.size()*sizeof(double);
    modelHops += message.kind == NativeInterfaceMessageKind::Model;
  }
  EXPECT_EQ(modelHops, 4u);
  EXPECT_EQ(result.applicationBytes, bytes);
  EXPECT_EQ(result.payloadBytes, payload);
  EXPECT_EQ(bytes, payload+40*result.messages.size());
  for (const auto& node : nodes) EXPECT_EQ(result.operatorApplications.at(node.robot), 2+result.iterations+1);
  const auto direct = solveNativeInterface(nodes, {{0,1},{1,2},{0,2}}, options);
  EXPECT_LE((gather(nodes,direct)-gather(nodes,result)).norm(), 1e-12);
  EXPECT_LT(direct.applicationBytes, result.applicationBytes);
  RecordProperty("pcg_iterations", int(result.iterations));
  RecordProperty("application_bytes", int(result.applicationBytes));
  std::ostringstream residual;
  residual << std::setprecision(17) << result.relativeResidual;
  RecordProperty("relative_residual", residual.str());
}

TEST(NativeInterface, FirstStepMatchesBlockJacobiAlgebra) {
  const auto nodes = fixture();
  const auto hg = dense(nodes);
  gtsam::Matrix m = gtsam::Matrix::Zero(6,6);
  for (int i=0; i<6; i+=2) m.block<2,2>(i,i) = hg.first.block<2,2>(i,i);
  const gtsam::Vector rhs = -hg.second, z = m.ldlt().solve(rhs);
  const gtsam::Vector expected = (rhs.dot(z)/z.dot(hg.first*z))*z;
  NativeInterfaceOptions options;
  options.maxIterations = 1;
  const auto result = solveNativeInterface(nodes, {{0,1},{1,2}}, options);
  EXPECT_EQ(result.iterations, 1u);
  EXPECT_EQ(result.status, "iteration_limit");
  EXPECT_LE((gather(nodes,result)-expected).norm(), 1e-12);
  EXPECT_NEAR(result.relativeResidual, (hg.first*expected+hg.second).norm()/hg.second.norm(), 1e-12);
}

TEST(NativeInterface, FusedOneStepPreservesUpdateAndSavesRoutedWork) {
  auto nodes=fixture();
  for (auto& node : nodes) for (const auto key : node.response->boundaryKeys()) {
    node.chartDisplacements[key]=gtsam::Point2(.02*gtsam::Symbol(key).index(),-.1);
  }
  NativeInterfaceOptions options; options.maxIterations=1;
  const auto old=solveNativeInterface(nodes,{{0,1},{1,2}},options);
  options.fusedOneStep=true;
  const auto fused=solveNativeInterface(nodes,{{0,1},{1,2}},options);
  EXPECT_EQ(fused.iterations,old.iterations);
  for (const auto& node : nodes) {
    EXPECT_TRUE(fused.localSteps.at(node.robot).isApprox(old.localSteps.at(node.robot),0.));
    EXPECT_EQ(fused.operatorApplications.at(node.robot)+1,old.operatorApplications.at(node.robot));
  }
  EXPECT_NEAR(fused.relativeResidual,old.relativeResidual,1e-12);
  EXPECT_LT(fused.applicationBytes,old.applicationBytes);
  for (const auto& message : fused.messages) {
    EXPECT_NE(message.kind,NativeInterfaceMessageKind::Update);
    EXPECT_NE(message.kind,NativeInterfaceMessageKind::VerifyProduct);
  }
  options.maxIterations=2;
  EXPECT_THROW(solveNativeInterface(nodes,{{0,1},{1,2}},options),std::invalid_argument);
  options.maxIterations=1;
  auto zero=endpoint(0,{L(0)},gtsam::Matrix2::Identity(),gtsam::Vector2::Zero());
  EXPECT_TRUE(solveNativeInterface({zero},{},options).localSteps.at(0).isZero());
}

TEST(NativeInterface, RigidSubspaceContainsOldDirectionAndCountsEveryHop) {
  auto nodes = fixture();
  const auto hg = dense(nodes);
  gtsam::Matrix basis = gtsam::Matrix::Zero(6,4);
  for (int i=0; i<3; ++i) {
    basis.block<2,2>(2*i,0).setIdentity();
    basis.block<2,1>(2*i,2) = gtsam::Vector2(-1.,double(i));
    basis.block<2,1>(2*i,3) = hg.first.block<2,2>(2*i,2*i).ldlt().solve(-hg.second.segment<2>(2*i));
  }
  const gtsam::Matrix4 h = basis.transpose()*hg.first*basis;
  const gtsam::Vector expected = -basis*h.ldlt().solve(basis.transpose()*hg.second);
  NativeInterfaceOptions options; options.maxIterations = 1;
  const auto old = solveNativeInterface(nodes, {{0,1},{1,2}}, options);
  options.rigidSubspace = true;
  const auto result = solveNativeInterface(nodes, {{0,1},{1,2}}, options);
  const auto actual = gather(nodes,result), previous = gather(nodes,old);
  EXPECT_LE((actual-expected).norm(), 1e-11);
  EXPECT_LE(hg.second.dot(actual)+.5*actual.dot(hg.first*actual),
            hg.second.dot(previous)+.5*previous.dot(hg.first*previous)+1e-12);
  EXPECT_EQ(result.iterations, 0u);
  EXPECT_EQ(result.subspaceSteps, 1u);
  EXPECT_EQ(result.subspaceRank, 4u);
  std::size_t bytes=0;
  for (const auto& message : result.messages) {
    EXPECT_EQ(std::max(message.from,message.to)-std::min(message.from,message.to),1u);
    bytes += message.bytes();
  }
  EXPECT_EQ(bytes,result.applicationBytes);
  EXPECT_NEAR(result.relativeResidual,(hg.first*actual+hg.second).norm()/hg.second.norm(),1e-12);
}

TEST(NativeInterface, RigidSubspaceHandlesRedundantDirectionsAndCurrentCharts) {
  auto node = endpoint(0,{L(0)},gtsam::Matrix2::Identity(),gtsam::Vector2(.3,-.7));
  node.chartDisplacements[L(0)] = gtsam::Point2(.2,.1);
  NativeInterfaceOptions options; options.maxIterations=1; options.rigidSubspace=true;
  const auto result = solveNativeInterface({node},{},options);
  EXPECT_LE((result.localSteps.at(0)-gtsam::Vector2(-.5,.6)).norm(),1e-12);
  EXPECT_EQ(result.subspaceRank,2u);
  EXPECT_LE(result.relativeResidual,1e-12);
  const auto zero = endpoint(0,{L(0)},gtsam::Matrix2::Identity(),gtsam::Vector2::Zero());
  EXPECT_TRUE(solveNativeInterface({zero},{},options).localSteps.at(0).isZero());
}

TEST(NativeInterface, CachedChartSolvesCorrectionsAtCurrentDisplacement) {
  auto nodes = fixture();
  const auto hg = dense(nodes);
  const gtsam::Vector6 displacement = (gtsam::Vector6() << .1,-.2,.3,.1,-.1,.4).finished();
  for (auto& node : nodes) for (auto key : node.response->boundaryKeys()) {
    node.chartDisplacements[key] = displacement.segment<2>(2*gtsam::Symbol(key).index());
  }
  const gtsam::Vector gradient = hg.second+hg.first*displacement;
  const gtsam::Vector exact = hg.first.ldlt().solve(-gradient);
  const auto solved = solveNativeInterface(nodes, {{0,1},{1,2}});
  EXPECT_LE((gather(nodes,solved)-exact).norm(), 1e-10);
  EXPECT_LE((displacement+gather(nodes,solved)+hg.first.ldlt().solve(hg.second)).norm(), 1e-10);
  gtsam::Matrix m = gtsam::Matrix::Zero(6,6);
  for (int i=0; i<6; i+=2) m.block<2,2>(i,i) = hg.first.block<2,2>(i,i);
  const gtsam::Vector z = m.ldlt().solve(-gradient);
  const gtsam::Vector expected = (-gradient.dot(z)/z.dot(hg.first*z))*z;
  NativeInterfaceOptions options; options.maxIterations = 1;
  const auto partial = solveNativeInterface(nodes, {{0,1},{1,2}}, options);
  const auto step = gather(nodes,partial);
  EXPECT_LE((step-expected).norm(), 1e-12);
  EXPECT_LT(gradient.dot(step)+.5*step.dot(hg.first*step), 0.);
  EXPECT_LT(partial.applicationBytes, solved.applicationBytes);
  nodes[2].chartDisplacements.at(L(0)).x() += .1;
  EXPECT_THROW(solveNativeInterface(nodes, {{0,1},{1,2}}), std::invalid_argument);
  nodes = fixture();
  nodes[0].chartDisplacements[L(0)] = gtsam::Point2::Zero();
  EXPECT_THROW(solveNativeInterface(nodes, {{0,1},{1,2}}), std::invalid_argument);
}

TEST(NativeInterface, LocalKeysAndNetworkIdentitiesAreDistinct) {
  const auto original = fixture();
  const auto reference = solveNativeInterface(original, {{0,1},{1,2}});
  std::vector<NativeInterfaceEndpoint> renamed;
  for (const auto& node : original) {
    const auto& keys = node.response->boundaryKeys();
    std::vector<gtsam::Key> localKeys{L(100+10*node.robot), L(101+10*node.robot)};
    const auto matrix = node.response->apply(gtsam::Matrix4::Identity());
    auto copy = endpoint(node.robot, localKeys, matrix, node.response->gradient());
    for (std::size_t i=0; i<keys.size(); ++i) {
      copy.interfaceKeys[localKeys[i]] = keys[i];
      copy.chartBases[localKeys[i]] = node.chartBases.at(keys[i]);
    }
    renamed.push_back(std::move(copy));
  }
  const auto actual = solveNativeInterface(renamed, {{0,1},{1,2}});
  for (const auto& node : renamed) EXPECT_TRUE(actual.localSteps.at(node.robot).isApprox(
      reference.localSteps.at(node.robot), 1e-13));
  EXPECT_EQ(actual.applicationBytes, reference.applicationBytes);
  for (const auto& message : actual.messages) {
    if (message.key) { EXPECT_LE(gtsam::Symbol(message.key).index(), 2u); }
  }
  auto invalid = renamed;
  invalid[0].interfaceKeys.erase(invalid[0].interfaceKeys.begin());
  EXPECT_THROW(solveNativeInterface(invalid, {{0,1},{1,2}}), std::invalid_argument);
  invalid = renamed;
  invalid[0].interfaceKeys.rbegin()->second = invalid[0].interfaceKeys.begin()->second;
  EXPECT_THROW(solveNativeInterface(invalid, {{0,1},{1,2}}), std::invalid_argument);

  // A new identity epoch can split a relation while reusing the immutable local model.
  renamed[0].interfaceKeys.begin()->second = L(3);
  gtsam::Matrix h = gtsam::Matrix::Zero(8,8);
  gtsam::Vector g = gtsam::Vector::Zero(8);
  for (const auto& node : renamed) {
    const auto& keys = node.response->boundaryKeys();
    const auto local = node.response->apply(gtsam::Matrix4::Identity());
    for (std::size_t i=0; i<2; ++i) {
      const auto gi = 2*gtsam::Symbol(node.interfaceKeys.at(keys[i])).index();
      g.segment<2>(gi) += node.response->gradient().segment<2>(2*i);
      for (std::size_t j=0; j<2; ++j) h.block<2,2>(gi,
          2*gtsam::Symbol(node.interfaceKeys.at(keys[j])).index()) += local.block<2,2>(2*i,2*j);
    }
  }
  const gtsam::Vector expected = h.ldlt().solve(-g);
  NativeInterfaceOptions options; options.epoch = 7;
  const auto split = solveNativeInterface(renamed, {{0,1},{1,2}}, options);
  for (const auto& node : renamed) for (std::size_t i=0; i<2; ++i) {
    const auto key = node.response->boundaryKeys()[i];
    EXPECT_LE((split.localSteps.at(node.robot).segment<2>(2*i)-expected.segment<2>(
        2*gtsam::Symbol(node.interfaceKeys.at(key)).index())).norm(), 1e-10);
  }
  for (const auto& message : split.messages) EXPECT_EQ(message.epoch, 7u);
}

TEST(NativeInterface, ScalarCollectiveUsesTheSameRoutedLedger) {
  const auto nodes = fixture();
  const std::map<std::uint32_t,gtsam::Vector> values{
      {0,gtsam::Vector2(1.,2.)}, {1,gtsam::Vector2(3.,-2.)}, {2,gtsam::Vector2(-1.,5.)}};
  const auto result = sumNativeInterfaceScalars(nodes, {{0,1},{1,2}}, values);
  EXPECT_TRUE(result.sum.isApprox(gtsam::Vector2(3.,5.), 0.));
  EXPECT_EQ(result.messages.size(), 4u);
  EXPECT_EQ(result.payloadBytes, 64u);
  EXPECT_EQ(result.applicationBytes, 224u);
  for (const auto& message : result.messages) {
    EXPECT_EQ(std::max(message.from,message.to)-std::min(message.from,message.to), 1u);
  }
  auto invalid = values;
  invalid.erase(2);
  EXPECT_THROW(sumNativeInterfaceScalars(nodes, {{0,1},{1,2}}, invalid), std::invalid_argument);
  invalid = values;
  invalid[1] = gtsam::Vector3::Zero();
  EXPECT_THROW(sumNativeInterfaceScalars(nodes, {{0,1},{1,2}}, invalid), std::invalid_argument);
}

TEST(NativeInterface, ZeroRhsAndInvalidInputsAreExplicit) {
  const auto zero = endpoint(7, {L(0)}, gtsam::Matrix2::Identity(), gtsam::Vector2::Zero());
  const auto result = solveNativeInterface({zero}, {});
  EXPECT_EQ(result.iterations, 0u);
  EXPECT_TRUE(result.localSteps.at(7).isZero());
  EXPECT_EQ(result.applicationBytes, 0u);
  auto nodes = fixture();
  EXPECT_THROW(solveNativeInterface(nodes, {{0,1}}), std::invalid_argument);
  EXPECT_THROW(solveNativeInterface(nodes, {{0,1},{1,2},{1,0}}), std::invalid_argument);
  nodes[2].chartBases.at(L(0)).x() += .01;
  EXPECT_THROW(solveNativeInterface(nodes, {{0,1},{1,2}}), std::invalid_argument);
  auto bad = endpoint(0, {L(0)}, gtsam::Matrix2::Zero(), gtsam::Vector2::Ones());
  EXPECT_THROW(solveNativeInterface({bad}, {}), std::runtime_error);
  NativeInterfaceOptions options;
  options.maxIterations = 0;
  EXPECT_THROW(solveNativeInterface({zero}, {}, options), std::invalid_argument);
  EXPECT_THROW(solveNativeInterface({zero,zero}, {}), std::invalid_argument);
  auto missing = zero;
  missing.chartBases.clear();
  EXPECT_THROW(solveNativeInterface({missing}, {}), std::invalid_argument);
}

TEST(NativeInterface, RejectsMixedDimensionsEvenWhenTotalSizeMatches) {
  NativeLinearSystem system;
  system.keys = {L(0), L(1)};
  system.offsets = {{L(0), 0}, {L(1), 1}};
  system.dimensions = {{L(0), 1}, {L(1), 3}};
  system.hessian = gtsam::Matrix4::Identity().sparseView();
  system.gradient = gtsam::Vector4::Ones();
  NativeInterfaceEndpoint node;
  node.response = std::make_shared<NativeSchurModel>(system, std::vector<gtsam::Key>{});
  node.chartBases = {{L(0), gtsam::Point2(0., 1.)}, {L(1), gtsam::Point2(1., 1.)}};
  EXPECT_THROW(solveNativeInterface({node}, {}), std::invalid_argument);
}

TEST(NativeInterface, RelayWithoutObservedObjectsAndNegativeCurvature) {
  const auto nodes = fixture();
  NativeLinearSystem empty;
  NativeInterfaceEndpoint relay;
  relay.robot = 3;
  relay.response = std::make_shared<NativeSchurModel>(empty, std::vector<gtsam::Key>{});
  auto relayed = nodes;
  relayed.push_back(relay);
  const auto direct = solveNativeInterface(nodes, {{0,1},{1,2}});
  const auto result = solveNativeInterface(relayed, {{0,3},{3,1},{1,2}});
  EXPECT_TRUE(result.localSteps.at(3).size() == 0);
  EXPECT_LE((gather(nodes,result)-gather(nodes,direct)).norm(), 1e-12);
  EXPECT_GT(result.applicationBytes, direct.applicationBytes);
  gtsam::Matrix4 indefinite = gtsam::Matrix4::Identity();
  indefinite.topRightCorner<2,2>() = 2.*gtsam::Matrix2::Identity();
  indefinite.bottomLeftCorner<2,2>() = 2.*gtsam::Matrix2::Identity();
  const auto bad = endpoint(0, {L(0),L(1)}, indefinite, gtsam::Vector4(1.,0.,-1.,0.));
  EXPECT_THROW(solveNativeInterface({bad}, {}), std::runtime_error);
}

TEST(NativeInterface, ResidualStopDoesNotCertifyObservability) {
  gtsam::Matrix4 laplacian = gtsam::Matrix4::Identity();
  laplacian.topRightCorner<2,2>() = -gtsam::Matrix2::Identity();
  laplacian.bottomLeftCorner<2,2>() = -gtsam::Matrix2::Identity();
  const auto node = endpoint(0, {L(0),L(1)}, laplacian, gtsam::Vector4::Zero());
  const auto result = solveNativeInterface({node}, {});
  EXPECT_EQ(result.status, "linear_residual_small");
  EXPECT_EQ(result.iterations, 0u);
  gtsam::Vector4 nullDirection(1.,0.,1.,0.);
  EXPECT_TRUE((laplacian*nullDirection).isZero());
}

void checkPhysicalSystem(std::uint32_t numRobots) {
  SCOPED_TRACE(numRobots);
  std::vector<NativeInterfaceEndpoint> nodes;
  std::vector<NativeLinearSystem> systems;
  std::vector<gtsam::Point2> objects;
  for (std::size_t j = 0; j < 2*(numRobots+1); ++j) {
    objects.emplace_back(1.+1.2*j, j%2 == 0 ? 1.5 : -1.);
  }
  const auto odomNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(.1, .1, .05));
  const auto polarNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05, .1));
  for (std::uint32_t robot = 0; robot < numRobots; ++robot) {
    NativeModelUpdate input;
    const std::vector<gtsam::Pose2> truth{
        gtsam::Pose2(.2*robot, -.5*robot, .1*robot),
        gtsam::Pose2(1.+.2*robot, -.5*robot, .1*robot)};
    input.newValues.insert(X(0), robot == 0 ? truth[0] : gtsam::Pose2(.5, -.3, .22));
    input.newValues.insert(X(1), gtsam::Pose2(1.1+.3*robot, -.4*robot, .08+.15*robot));
    input.addFactors[0] = boost::make_shared<NativeBetweenPose2Factor>(
        X(0), X(1), truth[0].between(truth[1]), odomNoise);
    for (std::size_t j = 2*robot; j < 2*robot+4; ++j) {
      input.newValues.insert(L(j), gtsam::Point2(objects[j]+gtsam::Point2(.1, -.08)));
      for (std::size_t p = 0; p < truth.size(); ++p) {
        input.addFactors[1+2*j+p] = boost::make_shared<
            gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(
            X(p), L(j), truth[p].bearing(objects[j]), truth[p].range(objects[j]), polarNoise);
      }
    }
    IncrementalLocalModel model;
    model.update(input);
    const std::set<gtsam::Key> fixed = robot == 0 ? std::set<gtsam::Key>{X(0)} : std::set<gtsam::Key>{};
    systems.push_back(model.linearSystem({}, fixed));
    const std::vector<gtsam::Key> privateKeys = robot == 0 ?
        std::vector<gtsam::Key>{X(1)} : std::vector<gtsam::Key>{X(0), X(1)};
    NativeInterfaceEndpoint node;
    node.robot = robot;
    node.response = std::make_shared<NativeSchurModel>(systems.back(), privateKeys);
    for (std::size_t j = 2*robot; j < 2*robot+4; ++j) {
      node.chartBases[L(j)] = input.newValues.at<gtsam::Point2>(L(j));
    }
    nodes.push_back(std::move(node));
  }
  // Dense oracle is test-only; the production interface never builds this matrix.
  const Eigen::Index privateSize = 3*(2*numRobots-1), size = privateSize+2*objects.size();
  gtsam::Matrix h = gtsam::Matrix::Zero(size,size);
  gtsam::Vector g = gtsam::Vector::Zero(size), recovered = gtsam::Vector::Zero(size);
  auto globalOffset = [privateSize](std::size_t robot, gtsam::Key key) -> Eigen::Index {
    const gtsam::Symbol symbol(key);
    return symbol.chr() == 'l' ? privateSize+2*symbol.index() :
        (robot == 0 ? 0 : 3*(2*robot-1)+3*symbol.index());
  };
  for (std::size_t robot = 0; robot < systems.size(); ++robot) {
    const auto& system = systems[robot];
    const gtsam::Matrix local(system.hessian);
    for (const auto a : system.keys) {
      const auto ga = globalOffset(robot, a), la = system.offsets.at(a), da = system.dimensions.at(a);
      g.segment(ga,da) += system.gradient.segment(la,da);
      for (const auto b : system.keys) {
        const auto gb = globalOffset(robot, b), lb = system.offsets.at(b), db = system.dimensions.at(b);
        h.block(ga,gb,da,db) += local.block(la,lb,da,db);
      }
    }
  }
  NativeInterfaceOptions options;
  options.maxIterations = 1000;
  options.relativeTolerance = 1e-12;
  std::vector<std::pair<std::uint32_t,std::uint32_t>> links;
  for (std::uint32_t r = 1; r < numRobots; ++r) links.emplace_back(r-1,r);
  const auto result = solveNativeInterface(nodes, links, options);
  ASSERT_EQ(result.status, "linear_residual_small");
  for (std::uint32_t r = 1; r < numRobots; ++r) {
    EXPECT_TRUE(result.localSteps.at(r-1).tail<4>().isApprox(result.localSteps.at(r).head<4>(), 0.));
  }
  for (std::size_t robot = 0; robot < systems.size(); ++robot) {
    const auto& system = systems[robot];
    const auto step = nodes[robot].response->recover(result.localSteps.at(robot));
    EXPECT_LE(nodes[robot].response->privateRelativeResidual(step), 1e-10);
    for (const auto key : system.keys) {
      recovered.segment(globalOffset(robot,key), system.dimensions.at(key)) =
          step.segment(system.offsets.at(key), system.dimensions.at(key));
    }
  }
  const gtsam::Vector direct = h.ldlt().solve(-g);
  const double error = (recovered-direct).norm()/std::max(1., direct.norm());
  const double residual = (h*recovered+g).norm()/std::max(1., g.norm());
  EXPECT_LE(error, 1e-9);
  EXPECT_LE(residual, 1e-10);
  EXPECT_GT(recovered.segment<3>(3).norm(), .01);
  const auto label = std::to_string(numRobots)+"_robots";
  ::testing::Test::RecordProperty("pcg_iterations_"+label, int(result.iterations));
  ::testing::Test::RecordProperty("application_bytes_"+label, int(result.applicationBytes));
  std::ostringstream values;
  values << std::setprecision(17) << error << "," << residual << "," << result.relativeResidual;
  ::testing::Test::RecordProperty("step_full_and_interface_residual_"+label, values.str());

  gtsam::Vector displacement(size);
  for (Eigen::Index i=0; i<size; ++i) displacement[i] = .02*std::sin(double(i+1));
  for (auto& node : nodes) for (auto key : node.response->boundaryKeys()) {
    node.chartDisplacements[key] = displacement.segment<2>(globalOffset(node.robot,key));
  }
  const auto corrected = solveNativeInterface(nodes, links, options);
  gtsam::Vector correction = gtsam::Vector::Zero(size);
  for (std::size_t robot=0; robot<systems.size(); ++robot) {
    const auto& system = systems[robot];
    auto boundary = corrected.localSteps.at(robot);
    for (std::size_t i=0; i<nodes[robot].response->boundaryKeys().size(); ++i) {
      boundary.segment<2>(2*i) += nodes[robot].chartDisplacements.at(nodes[robot].response->boundaryKeys()[i]);
    }
    const auto absolute = nodes[robot].response->recover(boundary);
    for (auto key : system.keys) {
      const auto offset = globalOffset(robot,key), dimension = system.dimensions.at(key);
      correction.segment(offset,dimension) = absolute.segment(system.offsets.at(key),dimension)-
          displacement.segment(offset,dimension);
    }
  }
  EXPECT_LE((correction+displacement-direct).norm()/std::max(1.,direct.norm()), 1e-9);
  EXPECT_LE((h*correction+g+h*displacement).norm()/std::max(1.,g.norm()), 1e-10);
}

TEST(NativeInterface, SingleGaugePhysicalSystemsMatchFullJointSolve) {
  for (const std::uint32_t numRobots : {2u,3u,5u,10u}) checkPhysicalSystem(numRobots);
}
}  // namespace
