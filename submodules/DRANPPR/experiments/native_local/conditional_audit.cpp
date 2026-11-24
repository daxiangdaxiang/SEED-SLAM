#include "joint_package.h"
#include <DPGO/IncrementalConditionalModel.h>
#include <gtsam/linear/JacobianFactor.h>

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace {
gtsam::GaussianFactor::shared_ptr linearizeFree(
    const gtsam::NonlinearFactor::shared_ptr& factor, const Local& local) {
  const auto linear = factor->linearize(local.initial);
  const auto ab = linear->jacobian();
  std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms;
  Eigen::Index col=0;
  for (auto key=linear->begin(); key!=linear->end(); ++key) {
    const auto dim=linear->getDim(key);
    if (!local.fixed.count(*key)) terms.emplace_back(*key, ab.first.middleCols(col,dim));
    col+=dim;
  }
  require(!terms.empty(), "constant-only physical factor is not supported in this audit");
  return boost::make_shared<gtsam::JacobianFactor>(terms,ab.second);
}

std::vector<gtsam::Key> privateOrder(const NativeLinearSystem& system) {
  std::vector<gtsam::Key> keys;
  for (auto key : system.keys) if (gtsam::Symbol(key).chr()=='x') keys.push_back(key);
  std::sort(keys.begin(),keys.end());
  return keys;
}

Json parity(const NativeLocalResponse& response, const NativeSchurModel& fresh) {
  require(response.boundaryKeys()==fresh.boundaryKeys(), "boundary order mismatch");
  require(response.boundaryDimensions()==fresh.boundaryDimensions(), "boundary dimensions mismatch");
  const auto n=fresh.gradient().size();
  const auto identity=gtsam::Matrix::Identity(n,n).eval();
  const auto expected=fresh.apply(identity);
  const double product=(response.apply(identity)-expected).norm()/std::max(1.,expected.norm());
  const double gradient=(response.gradient()-fresh.gradient()).norm()/std::max(1.,fresh.gradient().norm());
  const gtsam::Vector b=gtsam::Vector::LinSpaced(n,-.02,.03);
  const auto referenceStart=Clock::now();
  const gtsam::Vector exact=fresh.recover(b);
  const double referenceSeconds=seconds(referenceStart);
  const auto recoveryStart=Clock::now();
  const gtsam::Vector recovered=response.recover(b);
  const double recoverySeconds=seconds(recoveryStart);
  const double recovery=(recovered-exact).norm()/std::max(1.,exact.norm());
  const double residual=fresh.privateRelativeResidual(recovered);
  require(product<1e-9 && gradient<1e-9 && recovery<1e-8 && residual<1e-8,
          "conditional response parity failure: "+Json({product,gradient,recovery,residual}).dump());
  return {{"product_relative_error",product},{"gradient_relative_error",gradient},
      {"recovery_relative_error",recovery},{"private_relative_residual",residual},
      {"reference_recovery_seconds",referenceSeconds},{"conditional_recovery_seconds",recoverySeconds}};
}

Json statistics(const NativeConditionalStatistics& s) {
  return {{"revision",s.revision},{"conditionals_rebuilt",s.conditionalsRebuilt},
      {"conditionals_reused",s.conditionalsReused},{"full_rebuild",s.fullRebuild},
      {"factors_cloned",s.factorsCloned},{"original_factors_eliminated",s.originalFactorsEliminated},
      {"child_messages_eliminated",s.childMessagesEliminated},{"snapshot_ms",s.snapshotMs},
      {"preparation_ms",s.preparationMs},{"elimination_ms",s.eliminationMs},{"boundary_ms",s.boundaryMs}};
}

Json audit(Local& local, NativeInterfaceEndpoint& referenceEndpoint,
           NativeInterfaceEndpoint& conditionalEndpoint) {
  std::map<std::uint32_t,std::vector<NativeFactorId>> groups;
  for (const auto& time : local.times) groups[time.first];
  for (std::size_t id=0; id<local.graph.size(); ++id) {
    std::uint32_t pose=0;
    for (auto key : local.graph[id]->keys()) if (gtsam::Symbol(key).chr()=='x') {
      pose=std::max(pose,std::uint32_t(gtsam::Symbol(key).index()));
    }
    groups.at(pose).push_back(id);
  }
  IncrementalLocalModel reference;
  IncrementalConditionalModel conditional;
  NativeGaussianUpdate pending;
  Json rows=Json::array(), edits=Json::array();
  Json summary{{"robot",local.id},{"valid_prefixes",0},{"singular_prefixes",0},
      {"reference_update_seconds",0.},{"gaussian_preparation_seconds",0.},
      {"conditional_update_seconds",0.},{"reference_recovery_seconds",0.},
      {"conditional_recovery_seconds",0.},{"conditionals_rebuilt",0},{"conditionals_reused",0},
      {"max_product_relative_error",0.},{"max_gradient_relative_error",0.},
      {"max_recovery_relative_error",0.},{"max_private_relative_residual",0.}};
  NativeLinearSystem system;
  std::vector<gtsam::Key> privates;
  std::shared_ptr<NativeSchurModel> fresh;
  auto accumulateParity=[&](const Json& checked) {
    for (const auto* name : {"product_relative_error","gradient_relative_error","recovery_relative_error","private_relative_residual"}) {
      const auto target=std::string("max_")+name;
      summary[target]=std::max(summary.at(target).get<double>(),checked.at(name).get<double>());
    }
  };
  for (const auto& group : groups) {
    NativeModelUpdate update;
    update.newValues.insert(X(group.first),local.initial.at(X(group.first)));
    for (auto id : group.second) {
      update.addFactors[id]=local.graph.at(id);
      for (auto key : local.graph.at(id)->keys()) if (!reference.bases().exists(key) && !update.newValues.exists(key)) {
        update.newValues.insert(key,local.initial.at(key));
      }
    }
    const auto prepStart=Clock::now();
    for (const auto& factor : update.addFactors) pending.addFactors[factor.first]=linearizeFree(factor.second,local);
    const double prepSeconds=seconds(prepStart);
    summary["gaussian_preparation_seconds"]=summary.at("gaussian_preparation_seconds").get<double>()+prepSeconds;
    const auto referenceStart=Clock::now();
    reference.update(update);
    system=reference.linearSystem({},local.fixed); privates=privateOrder(system);
    std::string failure;
    try { fresh=std::make_shared<NativeSchurModel>(system,privates); }
    catch (const std::runtime_error& error) {
      failure=error.what();
      require(failure.find("private")!=std::string::npos,"unexpected reference failure: "+failure);
    }
    const double referenceSeconds=seconds(referenceStart);
    summary["reference_update_seconds"]=summary.at("reference_update_seconds").get<double>()+referenceSeconds;
    Json row{{"pose",group.first},{"timestamp_s",local.times.at(group.first)},
        {"factors",reference.factorCount()},{"private_keys",privates.size()},
        {"reference_update_seconds",referenceSeconds},{"gaussian_preparation_seconds",prepSeconds}};
    if (!failure.empty()) {
      row.update({{"status","singular_private_system"},{"reason",failure}});
      summary["singular_prefixes"]=summary.at("singular_prefixes").get<int>()+1;
      rows.push_back(row); continue;
    }
    const auto conditionalStart=Clock::now();
    const auto stats=conditional.update(pending,system.keys,system.dimensions,privates);
    const double conditionalSeconds=seconds(conditionalStart); pending={};
    summary["conditional_update_seconds"]=summary.at("conditional_update_seconds").get<double>()+conditionalSeconds;
    summary["conditionals_rebuilt"]=summary.at("conditionals_rebuilt").get<std::size_t>()+stats.conditionalsRebuilt;
    summary["conditionals_reused"]=summary.at("conditionals_reused").get<std::size_t>()+stats.conditionalsReused;
    const auto checked=parity(*conditional.response(),*fresh); accumulateParity(checked);
    for (const auto* name : {"reference_recovery_seconds","conditional_recovery_seconds"}) {
      summary[name]=summary.at(name).get<double>()+checked.at(name).get<double>();
    }
    row.update(statistics(stats)); row.update(checked);
    row.update({{"status","ok"},{"conditional_update_seconds",conditionalSeconds}});
    summary["valid_prefixes"]=summary.at("valid_prefixes").get<int>()+1;
    rows.push_back(row);
  }
  require(pending.addFactors.empty() && summary.at("valid_prefixes").get<int>()>0,
          "terminal graph has no valid private response");
  // Controlled information replacements, not an adaptive noise estimator.
  std::vector<NativeFactorId> eligible;
  for (const auto& group : groups) for (auto id : group.second) {
    bool touchesPrivate=false;
    for (auto key : local.graph.at(id)->keys()) touchesPrivate |=
        gtsam::Symbol(key).chr()=='x' && !local.fixed.count(key);
    if (touchesPrivate) eligible.push_back(id);
  }
  require(!eligible.empty(),"no private factors for edit audit");
  for (const auto& edit : std::vector<std::pair<std::string,NativeFactorId>>{
      {"late_factor",eligible.back()},{"early_factor",eligible.front()}}) {
    const auto factor=boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(local.graph.at(edit.second));
    const auto noise=boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(factor->noiseModel());
    const auto replacement=factor->cloneWithNewNoiseModel(
        gtsam::noiseModel::Gaussian::Information(1.1*noise->information()));
    NativeModelUpdate update; update.replaceFactors[edit.second]=replacement;
    const auto referenceStart=Clock::now();
    reference.update(update); system=reference.linearSystem({},local.fixed);
    fresh=std::make_shared<NativeSchurModel>(system,privates);
    const double referenceSeconds=seconds(referenceStart);
    NativeGaussianUpdate changed;
    const auto prepStart=Clock::now();
    changed.replaceFactors[edit.second]=linearizeFree(replacement,local);
    const double prepSeconds=seconds(prepStart);
    const auto conditionalStart=Clock::now();
    const auto stats=conditional.update(changed,system.keys,system.dimensions,privates);
    const double conditionalSeconds=seconds(conditionalStart);
    Json row=statistics(stats); const auto checked=parity(*conditional.response(),*fresh);
    accumulateParity(checked); row.update(checked);
    row.update({{"name",edit.first},{"factor_id",edit.second},{"reference_update_seconds",referenceSeconds},
        {"gaussian_preparation_seconds",prepSeconds},{"conditional_update_seconds",conditionalSeconds}});
    edits.push_back(row);
  }
  // Keep the routed identity/header revision consistent across equivalent models.
  system.revision=conditional.response()->revision();
  referenceEndpoint.robot=conditionalEndpoint.robot=local.id;
  referenceEndpoint.response=std::make_shared<NativeSchurModel>(system,privates);
  conditionalEndpoint.response=conditional.response();
  for (auto key : fresh->boundaryKeys()) {
    referenceEndpoint.chartBases[key]=conditionalEndpoint.chartBases[key]=local.initial.at<gtsam::Point2>(key);
  }
  summary.update({{"prefixes",rows},{"edits",edits},{"factors_final",reference.factorCount()}});
  return summary;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc==3,"usage: native-conditional-audit frozen.json output.json");
    std::ifstream input(argv[1]); require(bool(input),"cannot open package");
    Json package; input>>package;
    Links links; auto locals=parse(package,identifier(package,"root_robot"),links);
    Json result{{"scope","frozen_chart_incremental_response_not_online_estimator"},
        {"truth_used",false},{"robots",Json::array()}};
    std::vector<NativeInterfaceEndpoint> reference(locals.size()),conditional(locals.size());
    for (std::size_t i=0; i<locals.size(); ++i) result["robots"].push_back(audit(locals[i],reference[i],conditional[i]));
    NativeInterfaceOptions options; options.maxIterations=500;
    const auto a=solveNativeInterface(reference,links,options), b=solveNativeInterface(conditional,links,options);
    double stepError=0.;
    for (const auto& step : a.localSteps) stepError=std::max(stepError,
        (step.second-b.localSteps.at(step.first)).norm()/std::max(1.,step.second.norm()));
    require(a.relativeResidual<1e-8 && b.relativeResidual<1e-8 && stepError<1e-8,"routed response mismatch");
    result["routed"]={{"relative_residual",b.relativeResidual},{"reference_relative_residual",a.relativeResidual},
        {"step_relative_error",stepError},{"pcg_iterations",b.iterations},{"reference_pcg_iterations",a.iterations},
        {"application_bytes",b.applicationBytes},{"reference_application_bytes",a.applicationBytes},
        {"scope","same_terminal_models_after_controlled_information_replacements"}};
    std::ofstream output(argv[2]); output.exceptions(std::ios::failbit|std::ios::badbit);
    output<<std::setw(2)<<result<<'\n';
    std::cout<<result.at("routed").dump()<<'\n';
    return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
