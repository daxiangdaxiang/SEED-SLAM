#include <rimesa/rimesa.h>
#include <gtsam/slam/PriorFactor.h>
#include <nlohmann/json.hpp>
#include <iostream>

int main() {
  try {
    nlohmann::json rows=nlohmann::json::array();
    const gtsam::Key key=gtsam::Symbol('#',0);
    for (double decay : {.9,1.}) for (double sigma : {1.,.1}) {
      rimesa::RIMESAParams p; p.bp_kernel_params=std::nullopt; p.dual_decay_rate=decay;
      p.external_shared_var_init_strat=rimesa::init::InitializationStrategy::OTHER;
      risam::RISAM::Parameters local; local.optimization_params=std::nullopt;
      local.increment_outlier_mu=false; local.number_extra_iters=0;
      rimesa::RIMESA a('a',p,local),b('b',p,local);
      for (auto pair : {std::make_pair(&a,0.),std::make_pair(&b,1.)}) {
        gtsam::Values values; const gtsam::Point2 point(pair.second,0.); values.insert(key,point);
        gtsam::NonlinearFactorGraph factors;
        factors.emplace_shared<gtsam::PriorFactor<gtsam::Point2>>(key,point,gtsam::noiseModel::Isotropic::Sigma(2,sigma));
        pair.first->update(factors,values);
      }
      constexpr int rounds=5000;
      for (int i=0;i<rounds;++i) {
        rimesa::RIMESA::CommunicationHandler ha(a.getCommunicationStateCache('b')),hb(b.getCommunicationStateCache('a'));
        const auto da=ha.declareNewSharedVariables(),db=hb.declareNewSharedVariables();
        ha.receiveNewSharedVariables(db); hb.receiveNewSharedVariables(da);
        const auto sa=ha.sendCommunication(),sb=hb.sendCommunication();
        ha.receiveCommunication(sb); hb.receiveCommunication(sa);
        a.incorporateCommunicationResult(*ha.getResult()); b.incorporateCommunicationResult(*hb.getResult());
        a.update({},{}); b.update({},{});
      }
      const double x=a.getEstimate().at<gtsam::Point2>(key).x(),y=b.getEstimate().at<gtsam::Point2>(key).x();
      const double weight=1./(sigma*sigma);
      const double kappa=decay<1. ? .5*p.penalty_param*(1.+1./(1.-decay)) : 0.;
      rows.push_back({{"dual_decay",decay},{"sigma",sigma},{"weight",weight},{"rounds",rounds},
          {"predicted_gap",decay<1. ? weight/(weight+2*kappa) : 0.},
          {"numerical_gap",y-x},{"mean",.5*(x+y)},{"x1",x},{"x2",y},
          {"scope","Two Gaussian physical Point2 measurements; no outliers, robust kernels, or pose gauge. Official pinned RIMESA API; decay=1 is diagnostic."}});
    }
    std::cout << rows.dump(2) << std::endl;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
