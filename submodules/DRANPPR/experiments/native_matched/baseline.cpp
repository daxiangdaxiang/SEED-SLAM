#include <DPGO/NativeBetweenPose2Factor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <imesa/imesa.h>
#include <rimesa/rimesa.h>
#include <risam/GraduatedFactor.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;
using Links = std::vector<std::pair<int,int>>;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
int id(const Json& row, const std::string& key) {
  require(row.at(key).is_number_integer(), "noninteger " + key);
  const auto value = row.at(key).get<std::int64_t>();
  require(value >= 0 && value < (1LL<<24), "identifier out of range");
  return int(value);
}
double number(const Json& row, const std::string& key) {
  require(row.at(key).is_number(), "nonnumeric " + key);
  const double v = row.at(key).get<double>();
  require(std::isfinite(v), "nonfinite " + key);
  return v;
}
gtsam::Vector vector(const Json& row, std::size_t n) {
  require(row.is_array() && row.size() == n, "bad vector dimension");
  gtsam::Vector v(n);
  for (std::size_t i=0; i<n; ++i) {
    require(row[i].is_number(), "nonnumeric vector"); v[i]=row[i].get<double>();
  }
  require(v.allFinite(), "nonfinite vector"); return v;
}
char robotChar(int r) {
  require(r >= 1 && r <= 26, "adapter supports robot IDs 1..26"); return char('a'+r-1);
}
gtsam::Key X(int r, int k) { return gtsam::Symbol(robotChar(r), k); }
gtsam::Key L(int k) { return gtsam::Symbol('#', k); }

// Preload only common supplied values; no solver update or extra root prior.
struct SeededRiMesa : rimesa::RIMESA {
  SeededRiMesa(int r, const gtsam::Values& values, const std::string& mode)
      : RIMESA(robotChar(r), parameters(mode), localParameters(mode)) {
    current_estimate_ = values;
    for (auto k : values.keys()) if (gtsam::Symbol(k).chr() == '#') observed_global_variables_.insert(k);
  }
  static rimesa::RIMESAParams parameters(const std::string& mode) {
    rimesa::RIMESAParams p;
    if (mode != "rimesa") p.bp_kernel_params = std::nullopt;
    if (mode == "imesa_paper") p.external_shared_var_init_strat=rimesa::init::InitializationStrategy::OTHER;
    return p;
  }
  static risam::RISAM::Parameters localParameters(const std::string& mode) {
    risam::RISAM::Parameters p;
    if (mode == "imesa_paper") {
      p.optimization_params=std::nullopt;
      p.increment_outlier_mu=false;
      p.number_extra_iters=0;
    }
    return p;
  }
};
struct SeededIMesa : imesa::IMESAAgent {
  SeededIMesa(int r, const gtsam::Values& values, bool dogleg)
      : IMESAAgent(robotChar(r), parameters(), localParameters(dogleg)) {
    current_estimate_ = values;
    for (auto k : values.keys()) if (gtsam::Symbol(k).chr() == '#') observed_global_variables_.insert(k);
  }
  static imesa::IMESAAgentParams parameters() {
    imesa::IMESAAgentParams p;
    p.biased_prior_noise_model_sigmas = {{.1, 1.}};
    return p;
  }
  static gtsam::ISAM2Params localParameters(bool dogleg) {
    gtsam::ISAM2Params p;
    if (dogleg) p.optimizationParams=gtsam::ISAM2DoglegParams();
    return p;
  }
};

struct Local {
  int robot;
  gtsam::Values initial, current;
  gtsam::NonlinearFactorGraph physical, factors;
  std::map<int,double> times;
  std::map<int,int> labels;
  std::set<int> sources;
  std::unique_ptr<SeededIMesa> imesa;
  std::unique_ptr<SeededRiMesa> rimesa;
  void update(bool first) {
    const auto f = first ? factors : gtsam::NonlinearFactorGraph();
    const auto v = first ? initial : gtsam::Values();
    if (imesa) { imesa->update(f,v); current=imesa->getEstimate(); }
    else { rimesa->update(f,v); current=rimesa->getEstimate(); }
  }
  void append(const gtsam::NonlinearFactorGraph& f,const gtsam::Values& v) {
    if (imesa) { imesa->update(f,v); current=imesa->getEstimate(); }
    else { rimesa->update(f,v); current=rimesa->getEstimate(); }
  }
};

std::map<int,Local> parse(const Json& input, Links& links, const std::string& mode,bool stable=false) {
  require(input.at("schema") == "dranppr_frozen_joint_v1", "unsupported schema");
  const int root=id(input,"root_robot");
  std::map<int,Local> locals;
  std::map<int,std::set<int>> observers;
  std::map<int,gtsam::Point2> charts;
  std::map<int,int> labels;
  for (const auto& r : input.at("robots")) {
    const int robot=id(r,"id"); robotChar(robot);
    require(!locals.count(robot), "duplicate robot");
    Local local; local.robot=robot;
    for (const auto& row : r.at("poses")) {
      const int k=id(row,"index"); const double t=number(row,"timestamp_s");
      require(k == int(local.times.size()) && (k == 0 || t > local.times.at(k-1)), "invalid pose time/index");
      const auto v=vector(row.at("value"),3);
      local.times[k]=t; local.initial.insert(X(robot,k),gtsam::Pose2(v[0],v[1],v[2]));
    }
    require(!local.times.empty(), "empty trajectory");
    for (const auto& row : r.at("objects")) {
      const int c=id(row,"cluster"), label=id(row,"label");
      const gtsam::Point2 p=vector(row.at("point"),2);
      require(local.labels.emplace(c,label).second, "duplicate object");
      const auto prior=charts.emplace(c,p);
      require(prior.second || (prior.first->second-p).norm() == 0., "inconsistent object chart");
      const auto named=labels.emplace(c,label);
      require(named.second || named.first->second == label, "inconsistent label");
      observers[c].insert(robot); local.initial.insert(L(stable ? label : c),p);
    }
    std::set<int> odometry,sources,seen;
    for (const auto& row : r.at("odometry")) {
      const int a=id(row,"source"), b=id(row,"target"); const double dt=number(row,"dt");
      require(b == a+1 && local.times.count(b) && odometry.insert(b).second, "invalid odometry");
      require(dt > 0. && std::abs(local.times.at(b)-local.times.at(a)-dt)<1e-6, "invalid dt");
      const auto d=vector(row.at("delta"),3);
      local.physical.emplace_shared<DPGO::NativeBetweenPose2Factor>(X(robot,a),X(robot,b),
          gtsam::Pose2(d[0],d[1],d[2]),gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector3(.1,.1,.05)*std::sqrt(dt)).eval()));
    }
    require(odometry.size()+1 == local.times.size(), "missing odometry");
    local.factors=local.physical;
    for (const auto& row : r.at("observations")) {
      const int k=id(row,"pose"), c=id(row,"cluster"), source=id(row,"source");
      require(local.times.count(k) && local.labels.count(c) && sources.insert(source).second, "invalid observation source/key");
      const double range=number(row,"range"), bearing=number(row,"bearing");
      require(range > 0., "nonpositive range"); seen.insert(c);
      const auto noise=gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05,.1));
      const auto objectKey=L(stable ? local.labels.at(c) : c);
      const auto factor=boost::make_shared<Observation>(X(robot,k),objectKey,gtsam::Rot2(bearing),range,noise);
      local.physical.push_back(factor);
      if (mode == "rimesa") {
        auto kernel=boost::make_shared<risam::SIGKernel>(
            risam::SIGKernel::shapeParamFromInfThresh(.1,2,.95),
            risam::SIGKernel::MuUpdateStrategy::STABLE,risam::SIGKernel::MuInitIncrementStrategy::EQUAL_5);
        local.factors.push_back(risam::make_shared_graduated<Observation>(kernel,
            X(robot,k),objectKey,gtsam::Rot2(bearing),range,noise));
      } else local.factors.push_back(factor);
    }
    require(seen.size() == local.labels.size(), "unobserved object");
    local.sources=sources;
    if (robot == root) local.factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose2>>(
        X(robot,0),local.initial.at<gtsam::Pose2>(X(robot,0)),gtsam::noiseModel::Constrained::All(3));
    local.current=local.initial;
    locals.emplace(robot,std::move(local));
  }
  require(locals.size() >= 2 && locals.count(root), "invalid root/network");
  std::map<int,std::set<int>> adjacency;
  std::set<std::pair<int,int>> unique;
  for (const auto& edge : input.at("links")) {
    require(edge.is_array() && edge.size()==2 && edge[0].is_number_integer() && edge[1].is_number_integer(), "bad link");
    int a=edge[0].get<int>(),b=edge[1].get<int>(); if (a>b) std::swap(a,b);
    require(a!=b && locals.count(a) && locals.count(b) && unique.emplace(a,b).second,"invalid link");
    links.emplace_back(a,b); adjacency[a].insert(b); adjacency[b].insert(a);
  }
  auto connected=[&](const std::set<int>& group) {
    std::set<int> visited{*group.begin()}; std::vector<int> queue{*group.begin()};
    for (std::size_t i=0;i<queue.size();++i) for (auto b : adjacency[queue[i]])
      if (group.count(b) && visited.insert(b).second) queue.push_back(b);
    return visited == group;
  };
  std::set<int> robots; for (const auto& r : locals) robots.insert(r.first);
  require(connected(robots),"disconnected network");
  std::set<int> clusters;
  std::map<int,std::map<int,int>> labelClusters;
  for (const auto& row : input.at("clusters")) {
    const int c=id(row,"cluster"); std::set<int> group;
    for (const auto& r : row.at("observers")) { require(r.is_number_integer(),"bad observer"); require(group.insert(r.get<int>()).second,"duplicate observer"); }
    require(clusters.insert(c).second && observers.count(c) && group == observers.at(c),"invalid roster");
    require(connected(group) && id(row,"owner") == *group.begin() && id(row,"label") == labels.at(c),"invalid identity");
    for (auto robot : group) require(labelClusters[labels.at(c)].emplace(robot,c).second,"duplicate local identity");
  }
  require(clusters.size()==observers.size(),"incomplete roster");
  for (const auto& label : labelClusters) for (const auto& edge : links) {
    const auto& map=label.second;
    require(!map.count(edge.first) || !map.count(edge.second) || map.at(edge.first)==map.at(edge.second),
        "neighbor identity incorrectly split");
  }
  for (auto& r : locals) {
    if (stable) {
      auto& local=r.second;
      std::map<int,int> labels;
      for (const auto& p : local.labels) require(labels.emplace(p.second,p.second).second,"duplicate local barcode");
      local.labels=std::move(labels);
    }
    if (mode == "imesa" || mode == "imesa_dogleg")
      r.second.imesa=std::make_unique<SeededIMesa>(r.first,r.second.initial,mode=="imesa_dogleg");
    else r.second.rimesa=std::make_unique<SeededRiMesa>(r.first,r.second.initial,mode);
  }
  return locals;
}

struct Ledger {
  std::ofstream out;
  std::vector<Json> pending;
  std::uint64_t payload{0}, bytes{0}, messages{0};
  std::size_t round{0};
  explicit Ledger(const std::filesystem::path& path) : out(path) { out.exceptions(std::ios::failbit|std::ios::badbit); }
  void add(int a,int b,const char* kind,std::size_t size) {
    pending.push_back(Json{{"round",round},{"from",a},{"to",b},{"kind",kind},
        {"payload_bytes",size},{"application_bytes",size+40}});
    payload+=size; bytes+=size+40; ++messages;
  }
  void flush() {
    for (const auto& row : pending) out << row.dump() << '\n';
    pending.clear();
  }
};
std::size_t valueBytes(const gtsam::Values& values) {
  std::size_t n=0; for (const auto& v : values) n+=8+8*v.value.dim(); return n;
}
void communicate(Local& a, Local& b, Ledger& ledger) {
  if (a.imesa) {
    const auto da=a.imesa->declareNewSharedVariables(robotChar(b.robot));
    const auto db=b.imesa->declareNewSharedVariables(robotChar(a.robot));
    ledger.add(a.robot,b.robot,"declaration",8*(da.new_shared_variables.size()+da.observed_global_variables.size()));
    ledger.add(b.robot,a.robot,"declaration",8*(db.new_shared_variables.size()+db.observed_global_variables.size()));
    a.imesa->receiveNewSharedVariables(robotChar(b.robot),db);
    b.imesa->receiveNewSharedVariables(robotChar(a.robot),da);
    const auto sa=a.imesa->sendCommunication(robotChar(b.robot)), sb=b.imesa->sendCommunication(robotChar(a.robot));
    const auto pa=std::dynamic_pointer_cast<imesa::IMESACommunicationData>(sa);
    const auto pb=std::dynamic_pointer_cast<imesa::IMESACommunicationData>(sb);
    ledger.add(a.robot,b.robot,"solution",valueBytes(pa->shared_var_solution)+8*pa->vars_requested_for_init.size());
    ledger.add(b.robot,a.robot,"solution",valueBytes(pb->shared_var_solution)+8*pb->vars_requested_for_init.size());
    a.imesa->receiveCommunication(robotChar(b.robot),sb);
    b.imesa->receiveCommunication(robotChar(a.robot),sa);
  } else {
    rimesa::RIMESA::CommunicationHandler ha(a.rimesa->getCommunicationStateCache(robotChar(b.robot)));
    rimesa::RIMESA::CommunicationHandler hb(b.rimesa->getCommunicationStateCache(robotChar(a.robot)));
    const auto da=ha.declareNewSharedVariables(),db=hb.declareNewSharedVariables();
    const auto prior=ledger.payload;
    ledger.add(a.robot,b.robot,"declaration",8*(da.shared_variables.size()+da.observed_global_variables.size()));
    ledger.add(b.robot,a.robot,"declaration",8*(db.shared_variables.size()+db.observed_global_variables.size()));
    ha.receiveNewSharedVariables(db); hb.receiveNewSharedVariables(da);
    const auto sa=ha.sendCommunication(),sb=hb.sendCommunication();
    ledger.add(a.robot,b.robot,"solution",valueBytes(sa.shared_var_solution)+9*sa.vars_requested_for_init.size());
    ledger.add(b.robot,a.robot,"solution",valueBytes(sb.shared_var_solution)+9*sb.vars_requested_for_init.size());
    ha.receiveCommunication(sb); hb.receiveCommunication(sa);
    require(ha.getResult().has_value() && hb.getResult().has_value(),"incomplete handshake");
    require(std::abs(1000*(*ha.commSizeKB()+*hb.commSizeKB())-double(ledger.payload-prior))<1e-6,"upstream payload mismatch");
    a.rimesa->incorporateCommunicationResult(*ha.getResult());
    b.rimesa->incorporateCommunicationResult(*hb.getResult());
  }
}

Json metrics(const std::map<int,Local>& locals,int root) {
  double cost=0.,consensus=0.,rootMove=0.,otherMove=0.; std::size_t factors=0;
  std::map<int,gtsam::Point2> points;
  for (const auto& item : locals) {
    const auto& r=item.second; cost+=r.physical.error(r.current); factors+=r.physical.size();
    for (auto k : r.current.keys()) {
      if (gtsam::Symbol(k).chr()=='#') require(r.current.at<gtsam::Point2>(k).allFinite(),"nonfinite object");
      else require(r.current.at<gtsam::Pose2>(k).matrix().allFinite(),"nonfinite pose");
    }
    const double move=r.initial.at<gtsam::Pose2>(X(r.robot,0)).localCoordinates(r.current.at<gtsam::Pose2>(X(r.robot,0))).norm();
    if (r.robot==root) rootMove=move; else otherMove=std::max(otherMove,move);
    for (const auto& object : r.labels) {
      const auto p=r.current.at<gtsam::Point2>(L(object.first));
      const auto first=points.emplace(object.first,p); consensus=std::max(consensus,(first.first->second-p).norm());
    }
  }
  require(std::isfinite(cost),"nonfinite physical cost");
  return {{"physical_cost",cost},{"consensus_max_m",consensus},{"root_displacement",rootMove},
      {"nonroot_root_displacement_max",otherMove},{"physical_factors",factors}};
}
void save(const std::map<int,Local>& locals,const std::filesystem::path& output,int root) {
  std::ofstream out(output/"final_estimates.csv"); out.exceptions(std::ios::failbit|std::ios::badbit);
  out << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw,component_root\n";
  for (const auto& item : locals) {
    const auto& r=item.second;
    for (const auto& t : r.times) {
      const auto p=r.current.at<gtsam::Pose2>(X(r.robot,t.first));
      out << r.robot << ",pose," << t.first << ',' << t.second << ',' << p.x() << ',' << p.y() << ',' << p.theta() << ',' << root << '\n';
    }
    for (const auto& label : r.labels) {
      const auto p=r.current.at<gtsam::Point2>(L(label.first));
      out << r.robot << ",landmark," << label.second << ",," << p.x() << ',' << p.y() << ",," << root << '\n';
    }
  }
}

#include "online.h"

int main(int argc,char** argv) {
  try {
    if (argc==8 && std::string(argv[4])=="--online") return onlineMain(argv);
    require(argc==6,"usage: native-matched-baseline package output imesa|imesa_dogleg|imesa_paper|rimesa|rimesa_gaussian rounds byte_budget");
    const std::string mode=argv[3];
    require(mode=="imesa" || mode=="imesa_dogleg" || mode=="imesa_paper" || mode=="rimesa" || mode=="rimesa_gaussian","invalid mode");
    const int rounds=std::stoi(argv[4]); const auto budget=std::stoull(argv[5]);
    require(rounds>0 && rounds<=10000,"invalid round cap");
    const std::filesystem::path output=argv[2]; require(!std::filesystem::exists(output),"output exists");
    std::ifstream in(argv[1]); require(in.good(),"cannot open package"); Json input; in>>input;
    const auto setup=Clock::now(); Links links; auto locals=parse(input,links,mode);
    const double setupSeconds=elapsed(setup); const int root=id(input,"root_robot");
    std::filesystem::create_directories(output);
    Ledger ledger(output/"messages.jsonl");
    std::ofstream trace(output/"trace.jsonl"); trace.exceptions(std::ios::failbit|std::ios::badbit);
    const double initial=metrics(locals,root).at("physical_cost").get<double>();
    double service=0.,startup=0.; int completed=0; std::string stop="round_cap";
    for (int round=0;round<rounds;++round) {
      std::uint64_t expected=0;
      for (const auto& edge : links) {
        const auto& a=locals.at(edge.first); const auto& b=locals.at(edge.second);
        std::size_t shared=0; for (const auto& p : a.labels) shared+=b.labels.count(p.first);
        expected+=160+8*(a.labels.size()+b.labels.size())+48*shared;
        if (mode!="imesa" && mode!="imesa_dogleg" && round>0) expected+=16*shared;
      }
      if (budget && ledger.bytes+expected>budget) { stop="byte_budget"; break; }
      const auto start=Clock::now(); const auto prior=ledger.bytes; ledger.round=round;
      for (const auto& edge : links) communicate(locals.at(edge.first),locals.at(edge.second),ledger);
      require(ledger.bytes-prior==expected,"preflight byte count mismatch");
      for (auto& r : locals) r.second.update(round==0);
      const double work=elapsed(start); service+=work; if (round==0) startup=work;
      ledger.flush();
      auto row=metrics(locals,root);
      row.update({{"round",round+1},{"service_seconds",service},{"setup_seconds",setupSeconds},
          {"payload_bytes",ledger.payload},{"application_bytes",ledger.bytes},{"update_seconds",work}});
      trace << row.dump() << '\n'; ++completed;
    }
    require(completed>0,"budget cannot pay bootstrap transaction");
    auto summary=metrics(locals,root);
    summary["final_cost"]=summary.at("physical_cost"); summary.erase("physical_cost");
    summary.update({{"initial_cost",initial},{"mode",mode},{"rounds",completed},{"stop_reason",stop},
        {"setup_seconds",setupSeconds},{"service_seconds",service},{"startup_seconds",startup},
        {"application_bytes",ledger.bytes},{"payload_bytes",ledger.payload},{"messages",ledger.messages},
        {"gauge_priors",1},{"truth_used",false},{"byte_budget",budget},
        {"scope","Frozen common-past state; baseline startup and fixed-data updates, not online"},
        {"solver_parameters","imesa=RSS24 defaults with pose BP sigmas 0.1/1.0; imesa_dogleg changes its local step; imesa_paper=2026 paper GN/decay configuration without outlier oracle; rimesa_gaussian=riMESA without robust factors"},
        {"outlier_oracle_used",false},
        {"ledger_exclusions","Common bootstrap generation, transport framing/ACKs/retransmission; logical directed message headers are 40 bytes"}});
    save(locals,output,root);
    std::ofstream record(output/"summary.json"); record.exceptions(std::ios::failbit|std::ios::badbit);
    record << summary.dump(2) << '\n'; std::cout << summary.dump() << std::endl;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
