#pragma once
#include "../online_stream.h"

void appendFrame(Local& local,const std::vector<Json>& frame,const std::string& mode) {
  const auto& head=frame.front(); const int k=id(head,"pose_index");
  const double t=number(head,"timestamp_s"),dt=number(head,"duration_s");
  require(k>0 && k==int(local.times.size()) && id(head,"from_pose_index")==k-1,"invalid suffix pose index");
  require(dt>0. && std::abs(t-local.times.at(k-1)-dt)<1e-6,"invalid suffix dt");
  const auto d=vector(head.at("delta"),3); const gtsam::Pose2 relative(d[0],d[1],d[2]);
  const auto predicted=local.current.at<gtsam::Pose2>(X(local.robot,k-1)).compose(relative);
  gtsam::Values values; values.insert(X(local.robot,k),predicted);
  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<DPGO::NativeBetweenPose2Factor>(X(local.robot,k-1),X(local.robot,k),relative,
      gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector3(.1,.1,.05)*std::sqrt(dt)).eval()));
  local.physical.push_back(factors.back());
  for (std::size_t i=1;i<frame.size();++i) {
    const auto& e=frame[i]; const int source=id(e,"source_row"),label=id(e,"landmark_subject");
    require(e.at("kind")=="observation" && local.sources.insert(source).second,"duplicate suffix source");
    const double range=number(e,"range_m"),bearing=number(e,"bearing_rad");
    require(range>0.,"invalid suffix range");
    if (local.labels.emplace(label,label).second) values.insert(L(label),
        predicted.transformFrom(gtsam::Point2(range*std::cos(bearing),range*std::sin(bearing))));
    const auto noise=gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05,.1));
    const auto f=boost::make_shared<Observation>(X(local.robot,k),L(label),gtsam::Rot2(bearing),range,noise);
    local.physical.push_back(f);
    if (mode=="rimesa") {
      const auto kernel=boost::make_shared<risam::SIGKernel>(risam::SIGKernel::shapeParamFromInfThresh(.1,2,.95),
          risam::SIGKernel::MuUpdateStrategy::STABLE,risam::SIGKernel::MuInitIncrementStrategy::EQUAL_5);
      factors.push_back(risam::make_shared_graduated<Observation>(kernel,X(local.robot,k),L(label),
          gtsam::Rot2(bearing),range,noise));
    } else factors.push_back(f);
  }
  local.times[k]=t; local.append(factors,values);
}

double observerConsensus(const std::map<int,Local>& locals,const Links& links) {
  double maximum=0.; std::map<int,std::set<int>> observers,adjacency;
  for (const auto& r : locals) for (const auto& p : r.second.labels) observers[p.first].insert(r.first);
  for (const auto& e : links) { adjacency[e.first].insert(e.second); adjacency[e.second].insert(e.first); }
  for (auto item : observers) {
    auto& unseen=item.second;
    while (!unseen.empty()) {
      const int owner=*unseen.begin(); unseen.erase(owner); std::vector<int> queue{owner};
      const auto reference=locals.at(owner).current.at<gtsam::Point2>(L(item.first));
      for (std::size_t i=0;i<queue.size();++i) {
        const auto point=locals.at(queue[i]).current.at<gtsam::Point2>(L(item.first));
        maximum=std::max(maximum,(reference-point).norm());
        for (auto next : adjacency[queue[i]]) if (unseen.erase(next)) queue.push_back(next);
      }
    }
  }
  return maximum;
}

int onlineMain(char** argv) {
  const std::string mode=argv[3],schedule=argv[7];
  require(mode=="imesa_paper" || mode=="rimesa" || mode=="rimesa_gaussian","unsupported online mode");
  const double duration=std::stod(argv[6]);
  const bool credits=std::filesystem::is_regular_file(schedule);
  const int roundCap=credits ? 10000 : std::stoi(schedule);
  require(roundCap>0 && roundCap<=10000,"invalid online round cap");
  std::ifstream in(argv[1]); require(in.good(),"cannot open package"); Json package; in>>package;
  const std::filesystem::path output=argv[2]; require(!std::filesystem::exists(output),"output exists");
  std::filesystem::create_directories(output);
  const auto setup=Clock::now(); Links links; auto locals=parse(package,links,mode,true);
  const double setupSeconds=elapsed(setup),first=online::firstTime(package),cutoff=online::cutoffTime(package)-first;
  const int root=id(package,"root_robot"); const auto initial=metrics(locals,root);
  Ledger ledger(output/"messages.jsonl");
  std::map<std::pair<int,int>,std::size_t> previousShared;
  auto expected=[&]() {
    std::uint64_t bytes=0;
    for (const auto& edge : links) {
      const auto& a=locals.at(edge.first); const auto& b=locals.at(edge.second);
      std::size_t shared=0; for (const auto& p : a.labels) shared+=b.labels.count(p.first);
      bytes+=160+8*(a.labels.size()+b.labels.size())+16*previousShared[edge]+48*shared;
    }
    return bytes;
  };
  auto sweep=[&](bool bootstrap) {
    const auto predicted=expected(),before=ledger.bytes;
    for (const auto& e : links) {
      communicate(locals.at(e.first),locals.at(e.second),ledger);
      std::size_t shared=0;
      for (const auto& p : locals.at(e.first).labels) shared+=locals.at(e.second).labels.count(p.first);
      previousShared[e]=shared;
    }
    require(ledger.bytes-before==predicted,"online preflight byte mismatch");
    for (auto& r : locals) r.second.update(bootstrap);
    ++ledger.round;
  };
  const auto startup=Clock::now(); sweep(true);
  const double startupSeconds=elapsed(startup); ledger.flush();
  double service=startupSeconds,finish=cutoff+setupSeconds+startupSeconds,maxBacklog=setupSeconds+startupSeconds;
  std::vector<double> services;
  std::size_t frames=0,packets=0; std::uint64_t ceiling=0;
  std::ifstream grants; if (credits) { grants.open(schedule); require(grants.good(),"cannot open byte credits"); }
  std::ofstream causal(output/"causal_estimates.csv"),updates(output/"updates.csv"),trace(output/"trace.jsonl");
  causal.exceptions(std::ios::failbit|std::ios::badbit); updates.exceptions(std::ios::failbit|std::ios::badbit);
  trace.exceptions(std::ios::failbit|std::ios::badbit);
  causal << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw,component_root,all_connected\n";
  updates << std::setprecision(17) << "robot,timestamp_s,elapsed_s,total_ms,serial_backlog_s,application_bytes,rounds\n";
  online::suffix(argv[5],package,duration,[&](const std::vector<Json>& frame) {
    const int robot=id(frame.front(),"robot"),k=id(frame.front(),"pose_index");
    const double t=number(frame.front(),"timestamp_s"),arrival=t-first;
    if (credits) {
      std::string line; require(bool(std::getline(grants,line)),"missing byte credit"); const auto g=Json::parse(line);
      require(g.at("robot")==robot && g.at("timestamp_s")==t && g.at("application_bytes").get<std::uint64_t>()>=ceiling,
          "invalid byte credit frame"); ceiling=g.at("application_bytes");
      require(ledger.bytes<=ceiling,"byte credit cannot pay startup");
    }
    const auto start=Clock::now(); const auto beforeBytes=ledger.bytes;
    appendFrame(locals.at(robot),frame,mode);
    int rounds=0;
    while (rounds<roundCap && (!credits || ledger.bytes+expected()<=ceiling)) { sweep(false); ++rounds; }
    require(!credits || rounds<roundCap,"byte replay round safety cap reached");
    const double work=elapsed(start); ledger.flush(); service+=work; services.push_back(work);
    finish=std::max(finish,arrival)+work; maxBacklog=std::max(maxBacklog,finish-arrival);
    const auto p=locals.at(robot).current.at<gtsam::Pose2>(X(robot,k));
    causal << robot << ",pose," << k << ',' << t << ',' << p.x() << ',' << p.y() << ',' << p.theta() << ',' << root << ",1\n";
    updates << robot << ',' << t << ',' << arrival << ',' << 1000*work << ',' << finish-arrival << ','
        << ledger.bytes-beforeBytes << ',' << rounds << '\n';
    auto row=metrics(locals,root); row["consensus_max_m"]=observerConsensus(locals,links);
    row.update({{"robot",robot},{"timestamp_s",t},{"application_bytes",ledger.bytes},
        {"credit_bytes",credits ? Json(ceiling) : Json(nullptr)},{"rounds",rounds}});
    trace << row.dump() << '\n'; ++frames; packets+=frame.size();
  });
  require(frames>0,"empty suffix");
  std::sort(services.begin(),services.end());
  auto quantile=[&](double q) { return services.at(std::size_t(q*(services.size()-1))); };
  auto summary=metrics(locals,root); summary["consensus_max_m"]=observerConsensus(locals,links);
  summary.update({{"mode",mode},{"truth_used",false},{"outlier_oracle_used",false},{"gauge_priors",1},
      {"initial_cost",initial.at("physical_cost")},{"prefix_physical_factors",initial.at("physical_factors")},
      {"warm_start_cutoff_s",cutoff},{"setup_seconds",setupSeconds},{"startup_seconds",startupSeconds},
      {"service_seconds",service},{"frames",frames},{"packets",packets},{"rounds",ledger.round},
      {"application_bytes",ledger.bytes},{"payload_bytes",ledger.payload},{"messages",ledger.messages},
      {"frame_p50_s",quantile(.5)},{"frame_p95_s",quantile(.95)},{"frame_p99_s",quantile(.99)},
      {"frame_max_s",services.back()},{"serial_max_backlog_s",maxBacklog},
      {"schedule",credits ? "DRAN cumulative byte-credit replay; full sweeps only" : "fixed all-edge sweeps per frame"},
      {"rounds_per_frame_cap",roundCap},{"scope","Common-past warm start with persistent solvers and chronological suffix; single-process simulation"},
      {"timing_scope","Insertion/solver updates and in-memory messages; excludes JSON input, evaluation and trace serialization/file I/O"},
      {"ledger_exclusions","Common past-state generation, static topology, transport framing/ACKs/retransmission; 40 byte logical headers"}});
  save(locals,output,root);
  std::ofstream record(output/"summary.json"); record.exceptions(std::ios::failbit|std::ios::badbit);
  record << summary.dump(2) << '\n'; std::cout << summary.dump() << std::endl;
  return 0;
}
