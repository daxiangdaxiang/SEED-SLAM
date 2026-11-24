#include "joint_package.h"
#include "chart_audit.h"
#include "../online_stream.h"
#include <DPGO/NativeBetweenPose2Factor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>

using Observation = gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>;
using Group = std::set<Robot>;
using Packet = std::vector<double>;

struct ControlLedger {
  std::map<Robot, Group> adjacency;
  std::ofstream stream;
  std::vector<Json> pending;
  std::uint64_t epoch{0}, bytes{0}, payloadBytes{0}, messages{0};
  ControlLedger(const Links& links, const std::filesystem::path& path) : stream(path) {
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    for (const auto& edge : links) {
      adjacency[edge.first].insert(edge.second);
      adjacency[edge.second].insert(edge.first);
    }
  }
  Packet send(Robot from, Robot to, Robot origin, const std::string& kind, const Packet& packet) {
    jointRequire(adjacency.at(from).count(to), "control message outside topology");
    for (double value : packet) jointRequire(std::isfinite(value), "nonfinite control packet");
    payloadBytes += 8*packet.size();
    bytes += 40+8*packet.size();
    ++messages;
    pending.push_back(Json{{"epoch", epoch}, {"from", from}, {"to", to}, {"origin", origin},
        {"kind", kind}, {"payload", packet}, {"application_bytes", 40+8*packet.size()}});
    return packet;
  }
  void flush() {
    for (const auto& row : pending) stream << row.dump() << '\n';
    pending.clear();
  }
  std::map<Robot, Packet> flood(Robot origin, const Group& group, const std::string& kind, const Packet& packet) {
    jointRequire(group.count(origin), "flood origin absent");
    std::map<Robot, Packet> received{{origin, packet}};
    std::vector<Robot> queue{origin};
    for (std::size_t i=0; i<queue.size(); ++i) for (auto next : adjacency.at(queue[i])) {
      if (!group.count(next) || received.count(next)) continue;
      received[next] = send(queue[i], next, origin, kind, received.at(queue[i]));
      queue.push_back(next);
    }
    jointRequire(received.size() == group.size(), "disconnected control flood");
    return received;
  }
};

struct Replay {
  Links links;
  Group configured;
  std::map<Robot, Local> robots;
  std::map<Robot, Robot> roots;
  std::map<Robot, std::set<Robot>> sources;
  ControlLedger ledger;
  Json merges = Json::array();
  std::size_t frames{0}, packets{0}, optimizerBytes{0}, optimizerPayload{0}, pcgIterations{0};
  double first{-1.}, allConnected{-1.}, maxResidual{0.}, workSeconds{0.}, finish{0.}, maxBacklog{0.};
  double setupSeconds{0.}, warmCutoff{-1.}, initialCost{0.};
  std::size_t prefixFactors{0};
  gtsam::Pose2 rootInitial;
  std::vector<double> services;
  std::ofstream causal, updates;
  std::ofstream chartAudit;
  std::vector<Json> pendingAudit;
  std::string mode;
  bool joint, reuseCharts, conditionalResponse, lazyRefresh, rigidSubspace, fusedOneStep;
  double suffixAddedBearingVariance, suffixBearingSigma;
  std::array<double,5> suffixVarianceMultipliers;
  std::size_t interfaceIterations;
  Json solveTotals = {{"response_builds",0}, {"response_reuses",0}, {"refreshed_keys",0},
      {"factors_linearized",0}, {"model_seconds",0.}, {"interface_seconds",0.},
      {"publication_seconds",0.}, {"evaluation_and_collective_seconds",0.},
      {"update_snapshot_seconds",0.}, {"update_linearization_seconds",0.}, {"update_contribution_seconds",0.},
      {"conditionals_rebuilt",0}, {"conditionals_reused",0}, {"conditional_full_rebuilds",0},
      {"conditional_seconds",0.}, {"recovery_seconds",0.}, {"directional_seconds",0.},
      {"refresh_requests",0}, {"refresh_rounds",0}, {"subspace_steps",0}};

  Replay(const Links& graph, const std::filesystem::path& output, const std::string& variant,
      bool audit=false,bool lazy=false,double addedBearingVariance=0.,
      std::array<double,5> varianceMultipliers={{1.,1.,1.,1.,1.}},bool subspace=false,bool fused=false)
      : links(graph), ledger(graph, output/"control_messages.jsonl"),
        causal(output/"causal_estimates.csv"), updates(output/"updates.csv"), mode(variant),
        joint(mode != "independent"), reuseCharts(mode.find("cached") != std::string::npos),
        conditionalResponse(mode.find("conditional") != std::string::npos),
        lazyRefresh(lazy), rigidSubspace(subspace), fusedOneStep(fused),
        suffixAddedBearingVariance(addedBearingVariance),
        suffixBearingSigma(addedBearingVariance == 0. && varianceMultipliers[3] == 1. ? .05 :
            std::sqrt(.05*.05*varianceMultipliers[3]+addedBearingVariance)),
        suffixVarianceMultipliers(varianceMultipliers),
        interfaceIterations(mode.find("_one") != std::string::npos ? 1 : 500) {
    if (audit) {
      chartAudit.open(output/"chart_audit.jsonl");
      chartAudit.exceptions(std::ios::failbit|std::ios::badbit);
    }
    for (const auto& edge : links) { configured.insert(edge.first); configured.insert(edge.second); }
    causal.exceptions(std::ios::failbit|std::ios::badbit);
    updates.exceptions(std::ios::failbit|std::ios::badbit);
    causal << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw,component_root,all_connected\n";
    updates << std::setprecision(17) << "robot,timestamp_s,elapsed_s,components,component_size,cost_before,cost_after,accepted,pcg_iterations,pcg_relative_residual,optimizer_bytes,control_bytes,insert_ms,coordination_ms,solve_ms,total_ms,serial_backlog_s,model_ms,interface_ms,publication_ms,evaluation_collective_ms,response_builds,response_reuses,refreshed_keys,factors_linearized,update_snapshot_ms,update_linearization_ms,update_contribution_ms,conditionals_rebuilt,conditionals_reused,conditional_full_rebuilds,conditional_ms,recovery_ms,directional_ms,refresh_requests,refresh_rounds,full_step_prediction,full_step_decrease\n";
  }
  Group group(Robot robot) const {
    Group result;
    for (const auto& entry : roots) if (entry.second == roots.at(robot)) result.insert(entry.first);
    return result;
  }
  std::size_t components() const {
    Group values;
    for (const auto& entry : roots) values.insert(entry.second);
    return values.size();
  }
  void warmStart(const Json& package) {
    const auto start=Clock::now();
    jointRequire(package.at("schema")=="dranppr_frozen_joint_v1",
        "warm-start requires nominal v1 schema; weighted frozen graphs are audit-only");
    const Robot root=identifier(package,"root_robot"); Links graph;
    auto loaded=parse(package,root,graph);
    jointRequire(graph==links && joint,"warm-start topology/mode mismatch");
    first=online::firstTime(package); warmCutoff=online::cutoffTime(package)-first;
    allConnected=warmCutoff; finish=warmCutoff;
    for (auto& local : loaded) {
      const Robot robot=local.id;
      // Native factors retain stable local barcode keys as observer components grow.
      std::map<gtsam::Key,gtsam::Key> keyMap;
      for (auto k : local.initial.keys()) keyMap[k]=gtsam::Symbol(k).chr()=='l' ? L(local.labels.at(gtsam::Symbol(k).index())) : k;
      NativeModelUpdate update;
      for (const auto& key : keyMap) update.newValues.insert(key.second,local.initial.at(key.first));
      const auto& record=*std::find_if(package.at("robots").begin(),package.at("robots").end(),
          [&](const Json& r) { return identifier(r,"id")==robot; });
      for (const auto& factor : record.at("odometry")) {
        const auto target=identifier(factor,"target");
        update.addFactors[2*std::uint64_t(target)]=boost::make_shared<NativeBetweenPose2Factor>(
            X(target-1),X(target),gtsam::Pose2(factor.at("delta")[0],factor.at("delta")[1],factor.at("delta")[2]),
            gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector3(.1,.1,.05)*std::sqrt(number(factor,"dt"))).eval()));
      }
      for (const auto& o : record.at("observations")) {
        const auto source=identifier(o,"source"),c=identifier(o,"cluster"); sources[robot].insert(source);
        update.addFactors[2*std::uint64_t(source)+1]=boost::make_shared<Observation>(X(identifier(o,"pose")),
            keyMap.at(L(c)),gtsam::Rot2(number(o,"bearing")),number(o,"range"),
            gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(.05,.1)));
      }
      std::map<Robot,Robot> stable;
      for (const auto& p : local.labels) stable.emplace(p.second,p.second);
      local.labels=std::move(stable); local.reuseCharts=reuseCharts;
      local.conditionalResponse=conditionalResponse;
      local.lazyRefresh=lazyRefresh;
      local.model=std::make_unique<IncrementalLocalModel>(conditionalResponse); local.model->update(update);
      local.initial=local.model->bases(); local.estimate=local.initial; local.graph=local.model->factorGraph();
      if (chartAudit.is_open()) local.auditUpdate=[this](const Local& current,
          const NativeModelUpdate& edit,const std::string& stage) {
        auto record=auditChartRefresh(*current.model,edit,current.fixed);
        record["pose_origin"]=auditChartRefresh(*current.model,edit,current.fixed,true);
        record.update({{"robot",current.id},{"epoch",frames},{"stage",stage}});
        pendingAudit.push_back(std::move(record));
      };
      prefixFactors+=local.graph.size(); initialCost+=local.graph.error(local.current());
      roots[robot]=root; robots.emplace(robot,std::move(local));
    }
    rootInitial=robots.at(root).current().at<gtsam::Pose2>(X(0));
    setupSeconds=jointSeconds(start); finish+=setupSeconds;
  }
  void insert(const std::vector<Json>& frame) {
    const auto& head = frame.front();
    const auto robot = identifier(head, "robot"), index = identifier(head, "pose_index");
    const double t = number(head, "timestamp_s");
    jointRequire(configured.count(robot) && head.at("kind") == "pose", "invalid frame robot/kind");
    for (const auto& row : frame) {
      jointRequire(identifier(row, "robot") == robot && identifier(row, "pose_index") == index &&
          number(row, "timestamp_s") == t && number(row, "available_time_s") == t &&
          number(row, "support_max_time_s") <= t, "mixed or noncausal frame");
    }
    if (!robots.count(robot)) {
      Local local; local.id = robot;
      local.reuseCharts = reuseCharts;
      local.model = std::make_unique<IncrementalLocalModel>();
      robots.emplace(robot, std::move(local));
      roots[robot] = robot;
    }
    auto& local = robots.at(robot);
    jointRequire(index == local.times.size(), "nonconsecutive pose");
    NativeModelUpdate insertion;
    gtsam::Pose2 predicted;
    if (index == 0) {
      jointRequire(head.at("delta").is_null() && head.at("from_pose_index").is_null() &&
                   number(head, "duration_s") == 0., "invalid root");
    } else {
      jointRequire(identifier(head, "from_pose_index") == index-1, "invalid parent");
      const double dt = number(head, "duration_s");
      jointRequire(dt > 0. && std::abs(t-local.times.at(index-1)-dt) < 1e-6, "invalid duration");
      const auto d = head.at("delta").get<Packet>();
      jointRequire(d.size() == 3 && std::isfinite(d[0]) && std::isfinite(d[1]) && std::isfinite(d[2]), "invalid odometry");
      const gtsam::Pose2 relative(d[0], d[1], d[2]);
      predicted = local.current().at<gtsam::Pose2>(X(index-1)).compose(relative);
      gtsam::Vector3 sigmas=(gtsam::Vector3(.1,.1,.05)*std::sqrt(dt)).eval();
      for (int k=0;k<3;++k) sigmas[k]*=std::sqrt(suffixVarianceMultipliers[k]);
      insertion.addFactors[2*std::uint64_t(index)] = boost::make_shared<NativeBetweenPose2Factor>(
          X(index-1), X(index), relative, gtsam::noiseModel::Diagonal::Sigmas(sigmas));
    }
    insertion.newValues.insert(X(index), predicted);
    for (std::size_t i=1; i<frame.size(); ++i) {
      const auto& row = frame[i];
      jointRequire(row.at("kind") == "observation", "unexpected frame event");
      const auto source = identifier(row, "source_row"), label = identifier(row, "landmark_subject");
      jointRequire(sources[robot].insert(source).second, "duplicate observation source");
      const double range = number(row, "range_m"), bearing = number(row, "bearing_rad");
      jointRequire(range > 0., "nonpositive range");
      if (local.labels.emplace(label, label).second) insertion.newValues.insert(L(label),
          predicted.transformFrom(gtsam::Point2(range*std::cos(bearing), range*std::sin(bearing))));
      insertion.addFactors[2*std::uint64_t(source)+1] = boost::make_shared<Observation>(
          X(index), L(label), gtsam::Rot2(bearing), range,
          gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(
              suffixBearingSigma,.1*std::sqrt(suffixVarianceMultipliers[4]))));
    }
    local.updateModel(insertion,"insertion");
    if (reuseCharts) local.estimate.insert(insertion.newValues);
    for (const auto& factor : insertion.addFactors) local.graph.push_back(factor.second);
    local.times[index] = t;
  }

  std::map<Robot, gtsam::Point2> decodePoints(const Packet& packet) const {
    jointRequire(packet.size() >= 2 && packet.size() == 2+3*std::size_t(packet[1]), "malformed peer map");
    std::map<Robot, gtsam::Point2> points;
    for (std::size_t i=2; i<packet.size(); i+=3) {
      jointRequire(points.emplace(Robot(packet[i]), gtsam::Point2(packet[i+1],packet[i+2])).second, "duplicate peer object");
    }
    return points;
  }
  void connect(Robot active, double elapsed) {
    bool changed;
    do {
      changed = false;
      std::map<std::pair<Robot,Robot>, Packet> inbox;
      for (const auto& edge : links) for (const auto& direction : {edge, std::make_pair(edge.second,edge.first)}) {
        const auto from = direction.first, to = direction.second;
        if (!robots.count(from) || !robots.count(to) || roots.at(from) == roots.at(to)) continue;
        if (roots.at(from) != roots.at(active) && roots.at(to) != roots.at(active)) continue;
        const auto& local = robots.at(from);
        Packet packet{double(roots.at(from)), double(local.labels.size())};
        for (const auto& object : local.labels) {
          const auto& p = local.current().at<gtsam::Point2>(L(object.first));
          packet.insert(packet.end(), {double(object.first),p.x(),p.y()});
        }
        inbox[direction] = ledger.send(from, to, from, "peer_map", packet);
      }
      for (const auto& edge : links) {
        if (!inbox.count(edge)) continue;
        Robot target=edge.first, source=edge.second;
        if (roots.at(target) > roots.at(source)) std::swap(target, source);
        const auto a = decodePoints(inbox.at({target,source}));
        const auto b = decodePoints(inbox.at({source,target}));
        std::vector<Robot> common;
        gtsam::Point2 ca=gtsam::Point2::Zero(), cb=gtsam::Point2::Zero();
        for (const auto& point : a) if (b.count(point.first)) {
          common.push_back(point.first); ca += point.second; cb += b.at(point.first);
        }
        if (common.size() < 2) continue;
        ca /= double(common.size()); cb /= double(common.size());
        double cross=0., dot=0., spreadA=0., spreadB=0.;
        for (auto label : common) {
          const gtsam::Point2 x=b.at(label)-cb, y=a.at(label)-ca;
          dot += x.dot(y); cross += x.x()*y.y()-x.y()*y.x();
          spreadA += y.squaredNorm(); spreadB += x.squaredNorm();
        }
        if (spreadA <= 1e-16 || spreadB <= 1e-16 || std::hypot(cross,dot) <=
            std::numeric_limits<double>::epsilon()*std::sqrt(spreadA*spreadB)) continue;
        const double yaw = std::atan2(cross,dot);
        const auto rotation = gtsam::Rot2(yaw);
        const gtsam::Point2 translation = ca-rotation.rotate(cb);
        const auto targetGroup = group(target), sourceGroup = group(source);
        const Robot newRoot = roots.at(target), oldRoot = roots.at(source);
        Packet decision{double(newRoot), translation.x(), translation.y(), yaw, double(targetGroup.size()+sourceGroup.size())};
        for (auto r : targetGroup) decision.push_back(r);
        for (auto r : sourceGroup) decision.push_back(r);
        const auto delivered = ledger.send(target, source, target, "merge", decision);
        const auto transformed = ledger.flood(source, sourceGroup, "merge_transform", delivered);
        ledger.flood(target, targetGroup, "merge_membership", decision);
        double costChange = 0.;
        for (auto r : sourceGroup) {
          auto& local = robots.at(r);
          const auto& message = transformed.at(r);
          const gtsam::Pose2 transform(message[1], message[2], message[3]);
          const double before = local.graph.error(local.current());
          NativeModelUpdate update;
          for (auto key : local.model->bases().keys()) {
            if (gtsam::Symbol(key).chr() == 'x') update.relinearizeValues.insert(key,
                transform.compose(local.current().at<gtsam::Pose2>(key)));
            else update.relinearizeValues.insert(key, transform.transformFrom(local.current().at<gtsam::Point2>(key)));
          }
          local.updateModel(update,"gauge_merge");
          if (reuseCharts) local.estimate = local.model->bases();
          const double difference = std::abs(before-local.graph.error(local.current()));
          jointRequire(difference <= 1e-7*std::max(1.,before), "gauge transform changed physical loss");
          costChange += difference;
          roots[r] = newRoot;
        }
        merges.push_back({{"elapsed_s", elapsed}, {"old_root", oldRoot}, {"new_root", newRoot},
            {"target", target}, {"source", source}, {"common_labels", common},
            {"transform", {translation.x(),translation.y(),yaw}}, {"gauge_cost_change_abs", costChange}});
        changed = true;
        break;
      }
    } while (changed);
  }

  void identities(const Group& component) {
    std::map<Robot, Group> observers;
    // Explicit allgather of arrived rosters; every node can derive this same map.
    for (auto r : component) {
      Packet roster{double(r), double(robots.at(r).labels.size())};
      for (const auto& object : robots.at(r).labels) roster.push_back(object.first);
      const auto copies = ledger.flood(r, component, "roster", roster);
      const auto& received = copies.at(*component.begin());
      for (std::size_t i=2; i<received.size(); ++i) observers[Robot(received[i])].insert(Robot(received[0]));
      robots.at(r).interfaceKeys.clear();
    }
    std::map<Robot, NativeModelUpdate> updates;
    for (const auto& item : observers) {
      Group unseen=item.second;
      while (!unseen.empty()) {
        const Robot owner=*unseen.begin();
        Group cluster{owner}; unseen.erase(owner);
        std::vector<Robot> queue{owner};
        for (std::size_t i=0; i<queue.size(); ++i) for (auto next : ledger.adjacency.at(queue[i])) {
          if (unseen.erase(next)) { cluster.insert(next); queue.push_back(next); }
        }
        const auto& base = robots.at(owner).model->bases().at<gtsam::Point2>(L(item.first));
        Packet chart{double(item.first),base.x(),base.y()};
        if (reuseCharts) {
          const auto& point = robots.at(owner).current().at<gtsam::Point2>(L(item.first));
          chart.insert(chart.end(), {point.x(),point.y()});
        }
        const auto copies = ledger.flood(owner, cluster, "identity_chart", chart);
        for (auto r : cluster) {
          const auto& packet = copies.at(r);
          const auto localKey=L(item.first), identity=L((std::uint64_t(item.first)<<24)|owner);
          robots.at(r).interfaceKeys[localKey] = identity;
          const gtsam::Point2 point(packet[1],packet[2]);
          if (reuseCharts) robots.at(r).estimate.update(localKey, gtsam::Point2(packet[3],packet[4]));
          if ((robots.at(r).model->bases().at<gtsam::Point2>(localKey)-point).norm() != 0.) {
            updates[r].relinearizeValues.insert(localKey, point);
          }
        }
      }
    }
    for (auto& entry : updates) robots.at(entry.first).updateModel(entry.second,"identity_chart");
  }

  void push(const std::vector<Json>& frame) {
    const auto start = Clock::now();
    if (first < 0.) first = number(frame.front(),"timestamp_s");
    const auto robot=identifier(frame.front(),"robot"), index=identifier(frame.front(),"pose_index");
    const double t=number(frame.front(),"timestamp_s"), elapsed=t-first;
    const auto priorBytes=ledger.bytes;
    ledger.epoch = frames;
    insert(frame);
    const double insertMs=1000*jointSeconds(start);
    const auto coordinationStart=Clock::now();
    if (joint) connect(robot,elapsed);
    const auto members=group(robot);
    if (members.size() > 1) identities(members);
    if (allConnected < 0. && roots.size() == configured.size() && components() == 1) allConnected=elapsed;
    const double coordinationMs=1000*jointSeconds(coordinationStart);
    const auto solveStart=Clock::now();
    double before=0., after=0., residual=0.;
    std::size_t accepted=0, pcg=0, bytes=0;
    Json fullPrediction=nullptr,fullDecrease=nullptr;
    Json stages = {{"response_builds",0}, {"response_reuses",0}, {"refreshed_keys",0},
        {"factors_linearized",0}, {"model_seconds",0.}, {"interface_seconds",0.},
        {"publication_seconds",0.}, {"evaluation_and_collective_seconds",0.},
        {"update_snapshot_seconds",0.}, {"update_linearization_seconds",0.}, {"update_contribution_seconds",0.},
        {"conditionals_rebuilt",0}, {"conditionals_reused",0}, {"conditional_full_rebuilds",0},
        {"conditional_seconds",0.}, {"recovery_seconds",0.}, {"directional_seconds",0.},
        {"refresh_requests",0}, {"refresh_rounds",0}, {"subspace_steps",0}};
    for (auto r : members) {
      auto& local=robots.at(r);
      local.fixed.clear(); local.privateKeys.clear();
      if (r == roots.at(r)) local.fixed.insert(X(0));
      for (const auto& pose : local.times) if (!local.fixed.count(X(pose.first))) local.privateKeys.push_back(X(pose.first));
    }
    if (members.size() == 1) {
      auto& local=robots.at(robot);
      NativeLocalOptions options; options.maxIterations=1;
      const auto result = reuseCharts ?
          optimizeNativeCachedLocal(*local.model,local.estimate,local.privateKeys,local.fixed,1e-4,options) :
          optimizeNativeLocal(*local.model,local.privateKeys,local.fixed,options);
      before=result.initialCost; after=result.finalCost;
      for (const auto& step : result.trace) accepted += step.alpha > 0.;
    } else {
      std::vector<Local> active;
      for (auto r : members) active.push_back(std::move(robots.at(r)));
      Links activeLinks;
      for (const auto& edge : links) if (members.count(edge.first) && members.count(edge.second)) activeLinks.push_back(edge);
      const auto result=routed(active,activeLinks,1,false,frames,interfaceIterations,rigidSubspace,fusedOneStep);
      for (auto& local : active) { const auto id=local.id; robots.at(id)=std::move(local); }
      const auto& step=result.at("trace").at(0);
      before=step.at("cost_before"); after=step.at("cost_after"); accepted=step.at("alpha").get<double>() > 0.;
      fullPrediction=step.at("full_step_prediction"); fullDecrease=step.at("full_step_decrease");
      bytes=result.at("application_bytes"); optimizerBytes += bytes;
      optimizerPayload += result.at("payload_bytes").get<std::size_t>();
      pcg=result.at("pcg_iterations"); pcgIterations += pcg;
      residual=result.at("pcg_max_relative_residual"); maxResidual=std::max(maxResidual,residual);
      for (auto it=stages.begin(); it!=stages.end(); ++it) {
        it.value() = result.at(it.key());
        if (it.value().is_number_integer()) solveTotals[it.key()] =
            solveTotals.at(it.key()).get<std::size_t>()+it.value().get<std::size_t>();
        else solveTotals[it.key()] = solveTotals.at(it.key()).get<double>()+it.value().get<double>();
      }
    }
    const double solveMs=1000*jointSeconds(solveStart), work=jointSeconds(start);
    ledger.flush();
    for (const auto& record:pendingAudit) chartAudit<<record.dump()<<'\n';
    pendingAudit.clear();
    workSeconds += work; services.push_back(work);
    finish=std::max(finish,elapsed)+work; maxBacklog=std::max(maxBacklog,finish-elapsed);
    const auto& pose=robots.at(robot).current().at<gtsam::Pose2>(X(index));
    causal << robot << ",pose," << index << ',' << t << ',' << pose.x() << ',' << pose.y() << ',' << pose.theta()
           << ',' << roots.at(robot) << ',' << (allConnected >= 0.) << '\n';
    updates << robot << ',' << t << ',' << elapsed << ',' << components() << ',' << members.size() << ','
            << before << ',' << after << ',' << accepted << ',' << pcg << ',' << residual << ',' << bytes << ','
            << ledger.bytes-priorBytes << ',' << insertMs << ',' << coordinationMs << ',' << solveMs << ','
            << 1000*work << ',' << finish-elapsed << ','
            << 1000*stages.at("model_seconds").get<double>() << ','
            << 1000*stages.at("interface_seconds").get<double>() << ','
            << 1000*stages.at("publication_seconds").get<double>() << ','
            << 1000*stages.at("evaluation_and_collective_seconds").get<double>() << ','
            << stages.at("response_builds") << ',' << stages.at("response_reuses") << ','
            << stages.at("refreshed_keys") << ',' << stages.at("factors_linearized") << ','
            << 1000*stages.at("update_snapshot_seconds").get<double>() << ','
            << 1000*stages.at("update_linearization_seconds").get<double>() << ','
            << 1000*stages.at("update_contribution_seconds").get<double>() << ','
            << stages.at("conditionals_rebuilt") << ',' << stages.at("conditionals_reused") << ','
            << stages.at("conditional_full_rebuilds") << ','
            << 1000*stages.at("conditional_seconds").get<double>() << ','
            << 1000*stages.at("recovery_seconds").get<double>() << ','
            << 1000*stages.at("directional_seconds").get<double>() << ','
            << stages.at("refresh_requests") << ',' << stages.at("refresh_rounds") << ','
            << fullPrediction << ',' << fullDecrease << '\n';
    ++frames; packets += frame.size();
  }

  Json save(const std::filesystem::path& output) {
    std::ofstream final(output/"final_estimates.csv");
    final.exceptions(std::ios::failbit|std::ios::badbit);
    final << std::setprecision(17) << "robot,kind,index,timestamp_s,x,y,yaw,component_root\n";
    double cost=0., consensus=0.;
    std::map<gtsam::Key,gtsam::Point2> shared;
    for (const auto& item : robots) {
      const auto& local=item.second;
      cost += local.graph.error(local.current());
      for (const auto& p : local.times) {
        const auto& pose=local.current().at<gtsam::Pose2>(X(p.first));
        final << local.id << ",pose," << p.first << ',' << p.second << ',' << pose.x() << ',' << pose.y() << ','
              << pose.theta() << ',' << roots.at(local.id) << '\n';
      }
      for (const auto& label : local.labels) {
        const auto& p=local.current().at<gtsam::Point2>(L(label.first));
        final << local.id << ",landmark," << label.first << ",," << p.x() << ',' << p.y() << ",," << roots.at(local.id) << '\n';
        if (local.interfaceKeys.count(L(label.first))) {
          const auto first=shared.emplace(local.interfaceKeys.at(L(label.first)),p);
          consensus=std::max(consensus,(first.first->second-p).norm());
        }
      }
    }
    final.close(); causal.close(); updates.close(); ledger.stream.close();
    std::sort(services.begin(),services.end());
    auto quantile=[&](double q) { return services.at(std::size_t(q*(services.size()-1))); };
    Json summary{{"scope", conditionalResponse ?
        "Chronological conditional-elimination prototype with common-past warm start; not asynchronous deployment" :
        "Chronological synchronous single-process simulator; not incremental elimination or asynchronous deployment"},
        {"conditional_response",conditionalResponse},
        {"chart_audit",chartAudit.is_open()},
        {"mode",mode}, {"reuse_charts",reuseCharts}, {"interface_iteration_cap",interfaceIterations},
        {"chart_refresh_tolerance",reuseCharts && !lazyRefresh ? Json(1e-4) : Json(nullptr)},
        {"lazy_refresh",lazyRefresh},
        {"interface_solver",rigidSubspace ? "rigid_subspace" :
            (fusedOneStep ? "fused_one_step_pcg" : "block_jacobi_pcg")},
        {"interface_residual_evaluation",(fusedOneStep || rigidSubspace) ? "cached_linear_product" : "fresh_operator_application"},
        {"suffix_added_bearing_variance",suffixAddedBearingVariance},
        {"suffix_bearing_sigma",suffixBearingSigma}, {"noise_parameters_learned",false},
        {"suffix_variance_multipliers",suffixVarianceMultipliers},
        {"noise_model_scope","Fixed nominal prefix; externally supplied frozen suffix variance multipliers and optional added bearing variance; no internal noise learning"},
        {"refresh_pending",std::any_of(robots.begin(),robots.end(),[](const auto& r) { return r.second.refreshPending; })},
        {"truth_used",false}, {"frames",frames}, {"packets",packets},
        {"merges",merges}, {"final_components",components()}, {"physical_cost",cost}, {"consensus_max_m",consensus},
        {"network_identities",shared.size()},
        {"optimizer_application_bytes",optimizerBytes}, {"optimizer_payload_bytes",optimizerPayload},
        {"control_application_bytes",ledger.bytes}, {"control_payload_bytes",ledger.payloadBytes}, {"control_messages",ledger.messages},
        {"application_bytes",optimizerBytes+ledger.bytes}, {"pcg_iterations",pcgIterations}, {"pcg_max_relative_residual",maxResidual},
        {"service_seconds",workSeconds}, {"frame_p50_s",quantile(.5)}, {"frame_p95_s",quantile(.95)},
        {"frame_p99_s",quantile(.99)}, {"frame_max_s",services.back()}, {"serial_max_backlog_s",maxBacklog},
        {"ledger_exclusions","Static topology configuration, transport framing/retransmission, asynchronous revision recovery; no network latency simulated"},
        {"timing_scope","Native insertion, synchronous coordination including control trace writes, joint solve; excludes JSON parsing/output evaluation"}};
    summary.update(solveTotals);
    if (warmCutoff>=0.) {
      std::size_t factors=0; for (const auto& r : robots) factors+=r.second.graph.size();
      const Robot root=roots.begin()->second;
      summary.update({{"warm_start_cutoff_s",warmCutoff},{"setup_seconds",setupSeconds},
          {"startup_seconds",0.},{"prefix_physical_factors",prefixFactors},{"physical_factors",factors},
          {"initial_cost",initialCost},{"root_displacement",rootInitial.localCoordinates(
              robots.at(root).current().at<gtsam::Pose2>(X(0))).norm()}});
    }
    summary["timing_scope"]="Native insertion, in-memory coordination, joint solve; excludes JSON input and trace serialization/file I/O";
    if (chartAudit.is_open()) {
      summary["timing_scope"]="Instrumented chart-cause diagnostic; not a runtime benchmark";
      chartAudit.close();
    }
    summary["all_connected_elapsed_s"]=allConnected < 0. ? Json(nullptr) : Json(allConnected);
    std::ofstream record(output/"summary.json");
    record.exceptions(std::ios::failbit|std::ios::badbit); record << summary.dump(2) << '\n'; record.close();
    return summary;
  }
};

int main(int argc,char** argv) {
  try {
    jointRequire(argc >= 6,"usage: native-joint-replay events.jsonl output joint|joint_cached|joint_one|joint_cached_one|joint_conditional_cached|joint_conditional_cached_one|independent duration r:r ...");
    const std::string mode(argv[3]);
    jointRequire(mode == "joint" || mode == "joint_cached" || mode == "joint_one" ||
        mode == "joint_cached_one" || mode == "joint_conditional_cached" ||
        mode == "joint_conditional_cached_one" || mode == "independent","invalid mode");
    const double duration=std::stod(argv[4]);
    jointRequire(std::isfinite(duration) && duration > 0.,"invalid duration");
    Links links; std::set<std::pair<Robot,Robot>> edges; std::string warmPath; bool chartAudit=false,lazyRefresh=false;
    bool rigidSubspace=false,fusedOneStep=false;
    double addedBearingVariance=0.; bool varianceSpecified=false;
    std::array<double,5> varianceMultipliers={{1.,1.,1.,1.,1.}}; bool scalesSpecified=false;
    for (int i=5;i<argc;++i) {
      if (std::string(argv[i])=="--fused-one-step") {
        jointRequire(!fusedOneStep,"duplicate fused one-step flag"); fusedOneStep=true; continue;
      }
      if (std::string(argv[i])=="--rigid-subspace") {
        jointRequire(!rigidSubspace,"duplicate rigid subspace flag"); rigidSubspace=true; continue;
      }
      if (std::string(argv[i])=="--suffix-variance-multipliers") {
        jointRequire(!scalesSpecified,"duplicate suffix variance multipliers");
        jointRequire(i+5<argc,"invalid suffix variance multipliers");
        for (double& scale : varianceMultipliers) {
          const std::string value(argv[++i]); std::size_t consumed=0;
          try { scale=std::stod(value,&consumed); }
          catch (const std::exception&) { throw std::runtime_error("invalid suffix variance multipliers"); }
          jointRequire(consumed==value.size() && std::isfinite(scale) && scale>0.,
              "invalid suffix variance multipliers");
        }
        scalesSpecified=true; continue;
      }
      if (std::string(argv[i])=="--suffix-added-bearing-variance") {
        jointRequire(!varianceSpecified,"duplicate suffix bearing variance");
        jointRequire(i+1<argc,"invalid suffix bearing variance");
        const std::string value(argv[++i]); std::size_t consumed=0;
        try { addedBearingVariance=std::stod(value,&consumed); }
        catch (const std::exception&) { throw std::runtime_error("invalid suffix bearing variance"); }
        jointRequire(consumed==value.size() && std::isfinite(addedBearingVariance) &&
                     addedBearingVariance>=0.,"invalid suffix bearing variance");
        varianceSpecified=true; continue;
      }
      if (std::string(argv[i])=="--lazy-refresh") {
        jointRequire(!lazyRefresh,"duplicate lazy refresh flag"); lazyRefresh=true; continue;
      }
      if (std::string(argv[i])=="--chart-audit") {
        jointRequire(!chartAudit,"duplicate chart audit flag"); chartAudit=true; continue;
      }
      if (std::string(argv[i])=="--warm-start") {
        jointRequire(warmPath.empty() && i+1<argc,"invalid warm start"); warmPath=argv[++i]; continue;
      }
      const std::string value(argv[i]); const auto split=value.find(':');
      jointRequire(split != std::string::npos,"invalid link");
      const auto a=std::stoul(value.substr(0,split)), b=std::stoul(value.substr(split+1));
      jointRequire(a < (1u<<24) && b < (1u<<24) && a != b,"invalid link IDs");
      const auto edge=std::make_pair(Robot(std::min(a,b)),Robot(std::max(a,b)));
      jointRequire(edges.emplace(edge.first,edge.second).second,"duplicate link");
      links.emplace_back(edge.first,edge.second);
    }
    jointRequire(mode.find("conditional") == std::string::npos || !warmPath.empty(),
        "conditional replay prototype requires a connected common-past warm-start");
    jointRequire(!chartAudit || mode.find("conditional") != std::string::npos,
        "chart audit requires conditional square-root replay");
    jointRequire(!lazyRefresh || (mode.find("cached") != std::string::npos && !warmPath.empty()),
        "lazy refresh requires cached common-past replay");
    jointRequire(!varianceSpecified || !warmPath.empty(),"suffix bearing variance requires common-past");
    jointRequire(!scalesSpecified || !warmPath.empty(),"suffix variance multipliers require common-past");
    jointRequire(!rigidSubspace || (mode.find("_one") != std::string::npos && !warmPath.empty()),
        "rigid subspace requires one-step common-past replay");
    jointRequire(!fusedOneStep || (!rigidSubspace && mode.find("_one") != std::string::npos && !warmPath.empty()),
        "fused one-step requires exclusive one-step common-past replay");
    const std::filesystem::path output(argv[2]);
    jointRequire(!std::filesystem::exists(output),"output already exists");
    std::ifstream input(argv[1]); jointRequire(input.good(),"cannot read events");
    std::filesystem::create_directories(output);
    Replay replay(links,output,mode,chartAudit,lazyRefresh,addedBearingVariance,varianceMultipliers,rigidSubspace,fusedOneStep);
    if (!warmPath.empty()) {
      std::ifstream warm(warmPath); jointRequire(warm.good(),"cannot open warm start"); Json package; warm>>package;
      replay.warmStart(package);
      online::suffix(argv[1],package,duration,[&](const std::vector<Json>& frame) { replay.push(frame); });
      jointRequire(replay.frames>0,"empty suffix");
      std::cout << replay.save(output).dump() << std::endl; return 0;
    }
    std::map<Robot,std::vector<Json>> frames;
    std::string line; double first=-1.,previous=-1.,frameTime=-1.;
    auto flush=[&]() { for (const auto& frame : frames) replay.push(frame.second); frames.clear(); };
    while (std::getline(input,line)) {
      const auto row=Json::parse(line); const double t=number(row,"timestamp_s");
      if (first < 0.) first=t;
      jointRequire(t >= previous,"event time decreased"); previous=t;
      if (t-first > duration) break;
      if (!frames.empty() && t != frameTime) flush();
      frameTime=t;
      frames[identifier(row,"robot")].push_back(row);
    }
    flush();
    jointRequire(replay.frames > 0,"empty stream");
    std::cout << replay.save(output).dump() << std::endl;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
  return 0;
}
