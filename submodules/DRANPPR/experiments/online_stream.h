#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <limits>
#include <vector>

namespace online {
using Json = nlohmann::json;
inline void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline double firstTime(const Json& package) {
  double t=std::numeric_limits<double>::infinity();
  for (const auto& r : package.at("robots")) t=std::min(t,r.at("poses").at(0).at("timestamp_s").get<double>());
  return t;
}
inline double cutoffTime(const Json& package) {
  double t=-std::numeric_limits<double>::infinity();
  for (const auto& r : package.at("robots")) t=std::max(t,r.at("poses").back().at("timestamp_s").get<double>());
  return t;
}

// Validate the complete supplied past against raw frames before skipping it.
// Only the next timestamp is buffered; future measurements never enter a solve.
template<class Push>
void suffix(const std::string& path,const Json& package,double duration,Push push) {
  check(std::isfinite(duration) && duration>0.,"invalid duration");
  check(!package.value("truth_used",false),"truth-assisted warm start");
  const double first=firstTime(package), cutoff=cutoffTime(package);
  check(first+duration>cutoff,"duration does not extend prefix");
  std::map<int,Json> records;
  std::map<int,std::map<int,Json>> observations;
  std::map<int,std::map<int,int>> labels;
  std::map<int,std::set<int>> seenPoses,seenObservations;
  for (const auto& r : package.at("robots")) {
    const int robot=r.at("id"); records[robot]=r;
    for (const auto& o : r.at("objects")) labels[robot][o.at("cluster")]=o.at("label");
    for (const auto& o : r.at("observations")) observations[robot][o.at("source")]=o;
  }
  bool checked=false;
  auto verify=[&]() {
    if (checked) return;
    for (const auto& r : records) {
      check(seenPoses[r.first].size()==r.second.at("poses").size(),"prefix missing poses");
      check(seenObservations[r.first].size()==r.second.at("observations").size(),"prefix missing observations");
    }
    checked=true;
  };
  std::map<int,std::vector<Json>> frames;
  auto flush=[&]() {
    for (const auto& item : frames) {
      const auto& frame=item.second; const auto& head=frame.front();
      const int robot=item.first,k=head.at("pose_index"); const double t=head.at("timestamp_s");
      check(records.count(robot) && head.at("kind")=="pose","invalid frame robot/kind");
      for (const auto& e : frame) check(e.at("robot")==robot && e.at("pose_index")==k &&
          e.at("timestamp_s")==t && e.at("available_time_s")==t &&
          e.at("support_max_time_s").get<double>()<=t,"mixed or noncausal frame");
      if (t>cutoff) { verify(); push(frame); continue; }
      check(k>=0 && k<int(records.at(robot).at("poses").size()) && seenPoses[robot].insert(k).second,
          "prefix unexpected pose");
      check(records.at(robot).at("poses").at(k).at("timestamp_s")==t,"prefix pose time mismatch");
      if (k) {
        const auto& o=records.at(robot).at("odometry").at(k-1);
        check(head.at("from_pose_index")==o.at("source") && head.at("delta")==o.at("delta") &&
            head.at("duration_s")==o.at("dt"),"prefix odometry mismatch");
      } else check(head.at("delta").is_null() && head.at("from_pose_index").is_null() &&
          head.at("duration_s")==0.,"prefix invalid root");
      for (std::size_t i=1;i<frame.size();++i) {
        const auto& e=frame[i]; const int source=e.at("source_row");
        check(e.at("kind")=="observation" && observations[robot].count(source) &&
            seenObservations[robot].insert(source).second,"prefix unexpected observation");
        const auto& o=observations.at(robot).at(source);
        check(o.at("pose")==k && labels.at(robot).at(o.at("cluster"))==e.at("landmark_subject") &&
            o.at("range")==e.at("range_m") && o.at("bearing")==e.at("bearing_rad"),"prefix observation mismatch");
      }
    }
    frames.clear();
  };
  std::ifstream input(path); check(input.good(),"cannot open stream");
  std::string line; double previous=-std::numeric_limits<double>::infinity(),frameTime=previous;
  while (std::getline(input,line)) {
    const auto e=Json::parse(line); const double t=e.at("timestamp_s");
    check(std::isfinite(t) && t>=previous,"event time decreased"); previous=t;
    if (t-first>duration) break;
    if (!frames.empty() && t!=frameTime) flush();
    frameTime=t; frames[e.at("robot").get<int>()].push_back(e);
  }
  flush(); verify();
}
}  // namespace online
