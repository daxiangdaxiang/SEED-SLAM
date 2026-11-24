#include <gtsam/geometry/Point2.h>
#include <gtsam/slam/PriorFactor.h>
#include <risam/ExtendedISAM2.h>
#include <risam/GraduatedFactor.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using Point = gtsam::Point2;
using Prior = gtsam::PriorFactor<Point>;

int main() {
  bool passed = true;
  std::cout << "case,merit,objective,merit_gradient,linear_gradient,error,passed\n";
  for (const std::string mode : {"gaussian", "graduated", "mixed"}) {
    gtsam::NonlinearFactorGraph graph;
    auto kernel = boost::make_shared<risam::SIGKernel>();
    for (int i = 0; i < 2; ++i) {
      const Point center(i, .2 * i);
      const auto noise = gtsam::noiseModel::Isotropic::Sigma(2, i ? 1. : .1);
      if (mode == "graduated" || (mode == "mixed" && i == 1)) {
        graph.push_back(risam::make_shared_graduated<Prior>(kernel, 0, center, noise));
      } else {
        graph.emplace_shared<Prior>(0, center, noise);
      }
    }
    gtsam::Values values;
    values.insert(0, Point(.03, .07));
    ExtendedISAM2 solver;
    solver.update(graph, values);
    const double eps = 1e-6;
    gtsam::Values plus(values), minus(values);
    plus.update(0, Point(.03 + eps, .07));
    minus.update(0, Point(.03 - eps, .07));
    const double merit = solver.residual(values);
    const double derivative = (solver.residual(plus) - solver.residual(minus)) / (2 * eps);
    const auto linear = graph.linearize(values)->jacobian();
    const gtsam::Vector gradient = -linear.first.transpose() * linear.second;
    double objective = 0;
    for (const auto& factor : graph) {
      const auto robust = boost::dynamic_pointer_cast<risam::GraduatedFactor>(factor);
      objective += robust ? robust->robustResidual(values) : factor->error(values);
    }
    const double error = std::abs(derivative - gradient[0]);
    const bool valid = std::isfinite(merit) && std::abs(merit - objective) < 1e-12 && error < 1e-7;
    std::cout << mode << ',' << std::setprecision(17) << merit << ',' << objective << ','
              << derivative << ',' << gradient[0] << ',' << error << ',' << valid << '\n';
    passed = passed && valid;
  }
  return passed ? 0 : 1;
}
