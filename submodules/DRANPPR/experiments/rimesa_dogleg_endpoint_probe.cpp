#include <risam/DoglegLineSearch.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main() {
  bool passed = true;
  auto check = [&](const std::string& name, const gtsam::Vector2& u,
                   const gtsam::Vector2& n, double radius,
                   const gtsam::Vector2& expected) {
    gtsam::VectorValues gradient, newton;
    gradient.insert(0, u);
    newton.insert(0, n);
    const auto result = risam::dogleg_internal::ComputeDoglegPoint(
        radius, gradient, newton, false).at(0);
    const double error = (result - expected).norm();
    const bool valid = result.allFinite() && error <= 1e-12;
    std::cout << name << ',' << std::setprecision(17) << radius << ','
              << error << ',' << valid << '\n';
    passed = passed && valid;
  };
  std::cout << "case,radius,error,passed\n";
  // Captured from the first failed blend in the unchanged riMESA smoke.
  const gtsam::Vector2 u(-0.0002548417886291125, -0.00011734925498463766);
  const gtsam::Vector2 n(-0.00025484178862911254, -0.00011734925498463767);
  check("captured_endpoint", u, n, n.norm(), n);
  check("gradient_segment", {1., 0.}, {1., 2.}, .5, {.5, 0.});
  check("blend_segment", {1., 0.}, {1., 2.}, std::sqrt(2.), {1., 1.});
  check("newton_segment", {1., 0.}, {1., 2.}, 3., {1., 2.});
  check("stationary", {0., 0.}, {0., 0.}, 0., {0., 0.});
  return passed ? 0 : 1;
}
