#include "Fluid/Fluid.h"

#include "Common/Constants.h"
#include "Common/ThreadPool.h"
#include "Fluid/FluidConstants.h"
#include "Visual/PhongMaterial.h"

namespace {
  double SmoothingMin(const double i_first_sdf, const double i_second_sdf, const double i_k)
  {
    if (i_first_sdf < i_second_sdf) {
      const double temp = i_k + i_first_sdf - i_second_sdf;
      const double h = temp > 0.0 ? temp / i_k : 0.0;
      return i_first_sdf - h * h * i_k * 0.25;
    }

    const double temp = i_k + i_second_sdf - i_first_sdf;
    const double h = temp > 0.0 ? temp / i_k : 0.0;
    return i_second_sdf - h * h * i_k * 0.25;
  }
}

Fluid::Fluid(const std::size_t i_num_particles)
  : Object()
  , m_simulation(i_num_particles)
{
  mp_visual_material = new PhongMaterial(Color(0xffff0000));
  _UpdateBBox();
}

void Fluid::_UpdateBBox()
{
  const auto& system = m_simulation.GetParticleSystem();
  m_bbox.Reset();
  for (auto pos = system.BeginPositions(); pos != system.EndPositions(); ++pos) {
    m_bbox.AddPoint(*pos + Vector3d{1});
    m_bbox.AddPoint(*pos - Vector3d{1});
  }
}

bool Fluid::IntersectWithRay(double& o_distance, const Ray& i_ray, const double /*i_far*/) const
{
  static const auto& system = m_simulation.GetParticleSystem();
  static const auto start = system.BeginPositions();
  static const auto end = system.EndPositions();
  bool intersected = false;
  const double k = 0.1;
  auto sdf = [&](const Vector3d& i_ray_origin) {
    double res = MAX_DOUBLE;
    std::for_each(start, end, [&res, &i_ray_origin, &k](const Vector3d& i_pos) {
      const double local_sdf = i_pos.SquareDistance(i_ray_origin) - SMOOTHING_RADIUS_SQR;
      res = SmoothingMin(local_sdf, res, k);
    });
    return res;
  };
  auto ray_origin = i_ray.GetOrigin();
  const auto& ray_dir = i_ray.GetDirection();
  o_distance = 0.0;
  for (auto it = 0; it < 10; ++it) {
    const auto dist_to_fluid = sdf(ray_origin);
    o_distance += dist_to_fluid;
    if (dist_to_fluid <= 0.0) {
      intersected = true;
      break;
    }
    ray_origin += ray_dir * sqrt(dist_to_fluid);
  }
  return intersected;
}

Vector3d Fluid::GetNormalAtPoint(const Vector3d& /*i_point*/) const
{
  // TODO implement
  return {};
}

std::string Fluid::Serialize() const
{
  std::string res = "{ \"Fluid\" : { ";
  res += "\"NumOfParticles\" : " + std::to_string(GetNumParticles());
  res += "} }";
  return res;
}

BoundingBox3D Fluid::GetBoundingBox() const
{
  return m_bbox;
}

double Fluid::GetTimeStep() const
{
  return m_simulation.GetTimeStep();
}

std::size_t Fluid::GetNumParticles() const
{
  return m_simulation.GetParticleSystem().GetNumOfParticles();
}

void Fluid::Update()
{
  m_simulation.Update();
  _UpdateBBox();
}