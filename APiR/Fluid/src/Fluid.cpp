#include <Fluid/Fluid.h>
#include <Common/DefinesAndConstants.h>
#include <Fluid/FluidConstants.h>
#include <Sphere.h>
#include <ColorMaterial.h>
#include <RenderableObject.h>
#include <Common/ParallelUtils.h>
#include <Common/ThreadPool.h>

namespace
  {
  inline double SmoothingMin(double i_first_sdf, double i_second_sdf, double i_k)
    {
    if (i_first_sdf < i_second_sdf)
      {
      const double temp = i_k + i_first_sdf - i_second_sdf;
      const double h = temp > 0.0 ? temp / i_k : 0.0;
      return i_first_sdf - h * h * i_k * 0.25;
      }
    else
      {
      const double temp = i_k + i_second_sdf - i_first_sdf;
      const double h = temp > 0.0 ? temp / i_k : 0.0;
      return i_second_sdf - h * h * i_k * 0.25;
      }
    }
  }

Fluid::Fluid(std::size_t i_num_particles)
  : m_bbox()
  , m_simulation(i_num_particles)
  , m_material(std::make_shared<ColorMaterial>(Color(0x0000ff)))
  {
  _UpdateBBox();
  }

void Fluid::_UpdateBBox()
  {
  const auto& system = m_simulation.GetParticleSystem();
  const auto num_of_particles = system.GetNumOfParticles();
  m_bbox.Reset();
  for (auto pos = system.BeginPositions(); pos != system.EndPositions(); ++pos)
    {
    m_bbox.AddPoint(*pos + 1);
    m_bbox.AddPoint(*pos - 1);
    }
  }

bool Fluid::IntersectWithRay(IntersectionRecord& o_intersection, const Ray& i_ray) const
  {
  static const auto& system = m_simulation.GetParticleSystem();
  static const auto start = system.BeginPositions();
  static const auto end = system.EndPositions();
  bool intersected = false;
  const double k = 0.1;
  auto sdf = [&](const Vector3d& i_ray_origin)
    {
    double res = MAX_DOUBLE;
    std::for_each(
      start,
      end,
      [&res, &i_ray_origin, &k](const Vector3d& i_pos)
      {
      const double local_sdf =
        i_pos.SquareDistance(i_ray_origin) - SMOOTHING_RADIUS_SQR;
      res = SmoothingMin(local_sdf, res, k);
      });
    return res;
    };
  auto ray_origin = i_ray.GetOrigin();
  const auto& ray_dir = i_ray.GetDirection();
  for (auto it = 0; it < 10; ++it)
    {
    auto dist_to_fluid = sdf(ray_origin);
    if (dist_to_fluid <= 0)
      {
      intersected = true;
      break;
      }
    ray_origin += ray_dir * sqrt(dist_to_fluid);
    }
  if (intersected)
    o_intersection.m_material = m_material;
  return intersected;
  }