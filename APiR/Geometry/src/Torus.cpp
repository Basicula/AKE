#include "Geometry/Torus.h"

#include "Math/SolveEquations.h"

Torus::Torus(const Vector3d& i_center, double i_major_radius, double i_minor_radius)
  : ISurface()
  , m_major_radius(i_major_radius)
  , m_minor_radius(i_minor_radius)
{
  SetTranslation(i_center);
}

Vector3d Torus::_NormalAtLocalPoint(const Vector3d& i_local_point) const
{
  const double sqrt_sum_sqr = sqrt(i_local_point[0] * i_local_point[0] + i_local_point[1] * i_local_point[1]);
  const double dx = i_local_point[0] * (1 - m_major_radius / sqrt_sum_sqr);
  const double dy = i_local_point[1] * (1 - m_major_radius / sqrt_sum_sqr);
  const double dz = i_local_point[2];
  return Vector3d(dx, dy, dz).Normalized();
}

bool Torus::_IntersectWithRay(double& o_intersection_dist, const Ray& i_local_ray, const double i_far) const
{
  /*
  Let's make intersection equation by putting ray point into torus equation
  Ray point : O + D*t = P
  Torus equation : (z^2 - minor^2 + major^2 + x^2 + y^2)^2 = 4 * major^2 *(x^2 + y^2)
  in result we have equation c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0
  need to find t - distance to intersection
  */
  const auto& ray_origin = i_local_ray.GetOrigin();
  const auto& ray_direction = i_local_ray.GetDirection();
  const double sqr_dist_to_origin = ray_origin.SquareLength();
  const double origin_dot_direction = ray_direction.Dot(ray_origin);
  const double square_minor = m_minor_radius * m_minor_radius;
  const double square_major = m_major_radius * m_major_radius;
  const double square_start_major_minor = sqr_dist_to_origin + square_major - square_minor;
  const double c4 = 1.0;
  const double c3 = 4.0 * origin_dot_direction;
  const double c2 = 4.0 * origin_dot_direction * origin_dot_direction + 2.0 * square_start_major_minor -
                    4.0 * square_major * (1.0 - ray_direction[2] * ray_direction[2]);
  const double c1 = 4.0 * square_start_major_minor * origin_dot_direction -
                    8.0 * square_major * (origin_dot_direction - ray_direction[2] * ray_origin[2]);
  const double c0 = square_start_major_minor * square_start_major_minor -
                    4.0 * square_major * (sqr_dist_to_origin - ray_origin[2] * ray_origin[2]);
  double roots[4] = { INFINITY, INFINITY, INFINITY, INFINITY };
  const double quartic_coefs[5] = { c0, c1, c2, c3, c4 };
  int roots_count = Equations::SolveQuartic(quartic_coefs, roots);
  if (roots_count == 0)
    return false;
  // find smallest positive if exist temp solution
  double smallest_root = i_far;
  for (const auto& root : roots)
    if (root > 0.0 && root < smallest_root)
      smallest_root = root;
  if (smallest_root < i_far && smallest_root > 0.0) {
    o_intersection_dist = smallest_root;
    return true;
  }
  return false;
}

void Torus::_CalculateBoundingBox()
{
  m_bounding_box.AddPoint(Vector3d(0.0, 0.0, m_minor_radius));
  m_bounding_box.AddPoint(Vector3d(0.0, 0.0, -m_minor_radius));
  m_bounding_box.AddPoint(Vector3d(m_major_radius + m_minor_radius));
  m_bounding_box.AddPoint(Vector3d(-(m_major_radius + m_minor_radius)));
}
