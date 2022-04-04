#include "Geometry.3D/Plane.h"

Plane::Plane(const Vector3d& i_first, const Vector3d& i_second, const Vector3d& i_third)
  : Plane(i_first, (i_second - i_first).CrossProduct(i_third - i_first))
{}

Plane::Plane(const Vector3d& i_point, const Vector3d& i_normal)
  : ISurface()
  , m_normal(i_normal)
  , m_size_limit(100)
{
  _FillParams();
  SetTranslation(i_point);
}

void Plane::_FillParams()
{
  m_a = m_normal[0];
  m_b = m_normal[1];
  m_c = m_normal[2];
}

void Plane::_CalculateBoundingBox()
{
  Vector3d vec1(m_normal[2], m_normal[0], m_normal[1]);
  Vector3d vec2(m_normal[1], m_normal[2], m_normal[0]);
  m_bounding_box.AddPoint(m_normal * 0.01);
  m_bounding_box.AddPoint(-m_normal * 0.01);
  m_bounding_box.AddPoint(vec1 * m_size_limit);
  m_bounding_box.AddPoint(-vec1 * m_size_limit);
  m_bounding_box.AddPoint(vec2 * m_size_limit);
  m_bounding_box.AddPoint(-vec2 * m_size_limit);
}

bool Plane::_IntersectWithRay(double& o_intersection_dist, const Ray& i_ray, const double i_far) const
{
  const auto& ray_origin = i_ray.GetOrigin();
  const auto& ray_direction = i_ray.GetDirection();

  const double value_from_equation = ray_origin.Dot(m_normal);
  if (value_from_equation < 0.0 || value_from_equation * ray_direction.Dot(m_normal) >= 0.0)
    return false;

  const double distance_to_intersection = value_from_equation / m_normal.Dot(-ray_direction);
  if (distance_to_intersection < 0.0 || distance_to_intersection > i_far)
    return false;
  o_intersection_dist = distance_to_intersection;
  return true;
}
