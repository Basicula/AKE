#include <Plane.h>
#include <DefinesAndConstants.h>

Plane::Plane(
  const Vector3d& i_first, 
  const Vector3d& i_second, 
  const Vector3d& i_third)
  : m_point(i_first)
  , m_normal((i_second - i_first).CrossProduct(i_third - i_first))
  {
  _FillParams();
  }

Plane::Plane(
  const Vector3d& i_point, 
  const Vector3d& i_normal)
  : m_point(i_point)
  , m_normal(i_normal)
  {
  _FillParams();
  }

void Plane::_FillParams()
  {
  m_a = m_normal[0];
  m_b = m_normal[1];
  m_c = m_normal[2];
  m_d = -m_normal.Dot(m_point);
  }

Vector3d Plane::GetNormal() const
  {
  return m_normal;
  }

double Plane::GetValueFromEquation(const Vector3d& i_point) const
  {
  return i_point.Dot(m_normal) + m_d;
  }


bool Plane::_IntersectWithRay(IntersectionRecord& io_intersection, const Ray & i_ray) const
  {
  const auto& ray_origin = i_ray.GetOrigin();
  const auto& ray_direction = i_ray.GetDirection();

  const double value_from_equation = GetValueFromEquation(ray_origin);
  if (value_from_equation * ray_direction.Dot(m_normal) >= 0.0 ||
      value_from_equation < 0.0)
    return false;

  double distance_to_intersection = value_from_equation / m_normal.Dot(-ray_direction);

  if (distance_to_intersection > io_intersection.m_distance)
    return false;

  io_intersection.m_distance = distance_to_intersection;
  io_intersection.m_intersection = ray_origin + ray_direction * distance_to_intersection;
  io_intersection.m_normal = m_normal;
  return true;
  }
