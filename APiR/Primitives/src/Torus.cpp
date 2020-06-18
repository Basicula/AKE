#include <math.h>

#include <Torus.h>
#include <SolveEquations.h>
#include <DefinesAndConstants.h>

Torus::Torus(
  const Vector3d& i_center, 
  double i_major_radius, 
  double i_minor_radius)
  : m_center(i_center)
  , m_major_radius(i_major_radius)
  , m_minor_radius(i_minor_radius)
  , m_bounding_box()
  {
  _CalculateBoundingBox();
  }

BoundingBox Torus::_GetBoundingBox() const
  {
  return m_bounding_box;
  }

Vector3d Torus::_NormalAtPoint(const Vector3d& i_point) const
  {
  double sqrt_sum_sqr = sqrt(i_point[0] * i_point[0] + i_point[1] * i_point[1]);
  double dx = 2 * i_point[0] * (1 - m_major_radius / sqrt_sum_sqr);
  double dy = 2 * i_point[1] * (1 - m_major_radius / sqrt_sum_sqr);
  double dz = 2 * i_point[2];
  return Vector3d(dx, dy, dz).Normalized();
  }

bool Torus::_IntersectWithRay(
  IntersectionRecord& io_intersection, 
  const Ray & i_ray) const
  {
  /*
  Let's make intersection equation by putting ray point into torus equation
  Ray point : O + D*t = P
  Torus equation : (z^2 - minor^2 + major^2 + x^2 + y^2) = 4 * major^2 *(x^2 + y^2)
  in result we have equation c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0
  need to find t - distance to intersection
  */
  const auto& ray_origin = i_ray.GetOrigin();
  const auto& ray_direction = i_ray.GetDirection();
  Vector3d start = ray_origin - m_center;
  double square_start = start.SquareLength();
  double start_dot_direction = ray_direction.Dot(start);
  double square_minor = m_minor_radius * m_minor_radius;
  double square_major = m_major_radius * m_major_radius;
  double square_start_major_minor = square_start + square_major - square_minor;
  double c4 = 1;
  double c3 = 4 * start_dot_direction;
  double c2 = 4 * start_dot_direction * start_dot_direction
    + 2 * square_start_major_minor
    - 4 * square_major * (1 - ray_direction[2] * ray_direction[2]);
  double c1 = 4 * square_start_major_minor * start_dot_direction - 8 * square_major * (start_dot_direction - ray_direction[2] * start[2]);
  double c0 = square_start_major_minor * square_start_major_minor - 4 * square_major * (square_start - start[2] * start[2]);
  double roots[4] = { INFINITY,INFINITY ,INFINITY ,INFINITY };
  const double quartic_coefs[5] = { c0,c1,c2,c3,c4 };
  int roots_count = Equations::SolveQuartic(quartic_coefs, roots);
  if (roots_count == 0)
    return false;
  //find smallest positive if exist temp solution
  double distance = INFINITY;
  for (const auto& root : roots)
    if (root > 0.0 && root < distance)
      distance = root;
  if (distance == INFINITY || distance > io_intersection.m_distance)
    return false;
  io_intersection.m_distance = distance;
  io_intersection.m_intersection = ray_origin + ray_direction * distance;
  io_intersection.m_normal = _NormalAtPoint(io_intersection.m_intersection);
  return true;
  }

void Torus::_CalculateBoundingBox()
  {
  Vector3d vec(0, 0, 1);
  m_bounding_box.AddPoint(m_center + vec * m_minor_radius);
  m_bounding_box.AddPoint(m_center - vec * m_minor_radius);
  m_bounding_box.AddPoint(m_center + (m_major_radius + m_minor_radius));
  m_bounding_box.AddPoint(m_center - (m_major_radius + m_minor_radius));
  }
