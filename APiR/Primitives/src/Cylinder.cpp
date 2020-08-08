#include <Cylinder.h>
#include <SolveEquations.h>
#include <DefinesAndConstants.h>

Cylinder::Cylinder(
  const Vector3d& i_center,
  double i_radius,
  double i_height)
  : ISurface()
  , m_radius(i_radius)
  , m_height(i_height)
  , m_is_finite(i_height > 0)
  , m_zmax(MAX_DOUBLE)
  , m_zmin(-MAX_DOUBLE)
  , m_half_height(0)
  {
  SetTranslation(i_center);
  if (m_is_finite)
    {
    m_half_height = m_height / 2;
    m_zmax = +m_half_height;
    m_zmin = -m_half_height;
    m_top = Plane(
      Vector3d(0.0, 0.0, m_zmax), 
      Vector3d(0, 0, 1));
    m_bottom = Plane(
      Vector3d(0.0, 0.0, m_zmin), 
      Vector3d(0, 0, -1));
    }
  }

Vector3d Cylinder::_NormalAtLocalPoint(const Vector3d& i_local_point) const
  {
  if (m_is_finite && i_local_point[2] == m_zmin)
    return Vector3d(0, 0, -1);

  if (m_is_finite && i_local_point[2] == m_zmax)
    return Vector3d(0, 0, 1);

  Vector3d normal = i_local_point;
  normal[2] = 0;
  return normal.Normalized();
  }

bool Cylinder::_IntersectWithRay(
  double& io_nearest_intersection_dist,
  const Ray& i_local_ray) const
  {
  const auto& ray_direction = i_local_ray.GetDirection();
  const auto& ray_origin = i_local_ray.GetOrigin();

  // project ray to XY plane
  const Vector2d ray_origin_2d(ray_origin[0], ray_origin[1]);
  Vector2d projected_direction(ray_direction[0], ray_direction[1]);

  // normalize and store starting length for projected ray direction
  // corner case if direction projection is (0,0) then can assume 
  // that length is small enough but not zero for general approach
  const double projected_direction_len = 
    (projected_direction[0] == 0.0 && projected_direction[1] == 0.0) ? 
    MIN_DOUBLE : projected_direction.Length();
  projected_direction /= projected_direction_len;

  // solve ray sphere task in 2d
  const double dist_to_circle_point = -ray_origin_2d.Dot(projected_direction);
  if (dist_to_circle_point <= 0.0 && !m_is_finite || 
      dist_to_circle_point - m_radius > io_nearest_intersection_dist)
    return false;

  const auto pojnt_near_circle_2d = ray_origin_2d + projected_direction * dist_to_circle_point;
  const double sqr_radius = m_radius * m_radius;
  const double center_to_point = pojnt_near_circle_2d.SquareLength();
  if (center_to_point > sqr_radius)
    return false;

  const double half_chorde = sqrt(sqr_radius - center_to_point);
  const double close_dist_to_circle = dist_to_circle_point - half_chorde;
  if (close_dist_to_circle > io_nearest_intersection_dist)
    return false;

  // after solving intersection problem in 2D
  // extend result to 3D
  const double close_distance = close_dist_to_circle / projected_direction_len;

  if (close_distance > io_nearest_intersection_dist)
    return false;

  const auto close_intersection = ray_origin + ray_direction * close_distance;
  if (close_distance > 0.0)
    {
    if (!m_is_finite ||
        (close_intersection[2] <= m_zmax &&
         close_intersection[2] >= m_zmin))
      {
      io_nearest_intersection_dist = close_distance;
      return true;
      }
    }

  const double far_dist_to_circle = dist_to_circle_point + half_chorde;
  const double far_distance = far_dist_to_circle / projected_direction_len;
  if (far_distance <= 0.0 || far_distance > io_nearest_intersection_dist)
    return false;
  const auto far_intersection = ray_origin + ray_direction * far_distance;

  if (m_is_finite &&
      (far_intersection[2] > m_zmax && close_intersection[2] > m_zmax ||
       far_intersection[2] < m_zmin && close_intersection[2] < m_zmin))
    return false;

  if (!m_is_finite)
    {
    io_nearest_intersection_dist = far_distance;
    return true;
    }

  // find intersection with top or bottom plane
  IntersectionRecord intersection;
  bool is_intersected = m_top->IntersectWithRay(intersection, i_local_ray);
  is_intersected |= m_bottom->IntersectWithRay(intersection, i_local_ray);
  if (!is_intersected ||
      intersection.m_distance < 0 ||
      intersection.m_distance > io_nearest_intersection_dist)
    return false;

  // check that intersection point with plane satisfies cylinder equation
  // so transform intersection point to cylinder coordinate system
  const auto transformed_intersection = intersection.m_intersection;
  if (transformed_intersection[0] * transformed_intersection[0] +
      transformed_intersection[1] * transformed_intersection[1] <= m_radius * m_radius)
    {
    io_nearest_intersection_dist = intersection.m_distance;
    return true;
    }
  return false;
  }

void Cylinder::_CalculateBoundingBox()
  {
  const double z = m_is_finite ? m_height / 2 : MAX_DOUBLE;
  m_bounding_box.AddPoint(Vector3d(m_radius, m_radius, z));
  m_bounding_box.AddPoint(Vector3d(-m_radius, -m_radius, -z));
  }
