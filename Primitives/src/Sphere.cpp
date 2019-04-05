#include <Sphere.h>

Sphere::Sphere(const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(Vector3d(0, 0, 0))
  , m_radius(1)
  {}

Sphere::Sphere(double i_radius, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(Vector3d(0, 0, 0))
  , m_radius(i_radius)
  {}

Sphere::Sphere(const Vector3d& i_center, double i_radius, const ColorMaterial& i_material)
  : IObject(i_material, ObjectType::SPHERE)
  , m_center(i_center)
  , m_radius(i_radius)
  {}

bool Sphere::GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const
  {
  o_normal = (i_point - m_center).Normalized();
  return true;
  }


bool Sphere::IntersectWithRay(Vector3d& o_intersection, double &o_distance, const Ray& i_ray) const
  {
  Vector3d ray_start = i_ray.GetStart();
  Vector3d start_to_center = m_center - ray_start;

  double to_sphere = start_to_center.Dot(i_ray.GetDirection());
  Vector3d point_near_sphere = ray_start + i_ray.GetDirection()*to_sphere;

  const double sqr_radius = m_radius * m_radius;
  double sphere_center_to_point = m_center.SquareDistance(point_near_sphere);
  if (sphere_center_to_point > sqr_radius)
    return false;

  double distance_to_intersection = to_sphere - sqrt(sqr_radius - sphere_center_to_point);
  if (distance_to_intersection <= Epsilon3D)
    return false;
  o_distance = distance_to_intersection;
  o_intersection = ray_start + i_ray.GetDirection()*distance_to_intersection;
  return true;
  }