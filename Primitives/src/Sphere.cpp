#include <Sphere.h>

Sphere::Sphere()
  : m_center(Vector3d(0, 0, 0))
  , m_radius(1)
{}

Sphere::Sphere(double i_radius)
  : m_center(Vector3d(0, 0, 0))
  , m_radius(i_radius)
{}

Sphere::Sphere(const Vector3d& i_center, double i_radius)
  : m_center(i_center)
  , m_radius(i_radius)
{}