#include "IntersectionUtilities.h"

bool IntersectRayWithWave(Vector3d& o_intersection, double &o_distance, const Ray& i_ray)
  {
  // z + a = sin(x^2 + y^2)
  // z + a = sin(sqrt(x^2+y^2))
  const double PI = 3.14159265359;
  double tr = 1000, tl = 0;
  const double a = -210;
  const double eps = 1e-6;
  Vector3d far_point = i_ray.GetStart() + i_ray.GetDirection() * tr;
  double eqr = far_point[2] + a - sin(far_point[0] * far_point[0] + far_point[1] * far_point[1]);
  double eql = i_ray.GetStart()[2] + a - sin(i_ray.GetStart()[0] * i_ray.GetStart()[0] + i_ray.GetStart()[1] * i_ray.GetStart()[1]);
  if (eqr*eql > 0)
    return false;
  while (true)
    {
    double tm = (tr + tl) / 2;
    Vector3d probably_intersection = i_ray.GetStart() + i_ray.GetDirection() * tm;
    double x = probably_intersection[0] - PI * int(probably_intersection[0] / PI);
    double y = probably_intersection[1] - PI * int(probably_intersection[1] / PI);
    //double sqrx = x * x;
    double sqrx = probably_intersection[0] * probably_intersection[0];
    //double sqry = y * y;
    double sqry = probably_intersection[1] * probably_intersection[1];
    double eq = probably_intersection[2] + a - sin(sqrt(sqrx + sqry));
    if (abs(eq) <= eps)
      {
      o_intersection = probably_intersection;
      return true;
      }
    else if (eq > 0)
      tr = tm;
    else
      tl = tm;
    }
  }
