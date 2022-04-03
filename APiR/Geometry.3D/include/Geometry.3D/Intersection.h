#pragma once
#include <Math/Vector.h>
#include <Geometry/BoundingBox.h>
#include <Geometry.3D/Ray.h>

#include <memory>

struct RayBoxIntersectionRecord
  {
  bool m_intersected;
  double m_tmin;
  double m_tmax;

  RayBoxIntersectionRecord();

  void Reset();
  };

// if result is true 
// o_tmin and o_tmax contain near and far distance to box
// if result is false
// o_tmin and o_tmax contain garbage
void RayBoxIntersection(
  const Ray& i_ray,
  const BoundingBox3D& i_box,
  RayBoxIntersectionRecord& o_intersection);

bool RayIntersectBox(
  const Ray& i_ray,
  const BoundingBox3D& i_box);