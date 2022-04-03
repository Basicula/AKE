#include "Rendering/RenderableObject.h"

RenderableObject::RenderableObject(
  ISurface* i_surface,
  IVisualMaterial* i_material)
  : Object()
  , mp_surface(i_surface) {
  mp_visual_material = i_material;
  }

RenderableObject::~RenderableObject()
{
  if (mp_surface)
    delete mp_surface;
}

bool RenderableObject::IntersectWithRay(
  double& o_distance,
  const Ray& i_ray,
  const double i_far) const {
  return mp_surface->IntersectWithRay(o_distance, i_ray, i_far);
  }

inline Vector3d RenderableObject::GetNormalAtPoint(const Vector3d& i_point) const {
  return mp_surface->NormalAtPoint(i_point);
  }

BoundingBox RenderableObject::GetBoundingBox() const {
  return mp_surface->GetBoundingBox();
  }