#include <Rendering/RenderableObject.h>

RenderableObject::RenderableObject(
  ISurfaceSPtr i_surface, 
  IMaterialSPtr i_material)
  : mp_surface(i_surface)
  , mp_material(i_material)
  {}

bool RenderableObject::IntersectWithRay(
  IntersectionRecord& o_intersection, 
  const Ray& i_ray) const
  {
  if (mp_surface->IntersectWithRay(o_intersection, i_ray))
    {
    o_intersection.mp_material = mp_material;
    return true;
    }

  return false;
  }
