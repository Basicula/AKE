#include <Rendering/RenderableObject.h>

RenderableObject::RenderableObject(
  ISurface* i_surface,
  IMaterial* i_material)
  : mp_surface(i_surface)
  , mp_material(i_material) {
  }

RenderableObject::~RenderableObject()
{
  if (mp_surface)
    delete mp_surface;
  if (mp_material)
    delete mp_material;
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

const IMaterial* RenderableObject::GetMaterial() const {
  return mp_material;
  }

BoundingBox RenderableObject::GetBoundingBox() const {
  return mp_surface->GetBoundingBox();
  }

std::string RenderableObject::Serialize() const {
  std::string res = "{ \"RenderableObject\" : { ";
  res += " \"Surface\" : " + mp_surface->Serialize() + ", ";
  res += " \"Material\" : " + mp_material->Serialize();
  res += "} }";
  return res;
  }
