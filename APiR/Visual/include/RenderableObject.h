#pragma once

#include <IRenderable.h>
#include <Primitives/ISurface.h>
#include <IMaterial.h>

class RenderableObject : public IRenderable
  {
  public:
    RenderableObject(ISurfaceSPtr i_surface, IMaterialSPtr i_material);

    virtual bool IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const override;

    virtual BoundingBox GetBoundingBox() const override;

    virtual void Update() override;

    virtual std::string Serialize() const override;
  private:
    ISurfaceSPtr mp_surface;
    IMaterialSPtr mp_material;
  };

inline BoundingBox RenderableObject::GetBoundingBox() const
  {
  return mp_surface->GetBoundingBox();
  }

inline std::string RenderableObject::Serialize() const
  {
  std::string res = "{ \"RenderableObject\" : { ";
  res += " \"Surface\" : " + mp_surface->Serialize() + ", ";
  res += " \"Material\" : " + mp_material->Serialize();
  res += "} }";
  return res;
  }

inline void RenderableObject::Update()
  {

  }