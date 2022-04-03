#pragma once
#include "Geometry/ISurface.h"
#include "Rendering/Object.h"
#include "Visual/IVisualMaterial.h"

class RenderableObject : public Object
  {
  public:
    RenderableObject(ISurface* i_surface, IVisualMaterial* i_material);
    ~RenderableObject();

    virtual bool IntersectWithRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far) const override;

    virtual BoundingBox GetBoundingBox() const override;

    virtual Vector3d GetNormalAtPoint(const Vector3d& i_point) const override;

  private:
    ISurface* mp_surface;
  };