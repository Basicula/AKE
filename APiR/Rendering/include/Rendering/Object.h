#pragma once
#include <Geometry/BoundingBox.h>
#include <Geometry/Ray.h>

#include <Physics/IPhysicMaterial.h>

#include <Visual/IVisualMaterial.h>

// This and derivative classes will own dynamic allocated members inside (IVisualMaterial, IPhysicMaterial etc)
// Class contains all possible information that describes object in the world 
// i.e. how we see it, how it interacts with world, animations etc
class Object
{
public:
  Object();
  virtual ~Object();

  // Main function for ray based workflows (ray casting/tracing/marching etc)
  virtual bool IntersectWithRay(
    double& o_distance,
    const Ray& i_ray,
    const double i_far) const = 0;

  // Some helpful functions
  virtual Vector3d GetNormalAtPoint(const Vector3d& i_point) const = 0;
  virtual BoundingBox3D GetBoundingBox() const = 0;

  // It's about how object will be displayed throug different rendering approaches
  const IVisualMaterial* VisualRepresentation() const;
  // It's about how object will be interact with other world or changes by itself
  const IPhysicMaterial* PhysicRepresentation() const;

protected:
  IVisualMaterial* mp_visual_material;
  IPhysicMaterial* mp_physic_material;
};