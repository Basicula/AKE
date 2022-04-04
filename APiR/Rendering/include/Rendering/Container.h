#pragma once
#include "Rendering/Object.h"

#include <vector>

class Container
{
public:
  ~Container();

  void AddObject(Object* ip_object);

  HOSTDEVICE virtual const Object* TraceRay(double& o_distance, const Ray& i_ray, const double i_far) const;

  // Update internal object structure
  virtual void Update();

  std::size_t Size() const;

protected:
  std::vector<Object*> m_objects;
};