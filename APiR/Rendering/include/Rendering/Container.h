#pragma once
#include <Rendering/IRenderable.h>

#include <vector>

class Container
  {
  public:
    void AddObject(IRenderableSPtr i_object);

    HOSTDEVICE virtual bool TraceRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const;

    // Update internal object structure
    virtual void Update();

    std::size_t Size() const;

  protected:
    std::vector<IRenderableSPtr> m_objects;
  };