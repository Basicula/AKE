#pragma once
#include <Rendering/IRenderable.h>

#include <vector>

class Container
  {
  public:
    ~Container();

    void AddObject(IRenderable* ip_object);

    HOSTDEVICE virtual const IRenderable* TraceRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far) const;

    // Update internal object structure
    virtual void Update();

    std::size_t Size() const;

  protected:
    std::vector<IRenderable*> m_objects;
  };