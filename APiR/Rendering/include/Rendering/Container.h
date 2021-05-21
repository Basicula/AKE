#pragma once
#include <Rendering/IRenderable.h>

#include <vector>

class Container
  {
  public:
    void AddObject(IRenderableSPtr i_object);

    // Update internal object structure
    virtual void Update() = 0;

    std::size_t Size() const;

  protected:
    std::vector<IRenderableSPtr> m_objects;
  };