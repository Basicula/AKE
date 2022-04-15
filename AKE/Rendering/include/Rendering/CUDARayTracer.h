#pragma once
#include "CUDACore/KernelHandler.h"
#include "Rendering/IRenderer.h"

class CUDARayTracer : public IRenderer
  {
  public:
    CUDARayTracer();

    virtual void Render() override;

  protected:
    virtual void _OutputImageWasSet() override;
    virtual void _SceneWasSet() override;

  private:
    KernelHandler<void(Image* iop_image, const Scene* i_scene)> m_kernel;
  };