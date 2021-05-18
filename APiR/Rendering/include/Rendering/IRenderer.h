#pragma once
#include <Image/Image.h>

#include <Rendering/Scene.h>

class IRenderer
  {
  public:

    void SetOutputImage(Image* iop_image);

    virtual void Render(const Scene& m_scene) = 0;

    std::size_t GetRenderingDepth() const;
    void SetRenderingDepth(std::size_t i_depth);

  protected:
    virtual void _OutputImageWasSet() = 0;

  protected:
    size_t m_depth;
    Image* mp_frame_image;
  };