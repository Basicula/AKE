#pragma once
#include <Image/Image.h>

#include <Rendering/Scene.h>

#include <memory>

class IRenderer
  {
  public:

    void SetOutputImage(Image* iop_image);
    void SetScene(const Scene* ip_scene);

    virtual void Render() = 0;

    std::size_t GetRenderingDepth() const;
    void SetRenderingDepth(std::size_t i_depth);

  protected:
    // post processing function for derived classes
    virtual void _OutputImageWasSet() = 0;
    virtual void _SceneWasSet() = 0;

  protected:
    size_t m_depth;
    Image* mp_frame_image;
    const Scene* mp_scene;
  };