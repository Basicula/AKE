#pragma once
#include "Rendering/IRenderer.h"
#include "Rendering/Scene.h"

#include <memory>

class ImageRenderer : public IRenderer
{
public:
  ImageRenderer();

  void Render() override;

protected:
  virtual void _GenerateFrameImage() = 0;

private:
  void _OnWindowResize(int i_new_width, int i_new_height) override;

  void _BindImage();

protected:
  unsigned int m_image_binding;
  std::unique_ptr<Image> mp_image_source;
};