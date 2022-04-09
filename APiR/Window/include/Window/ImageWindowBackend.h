#pragma once
#include "Image/Image.h"
#include "Window/WindowBackend.h"

class ImageWindowBackend final : public WindowBackend
{
public:
  explicit ImageWindowBackend(const Image* ip_source_image);

  void PreDisplay() override;
  void Display() override;
  void PostDisplay() override;

  void _OnWindowResize(int i_new_width, int i_new_height) override;

private:
  const Image* mp_image;
  unsigned int m_image_binding;
};
