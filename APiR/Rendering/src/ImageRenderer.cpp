#include "Rendering/ImageRenderer.h"

#include <GL/glew.h>

ImageRenderer::ImageRenderer()
  : m_image_binding(0)
  , mp_image_source(std::make_unique<Image>(800, 600))
  {
  _BindImage();
  }

void ImageRenderer::Render()
{
  _GenerateFrameImage();
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindTexture(GL_TEXTURE_2D, m_image_binding);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGBA,
               static_cast<GLsizei>(mp_image_source->GetWidth()),
               static_cast<GLsizei>(mp_image_source->GetHeight()),
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               mp_image_source->GetRGBAData());
  glEnable(GL_TEXTURE_2D);

  glBegin(GL_QUADS);
  constexpr double x = 1.0;
  glTexCoord2d(0.0, 0.0);
  glVertex2d(-x, -x);
  glTexCoord2d(1.0, 0.0);
  glVertex2d(x, -x);
  glTexCoord2d(1.0, 1.0);
  glVertex2d(x, x);
  glTexCoord2d(0.0, 1.0);
  glVertex2d(-x, x);
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

void ImageRenderer::_OnWindowResize(const int i_new_width, const int i_new_height)
{
  glViewport(0, 0, i_new_width, i_new_height);
  mp_image_source = std::make_unique<Image>(i_new_width, i_new_height);
}

void ImageRenderer::_BindImage()
{
  glGenTextures(1, &m_image_binding);
  glBindTexture(GL_TEXTURE_2D, m_image_binding);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGBA,
               static_cast<GLsizei>(mp_image_source->GetWidth()),
               static_cast<GLsizei>(mp_image_source->GetHeight()),
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               mp_image_source->GetRGBAData());

  glBindTexture(GL_TEXTURE_2D, 0);
}
