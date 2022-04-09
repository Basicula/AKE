#include "Window/ImageWindowBackend.h"

#include <GL/glew.h>

ImageWindowBackend::ImageWindowBackend(const Image* ip_source_image)
  : mp_image(ip_source_image)
  , m_image_binding(0)
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
               static_cast<GLsizei>(mp_image->GetWidth()),
               static_cast<GLsizei>(mp_image->GetHeight()),
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               mp_image->GetRGBAData());

  glBindTexture(GL_TEXTURE_2D, 0);
}

void ImageWindowBackend::PreDisplay() {}

void ImageWindowBackend::Display()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (mp_image) {
    glBindTexture(GL_TEXTURE_2D, m_image_binding);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 static_cast<GLsizei>(mp_image->GetWidth()),
                 static_cast<GLsizei>(mp_image->GetHeight()),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 mp_image->GetRGBAData());
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
}

void ImageWindowBackend::PostDisplay() {}

void ImageWindowBackend::_OnWindowResize(int i_new_width, int i_new_height)
{
  glViewport(0, 0, i_new_width, i_new_height);
}
