#include <GLUTWindow.h>
#include <time.h>
#include <iostream>
#include <fstream>
#pragma warning(push)
#pragma warning(disable:4505)
#include <GL/glut.h>
#pragma warning(pop)


void Timer(int /*value*/) {
  glutTimerFunc(5, Timer, 0);
  glutPostRedisplay();
  }


GLUTWindow::GLUTWindow(int i_width, int i_height, const char* i_title)
  : m_width(i_width)
  , m_height(i_height)
  , m_title(i_title)
  , m_iterations_for_mandelbrot(256)
  , m_new_mandelbrot(true)
  {
  _Init();
  }

void GLUTWindow::Open()
  {
  glutMainLoop();
  }

void GLUTWindow::_Init()
  {
  mg_instance = this;
  int temp = 1;
  glutInit(&temp, nullptr);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(m_width, m_height);
  glutCreateWindow(m_title);

  glutDisplayFunc(GLUTWindow::_DisplayFuncWrapper);
  glutKeyboardFunc(GLUTWindow::_PressButtonWrapper);
  glutMouseFunc(GLUTWindow::_MouseEventWrapper);
  Timer(0);
  }

void GLUTWindow::_DisplayFunc()
  {
  glClearColor(1.0, 0, 0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_update_function();
  
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  if (mp_source)
    {
    gluBuild2DMipmaps(
      GL_TEXTURE_2D, 
      4, 
      static_cast<GLint>(mp_source->GetWidth()), 
      static_cast<GLint>(mp_source->GetHeight()), 
      GL_RGBA, 
      GL_UNSIGNED_BYTE, 
      mp_source->GetRGBAData());
    }

  glEnable(GL_TEXTURE_2D);

  glBegin(GL_QUADS);
  float x = 1;
  glTexCoord2d(0.0, 0.0); glVertex2d(-x, -x);
  glTexCoord2d(1.0, 0.0); glVertex2d(x, -x);
  glTexCoord2d(1.0, 1.0); glVertex2d(x, x);
  glTexCoord2d(0.0, 1.0); glVertex2d(-x, x);
  glEnd();

  glutSwapBuffers();
  }

void GLUTWindow::_PressButton(unsigned char /*i_key*/, int /*i_x*/, int /*i_y*/)
  {
  
  }

void GLUTWindow::_MouseEvent(int /*i_button*/, int /*i_state*/, int /*i_x*/, int /*i_y*/)
  {
  }

void GLUTWindow::_DisplayFuncWrapper()
  {
  mg_instance->_DisplayFunc();
  }

void GLUTWindow::_PressButtonWrapper(unsigned char i_key, int i_x, int i_y)
  {
  mg_instance->_PressButton(i_key,i_x,i_y);
  }

void GLUTWindow::_MouseEventWrapper(int i_button, int i_state, int i_x, int i_y)
  {
  mg_instance->_MouseEvent(i_button, i_state, i_x, i_y);
  }
