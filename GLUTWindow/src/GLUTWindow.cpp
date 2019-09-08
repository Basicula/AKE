#include <GLUTWindow.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <gl/glut.h>


void Timer(int value) {
  glutTimerFunc(25, Timer, 0);
  glutPostRedisplay();
  }


GLUTWindow::GLUTWindow(int i_width, int i_height, char* i_title)
  : m_width(i_width)
  , m_height(i_height)
  , m_title(i_title)
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
  glutInit(&temp, &m_title);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(m_width, m_height);
  glutCreateWindow(m_title);

  glutDisplayFunc(GLUTWindow::_DisplayFuncWrapper);
  Timer(0);

  m_kernel.Init();
  std::string path("D:\\Study\\RayTracing\\RayTracingBmp\\OpenCLKernel\\src\\OpenCLKernel.cl");
  std::ifstream reader(path);
  m_kernel.SetKernelSource(std::string(std::istreambuf_iterator<char>(reader),std::istreambuf_iterator<char>()));
  m_kernel.Build();
  }

void GLUTWindow::_DisplayFunc()
  {
  glClearColor(1.0, 0, 0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  GLuint texture;

  const size_t width = 1024;
  const size_t height = 768;
  const size_t bytes_per_pixel = 4;

  std::vector<unsigned char> picture = m_kernel.MandelbrotSet(width,height,256);

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  gluBuild2DMipmaps(GL_TEXTURE_2D, 4, width, height, GL_RGBA, GL_UNSIGNED_BYTE, picture.data());

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

void GLUTWindow::_DisplayFuncWrapper()
  {
  mg_instance->_DisplayFunc();
  }
