#include <GLUTWindow.h>

#include <gl/glut.h>

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
  int temp = 1;
  glutInit(&temp, &m_title);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(m_width, m_height);
  glutCreateWindow(m_title);

  glutDisplayFunc(GLUTWindow::_DisplayFunc);
  }

void GLUTWindow::_DisplayFunc()
  {
  glClearColor(0.5, 0.5, 0.5, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  glutSwapBuffers();
  }
