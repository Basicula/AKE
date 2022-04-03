#include "Window/GLUTWindow.h"

#pragma warning(push)
#pragma warning(disable : 4505)
#include <GL/glut.h>
#pragma warning(pop)

GLUTWindow* GLUTWindow::mg_instance = nullptr;

GLUTWindow::GLUTWindow(size_t i_width, size_t i_height, const std::string& i_title)
  : Window(i_width, i_height, i_title)
  , m_main_screen(0)
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
  glutInitWindowSize(static_cast<int>(m_width), static_cast<int>(m_height));
  glutCreateWindow(m_title.c_str());

  glutDisplayFunc(GLUTWindow::_DisplayFuncWrapper);
  glutIdleFunc(GLUTWindow::_DisplayFuncWrapper);
  glutKeyboardFunc(GLUTWindow::_PressButtonWrapper);
  glutMouseFunc(GLUTWindow::_MouseEventWrapper);
}

void GLUTWindow::_PreDisplay() {}

void GLUTWindow::_PostDisplay()
{
  glRasterPos2d(-0.99, 0.95);
  std::string fps_string = "FPS : " + std::to_string(m_fps_counter.GetFPS());
  for (auto c : fps_string)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

  glutSwapBuffers();
}

void GLUTWindow::_DisplayFuncWrapper()
{
  mg_instance->_RenderFrame();
}

void GLUTWindow::_PressButtonWrapper(unsigned char i_key, int /*i_x*/, int /*i_y*/)
{
  mg_instance->_OnKeyPressed(static_cast<KeyboardButton>(i_key));
}

void GLUTWindow::_MouseEventWrapper(int i_button, int i_state, int /*i_x*/, int /*i_y*/)
{
  if (i_state == GLUT_UP)
    mg_instance->_OnMouseButtonReleased(static_cast<MouseButton>(i_button));
  if (i_state == GLUT_DOWN)
    mg_instance->_OnMouseButtonPressed(static_cast<MouseButton>(i_button));
}
