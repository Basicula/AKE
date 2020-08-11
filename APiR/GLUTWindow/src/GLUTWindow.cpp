#include <GLUTWindow.h>

GLUTWindow::GLUTWindow(int i_width, int i_height, const char* i_title)
  : m_width(i_width)
  , m_height(i_height)
  , m_title(i_title)
  , m_main_screen()
  , mp_source(nullptr)
  , m_fps_counter(1)
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
  glutIdleFunc(GLUTWindow::_DisplayFuncWrapper);
  glutKeyboardFunc(GLUTWindow::_PressButtonWrapper);
  glutMouseFunc(GLUTWindow::_MouseEventWrapper);
  }

void GLUTWindow::_DisplayFunc()
  {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_update_function();
  m_fps_counter.Update();

  glBindTexture(GL_TEXTURE_2D, m_main_screen);
  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGBA,
    static_cast<GLsizei>(mp_source->GetWidth()),
    static_cast<GLsizei>(mp_source->GetHeight()),
    0,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    mp_source->GetRGBAData());
  glEnable(GL_TEXTURE_2D);

  glBegin(GL_QUADS);
  const double x = 1.0;
  glTexCoord2d(0.0, 0.0); glVertex2d(-x, -x);
  glTexCoord2d(1.0, 0.0); glVertex2d(x, -x);
  glTexCoord2d(1.0, 1.0); glVertex2d(x, x);
  glTexCoord2d(0.0, 1.0); glVertex2d(-x, x);
  glEnd();

  glDisable(GL_TEXTURE_2D);

  glRasterPos2d(-0.99, 0.95);
  std::string fps_string = "FPS : " + std::to_string(m_fps_counter.GetFPS());
  for (auto c : fps_string)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

  glutSwapBuffers();
  }

void GLUTWindow::SetImageSource(const Image* ip_source)
  {
  mp_source = ip_source;
  glGenTextures(1, &m_main_screen);
  glBindTexture(GL_TEXTURE_2D, m_main_screen);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(
    GL_TEXTURE_2D, 
    0, 
    GL_RGBA, 
    static_cast<GLsizei>(mp_source->GetWidth()), 
    static_cast<GLsizei>(mp_source->GetHeight()), 
    0, 
    GL_RGBA, 
    GL_UNSIGNED_BYTE, 
    mp_source->GetRGBAData());

  glBindTexture(GL_TEXTURE_2D, 0);
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
  mg_instance->_PressButton(i_key, i_x, i_y);
  }

void GLUTWindow::_MouseEventWrapper(int i_button, int i_state, int i_x, int i_y)
  {
  mg_instance->_MouseEvent(i_button, i_state, i_x, i_y);
  }
