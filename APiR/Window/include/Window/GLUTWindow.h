#pragma once
#include "Window/FPSCounter.h"
#include "Window/Window.h"
#include "Image/Image.h"

#pragma warning(push)
#pragma warning(disable:4505)
#include <GL/glut.h>
#pragma warning(pop)

class GLUTWindow : public Window
  {
  public:
    GLUTWindow(size_t i_width, size_t i_height, const std::string& i_title = "New window");

    virtual void Open() override;

  private:
    virtual void _Init() override;

    virtual void _PreDisplay() override;
    virtual void _PostDisplay() override;

    static void _DisplayFuncWrapper();
    static void _PressButtonWrapper(unsigned char i_key, int i_x, int i_y);
    static void _MouseEventWrapper(int i_button, int i_state, int i_x, int i_y);

  private:
    GLuint m_main_screen;
    static GLUTWindow* mg_instance;
  };
