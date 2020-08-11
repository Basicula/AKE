#pragma once
#include <functional>
#include <vector>
#include <string>

#pragma warning(push)
#pragma warning(disable:4505)
#include <GL/glut.h>
#pragma warning(pop)

#include <Image.h>
#include <FPSCounter.h>

class GLUTWindow
  {
  public:
    GLUTWindow(int i_width, int i_height, const char* i_title = "New window");

    void SetImageSource(const Image* ip_source);
    void SetUpdateFunction(std::function<void()> i_func);
    void Open();

  private:
    void _Init();
    void _DisplayFunc();
    void _PressButton(unsigned char i_key, int i_x, int i_y);
    void _MouseEvent(int i_button, int i_state, int i_x, int i_y);

    static void _DisplayFuncWrapper();
    static void _PressButtonWrapper(unsigned char i_key, int i_x, int i_y);
    static void _MouseEventWrapper(int i_button, int i_state, int i_x, int i_y);

  private:
    int m_width;
    int m_height;
    const char* m_title;

    GLuint m_main_screen;
    const Image* mp_source;
    FPSCounter m_fps_counter;

    std::function<void()> m_update_function;
  };

inline void GLUTWindow::SetUpdateFunction(std::function<void()> i_func)
  {
  m_update_function = i_func;
  }

 static GLUTWindow* mg_instance;