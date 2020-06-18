#pragma once
#include <functional>
#include <vector>

#include <Image.h>

class GLUTWindow
  {
  public:
    GLUTWindow(int i_width, int i_height, const char* i_title = "New window");

    void SetImageSource(const Image* ip_source);
    void SetUpdateImageFunc(std::function<void()> i_func);
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

    int m_iterations_for_mandelbrot;
    bool m_new_mandelbrot;

    const Image* mp_source;
    std::function<void()> m_update_function;
  };

inline void GLUTWindow::SetImageSource(const Image* ip_source)
  {
  mp_source = ip_source;
  }

inline void GLUTWindow::SetUpdateImageFunc(std::function<void()> i_func)
  {
  m_update_function = i_func;
  }

 static GLUTWindow* mg_instance;