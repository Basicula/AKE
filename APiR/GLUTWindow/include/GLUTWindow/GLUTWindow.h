#pragma once
#include <GLUTWindow/FPSCounter.h>
#include <Rendering/Image.h>

#pragma warning(push)
#pragma warning(disable:4505)
#include <GL/glut.h>
#pragma warning(pop)

#include <string>
#include <functional>

class GLUTWindow
  {
  public:
    using UpdateFunction = std::function<void()>;
    using KeyBoardFunction = std::function<void(unsigned char, int, int)>;
    using MouseFunction = std::function<void(int, int, int, int)>;

  public:
    GLUTWindow(int i_width, int i_height, const char* i_title = "New window");

    void SetImageSource(const Image* ip_source);
    void SetUpdateFunction(UpdateFunction i_func);
    void SetKeyBoardFunction(KeyBoardFunction i_func);
    void SetMouseFunction(MouseFunction i_func);
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

    UpdateFunction m_update_function;
    KeyBoardFunction m_keyboard_function;
    MouseFunction m_mouse_function;
  };

inline void GLUTWindow::SetUpdateFunction(UpdateFunction i_func)
  {
  m_update_function = i_func;
  }

inline void GLUTWindow::SetKeyBoardFunction(KeyBoardFunction i_func)
  {
  m_keyboard_function = i_func;
  }

inline void GLUTWindow::SetMouseFunction(MouseFunction i_func)
  {
  m_mouse_function = i_func;
  }

static GLUTWindow* mg_instance;