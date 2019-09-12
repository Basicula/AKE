#pragma once
#include <OpenCLKernel.h>

#include <vector>

class GLUTWindow
  {
  public:
    GLUTWindow(int i_width, int i_height, char* i_title = "New window");

    void Open();

  private:
    void _Init();
    void _DisplayFunc();
    void _PressButton(unsigned char i_key, int i_x, int i_y);

    static void _DisplayFuncWrapper();
    static void _PressButtonWrapper(unsigned char i_key, int i_x, int i_y);
  private:
    int m_width;
    int m_height;
    char* m_title;

    OpenCLKernel m_kernel;

    int m_iterations_for_mandelbrot;
    bool m_new_mandelbrot;
  };

 static GLUTWindow* mg_instance;