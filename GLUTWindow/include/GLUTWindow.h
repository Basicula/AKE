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

    static void _DisplayFuncWrapper();
  private:
    int m_width;
    int m_height;
    char* m_title;

    OpenCLKernel m_kernel;
  };

 static GLUTWindow* mg_instance;