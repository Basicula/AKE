#pragma once

class GLUTWindow
  {
  public:
    GLUTWindow(int i_width, int i_height, char* i_title = "New window");

    void Open();

  private:
    void _Init();
    static void _DisplayFunc();

  private:
    int m_width;
    int m_height;
    char* m_title;
  };