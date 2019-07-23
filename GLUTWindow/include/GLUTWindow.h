#pragma once
#include <vector>

class GLUTWindow
  {
  public:
    GLUTWindow(int i_width, int i_height, char* i_title = "New window");

    void Open();
    inline void SetPicture(const std::vector<unsigned char>& i_picture) { m_picture = i_picture; };

  private:
    void _Init();
    void _DisplayFunc();

    static void _DisplayFuncWrapper();
  private:
    int m_width;
    int m_height;
    char* m_title;

    std::vector<unsigned char> m_picture;

  };

 static GLUTWindow* mg_instance;