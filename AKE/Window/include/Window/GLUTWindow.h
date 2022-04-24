#pragma once
#include "Window/Window.h"

class GLUTWindow : public Window
{
public:
  GLUTWindow(size_t i_width, size_t i_height, std::string i_title = "New window");

  void Open() override;

private:
  void _Init() override;

  void _PreDisplay() override;
  void _PostDisplay() override;

  static void _DisplayFuncWrapper();
  static void _PressButtonWrapper(unsigned char i_key, int i_x, int i_y);
  static void _MouseEventWrapper(int i_button, int i_state, int i_x, int i_y);

private:
  static GLUTWindow* mg_instance;
};
