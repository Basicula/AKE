#pragma once
#include "Window/GUIView.h"

struct GLFWwindow;

class GLFWGUIView : public GUIView
{
public:
  explicit GLFWGUIView(GLFWwindow* ip_window);

  void NewFrame() override;
  void Display() override;
  void Clean() override;

protected:
  void _Init() override;

private:
  GLFWwindow* mp_window;
};