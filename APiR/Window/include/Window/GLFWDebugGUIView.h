#pragma once
#include "Window/GLFWGUIView.h"

class GLFWDebugGUIView final : public GLFWGUIView
{
public:
  explicit GLFWDebugGUIView(GLFWwindow* ip_window);

  void Render() override;
};