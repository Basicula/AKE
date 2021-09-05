#pragma once
#include <Window/GLFWGUIView.h>

class GLFWDebugGUIView : public GLFWGUIView {
public:
  GLFWDebugGUIView(GLFWwindow* ip_window);
  virtual ~GLFWDebugGUIView() = default;

  virtual void Render() override;
};