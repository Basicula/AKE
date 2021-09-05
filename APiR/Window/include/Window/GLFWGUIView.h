#pragma once
#include <Window/GUIView.h>

struct GLFWwindow;

class GLFWGUIView : public GUIView {
public:
  GLFWGUIView(GLFWwindow* ip_window);

  // Destructor will not free mp_window
  virtual ~GLFWGUIView() = default;

  virtual void NewFrame() override;
  virtual void Render() = 0;
  virtual void Display() override;
  virtual void Clean() override;

protected:
  virtual void _Init() override;

private:
  GLFWwindow* mp_window;
};