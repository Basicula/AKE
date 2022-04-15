#pragma once
#include "Window/Window.h"

#include <GLFW/glfw3.h>

class GLFWWindow : public Window
{
public:
  GLFWWindow(size_t i_width, size_t i_height, std::string i_title = "New GLFW window");
  ~GLFWWindow() override;

  void Open() override;

  // Commonly used for different opengl calling functions (for instance imgui init functions)
  [[nodiscard]] GLFWwindow* GetOpenGLWindow() const;

protected:
  void _Init() override;

  void _PreDisplay() override;
  void _PostDisplay() override;

private:
  static void _MouseMovedCallback(GLFWwindow* ip_window, double i_x, double i_y);
  static void _MouseButtonCallback(GLFWwindow* ip_window, int i_button, int i_action, int i_mods);
  static void _MouseScrollCallback(GLFWwindow* ip_window, double i_xoffset, double i_yoffset);
  static void _KeyboardCallback(GLFWwindow* ip_window, int i_key, int i_scancode, int i_action, int i_mods);
  static void _WindowResizeCallback(GLFWwindow* ip_window, int i_width, int i_height);
  static void _WindowCloseCallback(GLFWwindow* ip_window);

private:
  GLFWwindow* mp_window;
  static GLFWWindow* mg_instance;
};