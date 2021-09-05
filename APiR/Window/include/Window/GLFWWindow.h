#pragma once
#include <Window/Window.h>

#include <GLFW/glfw3.h>

class GLFWWindow : public Window {
public:
  GLFWWindow(const size_t i_width, const size_t i_height, const std::string& i_title = "New GLFW window");
  ~GLFWWindow();

  virtual void Open() override;

  // Commonly used for different opengl calling functions (for instance imgui init functions)
  GLFWwindow* GetOpenGLWindow() const;

protected:
  virtual void _Init() override;

  virtual void _PreDisplay() override;
  virtual void _PostDisplay() override;

private:
  static void _MouseMovedCallback(GLFWwindow* ip_window, const double i_x, const double i_y);
  static void _MouseButtonCallback(GLFWwindow* ip_window, const int i_button, const int i_action, const int i_mods);
  static void _MouseScrollCallback(GLFWwindow* ip_window, const double i_xoffset, const double i_yoffset);
  static void _KeyboardCallback(GLFWwindow* ip_window, const int i_key, const int i_scancode, const int i_action, const int i_mods);
  static void _WindowResizeCallback(GLFWwindow* ip_window, const int i_width, const int i_height);
  static void _WindowCloseCallback(GLFWwindow* ip_window);

private:
  GLFWwindow* mp_window;
  GLuint m_main_screen;
  static GLFWWindow* mg_instance;
};