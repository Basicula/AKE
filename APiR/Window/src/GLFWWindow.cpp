#include <Window/GLFWWindow.h>

GLFWWindow* GLFWWindow::mg_instance = nullptr;

GLFWWindow::GLFWWindow(const size_t i_width, const size_t i_height, const std::string& i_title)
  : Window(i_width, i_height, i_title)
  , mp_window(nullptr) {
  _Init();
}

GLFWWindow::~GLFWWindow() {
  glfwTerminate();
}

void GLFWWindow::Open() {
  while (!glfwWindowShouldClose(mp_window))
    _RenderFrame();
}

void GLFWWindow::_Init() {
  mg_instance = this;
  glfwInit();
  mp_window = glfwCreateWindow(static_cast<int>(m_width), static_cast<int>(m_height), m_title.c_str(), nullptr, nullptr);

  glfwMakeContextCurrent(mp_window);
  glfwSetWindowUserPointer(mp_window, this);
  glfwSetCursorPosCallback(mp_window, _MouseMovedCallback);
  glfwSetScrollCallback(mp_window, _MouseScrollCallback);
  glfwSetMouseButtonCallback(mp_window, _MouseButtonCallback);
  glfwSetKeyCallback(mp_window, _KeyboardCallback);
  glfwSetWindowSizeCallback(mp_window, _WindowResizeCallback);
  glfwSetWindowCloseCallback(mp_window, _WindowCloseCallback);
}

void GLFWWindow::_PreDisplay() {
  glfwPollEvents();
}

void GLFWWindow::_PostDisplay() {
  glfwSwapBuffers(mp_window);
}

void GLFWWindow::_MouseMovedCallback(GLFWwindow* /*ip_window*/, const double i_x, const double i_y) {
  mg_instance->_OnMouseMoved(i_x, i_y);
}

void GLFWWindow::_MouseButtonCallback(GLFWwindow* /*ip_window*/, const int i_button, const int i_action, const int /*i_mods*/) {
  if (i_action == GLFW_PRESS)
    mg_instance->_OnMouseButtonPressed(i_button);
  else if (i_action == GLFW_RELEASE)
    mg_instance->_OnMouseButtonReleased(i_button);
}

void GLFWWindow::_MouseScrollCallback(GLFWwindow* /*ip_window*/, const double /*i_xoffset*/, const double i_yoffset) {
  mg_instance->_OnMouseScroll(i_yoffset);
}

void GLFWWindow::_KeyboardCallback(GLFWwindow* /*ip_window*/, const int i_key, const int /*i_scancode*/, const int i_action, const int /*i_mods*/) {
  if (i_action == GLFW_KEY_UNKNOWN)
    return;
  if (i_action == GLFW_PRESS)
    mg_instance->_OnKeyPressed(static_cast<unsigned char>(i_key));
  else if (i_action == GLFW_REPEAT)
    mg_instance->_OnKeyRepeat(static_cast<unsigned char>(i_key));
  else if (i_action == GLFW_RELEASE)
    mg_instance->_OnKeyReleased(static_cast<unsigned char>(i_key));
}

void GLFWWindow::_WindowResizeCallback(GLFWwindow* /*ip_window*/, const int i_width, const int i_height) {
  mg_instance->_OnWindowResized(i_width, i_height);
}

void GLFWWindow::_WindowCloseCallback(GLFWwindow* /*ip_window*/) {
  mg_instance->_OnWindowClosed();
}
