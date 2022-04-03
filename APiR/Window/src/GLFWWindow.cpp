#include "Window/GLFWWindow.h"

namespace {
  // GLFW has the following defines for mouse buttons
  // GLFW_MOUSE_BUTTON_1   0
  // GLFW_MOUSE_BUTTON_2   1
  // GLFW_MOUSE_BUTTON_3   2
  // GLFW_MOUSE_BUTTON_4   3
  // GLFW_MOUSE_BUTTON_5   4
  // GLFW_MOUSE_BUTTON_6   5
  // GLFW_MOUSE_BUTTON_7   6
  // GLFW_MOUSE_BUTTON_8   7
  // GLFW_MOUSE_BUTTON_LAST   GLFW_MOUSE_BUTTON_8
  // GLFW_MOUSE_BUTTON_LEFT   GLFW_MOUSE_BUTTON_1
  // GLFW_MOUSE_BUTTON_RIGHT   GLFW_MOUSE_BUTTON_2
  // GLFW_MOUSE_BUTTON_MIDDLE   GLFW_MOUSE_BUTTON_3
  MouseButton GLFWIntToMouseButton(const int i_button) {
    switch (i_button) {
      case 0: return MouseButton::MOUSE_LEFT_BUTTON;
      case 1: return MouseButton::MOUSE_RIGHT_BUTTON;
      case 2: return MouseButton::MOUSE_MIDDLE_BUTTON;
      default: return MouseButton::MOUSE_UNDEFINED_BUTTON;
    }
  }

  KeyboardButton GLFWIntToKeyboardButton(const int i_key) {
    switch (i_key) {
      case GLFW_KEY_Q: return KeyboardButton::KEY_Q;
      case GLFW_KEY_W: return KeyboardButton::KEY_W;
      case GLFW_KEY_E: return KeyboardButton::KEY_E;
      case GLFW_KEY_R: return KeyboardButton::KEY_R;
      case GLFW_KEY_T: return KeyboardButton::KEY_T;
      case GLFW_KEY_Y: return KeyboardButton::KEY_Y;
      case GLFW_KEY_U: return KeyboardButton::KEY_U;
      case GLFW_KEY_I: return KeyboardButton::KEY_I;
      case GLFW_KEY_O: return KeyboardButton::KEY_O;
      case GLFW_KEY_P: return KeyboardButton::KEY_P;
      case GLFW_KEY_A: return KeyboardButton::KEY_A;
      case GLFW_KEY_S: return KeyboardButton::KEY_S;
      case GLFW_KEY_D: return KeyboardButton::KEY_D;
      case GLFW_KEY_F: return KeyboardButton::KEY_F;
      case GLFW_KEY_G: return KeyboardButton::KEY_G;
      case GLFW_KEY_H: return KeyboardButton::KEY_H;
      case GLFW_KEY_J: return KeyboardButton::KEY_J;
      case GLFW_KEY_K: return KeyboardButton::KEY_K;
      case GLFW_KEY_L: return KeyboardButton::KEY_L;
      case GLFW_KEY_Z: return KeyboardButton::KEY_Z;
      case GLFW_KEY_X: return KeyboardButton::KEY_X;
      case GLFW_KEY_C: return KeyboardButton::KEY_C;
      case GLFW_KEY_V: return KeyboardButton::KEY_V;
      case GLFW_KEY_B: return KeyboardButton::KEY_B;
      case GLFW_KEY_N: return KeyboardButton::KEY_N;
      case GLFW_KEY_M: return KeyboardButton::KEY_M;
      case GLFW_KEY_1: return KeyboardButton::KEY_1;
      case GLFW_KEY_2: return KeyboardButton::KEY_2;
      case GLFW_KEY_3: return KeyboardButton::KEY_3;
      case GLFW_KEY_4: return KeyboardButton::KEY_4;
      case GLFW_KEY_5: return KeyboardButton::KEY_5;
      case GLFW_KEY_6: return KeyboardButton::KEY_6;
      case GLFW_KEY_7: return KeyboardButton::KEY_7;
      case GLFW_KEY_8: return KeyboardButton::KEY_8;
      case GLFW_KEY_9: return KeyboardButton::KEY_9;
      case GLFW_KEY_0: return KeyboardButton::KEY_0;
      default: return KeyboardButton::KEY_UNDEFINED;
    }
  }
}

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

GLFWwindow* GLFWWindow::GetOpenGLWindow() const{
  return mp_window;
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
    mg_instance->_OnMouseButtonPressed(GLFWIntToMouseButton(i_button));
  else if (i_action == GLFW_RELEASE)
    mg_instance->_OnMouseButtonReleased(GLFWIntToMouseButton(i_button));
}

void GLFWWindow::_MouseScrollCallback(GLFWwindow* /*ip_window*/, const double /*i_xoffset*/, const double i_yoffset) {
  mg_instance->_OnMouseScroll(i_yoffset);
}

void GLFWWindow::_KeyboardCallback(GLFWwindow* /*ip_window*/, const int i_key, const int /*i_scancode*/, const int i_action, const int /*i_mods*/) {
  if (i_action == GLFW_KEY_UNKNOWN)
    return;
  if (i_action == GLFW_PRESS)
    mg_instance->_OnKeyPressed(GLFWIntToKeyboardButton(i_key));
  else if (i_action == GLFW_RELEASE)
    mg_instance->_OnKeyReleased(GLFWIntToKeyboardButton(i_key));
}

void GLFWWindow::_WindowResizeCallback(GLFWwindow* /*ip_window*/, const int i_width, const int i_height) {
  mg_instance->_OnWindowResized(i_width, i_height);
}

void GLFWWindow::_WindowCloseCallback(GLFWwindow* /*ip_window*/) {
  mg_instance->_OnWindowClosed();
}
