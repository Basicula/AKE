#include "Window/Window.h"

#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"
#include "Window/WindowEvent.h"

Window::Window(const size_t i_width, const size_t i_height, std::string i_title)
  : m_title(std::move(i_title))
  , m_width(i_width)
  , m_height(i_height)
  , m_frame_binding(0)
  , m_fps_counter(1)
  , mp_source(nullptr)
  , mp_event_listner(nullptr)
  , mp_gui_view(nullptr)
{}

void Window::SetUpdateFunction(UpdateFunction i_func)
{
  m_update_function = std::move(i_func);
}

void Window::_RenderFrame()
{
  _PreDisplay();

  if (mp_event_listner)
    mp_event_listner->PollEvents();

  if (m_update_function)
    m_update_function();

  m_fps_counter.Update();

  if (mp_renderer)
    mp_renderer->Render();

  if (mp_gui_view) {
    mp_gui_view->NewFrame();
    mp_gui_view->Render();
    mp_gui_view->Display();
  }

  _PostDisplay();
}

void Window::_OnMouseButtonPressed(const MouseButton i_button) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(MouseButtonPressedEvent(i_button));
}

void Window::_OnMouseButtonReleased(const MouseButton i_button) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(MouseButtonReleasedEvent(i_button));
}

void Window::_OnMouseMoved(const double i_x, const double i_y) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(MouseMovedEvent(i_x, i_y));
}

void Window::_OnMouseScroll(const double i_offset) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(MouseScrollEvent(i_offset));
}

void Window::_OnKeyPressed(const KeyboardButton i_key) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(KeyPressedEvent(i_key));
}

void Window::_OnKeyReleased(const KeyboardButton i_key) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(KeyReleasedEvent(i_key));
}

void Window::_OnWindowResized(const int i_width, const int i_height) const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(WindowResizeEvent(i_width, i_height));
  if (mp_renderer)
    mp_renderer->_OnWindowResize(i_width, i_height);
}

void Window::_OnWindowClosed() const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(WindowCloseEvent());
}
