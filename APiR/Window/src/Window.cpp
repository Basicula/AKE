#include "Window/Window.h"

#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"
#include "Window/WindowEvent.h"

#if defined(WIN32)
// need to include windows before gl for compiling gl stuff
#include <Windows.h>
#endif

#include <GL/gl.h>

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

Window::~Window()
{
  delete mp_event_listner;
  delete mp_gui_view;
}

void Window::SetEventListner(EventListner* ip_event_listner)
{
  mp_event_listner = ip_event_listner;
}

void Window::SetGUIView(GUIView* ip_gui_view)
{
  mp_gui_view = ip_gui_view;
}

void Window::SetImageSource(const Image* ip_source)
{
  mp_source = ip_source;
  glGenTextures(1, &m_frame_binding);
  glBindTexture(GL_TEXTURE_2D, m_frame_binding);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGBA,
               static_cast<GLsizei>(mp_source->GetWidth()),
               static_cast<GLsizei>(mp_source->GetHeight()),
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               mp_source->GetRGBAData());

  glBindTexture(GL_TEXTURE_2D, 0);
}

void Window::SetUpdateFunction(UpdateFunction i_func)
{
  m_update_function = std::move(i_func);
}

void Window::_RenderFrame()
{
  if (mp_event_listner)
    mp_event_listner->PollEvents();
  _PreDisplay();
  _Display();
  _PostDisplay();
}

void Window::_Display()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (m_update_function)
    m_update_function();
  m_fps_counter.Update();

  if (mp_source) {
    glBindTexture(GL_TEXTURE_2D, m_frame_binding);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 static_cast<GLsizei>(mp_source->GetWidth()),
                 static_cast<GLsizei>(mp_source->GetHeight()),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 mp_source->GetRGBAData());
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    constexpr double x = 1.0;
    glTexCoord2d(0.0, 0.0);
    glVertex2d(-x, -x);
    glTexCoord2d(1.0, 0.0);
    glVertex2d(x, -x);
    glTexCoord2d(1.0, 1.0);
    glVertex2d(x, x);
    glTexCoord2d(0.0, 1.0);
    glVertex2d(-x, x);
    glEnd();

    glDisable(GL_TEXTURE_2D);
  }

  if (mp_gui_view) {
    mp_gui_view->NewFrame();
    mp_gui_view->Render();
    mp_gui_view->Display();
  }
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
}

void Window::_OnWindowClosed() const
{
  if (mp_event_listner)
    mp_event_listner->ProcessEvent(WindowCloseEvent());
}
