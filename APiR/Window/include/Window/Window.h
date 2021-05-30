#pragma once
#include <Image/Image.h>

#include <Window/EventListner.h>
#include <Window/FPSCounter.h>

#include <functional>
#include <string>

class Window {
public:
  // Lambda or function that need to be run each frame
  using UpdateFunction = std::function<void()>;

public:
  Window(const size_t i_width, const size_t i_height, const std::string& i_title = "New window");
  virtual ~Window();

  virtual void Open() = 0;

  void SetEventListner(EventListner* ip_event_listner);
  void SetImageSource(const Image* ip_source);
  void SetUpdateFunction(UpdateFunction i_func);

protected:
  virtual void _Init() = 0;

  void _RenderFrame();
  virtual void _PreDisplay() = 0;
  virtual void _Display();
  virtual void _PostDisplay() = 0;

  void _OnMouseButtonPressed(const int i_button);
  void _OnMouseButtonReleased(const int i_button);
  void _OnMouseMoved(const double i_x, const double i_y);
  void _OnMouseScroll(const double i_offset);
  void _OnKeyPressed(const unsigned char i_key);
  void _OnKeyRepeat(const unsigned char i_key);
  void _OnKeyReleased(const unsigned char i_key);
  void _OnWindowResized(const int i_width, const int i_height);
  void _OnWindowClosed();


protected:
  std::string m_title;
  size_t m_width;
  size_t m_height;

  unsigned int m_frame_binding;

  FPSCounter m_fps_counter;

  UpdateFunction m_update_function;

  const Image* mp_source;
  EventListner* mp_event_listner;
};