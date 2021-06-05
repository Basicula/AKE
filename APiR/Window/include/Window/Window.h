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

  // Run main infinity loop for window
  virtual void Open() = 0;

  void SetEventListner(EventListner* ip_event_listner);
  // Image source - basically image that will be updated through update function
  // or can be static image to visualize
  void SetImageSource(const Image* ip_source);
  void SetUpdateFunction(UpdateFunction i_func);

protected:
  // Init for window specific options etc
  virtual void _Init() = 0;

  // Base function for rendering worflow that consists of below function
  void _RenderFrame();
  // Rendering preprocessing (input events etc)
  virtual void _PreDisplay() = 0;
  // Actual display func that updates screen with new/updated image
  virtual void _Display();
  // Rendering postprocessing rutine
  virtual void _PostDisplay() = 0;

  void _OnMouseButtonPressed(const MouseButton i_button);
  void _OnMouseButtonReleased(const MouseButton i_button);
  void _OnMouseMoved(const double i_x, const double i_y);
  void _OnMouseScroll(const double i_offset);
  void _OnKeyPressed(const KeyboardButton i_key);
  void _OnKeyReleased(const KeyboardButton i_key);
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