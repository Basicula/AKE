#pragma once
#include "Image/Image.h"
#include "Window/EventListner.h"
#include "Window/FPSCounter.h"
#include "Window/GUIView.h"

#include <functional>
#include <string>

class Window
{
public:
  // Lambda or function that need to be run each frame
  using UpdateFunction = std::function<void()>;

public:
  Window(size_t i_width, size_t i_height, std::string i_title = "New window");
  virtual ~Window();

  // Run main infinity loop for window
  virtual void Open() = 0;

  // Set event listner to process different user inputs like keyboard button press etc
  void SetEventListner(EventListner* ip_event_listner);
  // Set custom gui view
  void SetGUIView(GUIView* ip_gui_view);

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

  void _OnMouseButtonPressed(MouseButton i_button) const;
  void _OnMouseButtonReleased(MouseButton i_button) const;
  void _OnMouseMoved(double i_x, double i_y) const;
  void _OnMouseScroll(double i_offset) const;
  void _OnKeyPressed(KeyboardButton i_key) const;
  void _OnKeyReleased(KeyboardButton i_key) const;
  void _OnWindowResized(int i_width, int i_height) const;
  void _OnWindowClosed() const;

protected:
  std::string m_title;
  size_t m_width;
  size_t m_height;

  unsigned int m_frame_binding;

  FPSCounter m_fps_counter;

  UpdateFunction m_update_function;

  const Image* mp_source;
  EventListner* mp_event_listner;
  GUIView* mp_gui_view;
};