#pragma once

// Class that determines way of visualizing information to window
class WindowBackend
{
public:
  virtual ~WindowBackend() = default;

  virtual void PreDisplay() = 0;
  virtual void Display() = 0;
  virtual void PostDisplay() = 0;

  virtual void _OnWindowResize(int i_new_width, int i_new_height) = 0;
};