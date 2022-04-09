#pragma once

class IRenderer
{
public:
  virtual ~IRenderer() = default;

  virtual void Render() = 0;

  virtual void _OnWindowResize(int i_new_width, int i_new_height) = 0;
};