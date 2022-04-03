#pragma once

class GUIView
{
public:
  virtual ~GUIView() = default;

  virtual void NewFrame() = 0;
  virtual void Render() = 0;
  virtual void Display() = 0;
  virtual void Clean() = 0;

protected:
  virtual void _Init() = 0;
};