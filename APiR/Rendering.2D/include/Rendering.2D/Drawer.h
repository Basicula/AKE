#pragma once

class Drawer
{
public:
  virtual ~Drawer() = default;

  virtual void Draw() const = 0;
};