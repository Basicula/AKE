#pragma once
#include "Geometry.2D/Shape.h"
#include "Rendering.2D/Drawer.h"

#include <memory>
#include <string>
#include <vector>

class Scene2D
{
public:
  struct Object
  {
    std::shared_ptr<Shape> mp_shape;
    std::shared_ptr<Drawer> mp_drawer;
  };

public:
  explicit Scene2D(std::string i_name);

  [[nodiscard]] std::string GetName() const;

  void AddObject(std::unique_ptr<Object>&& ip_object);
  [[nodiscard]] const std::vector<std::unique_ptr<Object>>& GetObjects() const;

private:
  std::string m_name;
  std::vector<std::unique_ptr<Object>> m_objects;
};
