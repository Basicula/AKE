#pragma once
#include "World.2D/Object2D.h"

#include <memory>
#include <string>
#include <vector>

class Scene2D
{
public:
  explicit Scene2D(std::string i_name);

  [[nodiscard]] std::string GetName() const;

  void AddObject(std::unique_ptr<Object2D>&& ip_object);
  [[nodiscard]] const std::vector<std::unique_ptr<Object2D>>& GetObjects() const;
  void Clear();

  void Update();

private:
  std::string m_name;
  std::vector<std::unique_ptr<Object2D>> m_objects;
};
