#include "Rendering.2D/Scene2D.h"

Scene2D::Scene2D(std::string i_name)
  : m_name(std::move(i_name))
{}

std::string Scene2D::GetName() const
{
  return m_name;
}

void Scene2D::AddObject(std::unique_ptr<Object>&& ip_object)
{
  m_objects.emplace_back(std::move(ip_object));
}

const std::vector<std::unique_ptr<Scene2D::Object>>& Scene2D::GetObjects() const
{
  return m_objects;
}
