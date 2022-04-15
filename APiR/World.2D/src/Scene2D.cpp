#include "World.2D/Scene2D.h"

#include "Physics.2D/GJKCollisionDetection2D.h"

Scene2D::Scene2D(std::string i_name)
  : m_name(std::move(i_name))
{}

std::string Scene2D::GetName() const
{
  return m_name;
}

void Scene2D::AddObject(std::unique_ptr<Object2D>&& ip_object)
{
  m_objects.emplace_back(std::move(ip_object));
}

const std::vector<std::unique_ptr<Object2D>>& Scene2D::GetObjects() const
{
  return m_objects;
}

void Scene2D::Clear()
{
  m_objects.clear();
}

void Scene2D::Update()
{
  std::vector<bool> in_collision(m_objects.size(), false);
  for (std::size_t first_object_id = 0; first_object_id < m_objects.size(); ++first_object_id) {
    const auto& first_object = m_objects[first_object_id];
    if (first_object->GetCollider() == nullptr)
      continue;
    for (std::size_t second_object_id = first_object_id + 1; second_object_id < m_objects.size(); ++second_object_id) {
      const auto& second_object = m_objects[second_object_id];
      if (second_object->GetCollider() == nullptr)
        continue;

      const auto collision =
        GJKCollisionDetection2D::GetCollision(*first_object->GetCollider(), *second_object->GetCollider());
      in_collision[first_object_id] = in_collision[first_object_id] || collision;
      in_collision[second_object_id] = in_collision[second_object_id] || collision;
    }
  }

  for (std::size_t object_id = 0; object_id < m_objects.size(); ++object_id)
    m_objects[object_id]->GetDrawer().m_fill = !in_collision[object_id];
}
