#include "Rendering/Container.h"

Container::~Container()
{
  for (auto& object : m_objects)
    delete object;
}

void Container::AddObject(Object* ip_object) {
  m_objects.emplace_back(ip_object);
  Update();
  }

const Object* Container::TraceRay(double& o_distance, const Ray& i_ray, const double i_far) const {
  const Object* p_intersected_object = nullptr;
  double dist;
  o_distance = i_far;
  for (const auto& object : m_objects)
    if (object->IntersectWithRay(dist, i_ray, o_distance) && dist < o_distance) {
      o_distance = dist;
      p_intersected_object = object;
      }
  return p_intersected_object;
  }

void Container::Update() {
  }

std::size_t Container::Size() const {
  return m_objects.size();
  }