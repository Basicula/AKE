#include <Rendering/Container.h>

void Container::AddObject(IRenderableSPtr i_object) {
  m_objects.emplace_back(i_object);
  Update();
  }

bool Container::TraceRay(IntersectionRecord& io_intersection, const Ray& i_ray) const {
  bool is_intersected = false;
  for (const auto& object : m_objects)
    is_intersected |= object->IntersectWithRay(io_intersection, i_ray);
  return is_intersected;
  }

void Container::Update() {
  }

std::size_t Container::Size() const {
  return m_objects.size();
  }