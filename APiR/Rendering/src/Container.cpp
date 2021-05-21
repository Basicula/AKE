#include <Rendering/Container.h>

void Container::AddObject(IRenderableSPtr i_object) {
  m_objects.emplace_back(i_object);
  Update();
  }

std::size_t Container::Size() const {
  return m_objects.size();
  }