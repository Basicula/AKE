#include <Rendering/Object.h>

Object::Object()
  : mp_visual_material(nullptr)
  , mp_physic_material(nullptr) {}

Object::~Object() {
  if (mp_visual_material)
    delete mp_visual_material;
  if (mp_physic_material)
    delete mp_physic_material;
}

const IVisualMaterial* Object::VisualRepresentation() const {
  return mp_visual_material;
}

const IPhysicMaterial* Object::PhysicRepresentation() const {
  return mp_physic_material;
}
