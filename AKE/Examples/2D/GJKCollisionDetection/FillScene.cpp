#include "FillScene.h"

#include "Common/Randomizer.h"
#include "Math/Constants.h"
#include "Utils2D/Utils2D.h"

void FillScene(Scene2D& io_scene, const std::size_t i_objects_count, const double i_max_x, const double i_max_y)
{
  io_scene.Clear();
  for (std::size_t object_id = 0; object_id < i_objects_count; ++object_id) {
    const Vector2d center(Randomizer::Get(0.0, i_max_x), Randomizer::Get(0.0, i_max_y));
    auto p_object = Utils2D::RandomObject();
    p_object->GetTransformation().SetTranslation(center);
    p_object->GetTransformation().SetRotation(Randomizer::Get(0.0, Math::Constants::TWOPI));
    io_scene.AddObject(std::move(p_object));
  }
}