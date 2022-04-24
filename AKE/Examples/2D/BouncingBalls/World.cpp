#include "World.h"

#include "Geometry.2D/Circle.h"
#include "Geometry.2D/Rectangle.h"
#include "Math/VectorOperations.h"
#include "Physics.2D/CommonPhysicalProperties.h"
#include "Physics.2D/GJKCircleConvex2D.h"
#include "Physics.2D/GJKCollisionDetection2D.h"
#include "Physics.2D/GJKRectangleConvex2D.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/RectangleDrawer.h"

World::World(const std::size_t i_width, const std::size_t i_height)
  : m_width(i_width)
  , m_height(i_height)
  , m_screen_origin(static_cast<double>(m_width) / 2.0, static_cast<double>(m_height) / 2.0)
{
  _Init();
}

std::pair<std::size_t, std::size_t> World::GetSize() const
{
  return { m_width, m_height };
}

const std::vector<std::unique_ptr<Object2D>>& World::GetObjects() const
{
  return m_objects;
}

void World::SpawnBall(const Vector2d& i_position, const Vector2d& i_init_velocity)
{
  auto p_ball = std::make_unique<Object2D>();
  p_ball->InitShape<Circle>(m_options.m_ball_radius);
  p_ball->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_ball->GetShape()), Color::Red, true);
  p_ball->InitCollider<GJKCircleConvex2D>(static_cast<const Circle&>(p_ball->GetShape()));
  p_ball->InitPhysicalProperty<CommonPhysicalProperties>(
    CommonPhysicalProperties{ m_options.m_ball_radius, i_init_velocity, m_options.m_gravity_acceleration });
  p_ball->GetTransformation().SetTranslation(i_position);
  m_objects.emplace_back(std::move(p_ball));
}

void World::Update()
{
  for (const auto& p_object : m_objects) {
    if (p_object->GetPhysicalProperty()) {
      auto* p_object_props = static_cast<CommonPhysicalProperties*>(p_object->GetPhysicalProperty());
      p_object_props->m_acceleration = m_options.m_gravity_acceleration;
      p_object->GetPhysicalProperty()->Apply(m_options.m_time_delta);
    }
  }

  std::vector<Vector2d> accumulated_offsets(m_objects.size(), Vector2d{ 0.0, 0.0 }),
    accumulated_velocities(m_objects.size(), Vector2d{ 0.0, 0.0 });
  for (std::size_t ball_id = 4; ball_id < m_objects.size(); ++ball_id) {
    const auto& p_ball = m_objects[ball_id];
    for (std::size_t wall_id = 0; wall_id < 4; ++wall_id) {
      const auto& p_wall = m_objects[wall_id];
      const auto collision = GJKCollisionDetection2D::GetCollision(*p_ball->GetCollider(), *p_wall->GetCollider());
      if (collision.m_is_collided) {
        auto* p_ball_props = static_cast<CommonPhysicalProperties*>(p_ball->GetPhysicalProperty());
        const auto collision_overlap_offset = -collision.m_normal * collision.m_depth;
        accumulated_offsets[ball_id] += collision_overlap_offset;
        p_ball_props->m_velocity =
          Math::Reflected(collision.m_normal, p_ball_props->m_velocity) * m_options.m_bouncing_coef;
      }
    }
  }

  for (std::size_t first_ball_id = 4; first_ball_id < m_objects.size(); ++first_ball_id) {
    const auto& p_first_ball = m_objects[first_ball_id];
    for (std::size_t second_ball_id = first_ball_id + 1; second_ball_id < m_objects.size(); ++second_ball_id) {
      const auto& p_second_ball = m_objects[second_ball_id];
      const auto collision =
        GJKCollisionDetection2D::GetCollision(*p_first_ball->GetCollider(), *p_second_ball->GetCollider());
      if (collision.m_is_collided) {
        auto* p_first_ball_props = static_cast<CommonPhysicalProperties*>(p_first_ball->GetPhysicalProperty());
        auto* p_second_ball_props = static_cast<CommonPhysicalProperties*>(p_second_ball->GetPhysicalProperty());

        const auto collision_overlap_offset = collision.m_normal * collision.m_depth * 0.1 / 2;
        accumulated_offsets[first_ball_id] -= collision_overlap_offset;
        accumulated_offsets[second_ball_id] += collision_overlap_offset;

        const auto relative_velocity = p_first_ball_props->m_velocity - p_second_ball_props->m_velocity;
        const auto impulse_magnitude = -(1.0 + m_options.m_bouncing_coef) * relative_velocity.Dot(collision.m_normal) /
                                       (1.0 / p_first_ball_props->m_mass + 1.0 / p_second_ball_props->m_mass);
        const auto impulse = collision.m_normal * impulse_magnitude;
        accumulated_velocities[first_ball_id] += impulse / p_first_ball_props->m_mass;
        accumulated_velocities[second_ball_id] -= impulse / p_second_ball_props->m_mass;
      }
    }
  }

  for (std::size_t ball_id = 4; ball_id < m_objects.size(); ++ball_id) {
    const auto& p_ball = m_objects[ball_id];
    p_ball->GetTransformation().Translate(accumulated_offsets[ball_id]);
    auto* p_ball_props = static_cast<CommonPhysicalProperties*>(p_ball->GetPhysicalProperty());
    p_ball_props->m_velocity += accumulated_velocities[ball_id];
  }
}

void World::Clear()
{
  m_objects.erase(m_objects.begin() + 4, m_objects.end());
}

void World::_Init()
{
  m_objects.emplace_back(
    _CreateWall({ -static_cast<double>(m_width) / 2.0, 0.0 }, { 10.0, static_cast<double>(m_height) }));
  m_objects.emplace_back(
    _CreateWall({ static_cast<double>(m_width) / 2.0, 0.0 }, { 10.0, static_cast<double>(m_height) }));
  m_objects.emplace_back(
    _CreateWall({ 0.0, static_cast<double>(m_height) / 2.0 }, { static_cast<double>(m_width), 10.0 }));
  m_objects.emplace_back(
    _CreateWall({ 0.0, -static_cast<double>(m_height) / 2.0 }, { static_cast<double>(m_width), 10.0 }));
}

std::unique_ptr<Object2D> World::_CreateWall(const Vector2d& i_position, const Vector2d& i_size)
{
  auto p_wall = std::make_unique<Object2D>();
  p_wall->InitShape<Rectangle>(i_size[0], i_size[1]);
  p_wall->InitDrawer<RectangleDrawer>(static_cast<const Rectangle&>(p_wall->GetShape()), Color::White, true);
  p_wall->InitCollider<GJKRectangleConvex2D>(static_cast<const Rectangle&>(p_wall->GetShape()));
  p_wall->GetTransformation().SetTranslation(i_position);
  return p_wall;
}
