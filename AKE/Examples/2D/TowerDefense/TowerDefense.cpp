#include "TowerDefense.h"

#include "Common/Randomizer.h"
#include "Geometry.2D/Circle.h"
#include "Geometry.2D/Triangle2D.h"
#include "Physics.2D/CommonPhysicalProperties.h"
#include "Physics.2D/GJKCircleConvex2D.h"
#include "Physics.2D/GJKCollisionDetection2D.h"
#include "Physics.2D/GJKTriangleConvex2D.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/Triangle2DDrawer.h"

TowerDefense::TowerDefense(const std::size_t i_window_width, const std::size_t i_window_height)
  : m_width(static_cast<double>(i_window_width))
  , m_height(static_cast<double>(i_window_height))
  , m_time_delta(0.01)
  , m_shoot_power(10.0)
  , m_mob_speed(7.0)
{
  _Init();
}


void TowerDefense::Shoot(const Vector2d& i_direction)
{
  m_bullets.emplace_back(_CreateBullet(i_direction));
}

void TowerDefense::Update()
{
  if (Randomizer::Get(0, 10000) < 5)
    m_mobs.emplace_back(_CreateMob());
  for (const auto& p_bullet : m_bullets)
    p_bullet->GetPhysicalProperty()->Apply(m_time_delta);
  for (const auto& p_mob : m_mobs)
    p_mob->GetPhysicalProperty()->Apply(m_time_delta);

  std::vector<bool> bullets_to_remove(m_bullets.size(), false), mobs_to_remove(m_mobs.size(), false);
  for (std::size_t bullet_id = 0; bullet_id < m_bullets.size(); ++bullet_id)
    for (std::size_t mob_id = 0; mob_id < m_mobs.size(); ++mob_id)
      if (GJKCollisionDetection2D::GetCollision(*m_bullets[bullet_id]->GetCollider(), *m_mobs[mob_id]->GetCollider())
            .m_is_collided) {
        bullets_to_remove[bullet_id] = true;
        mobs_to_remove[mob_id] = true;
      }

  for (int bullet_id = static_cast<int>(m_bullets.size()) - 1; bullet_id >= 0; --bullet_id)
    if (bullets_to_remove[bullet_id])
      m_bullets.erase(m_bullets.begin() + bullet_id);
  for (int mob_id = static_cast<int>(m_mobs.size()) - 1; mob_id >= 0; --mob_id)
    if (mobs_to_remove[mob_id])
      m_mobs.erase(m_mobs.begin() + mob_id);
}

void TowerDefense::_Init()
{
  m_player.InitShape<Circle>(50);
  m_player.InitDrawer<CircleDrawer>(static_cast<const Circle&>(m_player.GetShape()), Color::White, true);
  m_player.InitCollider<GJKCircleConvex2D>(static_cast<const Circle&>(m_player.GetShape()));
  m_player.GetTransformation().SetTranslation({ m_width / 2.0, m_height / 2.0 });
}

[[nodiscard]] std::unique_ptr<Object2D> TowerDefense::_CreateBullet(const Vector2d& i_direction) const
{
  auto p_bullet = std::make_unique<Object2D>();
  p_bullet->InitShape<Circle>(5);
  p_bullet->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_bullet->GetShape()), Color::Red, true);
  p_bullet->InitCollider<GJKCircleConvex2D>(static_cast<const Circle&>(p_bullet->GetShape()));
  p_bullet->InitPhysicalProperty<CommonPhysicalProperties>(
    CommonPhysicalProperties{ 1.0, i_direction * m_shoot_power, Vector2d{ 0.0, 0.0 } });
  p_bullet->GetTransformation().SetTranslation(Vector2d{ m_width / 2.0, m_height / 2.0 } + i_direction * 50);
  return p_bullet;
}

[[nodiscard]] std::unique_ptr<Object2D> TowerDefense::_CreateMob() const
{
  auto p_mob = std::make_unique<Object2D>();
  p_mob->InitShape<Triangle2D>(Vector2d{ 0.0, 10.0 }, Vector2d{ -10.0, -10.0 }, Vector2d{ 10.0, -10.0 });
  p_mob->InitDrawer<Triangle2DDrawer>(static_cast<const Triangle2D&>(p_mob->GetShape()), Color::Blue, true);
  p_mob->InitCollider<GJKTriangleConvex2D>(static_cast<const Triangle2D&>(p_mob->GetShape()));
  const Vector2d target_position(m_width / 2.0, m_height / 2.0);
  Vector2d position;
  while (position[0] >= 0.0 && position[0] <= m_width && position[1] >= 0.0 && position[1] <= m_height)
    position =
      Vector2d{ Randomizer::Get<double>(-m_width, 2 * m_width), Randomizer::Get<double>(-m_height, 2 * m_height) };
  p_mob->GetTransformation().SetTranslation(position);
  p_mob->InitPhysicalProperty<CommonPhysicalProperties>(
    CommonPhysicalProperties{ 1.0, (target_position - position).Normalized() * m_mob_speed, Vector2d{ 0.0, 0.0 } });
  return p_mob;
}