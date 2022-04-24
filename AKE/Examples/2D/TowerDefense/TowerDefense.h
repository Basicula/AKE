#pragma once
#include "World.2D/Object2D.h"

#include <vector>

struct TowerDefense
{
  Object2D m_player;
  std::vector<std::unique_ptr<Object2D>> m_bullets;
  std::vector<std::unique_ptr<Object2D>> m_mobs;
  double m_width;
  double m_height;
  double m_time_delta;
  double m_shoot_power;
  double m_mob_speed;

  TowerDefense(std::size_t i_window_width, std::size_t i_window_height);

  void Shoot(const Vector2d& i_direction);

  void Update();

private:
  void _Init();

  [[nodiscard]] std::unique_ptr<Object2D> _CreateBullet(const Vector2d& i_direction) const;
  [[nodiscard]] std::unique_ptr<Object2D> _CreateMob() const;
};