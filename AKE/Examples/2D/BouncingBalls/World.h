#pragma once
#include "Math/Vector.h"
#include "World.2D/Object2D.h"

#include <memory>
#include <vector>

struct World
{
  struct SceneOptions
  {
    Vector2d m_gravity_acceleration{ 0.0, 0.0 };
    double m_bouncing_coef = .5;
    double m_ball_radius = 10.0;
    double m_time_delta = 0.015;
  };

  World(std::size_t i_width, std::size_t i_height);

  [[nodiscard]] std::pair<std::size_t, std::size_t> GetSize() const;
  [[nodiscard]] const std::vector<std::unique_ptr<Object2D>>& GetObjects() const;
  
  void SpawnBall(const Vector2d& i_position, const Vector2d& i_init_velocity);

  void Update();

  void Clear();

  std::size_t m_width;
  std::size_t m_height;
  std::vector<std::unique_ptr<Object2D>> m_objects;
  SceneOptions m_options;
  Vector2d m_screen_origin;

private:
  void _Init();

  std::unique_ptr<Object2D> _CreateWall(const Vector2d& i_position, const Vector2d& i_size);
};