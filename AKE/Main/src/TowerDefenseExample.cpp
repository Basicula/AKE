#include "Common/Randomizer.h"
#include "Geometry.2D/Circle.h"
#include "Geometry.2D/Triangle2D.h"
#include "Main/Example2D.h"
#include "Physics.2D/CommonPhysicalProperties.h"
#include "Physics.2D/GJKCircleConvex2D.h"
#include "Physics.2D/GJKCollisionDetection2D.h"
#include "Physics.2D/GJKTriangleConvex2D.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/Triangle2DDrawer.h"
#include "Visual/Color.h"
#include "Window/EventListner.h"
#include "Window/GLFWWindow.h"
#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"
#include "World.2D/Scene2D.h"

namespace {
  class TowerDefense
  {
  public:
    TowerDefense(const std::size_t i_window_width, const std::size_t i_window_height)
      : m_width(static_cast<double>(i_window_width))
      , m_height(static_cast<double>(i_window_height))
      , m_time_delta(0.01)
      , m_shoot_power(10.0)
      , m_mob_speed(7.0)
    {
      _Init();
    }

    void Start()
    {
      GLFWWindow window(static_cast<std::size_t>(m_width), static_cast<std::size_t>(m_height), "Tower defense");
      window.InitEventListner<GameController>(this);
      window.InitRenderer<GameRenderer>(this);
      auto update_wrapper = [&]() { Update(); };
      window.SetUpdateFunction(update_wrapper);
      window.Open();
    }

    void Shoot(const Vector2d& i_direction) { m_bullets.emplace_back(_CreateBullet(i_direction)); }

    void Update()
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
          if (GJKCollisionDetection2D::GetCollision(*m_bullets[bullet_id]->GetCollider(),
                                                    *m_mobs[mob_id]->GetCollider())
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

  private:
    void _Init()
    {
      m_player.InitShape<Circle>(50);
      m_player.InitDrawer<CircleDrawer>(static_cast<const Circle&>(m_player.GetShape()), Color::White, true);
      m_player.InitCollider<GJKCircleConvex2D>(static_cast<const Circle&>(m_player.GetShape()));
      m_player.GetTransformation().SetTranslation({ m_width / 2.0, m_height / 2.0 });
    }

    [[nodiscard]] std::unique_ptr<Object2D> _CreateBullet(const Vector2d& i_direction) const
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

    [[nodiscard]] std::unique_ptr<Object2D> _CreateMob() const
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

  private:
    class GameRenderer : public IRenderer
    {
    public:
      explicit GameRenderer(TowerDefense* ip_game)
        : mp_game(ip_game)
      {
        _OnWindowResize(static_cast<int>(mp_game->m_width), static_cast<int>(mp_game->m_height));
      }

      void Render() override
      {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        mp_game->m_player.GetDrawer().Draw();
        for (const auto& p_bullet : mp_game->m_bullets)
          p_bullet->GetDrawer().Draw();
        for (const auto& p_mob : mp_game->m_mobs)
          p_mob->GetDrawer().Draw();
      }

    private:
      void _OnWindowResize(const int i_new_width, const int i_new_height) override
      {
        glViewport(0, 0, i_new_width, i_new_height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, i_new_width, i_new_height, 0.0, 0.0, 1.0);
        // glOrtho(-i_new_width, i_new_width, i_new_height, -i_new_height, 0.0, 1.0);
      }

    private:
      TowerDefense* mp_game;
    };

    class GameController : public EventListner
    {
    public:
      explicit GameController(TowerDefense* ip_game)
        : mp_game(ip_game)
      {}

      void PollEvents() override {}

    private:
      void _ProcessEvent(const Event& i_event) override
      {
        if (i_event.Type() == Event::EventType::MOUSE_BUTTON_PRESSED_EVENT) {
          const auto& mouse_button_pressed_event = static_cast<const MouseButtonPressedEvent&>(i_event);
          if (mouse_button_pressed_event.Button() == MouseButton::MOUSE_LEFT_BUTTON) {
            auto shoot_direction = Vector2d{ m_mouse_position.first, m_mouse_position.second } -
                                   mp_game->m_player.GetTransformation().GetTranslation();
            shoot_direction.Normalize();
            mp_game->Shoot(shoot_direction);
          }
        }
        if (i_event.Type() == Event::EventType::MOUSE_MOVED_EVENT) {
          const auto& mouse_moved_event = static_cast<const MouseMovedEvent&>(i_event);
          const auto& position = mouse_moved_event.Position();
          m_mouse_position = { position.first, position.second };
        }
      }

    private:
      TowerDefense* mp_game;
    };

  private:
    Object2D m_player;
    std::vector<std::unique_ptr<Object2D>> m_bullets;
    std::vector<std::unique_ptr<Object2D>> m_mobs;
    double m_width;
    double m_height;
    double m_time_delta;
    double m_shoot_power;
    double m_mob_speed;
  };

}

namespace Example2D {
  void TowerDefenseGame(const std::size_t i_window_width, const std::size_t i_window_height)
  {
    TowerDefense game(i_window_width, i_window_height);
    game.Start();
  }
}
