#include "Common/Randomizer.h"
#include "Main/Example2D.h"
#include "Math/Constants.h"
#include "Renderer.2D/OpenGLRenderer.h"
#include "Utils2D.h"
#include "Window/EventListner.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"
#include "World.2D/Scene2D.h"

namespace {
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
}

namespace Example2D {
  void CollisionDetection(const std::size_t i_window_width,
                          const std::size_t i_window_height,
                          const std::size_t i_objects_count)
  {
    Scene2D scene("Test 2D scene: CollisionDetection");
    FillScene(scene, i_objects_count, static_cast<double>(i_window_width), static_cast<double>(i_window_height));

    class EventTracker : public EventListner
    {
    public:
      EventTracker(Scene2D& i_scene,
                   const std::size_t i_objects_cnt,
                   const std::size_t i_window_width,
                   const std::size_t i_window_height)
        : m_scene(i_scene)
        , mp_focused_object(m_scene.GetObjects()[0].get())
        , m_objects_cnt(i_objects_cnt)
        , m_window_width(i_window_width)
        , m_window_height(i_window_height)
      {}

      void PollEvents() override {}

    private:
      void _ProcessEvent(const Event& i_event) override
      {
        if (i_event.Type() == Event::EventType::KEY_PRESSED_EVENT) {
          const auto& key_pressed_event = static_cast<const KeyPressedEvent&>(i_event);
          if (key_pressed_event.Key() == KeyboardButton::KEY_R) {
            FillScene(
              m_scene, m_objects_cnt, static_cast<double>(m_window_width), static_cast<double>(m_window_height));
            mp_focused_object = m_scene.GetObjects()[0].get();
          }
        }

        if (i_event.Type() == Event::EventType::MOUSE_MOVED_EVENT) {
          const auto& mouse_moved_event = static_cast<const MouseMovedEvent&>(i_event);
          const auto new_pos = mouse_moved_event.Position();
          if (mp_focused_object)
            mp_focused_object->GetTransformation().SetTranslation({ new_pos.first, new_pos.second });
        }
      }

    private:
      Scene2D& m_scene;
      Object2D* mp_focused_object;
      std::size_t m_objects_cnt;
      std::size_t m_window_width;
      std::size_t m_window_height;
    };

    auto update_function = [&scene]() { scene.Update(); };

    GLFWWindow window(i_window_width, i_window_height, scene.GetName());
    window.SetUpdateFunction(std::move(update_function));
    window.InitGUIView<GLFWDebugGUIView>(window.GetOpenGLWindow());
    window.InitRenderer<OpenGLRenderer>(static_cast<int>(i_window_width), static_cast<int>(i_window_height), scene);
    window.InitEventListner<EventTracker>(scene, i_objects_count, i_window_width, i_window_height);
    window.Open();
  }
}