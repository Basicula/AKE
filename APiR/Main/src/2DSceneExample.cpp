#include "Main/2DSceneExample.h"

#include "Common/Randomizer.h"
#include "Math/Constants.h"
#include "Physics.2D/GJKCircleConvex2D.h"
#include "Physics.2D/GJKRectangleConvex2D.h"
#include "Renderer.2D/OpenGLRenderer.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/RectangleDrawer.h"
#include "Rendering.2D/Triangle2DDrawer.h"
#include "Window/EventListner.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"
#include "World.2D/Scene2D.h"

namespace {
  Vector2d RandomUnitVector(const double i_length = 1.0)
  {
    const auto angle = Randomizer::Get(0.0, Math::Constants::TWOPI);
    const auto x = cos(angle);
    const auto y = sin(angle);
    return { i_length * x, i_length * y };
  }

  void OpenWindow(const std::size_t i_window_width,
                  const std::size_t i_window_height,
                  const Scene2D& i_scene,
                  std::function<void()> i_update_function = nullptr)
  {
    GLFWWindow window(i_window_width, i_window_height, i_scene.GetName());
    window.SetUpdateFunction(std::move(i_update_function));
    window.InitGUIView<GLFWDebugGUIView>(window.GetOpenGLWindow());
    window.InitRenderer<OpenGLRenderer>(static_cast<int>(i_window_width), static_cast<int>(i_window_height), i_scene);
    window.Open();
  }

  std::unique_ptr<Object2D> RandomObject()
  {
    const auto rand = Randomizer::Get(0.0, 1.0);
    auto p_object = std::make_unique<Object2D>();
    if (rand < 0.33) {
      const auto radius = Randomizer::Get(25, 100);
      p_object->InitShape<Circle>(radius);
      p_object->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_object->GetShape()), Color::RandomColor(), true);
      p_object->InitCollider<GJKCircleConvex2D>(radius);
    } else if (rand < 0.66) {
      const auto width = Randomizer::Get(50, 250);
      const auto height = Randomizer::Get(50, 250);
      p_object->InitShape<Rectangle>(width, height);
      p_object->InitDrawer<RectangleDrawer>(
        static_cast<const Rectangle&>(p_object->GetShape()), Color::RandomColor(), true);
      p_object->InitCollider<GJKRectangleConvex2D>(width, height);
    } else {
      const Vector2d vertices[] = { RandomUnitVector(50), RandomUnitVector(50), RandomUnitVector(50) };
      p_object->InitShape<Triangle2D>(vertices[0], vertices[1], vertices[2]);
      p_object->InitDrawer<Triangle2DDrawer>(
        static_cast<const Triangle2D&>(p_object->GetShape()), Color::RandomColor(), true);
    }
    return p_object;
  }
}

namespace Scene2DExamples {
  void Rectangles(const std::size_t i_window_width,
                  const std::size_t i_window_height,
                  const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: Rectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = Randomizer::Get(0.0, static_cast<double>(i_window_width));
      const auto y = Randomizer::Get(0.0, static_cast<double>(i_window_height));
      const auto rect_width = Randomizer::Get(100, 250);
      const auto rect_height = Randomizer::Get(100, 250);
      auto p_rectangle = std::make_unique<Object2D>();
      p_rectangle->InitShape<Rectangle>(rect_width, rect_height);
      p_rectangle->InitDrawer<RectangleDrawer>(
        static_cast<const Rectangle&>(p_rectangle->GetShape()), Color::Red, false);
      p_rectangle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_rectangle));
    }

    OpenWindow(i_window_width, i_window_height, scene);
  }

  void Circles(const std::size_t i_window_width, const std::size_t i_window_height, const std::size_t i_circles_count)
  {
    Scene2D scene("Test 2D scene: Circles");

    for (std::size_t circle_id = 0; circle_id < i_circles_count; ++circle_id) {
      const auto x = Randomizer::Get(0.0, static_cast<double>(i_window_width));
      const auto y = Randomizer::Get(0.0, static_cast<double>(i_window_height));
      const auto radius = Randomizer::Get(100, 250);
      auto p_circle = std::make_unique<Object2D>();
      p_circle->InitShape<Circle>(radius);
      p_circle->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_circle->GetShape()), Color::Blue, false);
      p_circle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_circle));
    }

    OpenWindow(i_window_width, i_window_height, scene);
  }

  void RotatedRectangles(const std::size_t i_window_width,
                         const std::size_t i_window_height,
                         const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: RotatedRectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = Randomizer::Get(0.0, static_cast<double>(i_window_width));
      const auto y = Randomizer::Get(0.0, static_cast<double>(i_window_height));
      const auto rect_width = Randomizer::Get(100, 250);
      const auto rect_height = Randomizer::Get(100, 250);
      auto p_rectangle = std::make_unique<Object2D>();
      p_rectangle->InitShape<Rectangle>(rect_width, rect_height);
      p_rectangle->InitDrawer<RectangleDrawer>(
        static_cast<const Rectangle&>(p_rectangle->GetShape()), Color::Red, false);
      p_rectangle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_rectangle));
    }

    auto update_func = [&scene]() {
      for (const auto& object : scene.GetObjects())
        object->GetTransformation().Rotate(0.001);
    };
    OpenWindow(i_window_width, i_window_height, scene, update_func);
  }

  void RotatedTriangles(const std::size_t i_window_width,
                        const std::size_t i_window_height,
                        const std::size_t i_triangles_count)
  {
    Scene2D scene("Test 2D scene: RotatedTriangles");

    for (std::size_t triangle_id = 0; triangle_id < i_triangles_count; ++triangle_id) {
      const Vector2d center(Randomizer::Get(0.0, static_cast<double>(i_window_width)),
                            Randomizer::Get(0.0, static_cast<double>(i_window_height)));
      Vector2d vertices[] = { RandomUnitVector(50), RandomUnitVector(50), RandomUnitVector(50) };
      auto p_triangle = std::make_unique<Object2D>();
      p_triangle->InitShape<Triangle2D>(vertices[0], vertices[1], vertices[2]);
      p_triangle->InitDrawer<Triangle2DDrawer>(
        static_cast<const Triangle2D&>(p_triangle->GetShape()), Color::Green, false);
      p_triangle->GetTransformation().SetTranslation(center);
      scene.AddObject(std::move(p_triangle));
    }

    auto update_func = [&]() {
      for (const auto& object : scene.GetObjects())
        object->GetTransformation().Rotate(0.001);
    };
    OpenWindow(i_window_width, i_window_height, scene, update_func);
  }

  void CollisionDetectionExample(const std::size_t i_window_width,
                                 const std::size_t i_window_height,
                                 const std::size_t i_objects_count)
  {
    Scene2D scene("Test 2D scene: CollisionDetection");

    class EventTracker : public EventListner
    {
    public:
      EventTracker(Scene2D& i_scene,
                   const std::size_t i_objects_cnt,
                   const std::size_t i_window_width,
                   const std::size_t i_window_height)
        : m_scene(i_scene)
        , mp_focused_object(nullptr)
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
            m_scene.Clear();
             for (std::size_t object_id = 0; object_id < m_objects_cnt; ++object_id) {
              const Vector2d center(Randomizer::Get(0.0, static_cast<double>(m_window_width)),
                                    Randomizer::Get(0.0, static_cast<double>(m_window_height)));
              auto p_object = RandomObject();
              if (object_id == 0)
                mp_focused_object = p_object.get();
              p_object->GetTransformation().SetTranslation(center);
              p_object->GetTransformation().SetRotation(Randomizer::Get(0.0, Math::Constants::TWOPI));
              m_scene.AddObject(std::move(p_object));
            }
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