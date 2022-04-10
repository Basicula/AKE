#include "Main/2DSceneExample.h"

#include "Math/Constants.h"
#include "Renderer.2D/OpenGLRenderer.h"
#include "Rendering.2D/CircleDrawer.h"
#include "Rendering.2D/RectangleDrawer.h"
#include "Rendering.2D/Triangle2DDrawer.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "World.2D/Scene2D.h"

namespace {
  double Random(const double i_from = 0.0, const double i_to = 1.0)
  {
    if (i_to < i_from)
      return 0.0;

    const auto range = i_to - i_from;
    return range * static_cast<double>(rand()) / RAND_MAX - i_to;
  }

  Vector2d RandomUnitVector(const double i_length = 1.0)
  {
    const auto angle = Random(0.0, Math::Constants::TWOPI);
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
    window.InitRenderer<OpenGLRenderer>(i_scene);
    window.Open();
  }
}

namespace Scene2DExamples {
  void Rectangles(const std::size_t i_window_width,
                  const std::size_t i_window_height,
                  const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: Rectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = Random(-1.0, 1.0);
      const auto y = Random(-1.0, 1.0);
      const auto rect_width = Random(-1.0, 1.0);
      const auto rect_height = Random(-1.0, 1.0);
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
      const auto x = Random(-1.0, 1.0);
      const auto y = Random(-1.0, 1.0);
      const auto radius = Random(-1.0, 1.0);
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
      const auto x = Random(-1.0, 1.0);
      const auto y = Random(-1.0, 1.0);
      const auto rect_width = Random(-1.0, 1.0);
      const auto rect_height = Random(-1.0, 1.0);
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
      const Vector2d center(Random(-1.0, 1.0), Random(-1.0, 1.0));
      Vector2d vertices[] = { RandomUnitVector(), RandomUnitVector(), RandomUnitVector() };
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
}