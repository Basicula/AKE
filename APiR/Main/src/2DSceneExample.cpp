#include "Main/2DSceneExample.h"

#include <Geometry.2D/Circle.h>
#include <Geometry.2D/Rectangle.h>
#include <Rendering.2D/CircleDrawer.h>
#include <Rendering.2D/OpenGLRenderer.h>
#include <Rendering.2D/RectangleDrawer.h>
#include <Rendering.2D/Scene2D.h>
#include <Window/GLFWDebugGUIView.h>
#include <Window/GLFWWindow.h>

namespace {
  double UniformRandom() { return 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0; }
}

namespace Scene2DExamples {
  void Rectangles(const std::size_t i_window_width,
                  const std::size_t i_window_height,
                  const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: Rectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = UniformRandom();
      const auto y = UniformRandom();
      const auto rect_width = UniformRandom();
      const auto rect_height = UniformRandom();
      auto p_rectangle = std::make_unique<Object2D>();
      p_rectangle->InitShape<Rectangle>(rect_width, rect_height);
      p_rectangle->InitDrawer<RectangleDrawer>(
        static_cast<const Rectangle&>(p_rectangle->GetShape()), Color::Red, false);
      p_rectangle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_rectangle));
    }

    const OpenGLRenderer renderer(scene);

    auto update_func = [&]() { renderer.Render(); };

    GLFWWindow window(i_window_width, i_window_height, scene.GetName());
    window.SetUpdateFunction(update_func);
    window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
    window.Open();
  }

  void Circles(const std::size_t i_window_width, const std::size_t i_window_height, const std::size_t i_circles_count)
  {
    Scene2D scene("Test 2D scene: Circles");

    for (std::size_t circle_id = 0; circle_id < i_circles_count; ++circle_id) {
      const auto x = UniformRandom();
      const auto y = UniformRandom();
      const auto radius = UniformRandom();
      auto p_circle = std::make_unique<Object2D>();
      p_circle->InitShape<Circle>(radius);
      p_circle->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_circle->GetShape()), Color::Blue, false);
      p_circle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_circle));
    }

    const OpenGLRenderer renderer(scene);

    auto update_func = [&]() { renderer.Render(); };

    GLFWWindow window(i_window_width, i_window_height, scene.GetName());
    window.SetUpdateFunction(update_func);
    window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
    window.Open();
  }

  void RotatedRectangles(const std::size_t i_window_width,
                         const std::size_t i_window_height,
                         const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: RotatedRectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = UniformRandom();
      const auto y = UniformRandom();
      const auto rect_width = UniformRandom();
      const auto rect_height = UniformRandom();
      auto p_rectangle = std::make_unique<Object2D>();
      p_rectangle->InitShape<Rectangle>(rect_width, rect_height);
      p_rectangle->InitDrawer<RectangleDrawer>(
        static_cast<const Rectangle&>(p_rectangle->GetShape()), Color::Red, false);
      p_rectangle->GetTransformation().SetTranslation({ x, y });
      scene.AddObject(std::move(p_rectangle));
    }

    const OpenGLRenderer renderer(scene);

    auto update_func = [&]() {
      renderer.Render();
      for (const auto& object : scene.GetObjects())
        object->GetTransformation().Rotate(0.001);
    };

    GLFWWindow window(i_window_width, i_window_height, scene.GetName());
    window.SetUpdateFunction(update_func);
    window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
    window.Open();
  }
}