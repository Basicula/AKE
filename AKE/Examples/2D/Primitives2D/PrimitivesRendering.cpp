#include "PrimitivesRendering.h"

#include "Common/Randomizer.h"
#include "Renderer.2D/OpenGLRenderer.h"
#include "Utils2D/Utils2D.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "World.2D/Scene2D.h"

namespace {
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
}

namespace PrimitivesRendering {
  void Rectangles(const std::size_t i_window_width,
                  const std::size_t i_window_height,
                  const std::size_t i_rectangles_count)
  {
    Scene2D scene("Test 2D scene: Rectangles");

    for (std::size_t rectangle_id = 0; rectangle_id < i_rectangles_count; ++rectangle_id) {
      const auto x = Randomizer::Get(0.0, static_cast<double>(i_window_width));
      const auto y = Randomizer::Get(0.0, static_cast<double>(i_window_height));
      auto p_rectangle = Utils2D::RandomRectangle(100, 250);
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
      auto p_circle = Utils2D::RandomCircle(100, 250);
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
      auto p_rectangle = Utils2D::RandomRectangle(100, 250);
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
      auto p_triangle = Utils2D::RandomTriangle(50);
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