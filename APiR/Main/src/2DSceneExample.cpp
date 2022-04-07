#include "Main/2DSceneExample.h"

#include <Geometry.2D/Circle.h>
#include <Geometry.2D/Rectangle.h>
#include <Rendering.2D/CircleDrawer.h>
#include <Rendering.2D/OpenGLRenderer.h>
#include <Rendering.2D/RectangleDrawer.h>
#include <Rendering.2D/Scene2D.h>
#include <Window/GLFWDebugGUIView.h>
#include <Window/GLFWWindow.h>

void SceneExample2D()
{
  constexpr std::size_t width = 400;
  constexpr std::size_t height = 400;

  Scene2D scene("Test 2D scene");
  auto unif_rand = []() { return (static_cast<double>(rand()) / RAND_MAX) - 0.5; };
  for (std::size_t circle_id = 0; circle_id < 10; ++circle_id) {
    const auto x = unif_rand();
    const auto y = unif_rand();
    const auto radius = unif_rand();
    auto p_circle = std::make_unique<Object2D>();
    p_circle->InitShape<Circle>(Vector2d(x, y), radius);
    p_circle->InitDrawer<CircleDrawer>(static_cast<const Circle&>(p_circle->GetShape()), Color::Blue, false);
    scene.AddObject(std::move(p_circle));
  }

  for (std::size_t rectangle_id = 0; rectangle_id < 10; ++rectangle_id) {
    const auto x = unif_rand();
    const auto y = unif_rand();
    const auto rect_width = unif_rand();
    const auto rect_height = unif_rand();
    auto p_rectangle = std::make_unique<Object2D>();
    p_rectangle->InitShape<Rectangle>(Vector2d(x, y), rect_width, rect_height);
    p_rectangle->InitDrawer<RectangleDrawer>(static_cast<const Rectangle&>(p_rectangle->GetShape()), Color::Red, false);
    scene.AddObject(std::move(p_rectangle));
  }

  const OpenGLRenderer renderer(scene);

  auto update_func = [&]() { renderer.Render(); };

  GLFWWindow window(width, height, scene.GetName());
  window.SetUpdateFunction(update_func);
  window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
  window.Open();
}
