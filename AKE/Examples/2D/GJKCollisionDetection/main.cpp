#include "FillScene.h"
#include "Renderer.2D/OpenGLRenderer.h"
#include "SceneController.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"

int main()
{
  constexpr std::size_t width = 800, height = 600;
  constexpr std::size_t objects_count = 10;

  Scene2D scene("Test 2D scene: CollisionDetection");
  FillScene(scene, objects_count, static_cast<double>(width), static_cast<double>(height));

  GLFWWindow window(width, height, scene.GetName());
  window.SetUpdateFunction([&scene]() { scene.Update(); });
  window.InitGUIView<GLFWDebugGUIView>(window.GetOpenGLWindow());
  window.InitRenderer<OpenGLRenderer>(static_cast<int>(width), static_cast<int>(height), scene);
  window.InitEventListner<SceneController>(scene, objects_count, width, height);
  window.Open();
}