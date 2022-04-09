#include "Main/3DSceneExample.h"

#include "Main/SceneExamples.h"
#include "Main/SimpleCameraController.h"
#include "Rendering/CPURayTracer.h"
#include "Rendering/Scene.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/GLUTWindow.h"

void SceneExample3D()
{
  const std::size_t width = 800;
  const std::size_t height = 600;

  Scene scene = ExampleScene::RandomSpheres(20);

  Image image(width, height);
  CPURayTracer renderer;
  renderer.SetOutputImage(&image);
  renderer.SetScene(&scene);

  auto update_func = [&]() { renderer.Render(); };
  // GLUTWindow window(width, height, scene.GetName());
  GLFWWindow window(width, height, scene.GetName());
  window.SetImageSource(&image);
  window.SetUpdateFunction(update_func);
  window.SetEventListner(new SimpleCameraController(scene.GetActiveCamera()));
  window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
  window.Open();
}
