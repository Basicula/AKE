#include "Main/3DSceneExample.h"

#include "Main/SceneExamples.h"
#include "Main/SimpleCameraController.h"
#include "Rendering/CPURayTracer.h"
#include "Rendering/Scene.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/GLUTWindow.h"
#include "Window/ImageWindowBackend.h"

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
  window.InitWindowBackend<ImageWindowBackend>(&image);
  window.SetUpdateFunction(update_func);
  window.InitEventListner<SimpleCameraController>(scene.GetActiveCamera());
  window.InitGUIView<GLFWDebugGUIView>(window.GetOpenGLWindow());
  window.Open();
}
