#include "GUI.h"
#include "Window/GLFWWindow.h"
#include "World.h"
#include "WorldController.h"
#include "WorldRenderer.h"

int main()
{
  World world(800, 600);
  GLFWWindow window(800, 600, "Bouncing balls");
  window.InitEventListner<WorldController>(&world);
  window.InitRenderer<WorldRenderer>(&world);
  window.InitGUIView<Gui>(window, &world);
  auto update_wrapper = [&]() { world.Update(); };
  window.SetUpdateFunction(update_wrapper);
  window.Open();
  return 0;
}
