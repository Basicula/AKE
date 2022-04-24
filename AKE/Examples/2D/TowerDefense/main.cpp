#include "GameController.h"
#include "GameRenderer.h"
#include "TowerDefense.h"
#include "Window/GLFWWindow.h"

int main()
{
  constexpr std::size_t width = 800, height = 600;
  TowerDefense game(width, height);
  GLFWWindow window(width, height, "Tower defense");
  window.InitEventListner<GameController>(&game);
  window.InitRenderer<GameRenderer>(&game);
  auto update_wrapper = [&]() { game.Update(); };
  window.SetUpdateFunction(update_wrapper);
  window.Open();
}