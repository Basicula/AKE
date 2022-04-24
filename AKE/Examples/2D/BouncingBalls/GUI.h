#pragma once
#include "Window/GLFWGUIView.h"
#include "Window/GLFWWindow.h"
#include "World.h"

class Gui : public GLFWGUIView
{
public:
  Gui(const GLFWWindow& i_window, World* ip_scene);

  void Render() override;

private:
  World* mp_scene;
};