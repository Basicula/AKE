#include "Main/WindowExample.h"

#include "Main/ConsoleLogEventListner.h"
#include "Window/GLFWDebugGUIView.h"
#include "Window/GLFWWindow.h"
#include "Window/GLUTWindow.h"

void ConsoleLogEventsExample()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  // GLUTWindow window(width, height, "EventListnerTest");
  GLFWWindow window(width, height, "EventListnerTest");
  window.SetEventListner(new ConsoleLogEventListner);
  window.Open();
}

void GUIViewExample()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  GLFWWindow window(width, height, "GUIView");
  window.SetGUIView(new GLFWDebugGUIView(window.GetOpenGLWindow()));
  window.Open();
}
