#include "Window/GLFWGUIView.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

GLFWGUIView::GLFWGUIView(GLFWwindow* ip_window)
  : mp_window(ip_window)
{
  // Init OpenGL related functional
  glewInit();
  _Init();
}

void GLFWGUIView::_Init()
{
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(mp_window, true);
  ImGui_ImplOpenGL3_Init();
}

void GLFWGUIView::NewFrame()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
}

void GLFWGUIView::Display()
{
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWGUIView::Clean()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
