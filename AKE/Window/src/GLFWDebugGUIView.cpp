#include "Window/GLFWDebugGUIView.h"

#include <imgui.h>

GLFWDebugGUIView::GLFWDebugGUIView(GLFWwindow* ip_window)
  : GLFWGUIView(ip_window)
{}

void GLFWDebugGUIView::Render()
{
  ImGui::NewFrame();

  ImGui::SetNextWindowSize({ 0, 0 });
  ImGui::SetNextWindowSizeConstraints({ 128, 64 }, { 512, 512 });
  ImGui::SetNextWindowPos({ 8, 8.0f }, 0, { 0.f, 0.f });
  ImGui::Begin("Debug view", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

  ImGui::Text(
    "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  ImGui::End();

  ImGui::Render();
}