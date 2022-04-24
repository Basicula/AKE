#include "GUI.h"

#include "imgui.h"

Gui::Gui(const GLFWWindow& i_window, World* ip_scene)
  : GLFWGUIView(i_window.GetOpenGLWindow())
  , mp_scene(ip_scene)
{}

void Gui::Render()
{
  ImGui::NewFrame();

  ImGui::SetNextWindowSize({ 0, 0 });
  ImGui::SetNextWindowSizeConstraints({ 128, 64 }, { 512, 512 });
  ImGui::SetNextWindowPos({ 8, 8.0f }, 0, { 0.f, 0.f });
  ImGui::Begin("World options", nullptr, ImGuiWindowFlags_NoResize);

  static float bouncing_coef = static_cast<float>(mp_scene->m_options.m_bouncing_coef);
  mp_scene->m_options.m_bouncing_coef = static_cast<double>(bouncing_coef);
  ImGui::SliderFloat("Ball bouncing coef", &bouncing_coef, 0.0f, 1.0f);

  static float time_delta = static_cast<float>(mp_scene->m_options.m_time_delta);
  mp_scene->m_options.m_time_delta = static_cast<double>(time_delta);
  ImGui::SliderFloat("Time delta", &time_delta, 0.001f, 0.1f);

  static float acceleration[2] = { static_cast<float>(mp_scene->m_options.m_gravity_acceleration[0]),
                                   static_cast<float>(mp_scene->m_options.m_gravity_acceleration[1]) };
  mp_scene->m_options.m_gravity_acceleration = { static_cast<double>(acceleration[0]),
                                                 static_cast<double>(acceleration[1]) };
  ImGui::SliderFloat2("Acceleration", acceleration, -10.0f, 10.0f);

  if (ImGui::Button("Clear"))
    mp_scene->Clear();

  ImGui::Text(
    "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  ImGui::End();

  ImGui::Render();
}
