#include "SceneController.h"

#include "FillScene.h"
#include "Window/KeyboardEvent.h"
#include "Window/MouseEvent.h"

SceneController::SceneController(Scene2D& i_scene,
                                 const std::size_t i_objects_cnt,
                                 const std::size_t i_window_width,
                                 const std::size_t i_window_height)
  : m_scene(i_scene)
  , mp_focused_object(m_scene.GetObjects()[0].get())
  , m_objects_cnt(i_objects_cnt)
  , m_window_width(i_window_width)
  , m_window_height(i_window_height)
{}

void SceneController::_ProcessEvent(const Event& i_event)
{
  if (i_event.Type() == Event::EventType::KEY_PRESSED_EVENT) {
    const auto& key_pressed_event = static_cast<const KeyPressedEvent&>(i_event);
    if (key_pressed_event.Key() == KeyboardButton::KEY_R) {
      FillScene(m_scene, m_objects_cnt, static_cast<double>(m_window_width), static_cast<double>(m_window_height));
      mp_focused_object = m_scene.GetObjects()[0].get();
    }
  }

  if (i_event.Type() == Event::EventType::MOUSE_MOVED_EVENT) {
    const auto& mouse_moved_event = static_cast<const MouseMovedEvent&>(i_event);
    const auto new_pos = mouse_moved_event.Position();
    if (mp_focused_object)
      mp_focused_object->GetTransformation().SetTranslation({ new_pos.first, new_pos.second });
  }
}