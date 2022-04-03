#pragma once

class Event
{
public:
  enum class EventType
  {
    MOUSE_MOVED_EVENT,
    MOUSE_SCOLLED_EVENT,
    MOUSE_BUTTON_PRESSED_EVENT,
    MOUSE_BUTTON_RELEASED_EVENT,

    KEY_PRESSED_EVENT,
    KEY_REPEAT_EVENT,
    KEY_RELEASED_EVENT,

    WINDOW_RESIZED_EVENT,
    WINDOW_CLOSED_EVENT
  };

public:
  explicit Event(EventType i_type);
  virtual ~Event() = default;

  [[nodiscard]] EventType Type() const;

private:
  EventType m_type;
};