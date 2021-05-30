#pragma once
#include <utility>

class Event {
public:
  enum class EventType {
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
  Event(const EventType i_type);
  virtual ~Event() = default;

  EventType Type() const;

private:
  EventType m_type;
};

class KeyPressedEvent : public Event {
public:
  KeyPressedEvent(const unsigned char i_key);

  unsigned char Key() const;

private:
  unsigned char m_key;
};

class KeyRepeatEvent : public Event {
public:
  KeyRepeatEvent(const unsigned char i_key);

  unsigned char Key() const;

private:
  unsigned char m_key;
};

class KeyReleasedEvent : public Event {
public:
  KeyReleasedEvent(const unsigned char i_key);

  unsigned char Key() const;

private:
  unsigned char m_key;
};

class MouseButtonPressedEvent : public Event {
public:
  MouseButtonPressedEvent(const int i_button);

  int Button() const;

private:
  int m_button;
};

class MouseButtonReleasedEvent : public Event {
public:
  MouseButtonReleasedEvent(const int i_button);

  int Button() const;

private:
  int m_button;
};

class MouseMovedEvent : public Event {
public:
  MouseMovedEvent(const double i_x, const double i_y);

  std::pair<double, double> Position() const;

private:
  double m_x;
  double m_y;
};

class MouseScrollEvent : public Event {
public:
  MouseScrollEvent(const double i_offset);

  double Offset() const;

private:
  double m_offset;
};

class WindowResizeEvent : public Event {
public:
  WindowResizeEvent(const int i_width, const int i_height);

  std::pair<int, int> Size() const;

private:
  int m_width;
  int m_heigth;
};

class WindowCloseEvent : public Event {
public:
  WindowCloseEvent();
};