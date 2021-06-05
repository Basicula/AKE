#pragma once
#include <Window/Event.h>
#include <Window/Keys.h>

class KeyPressedEvent : public Event {
public:
  KeyPressedEvent(const KeyboardButton i_key);

  KeyboardButton Key() const;

private:
  KeyboardButton m_key;
};

class KeyReleasedEvent : public Event {
public:
  KeyReleasedEvent(const KeyboardButton i_key);

  KeyboardButton Key() const;

private:
  KeyboardButton m_key;
};