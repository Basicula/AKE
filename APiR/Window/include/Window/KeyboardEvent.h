#pragma once
#include "Window/Event.h"
#include "Window/Keys.h"

class KeyPressedEvent final : public Event
{
public:
  explicit KeyPressedEvent(KeyboardButton i_key);

  [[nodiscard]] KeyboardButton Key() const;

private:
  KeyboardButton m_key;
};

class KeyReleasedEvent final : public Event
{
public:
  explicit KeyReleasedEvent(const KeyboardButton i_key);

  [[nodiscard]] KeyboardButton Key() const;

private:
  KeyboardButton m_key;
};