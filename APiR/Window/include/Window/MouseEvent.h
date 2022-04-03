#pragma once
#include "Window/Event.h"
#include "Window/Keys.h"

#include <utility>

class MouseButtonPressedEvent final : public Event
{
public:
  explicit MouseButtonPressedEvent(MouseButton i_button);

  [[nodiscard]] MouseButton Button() const;

private:
  MouseButton m_button;
};

class MouseButtonReleasedEvent final : public Event
{
public:
  explicit MouseButtonReleasedEvent(MouseButton i_button);

  [[nodiscard]] MouseButton Button() const;

private:
  MouseButton m_button;
};

class MouseMovedEvent final : public Event
{
public:
  MouseMovedEvent(double i_x, double i_y);

  [[nodiscard]] std::pair<double, double> Position() const;

private:
  double m_x;
  double m_y;
};

class MouseScrollEvent final : public Event
{
public:
  explicit MouseScrollEvent(double i_offset);

  [[nodiscard]] double Offset() const;

private:
  double m_offset;
};