#pragma once
#include "Window/Event.h"
#include "Window/Keys.h"

#include <utility>

class MouseButtonPressedEvent : public Event
{
public:
  MouseButtonPressedEvent(const MouseButton i_button);

  MouseButton Button() const;

private:
  MouseButton m_button;
};

class MouseButtonReleasedEvent : public Event
{
public:
  MouseButtonReleasedEvent(const MouseButton i_button);

  MouseButton Button() const;

private:
  MouseButton m_button;
};

class MouseMovedEvent : public Event
{
public:
  MouseMovedEvent(const double i_x, const double i_y);

  std::pair<double, double> Position() const;

private:
  double m_x;
  double m_y;
};

class MouseScrollEvent : public Event
{
public:
  MouseScrollEvent(const double i_offset);

  double Offset() const;

private:
  double m_offset;
};