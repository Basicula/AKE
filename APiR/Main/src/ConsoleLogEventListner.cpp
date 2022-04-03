#include "Main/ConsoleLogEventListner.h"

#include <iostream>

void ConsoleLogEventListner::PollEvents() {}

void ConsoleLogEventListner::_ProcessEvent(const Event& i_event)
{
  switch (i_event.Type()) {
    case Event::EventType::KEY_PRESSED_EVENT:
      std::cout << "Key pressed " << static_cast<char>(static_cast<const KeyPressedEvent&>(i_event).Key()) << std::endl;
      return;
    case Event::EventType::KEY_RELEASED_EVENT:
      std::cout << "Key released " << static_cast<char>(static_cast<const KeyReleasedEvent&>(i_event).Key())
                << std::endl;
      return;
    case Event::EventType::MOUSE_BUTTON_PRESSED_EVENT:
      std::cout << "Mouse button pressed "
                << static_cast<int>(static_cast<const MouseButtonPressedEvent&>(i_event).Button()) << std::endl;
      return;
    case Event::EventType::MOUSE_BUTTON_RELEASED_EVENT:
      std::cout << "Mouse button released "
                << static_cast<int>(static_cast<const MouseButtonReleasedEvent&>(i_event).Button()) << std::endl;
      return;
    case Event::EventType::MOUSE_MOVED_EVENT: {
      auto mouse_position = static_cast<const MouseMovedEvent&>(i_event).Position();
      std::cout << "Mouse moved " << mouse_position.first << " " << mouse_position.second << std::endl;
      return;
    }
    case Event::EventType::MOUSE_SCOLLED_EVENT:
      std::cout << "Mouse scrolled " << static_cast<const MouseScrollEvent&>(i_event).Offset() << std::endl;
      return;
    case Event::EventType::WINDOW_CLOSED_EVENT:
      std::cout << "Window closed" << std::endl;
      return;
    case Event::EventType::WINDOW_RESIZED_EVENT: {
      auto window_size = static_cast<const WindowResizeEvent&>(i_event).Size();
      std::cout << "Window resize " << window_size.first << " " << window_size.second << std::endl;
      return;
    }
    default:
      break;
  }
}