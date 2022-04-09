#pragma once

template <class TEventListner, class... Args>
void Window::InitEventListner(Args&&... i_args)
{
  mp_event_listner = std::make_unique<TEventListner>(std::forward<Args>(i_args)...);
}

template <class TGUIView, class... Args>
void Window::InitGUIView(Args&&... i_args)
{
  mp_gui_view = std::make_unique<TGUIView>(std::forward<Args>(i_args)...);
}

template <class TRenderer, class... Args>
void Window::InitRenderer(Args&&... i_args)
{
  mp_renderer = std::make_unique<TRenderer>(std::forward<Args>(i_args)...);
}
