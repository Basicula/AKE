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

template <class TWindowBackend, class... Args>
void Window::InitWindowBackend(Args&&... i_args)
{
  mp_window_backend = std::make_unique<TWindowBackend>(std::forward<Args>(i_args)...);
}
