#include "Window/FPSCounter.h"

FPSCounter::FPSCounter(std::ostream& io_output_stream, const std::size_t i_update_interval_in_frames)
  : m_start(std::chrono::system_clock::now())
  , m_frames_cnt(0)
  , m_update_interval(i_update_interval_in_frames)
  , m_fps(0.0)
  , mp_output_stream(&io_output_stream)
{}

FPSCounter::FPSCounter(const std::size_t i_update_interval_in_frames, std::function<void(double)> i_fps_logging_function)
  : m_start(std::chrono::system_clock::now())
  , m_frames_cnt(0)
  , m_update_interval(i_update_interval_in_frames)
  , m_fps(0.0)
  , m_logging_function(std::move(i_fps_logging_function))
  , mp_output_stream(nullptr)
{}

void FPSCounter::Update()
{
  ++m_frames_cnt;
  if (m_frames_cnt > m_update_interval) {
    const auto now = std::chrono::system_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count();
    m_fps = 1000.0 * static_cast<double>(m_frames_cnt) / static_cast<double>(elapsed);
    if (m_logging_function)
      m_logging_function(m_fps);
    else if (mp_output_stream)
      *mp_output_stream << "FPS : " << m_fps << std::endl;
    m_frames_cnt = 0;
    m_start = now;
  }
}

double FPSCounter::GetFPS() const
{
  return m_fps;
}
