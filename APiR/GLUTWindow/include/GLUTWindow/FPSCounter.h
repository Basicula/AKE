#pragma once
#include <ostream>
#include <chrono>
#include <functional>

class FPSCounter
  {
  public:
    FPSCounter(
      std::ostream& io_output_stream, 
      std::size_t i_update_interval_in_frames = 10);
    FPSCounter(
      std::size_t i_update_interval_in_frames = 10,
      std::function<void(double)> i_fps_logging_function = [](double) {});

    void Update();
    double GetFPS() const;

  private:
    std::chrono::system_clock::time_point m_start;
    std::size_t m_frames_cnt;
    std::size_t m_update_interval;
    double m_fps;
    std::function<void(double)> m_logging_function;
    std::ostream* mp_output_stream;
  };

inline double FPSCounter::GetFPS() const
  {
  return m_fps;
  }