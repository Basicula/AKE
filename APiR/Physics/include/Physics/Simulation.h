#pragma once
#include <memory>

class Simulation
{
public:
  Simulation();
  Simulation(double i_time_step);
  ~Simulation() = default;

  // get/set time step in seconds
  void SetTimeStep(double i_new_time_step);
  double GetTimeStep() const;

  void Update();

protected:
  virtual void _PreProcessing() = 0;
  virtual void _Update() = 0;
  virtual void _PostProcessing() = 0;

private:
  double m_time_step;
};

inline void Simulation::SetTimeStep(double i_new_time_step)
{
  m_time_step = i_new_time_step;
}

inline double Simulation::GetTimeStep() const
{
  return m_time_step;
}