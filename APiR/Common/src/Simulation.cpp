#include <Common/DefinesAndConstants.h>
#include <Common/Simulation.h>

Simulation::Simulation()
  : Simulation(DEFAULT_TIME_STEP)
  {}

Simulation::Simulation(double i_time_step)
  : m_time_step(i_time_step)
  {}

void Simulation::Update()
  {
  _PreProcessing();
  _Update();
  _PostProcessing();
  }