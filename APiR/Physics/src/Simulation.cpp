#include "Physics/Simulation.h"

#include "Physics/Constants.h"

Simulation::Simulation()
  : Simulation(Physics::Constants::DEFAULT_TIME_STEP)
{}

Simulation::Simulation(const double i_time_step)
  : m_time_step(i_time_step)
{}

void Simulation::Update()
{
  _PreProcessing();
  _Update();
  _PostProcessing();
}

void Simulation::SetTimeStep(const double i_new_time_step)
{
  m_time_step = i_new_time_step;
}

double Simulation::GetTimeStep() const
{
  return m_time_step;
}