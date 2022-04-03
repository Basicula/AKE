#pragma once
#include "Fluid/SPHSystem.h"
#include "Physics/Simulation.h"

class SPHSimulation : public Simulation
{
public:
  SPHSimulation(std::size_t i_num_particles);

  const SPHSystem& GetParticleSystem() const;

  double GetEOSExponent() const;
  void SetEOSExponent(double i_eos_exponent);

  double GetNegativePressureScale() const;
  void SetNegativePressureScale(double i_scale);

protected:
  virtual void _PreProcessing() override;
  virtual void _Update() override;
  virtual void _PostProcessing() override;

private:
  void _InitParticles();

  void _AccumulateExternalForces();

  void _AccumulateViscosityForce();

  void _ComputePressure();
  void _AccumulatePressureForeces();

  void _TimeIntegration();

  void _UpdatePositionsAndVelocities();

  void _ResolveCollisions();

private:
  SPHSystem m_particle_system;

  SPHSystem::VectorData m_new_positions;
  SPHSystem::VectorData m_new_velocities;

  double m_eos_exponent;
  double m_negative_pressure_scale;
};

inline const SPHSystem& SPHSimulation::GetParticleSystem() const
{
  return m_particle_system;
}

inline double SPHSimulation::GetEOSExponent() const
{
  return m_eos_exponent;
}

inline void SPHSimulation::SetEOSExponent(double i_eos_exponent)
{
  m_eos_exponent = i_eos_exponent;
}

inline double SPHSimulation::GetNegativePressureScale() const
{
  return m_negative_pressure_scale;
}

inline void SPHSimulation::SetNegativePressureScale(double i_scale)
{
  m_negative_pressure_scale = i_scale;
}
