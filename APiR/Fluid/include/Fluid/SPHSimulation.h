#pragma once
#include "Fluid/SPHSystem.h"
#include "Physics/Simulation.h"

class SPHSimulation final : public Simulation
{
public:
  explicit SPHSimulation(std::size_t i_num_particles);

  [[nodiscard]] const SPHSystem& GetParticleSystem() const;

  [[nodiscard]] double GetEOSExponent() const;
  void SetEOSExponent(double i_eos_exponent);

  [[nodiscard]] double GetNegativePressureScale() const;
  void SetNegativePressureScale(double i_scale);

protected:
  void _PreProcessing() override;
  void _Update() override;
  void _PostProcessing() override;

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
