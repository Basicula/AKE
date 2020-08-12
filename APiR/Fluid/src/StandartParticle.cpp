#include <Fluid/StandartParticle.h>

StandartParticle::StandartParticle(
  const Vector3d& i_position,
  const Vector3d& i_velocity)
  : Particle()
  {
  m_position_id = AddVectorData(i_position);
  m_velocity_id = AddVectorData(i_velocity);
  m_force_id = AddVectorData();
  m_acceleration_id = AddVectorData();

  m_density_id = AddScalarData();
  m_pressure_id = AddScalarData();
  }