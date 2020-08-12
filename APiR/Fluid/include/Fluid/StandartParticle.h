#pragma once

#include <Fluid/Particle.h>

class StandartParticle : public Particle
  {
  public:
    StandartParticle(
      const Vector3d& i_position,
      const Vector3d& i_velocity = Vector3d(0));

    double GetDensity() const;
    void SetDensity(double i_density);

    Vector3d GetPosition() const;
    void SetPosition(const Vector3d& i_position);

    double GetPressure() const;
    void SetPressure(double i_pressure);

    Vector3d GetVelocity() const;
    void SetVelocity(const Vector3d& i_velocity);

    Vector3d GetAcceleration() const;
    void SetAcceleration(const Vector3d& i_acceleration);

  private:
    std::size_t m_position_id;
    std::size_t m_velocity_id;
    std::size_t m_force_id;
    std::size_t m_acceleration_id;

    std::size_t m_density_id;
    std::size_t m_pressure_id;
  };

inline double StandartParticle::GetDensity() const
  {
  return m_scalar_data[m_density_id];
  }

inline void StandartParticle::SetDensity(double i_density)
  {
  m_scalar_data[m_density_id] = i_density;
  }

inline Vector3d StandartParticle::GetPosition() const
  {
  return m_vector_data[m_position_id];
  }

inline void StandartParticle::SetPosition(const Vector3d& i_position)
  {
  m_vector_data[m_position_id] = i_position;
  }

inline double StandartParticle::GetPressure() const
  {
  return m_scalar_data[m_pressure_id];
  }

inline void StandartParticle::SetPressure(double i_pressure)
  {
  m_scalar_data[m_pressure_id] = i_pressure;
  }

inline Vector3d StandartParticle::GetVelocity() const
  {
  return m_vector_data[m_velocity_id];
  }

inline void StandartParticle::SetVelocity(const Vector3d& i_velocity)
  {
  m_vector_data[m_velocity_id] = i_velocity;
  }

inline Vector3d StandartParticle::GetAcceleration() const
  {
  return m_vector_data[m_acceleration_id];
  }

inline void StandartParticle::SetAcceleration(const Vector3d& i_acceleration)
  {
  m_vector_data[m_acceleration_id] = i_acceleration;
  }