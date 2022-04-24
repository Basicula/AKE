#include "Physics.2D/CommonPhysicalProperties.h"

CommonPhysicalProperties::CommonPhysicalProperties(const double i_mass,
                                                   const Vector2d& i_velocity,
                                                   const Vector2d& i_acceleration)
  : m_mass(i_mass)
  , m_velocity(i_velocity)
  , m_acceleration(i_acceleration)
{}

void CommonPhysicalProperties::Apply(const double i_time_delta)
{
  m_velocity += m_acceleration * i_time_delta;
  if (mp_transformation_source)
    mp_transformation_source->Translate(m_velocity * i_time_delta);
}