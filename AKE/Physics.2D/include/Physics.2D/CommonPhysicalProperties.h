#pragma once
#include "Math/Vector.h"
#include "Physics.2D/PhysicalProperty2D.h"

struct CommonPhysicalProperties : public PhysicalProperty2D
{
  double m_mass;
  Vector2d m_velocity;
  Vector2d m_acceleration;

  CommonPhysicalProperties(double i_mass, const Vector2d& i_velocity, const Vector2d& i_acceleration);

  void Apply(double i_time_delta) override;
};
