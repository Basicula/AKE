#include "Visual/ILight.h"

ILight::ILight(const bool i_state, const double i_intensity)
  : m_state(i_state)
  , m_intensity(i_intensity)
{}

double ILight::GetIntensity() const
{
  return m_intensity;
};

void ILight::SetIntensity(const double i_intensity)
{
  m_intensity = i_intensity;
};

void ILight::SetState(const bool i_state)
{
  m_state = i_state;
}

bool ILight::GetState() const
{
  return m_state;
}