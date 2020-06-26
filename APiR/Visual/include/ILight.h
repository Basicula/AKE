#pragma once
#include <string>

#include <Vector.h>

class ILight
  {
  public:
    ILight(bool i_state, double i_intensity);
    virtual ~ILight() = default;

    virtual void SetState(bool i_state);
    virtual bool GetState() const;

    virtual void SetIntensity(double i_intensity);
    virtual double GetIntensity() const;
    virtual double GetIntensityAtPoint(const Vector3d& i_point) const = 0;

    virtual Vector3d GetDirection(const Vector3d& i_point) const = 0;

    virtual std::string Serialize() const = 0;

  protected:
    bool m_state;
    double m_intensity;
  };

inline ILight::ILight(bool i_state, double i_intensity)
  : m_state(i_state)
  , m_intensity(i_intensity)
  {}

inline double ILight::GetIntensity() const
  {
  return m_intensity;
  };

inline void ILight::SetIntensity(double i_intensity)
  {
  m_intensity = i_intensity;
  };

inline void ILight::SetState(bool i_state)
  {
  m_state = i_state;
  }

inline bool ILight::GetState() const
  {
  return m_state;
  }