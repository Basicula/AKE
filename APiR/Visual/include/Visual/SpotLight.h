#pragma once
#include "Math/Vector.h"

#include "Visual/Color.h"
#include "Visual/ILight.h"

class SpotLight : public ILight
  {
  public:
    SpotLight(
      const Vector3d& i_location, 
      const Color& i_color = Color(255, 255, 255), 
      double i_intensity = 1.0,
      bool i_state = true);

    Vector3d GetLocation() const;
    void SetLocation(const Vector3d& i_location);

    Color GetColor() const;
    void SetColor(const Color& i_color);

    virtual Vector3d GetDirection(const Vector3d& i_point) const override;
    
    double GetIntensityAtPoint(const Vector3d& i_point) const override;

    virtual std::string Serialize() const override;

  private:
    Vector3d m_location;
    Color m_color;
  };
  
inline Vector3d SpotLight::GetLocation() const 
  { 
  return m_location; 
  };
  
inline void SpotLight::SetLocation(const Vector3d& i_location)
  { 
  m_location = i_location;
  };

inline Color SpotLight::GetColor() const 
  { 
  return m_color; 
  };
  
inline void SpotLight::SetColor(const Color& i_color)
  { 
  m_color = i_color; 
  };

inline std::string SpotLight::Serialize() const
  {
  std::string res = "{ \"SpotLight\" : { ";
  res += " \"Location\" : " + m_location.Serialize() + ", ";
  res += " \"Color\" : " + m_color.Serialize() + ", ";
  res += " \"Intensity\" : " + std::to_string(m_intensity) + ", ";
  res += " \"State\" : " + std::to_string(m_state);
  res += " } }";
  return res;
  }

inline Vector3d SpotLight::GetDirection(const Vector3d& i_point) const
  {
  return (i_point - m_location).Normalized();
  }