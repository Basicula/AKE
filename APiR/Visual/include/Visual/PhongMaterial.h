#pragma once
#include "Math/Vector.h"

#include "Visual/IVisualMaterial.h"
#include "Visual/Color.h"

class PhongMaterial : public IVisualMaterial
  {
  public:
    PhongMaterial(
      const Color& i_color = Color(0xffaaaaaa),
      const Vector3d& i_ambient = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_diffuse = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_specular = Vector3d(1.0, 1.0, 1.0),
      double i_shinines = 1.0,
      double i_reflection = 0.0,
      double i_refraction = 0.0);

    virtual Color GetPrimitiveColor() const override;

    virtual Color CalculateColor(
      const Vector3d& i_normal,
      const Vector3d& i_view_direction,
      const Vector3d& i_light_direction) const override;

    virtual bool IsReflectable() const override;
    virtual Vector3d ReflectedDirection(
      const Vector3d& i_normal_at_point,
      const Vector3d& i_view_direction) const override;
    virtual double ReflectionInfluence() const override;

    virtual bool IsRefractable() const override;
    virtual Vector3d RefractedDirection() const override;
    virtual double RefractionInfluence() const override;

    Color GetAmbientColor() const;
    Color GetDiffuseColor() const;
    Color GetSpecularColor() const;

  private:
    Color m_color;
    double m_shinines;

    //x,y,z == r,g,b(coefs) in [0,1]
    Vector3d m_ambient;
    Vector3d m_diffuse;
    Vector3d m_specular;

    // TODO remove
    //[0,1]
    double m_reflection;
    double m_refraction;
  };

inline Color PhongMaterial::GetAmbientColor() const 
  { 
  return m_color * m_ambient; 
  };

inline Color PhongMaterial::GetDiffuseColor() const 
  { 
  return m_color * m_diffuse; 
  };

inline Color PhongMaterial::GetSpecularColor() const 
  { 
  return m_color * m_specular; 
  };

inline bool PhongMaterial::IsReflectable() const
  {
  return m_reflection > 0.0;
  }

inline double PhongMaterial::ReflectionInfluence() const
  {
  return m_reflection;
  }

inline bool PhongMaterial::IsRefractable() const
  {
  return m_refraction > 0.0;
  }

inline double PhongMaterial::RefractionInfluence() const
  {
  return m_refraction;
  }

inline Color PhongMaterial::GetPrimitiveColor() const
  {
  return GetAmbientColor();
  }