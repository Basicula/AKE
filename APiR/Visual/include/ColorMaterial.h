#pragma once

#include <IMaterial.h>
#include <Color.h>
#include <Vector.h>

class ColorMaterial : public IMaterial
  {
  public:
    ColorMaterial(
      const Color& i_color = Color(0xaaaaaa),
      const Vector3d& i_ambient = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_diffuse = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_specular = Vector3d(1.0, 1.0, 1.0),
      double i_shinines = 1.0,
      double i_reflection = 0.0,
      double i_refraction = 0.0);

    Color GetResultColor(
      const Vector3d& i_normal, 
      const Vector3d& i_light, 
      const Vector3d& i_view) const;

    virtual Color GetPrimitiveColor() const override;

    virtual Color GetLightInfluence(
      const Vector3d& i_point,
      const Vector3d& i_normal, 
      std::shared_ptr<ILight> i_light) const override;

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
    
    Color GetColor() const;
    void SetColor(const Color& i_color);
    
    Vector3d GetAmbient() const;
    void SetAmbient(const Vector3d& i_ambient);
    
    Vector3d GetDiffuse() const;
    void SetDiffuse(const Vector3d& i_diffuse);
    
    Vector3d GetSpecular() const;
    void SetSpecular(const Vector3d& i_specular);
    
    double GetReflection() const;
    void SetReflection(double i_reflection);
    
    double GetRefraction() const;
    void SetRefraction(double i_refraction);
    
    double GetShinines() const;
    void SetShinines(double i_shinines);

    virtual std::string Serialize() const override;

  private:
    Color m_color;
    double m_shinines;

    //x,y,z == r,g,b(coefs) in [0,1]
    Vector3d m_ambient;
    Vector3d m_diffuse;
    Vector3d m_specular;
    
    //[0,1]
    double m_reflection;
    double m_refraction;
  };

inline Color ColorMaterial::GetColor() const 
  { 
  return m_color; 
  };
  
inline void ColorMaterial::SetColor(const Color& i_color)
  { 
  m_color = i_color;
  };
  
inline Color ColorMaterial::GetAmbientColor() const 
  { 
  return m_color * m_ambient; 
  };
  
inline Vector3d ColorMaterial::GetAmbient() const 
  { 
  return m_ambient; 
  };
  
inline void ColorMaterial::SetAmbient(const Vector3d& i_ambient) 
  { 
  m_ambient = i_ambient; 
  };
  
inline Color ColorMaterial::GetDiffuseColor() const 
  { 
  return m_color * m_diffuse; 
  };
    
inline Vector3d ColorMaterial::GetDiffuse() const 
  { 
  return m_diffuse; 
  };
  
inline void ColorMaterial::SetDiffuse(const Vector3d& i_diffuse) 
  { 
  m_diffuse = i_diffuse; 
  };
  
inline Color ColorMaterial::GetSpecularColor() const 
  { 
  return m_color * m_specular; 
  };
  
inline Vector3d ColorMaterial::GetSpecular() const 
  { 
  return m_specular; 
  };
  
inline void ColorMaterial::SetSpecular(const Vector3d& i_specular) 
  { 
  m_specular = i_specular; 
  };
  
inline double ColorMaterial::GetReflection() const 
  { 
  return m_reflection; 
  };
  
inline void ColorMaterial::SetReflection(double i_reflection)
  { 
  m_reflection = i_reflection; 
  };

inline double ColorMaterial::GetRefraction() const 
  { 
  return m_refraction; 
  };
  
inline void ColorMaterial::SetRefraction(double i_refraction)
  { 
  m_refraction = i_refraction; 
  };
  
inline double ColorMaterial::GetShinines() const
  {
  return m_shinines;
  }
inline void ColorMaterial::SetShinines(double i_shinines)
  {
  m_shinines = i_shinines;
  }

inline bool ColorMaterial::IsReflectable() const
  {
  return m_reflection > 0.0;
  }

inline double ColorMaterial::ReflectionInfluence() const
  {
  return m_reflection;
  }

inline bool ColorMaterial::IsRefractable() const
  {
  return m_refraction > 0.0;
  }

inline double ColorMaterial::RefractionInfluence() const
  {
  return m_refraction;
  }

inline std::string ColorMaterial::Serialize() const
  {
  std::string res = "{ \"ColorMaterial\" : { ";
  res += "\"Color\" : " + m_color.Serialize() + ", ";
  res += "\"Ambient\" : " + m_ambient.Serialize() + ", ";
  res += "\"Diffuse\" : " + m_diffuse.Serialize() + ", ";
  res += "\"Specular\" : " + m_specular.Serialize() + ", ";
  res += "\"Shinines\" : " + std::to_string(m_shinines) + ", ";
  res += "\"Reflection\" : " + std::to_string(m_reflection) + ", ";
  res += "\"Refraction\" : " + std::to_string(m_refraction);
  res += "} }";
  return res;
  }

inline Color ColorMaterial::GetPrimitiveColor() const
  {
  return GetAmbientColor();
  }