#pragma once

#include <Color.h>
#include <Vector.h>
#include <SpotLight.h>

class ColorMaterial
  {
  public:
    ColorMaterial() = delete;
    ColorMaterial(
      const Color& i_color,
      const Vector3d& i_ambient = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_diffuse = Vector3d(1.0, 1.0, 1.0),
      const Vector3d& i_specular = Vector3d(1.0, 1.0, 1.0),
      double i_shinines = 1.0,
      double i_reflection = 0.0,
      double i_refraction = 0.0);

    Color GetResultColor(const Vector3d& i_normal, const Vector3d& i_light, const Vector3d& i_view) const;
    Color GetResultColor(const Vector3d& i_normal, const Vector3d& i_point, const std::vector<SpotLight*>& i_light, const Vector3d& i_view) const;

    inline Color GetBaseColor() const { return m_color; };
    inline Color GetAmbientColor() const { return m_color * m_ambient; };
    inline Vector3d GetAmbient() const { return m_ambient; };
    inline void SetAmbient(const Vector3d& i_ambient) { m_ambient = i_ambient; };
    inline Color GetDiffuseColor() const { return m_color * m_diffuse; };
    inline Vector3d GetDiffuse() const { return m_diffuse; };
    inline void SetDiffuse(const Vector3d& i_diffuse) { m_diffuse = i_diffuse; };
    inline Color GetSpecularColor() const { return m_color * m_specular; };
    inline Vector3d GetSpecular() const { return m_specular; };
    inline void SetSpecular(const Vector3d& i_specular) { m_specular = i_specular; };
    inline double GetReflection() const { return m_reflection; };
    inline double GetRefraction() const { return m_refraction; };


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

const ColorMaterial defaultMaterial(Color(255,255,255));