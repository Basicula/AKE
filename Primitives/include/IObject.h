#pragma once
#include <vector>

#include <Vector.h>
#include <Ray.h>
#include <ColorMaterial.h>

class IObject
  {
  public:
    enum class ObjectType
      {
      SPHERE,
      PLANE,
      UNDEFINED
      };

  protected:
    ObjectType m_type;
    IObject(const ColorMaterial& i_material, ObjectType i_type = ObjectType::UNDEFINED) : m_type(i_type), m_material(i_material) {};
    IObject(const Color& i_color = Color(255,255,255), ObjectType i_type = ObjectType::UNDEFINED) : m_type(i_type), m_material(i_color) {};

  public:
    inline ObjectType GetType() const { return m_type; };
    virtual bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const = 0;

    inline Color GetColor() const { return m_material.GetBaseColor(); };
    inline ColorMaterial GetMaterial() const { return m_material; };
    Color GetColorInPoint(const Vector3d& i_point, const std::vector<SpotLight*>& i_lights, const Vector3d& i_to_viewer) const;

  private:
    ColorMaterial m_material;
  };

inline Color IObject::GetColorInPoint(const Vector3d& i_point, const std::vector<SpotLight*>& i_lights, const Vector3d& i_to_viewer) const
  {
  Vector3d normal;
  GetNormalInPoint(normal, i_point);
  return m_material.GetResultColor(normal,i_point,i_lights,i_to_viewer);
  }