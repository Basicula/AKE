#pragma once

#include <Vector.h>
#include <Ray.h>

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
    IObject(ObjectType i_type = ObjectType::UNDEFINED) : m_type(i_type) {};

  public:
    virtual bool GetNormalInPoint(Vector3d& o_normal, const Vector3d& i_point) const = 0;
    inline virtual ObjectType GetType() const { return m_type; };

  };