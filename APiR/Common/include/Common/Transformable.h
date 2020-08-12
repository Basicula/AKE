#pragma once

#include <Common/Intersection.h>
#include <Common/Transformation.h>
#include <Math/Vector.h>

class Transformable
  {
  public:
    Transformable() = default;
    ~Transformable() = default;

    Transformation GetTransformation() const;

    Vector3d PointToLocal(const Vector3d& i_world_point) const;
    Vector3d PointToWorld(const Vector3d& i_local_point) const;

    Vector3d DirectionToLocal(const Vector3d& i_world_dir) const;
    Vector3d DirectionToWorld(const Vector3d& i_local_dir) const;

    // Scalings
    Vector3d GetScale() const;
    void SetScale(const Vector3d& i_factors);
    void Scale(double i_factor);
    void Scale(const Vector3d& i_factors);
    Vector3d ApplyScaling(const Vector3d& i_point) const;

    // Rotations
    Matrix3d GetRotation() const;
    void SetRotation(const Vector3d& i_axis, double i_degree_in_rad);
    void Rotate(const Vector3d& i_axis, double i_degree_in_rad);
    Vector3d ApplyRotation(const Vector3d& i_point) const;

    // Translations
    Vector3d GetTranslation() const;
    void SetTranslation(const Vector3d& i_translation);
    void Translate(const Vector3d& i_translation);
    Vector3d ApplyTranslation(const Vector3d& i_point) const;

    Ray RayToLocal(const Ray& i_world_ray) const;

    BoundingBox BBoxToWorld(const BoundingBox& i_local_bbox) const;

  protected:
    virtual void _OnTransformationChange() = 0;

    void _LocalIntersectionToWorld(IntersectionRecord& io_intersection) const;

  private:
    Transformation m_transformation;
  };

inline Transformation Transformable::GetTransformation() const
  {
  return m_transformation;
  }

inline Vector3d Transformable::PointToLocal(const Vector3d& i_world_point) const
  {
  return m_transformation.PointToLocal(i_world_point);
  }

inline Vector3d Transformable::PointToWorld(const Vector3d& i_local_point) const
  {
  return m_transformation.PointToWorld(i_local_point);
  }

inline Vector3d Transformable::DirectionToLocal(const Vector3d& i_world_dir) const
  {
  return m_transformation.DirectionToLocal(i_world_dir);
  }

inline Vector3d Transformable::DirectionToWorld(const Vector3d& i_local_dir) const
  {
  return m_transformation.DirectionToWorld(i_local_dir);
  }

inline Vector3d Transformable::GetScale() const
  {
  return m_transformation.GetScale();
  }

inline void Transformable::SetScale(const Vector3d& i_factors)
  {
  m_transformation.SetScale(i_factors);
  _OnTransformationChange();
  }

inline Matrix3d Transformable::GetRotation() const
  {
  return m_transformation.GetRotation();
  }

inline void Transformable::SetRotation(const Vector3d& i_axis, double i_degree_in_rad)
  {
  m_transformation.SetRotation(i_axis, i_degree_in_rad);
  _OnTransformationChange();
  }

inline Vector3d Transformable::GetTranslation() const
  {
  return m_transformation.GetTranslation();
  }

inline void Transformable::SetTranslation(const Vector3d& i_translation)
  {
  m_transformation.SetTranslation(i_translation);
  _OnTransformationChange();
  }
