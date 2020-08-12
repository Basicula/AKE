#pragma once

#include <Math/Matrix3.h>
#include <Math/Vector.h>
#include <Common/Ray.h>
#include <Common/BoundingBox.h>

class Transformation
  {
  public:
    Transformation();
    Transformation(const Transformation& i_other);

    Vector3d GetScale() const;
    void SetScale(const Vector3d& i_scale);

    Vector3d GetTranslation() const;
    void SetTranslation(const Vector3d& i_translation);

    Matrix3d GetRotation() const;
    void SetRotation(const Matrix3d& i_rotation_matrix);
    void SetRotation(const Vector3d& i_axis, double i_degree_in_rad);

    void Inverse();
    Transformation GetInversed() const;

    Vector3d PointToLocal(const Vector3d& i_world_point) const;
    Vector3d PointToWorld(const Vector3d& i_local_point) const;

    Vector3d DirectionToLocal(const Vector3d& i_world_dir) const;
    Vector3d DirectionToWorld(const Vector3d& i_local_dir) const;

  private:    
    Vector3d m_translation;
    Vector3d m_scale;

    Matrix3d m_rotation;
    Matrix3d m_inverse_rotation;
  };

inline Vector3d Transformation::GetScale() const
  {
  return m_scale;
  }

inline void Transformation::SetScale(const Vector3d& i_scale)
  {
  m_scale = i_scale;
  }

inline Vector3d Transformation::GetTranslation() const
  {
  return m_translation;
  }

inline void Transformation::SetTranslation(const Vector3d& i_translation)
  {
  m_translation = i_translation;
  }

inline Matrix3d Transformation::GetRotation() const
  {
  return m_rotation;
  }

inline void Transformation::SetRotation(const Matrix3d& i_rotation_matrix)
  {
  m_rotation = i_rotation_matrix;
  m_inverse_rotation = i_rotation_matrix.Transposed();
  }

inline void Transformation::Inverse()
  {
  m_translation = -m_translation;
  m_scale = Vector3d(1.0 / m_scale[0], 1.0 / m_scale[1], 1.0 / m_scale[2]);
  std::swap(m_rotation, m_inverse_rotation);
  }