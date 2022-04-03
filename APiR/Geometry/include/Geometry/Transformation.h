#pragma once
#include <Math/SquareMatrix.h>
#include <Math/Vector.h>

class Transformation
  {
  public:
    Transformation();
    Transformation(const Transformation& i_other);

    Vector3d GetScale() const;
    void SetScale(const Vector3d& i_scale);

    Vector3d GetTranslation() const;
    void SetTranslation(const Vector3d& i_translation);

    Matrix3x3d GetRotation() const;
    void SetRotation(const Matrix3x3d& i_rotation_matrix);
    void SetRotation(const Vector3d& i_axis, double i_degree_in_rad);

    void Inverse();
    Transformation GetInversed() const;

    // Transforms i_vector
    // if i_is_vector is true then i_vector treats as vector(direction)
    // i.e. only rotation applies to vector
    // if i_is_vector is false then i_vector treats as point
    Vector3d Transform(const Vector3d& i_vector, const bool i_is_vector = false) const;
    void Transform(Vector3d& io_vector, const bool i_is_vector = false) const;

    Vector3d InverseTransform(const Vector3d& i_vector, const bool i_is_vector = false) const;
    void InverseTransform(Vector3d& io_vector, const bool i_is_vector = false) const;

    Vector3d Rotate(const Vector3d& i_vector) const;
    void Rotate(Vector3d& io_vector) const;
    Vector3d Translate(const Vector3d& i_vector) const;
    void Translate(Vector3d& io_vector) const;
    Vector3d Scale(const Vector3d& i_vector) const;
    void Scale(Vector3d& io_vector) const;

  private:
    Vector3d m_translation;
    Vector3d m_scale;

    Matrix3x3d m_rotation;
    Matrix3x3d m_inverse_rotation;
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

inline Matrix3x3d Transformation::GetRotation() const
  {
  return m_rotation;
  }

inline void Transformation::SetRotation(const Matrix3x3d& i_rotation_matrix)
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