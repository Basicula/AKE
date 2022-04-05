#pragma once
#include "Math/SquareMatrix.h"
#include "Math/Vector.h"

template <size_t Dimension>
class Transformation
{
public:
    using VectorType = Vector<double, Dimension>;
    using MatrixType = SquareMatrix<double, Dimension>;

public:
  Transformation();
  Transformation(const Transformation& i_other);

  VectorType GetScale() const;
  void SetScale(const VectorType& i_scale);

  VectorType GetTranslation() const;
  void SetTranslation(const VectorType& i_translation);

  MatrixType GetRotation() const;
  void SetRotation(const MatrixType& i_rotation_matrix);
  void SetRotation(const VectorType& i_axis, double i_degree_in_rad);

  void Inverse();
  Transformation GetInversed() const;

  // Transforms i_vector
  // if i_is_vector is true then i_vector treats as vector(direction)
  // i.e. only rotation applies to vector
  // if i_is_vector is false then i_vector treats as point
  VectorType Transform(const VectorType& i_vector, bool i_is_vector = false) const;
  void Transform(VectorType& io_vector, bool i_is_vector = false) const;

  VectorType InverseTransform(const VectorType& i_vector, bool i_is_vector = false) const;
  void InverseTransform(VectorType& io_vector, bool i_is_vector = false) const;

  VectorType Rotate(const VectorType& i_vector) const;
  void Rotate(VectorType& io_vector) const;
  VectorType Translate(const VectorType& i_vector) const;
  void Translate(VectorType& io_vector) const;
  VectorType Scale(const VectorType& i_vector) const;
  void Scale(VectorType& io_vector) const;

private:
  VectorType m_translation;
  VectorType m_scale;

  MatrixType m_rotation;
  MatrixType m_inverse_rotation;
};

using Transformation3D = Transformation<3>;

#include "impl/TransformationImpl.h"
#include "impl/Transformation2DImpl.h"
#include "impl/Transformation3DImpl.h"