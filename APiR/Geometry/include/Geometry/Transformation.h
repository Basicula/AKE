#pragma once
#include "Math/SquareMatrix.h"
#include "Math/Vector.h"

template <std::size_t Dimension>
class Transformation
{
public:
  using VectorType = Vector<double, Dimension>;
  using MatrixType = SquareMatrix<double, Dimension>;

public:
  Transformation();
  Transformation(const Transformation& i_other);

  [[nodiscard]] VectorType GetScale() const;
  void SetScale(const VectorType& i_scale);

  [[nodiscard]] VectorType GetTranslation() const;
  void SetTranslation(const VectorType& i_translation);

  [[nodiscard]] MatrixType GetRotation() const;
  void SetRotation(const MatrixType& i_rotation_matrix);
  template <std::size_t D = Dimension, std::enable_if_t<D == 3, bool> = true>
  void SetRotation(const VectorType& i_axis, double i_degree_in_rad);
  template <std::size_t D = Dimension, std::enable_if_t<D == 2, bool> = true>
  void SetRotation(double i_degree_in_rad);

  void Inverse();
  [[nodiscard]] Transformation GetInversed() const;

  // Transforms i_vector
  // if i_is_vector is true then i_vector treats as vector(direction)
  // i.e. only rotation applies to vector
  // if i_is_vector is false then i_vector treats as point
  [[nodiscard]] VectorType Transform(const VectorType& i_vector, bool i_is_vector = false) const;
  void Transform(VectorType& io_vector, bool i_is_vector = false) const;

  [[nodiscard]] VectorType InverseTransform(const VectorType& i_vector, bool i_is_vector = false) const;
  void InverseTransform(VectorType& io_vector, bool i_is_vector = false) const;

  [[nodiscard]] VectorType Rotate(const VectorType& i_vector) const;
  void Rotate(VectorType& io_vector) const;
  [[nodiscard]] VectorType Translate(const VectorType& i_vector) const;
  void Translate(VectorType& io_vector) const;
  [[nodiscard]] VectorType Scale(const VectorType& i_vector) const;
  void Scale(VectorType& io_vector) const;

private:
  VectorType m_translation;
  VectorType m_scale;

  MatrixType m_rotation;
  MatrixType m_inverse_rotation;
};

using Transformation2D = Transformation<2>;
using Transformation3D = Transformation<3>;

#include "impl/Transformation2DImpl.h"
#include "impl/Transformation3DImpl.h"
#include "impl/TransformationImpl.h"