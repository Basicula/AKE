#pragma once
#include "Math/Vector.h"

template <size_t Dimension>
struct BoundingBox
{
  using VectorType = Vector<double, Dimension>;

  VectorType m_min;
  VectorType m_max;

  template <class Iterator>
  static BoundingBox Make(Iterator i_begin, Iterator i_end);

  BoundingBox();
  BoundingBox(const VectorType& i_min, const VectorType& i_max);

  [[nodiscard]] VectorType Center() const;
  // iterate from min to max using bitmask depending on Dimension
  // i.e. (0,0,0) -> (1,1,1) for 3D or (0, 0) -> (1, 1) for 2D
  [[nodiscard]] VectorType GetCorner(std::size_t i_corner_id) const;

  [[nodiscard]] VectorType Delta() const;

  void Merge(const BoundingBox& i_other);
  void Reset();

  void AddPoint(const VectorType& i_point);
  [[nodiscard]] bool Contains(const VectorType& i_point) const;

  [[nodiscard]] bool IsValid() const;
};

using BoundingBox2D = BoundingBox<2>;
using BoundingBox3D = BoundingBox<3>;

#include "impl/BoundingBoxImpl.h"