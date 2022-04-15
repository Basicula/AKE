#pragma once
#include <Common/Constants.h>

template <size_t Dimension>
template <class Iterator>
BoundingBox<Dimension> BoundingBox<Dimension>::Make(Iterator i_begin, Iterator i_end)
{
  BoundingBox<Dimension> res;
  while (i_begin != i_end)
    res.AddPoint(*i_begin++);
  return res;
}

template <size_t Dimension>
BoundingBox<Dimension>::BoundingBox()
  : m_min(Common::Constants::MAX_INT)
  , m_max(Common::Constants::MIN_INT)
{}

template <size_t Dimension>
BoundingBox<Dimension>::BoundingBox(const VectorType& i_min, const VectorType& i_max)
  : m_min(i_min)
  , m_max(i_max)
{}

template <size_t Dimension>
void BoundingBox<Dimension>::AddPoint(const VectorType& i_point)
{
  for (int i = 0; i < Dimension; ++i) {
    m_min[i] = std::min(m_min[i], i_point[i]);
    m_max[i] = std::max(m_max[i], i_point[i]);
  }
}

template <size_t Dimension>
void BoundingBox<Dimension>::Merge(const BoundingBox<Dimension>& i_other)
{
  for (int i = 0; i < Dimension; ++i) {
    m_min[i] = std::min(m_min[i], i_other.m_min[i]);
    m_max[i] = std::max(m_max[i], i_other.m_max[i]);
  }
}

template <size_t Dimension>
void BoundingBox<Dimension>::Reset()
{
  m_min = BoundingBox<Dimension>::VectorType(Common::Constants::MAX_INT);
  m_max = BoundingBox<Dimension>::VectorType(Common::Constants::MIN_INT);
}

template <size_t Dimension>
typename BoundingBox<Dimension>::VectorType BoundingBox<Dimension>::GetCorner(const std::size_t i_corner_id) const
{
  BoundingBox<Dimension>::VectorType res;
  for (size_t i = 0; i < Dimension; ++i)
    res[i] = ((i_corner_id >> (Dimension - 1 - i)) & 1) == 1 ? m_max[i] : m_min[i];
  return res;
}

template <size_t Dimension>
typename BoundingBox<Dimension>::VectorType BoundingBox<Dimension>::Center() const
{
  return (m_min + m_max) / 2;
}

template <size_t Dimension>
bool BoundingBox<Dimension>::IsValid() const
{
  return (m_min < m_max);
}

template <size_t Dimension>
bool BoundingBox<Dimension>::Contains(const VectorType& i_point) const
{
  return (m_min <= i_point && i_point <= m_max);
}

template <size_t Dimension>
typename BoundingBox<Dimension>::VectorType BoundingBox<Dimension>::Delta() const
{
  return m_max - m_min;
}