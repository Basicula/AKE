#include <Geometry/BoundingBox.h>
#include <Common/Constants.h>

BoundingBox::BoundingBox()
  : m_min(MAX_INT)
  , m_max(MIN_INT)
  {}

BoundingBox::BoundingBox(
  const Vector3d& i_min,
  const Vector3d& i_max)
  : m_min(i_min)
  , m_max(i_max)
  {}

void BoundingBox::AddPoint(const Vector3d& i_point)
  {
  for (int i = 0; i < 3; ++i)
    {
    m_min[i] = std::min(m_min[i], i_point[i]);
    m_max[i] = std::max(m_max[i], i_point[i]);
    }
  }

void BoundingBox::Merge(const BoundingBox& i_other)
  {
  for (int i = 0; i < 3; ++i)
    {
    m_min[i] = std::min(m_min[i], i_other.m_min[i]);
    m_max[i] = std::max(m_max[i], i_other.m_max[i]);
    }
  }

void BoundingBox::Reset()
  {
  m_min = Vector3d(MAX_INT);
  m_max = Vector3d(MIN_INT);
  }

Vector3d BoundingBox::GetCorner(std::size_t i_corner_id) const
  {
  return Vector3d(
    ((i_corner_id >> 2) & 1) == 1 ? m_max[0] : m_min[0],
    ((i_corner_id >> 1) & 1) == 1 ? m_max[1] : m_min[1],
    ((i_corner_id >> 0) & 1) == 1 ? m_max[2] : m_min[2]);
  }