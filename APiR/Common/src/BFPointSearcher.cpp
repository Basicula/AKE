#include <BFPointSearcher.h>

BFPointSearcher::BFPointSearcher(const Points& i_points)
  : m_points(i_points)
  {}

BFPointSearcher::BFPointSearcher(
  const PointsIteratorC& i_points_begin,
  std::size_t i_size)
  : m_points(i_points_begin, i_points_begin + i_size)
  {}

void BFPointSearcher::_Build(const Points&)
  {}

bool BFPointSearcher::HasNeighborPoint(
  const Vector3d& i_point,
  double i_search_radius)
  {
  const double square_search_radius = i_search_radius * i_search_radius;
  for (const auto& point : m_points)
    {
    if (i_point.SquareDistance(point) <= square_search_radius)
      return true;
    }
  return false;
  }

void BFPointSearcher::ForEachNearbyPoint(
  const Vector3d& i_point,
  double i_search_radius,
  const ForEachNearbyPointFunc& i_callback)
  {
  const double square_search_radius = i_search_radius * i_search_radius;
  for (auto i = 0u; i < m_points.size(); ++i)
    {
    if (i_point.SquareDistance(m_points[i]) <= square_search_radius)
      i_callback(i,m_points[i]);
    }
  }