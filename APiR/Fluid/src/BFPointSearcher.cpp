#include "Fluid/BFPointSearcher.h"

BFPointSearcher::BFPointSearcher(Points i_points)
  : m_points(std::move(i_points))
{}

BFPointSearcher::BFPointSearcher(const PointsIteratorC& i_begin, const std::size_t i_size)
  : m_points(i_begin, i_begin + i_size)
{}

void BFPointSearcher::_Build(const Points&) {}

bool BFPointSearcher::HasNeighborPoint(const Vector3d& i_point, const double i_search_radius)
{
  const double square_search_radius = i_search_radius * i_search_radius;
  return std::any_of(m_points.begin(), m_points.end(), [&i_point, &square_search_radius](const Vector3d& point) {
    return i_point.SquareDistance(point) <= square_search_radius;
  });
}

void BFPointSearcher::ForEachNearbyPoint(const Vector3d& i_point,
                                         const double i_search_radius,
                                         const ForEachNearbyPointFunc& i_callback)
{
  const double square_search_radius = i_search_radius * i_search_radius;
  for (auto i = 0u; i < m_points.size(); ++i) {
    if (i_point.SquareDistance(m_points[i]) <= square_search_radius)
      i_callback(i, m_points[i]);
  }
}