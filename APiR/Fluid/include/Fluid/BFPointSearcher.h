#pragma once
#include "Fluid/PointNeighborSearcher.h"

class BFPointSearcher : public PointNeighborSearcher
{
public:
  BFPointSearcher(const Points& i_points);
  BFPointSearcher(const PointsIteratorC& i_begin, std::size_t i_size);

  virtual bool HasNeighborPoint(const Vector3d& i_point, double i_search_radius) override;

  virtual void ForEachNearbyPoint(const Vector3d& i_point,
                                  double i_search_radius,
                                  const ForEachNearbyPointFunc& i_callback) override;

protected:
  virtual void _Build(const Points& i_points) override;

private:
  Points m_points;
};