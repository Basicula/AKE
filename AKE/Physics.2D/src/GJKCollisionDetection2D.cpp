#include "Physics.2D/GJKCollisionDetection2D.h"

#include "Common/Constants.h"
#include "Physics.2D/Collision2D.h"

#include <array>
#include <vector>

namespace {
  struct Simplex2D
  {
    std::array<Vector2d, 3> m_points;
    std::size_t m_size = 0;

    void Push(const Vector2d& i_point)
    {
      m_points = { m_points[1], m_points[2], i_point };
      if (m_size < 3)
        ++m_size;
    }
  };

  Vector2d GetSupportPoint(const GJKConvex2D& i_first, const GJKConvex2D& i_second, const Vector2d& i_direction)
  {
    return i_first.GetFurthestPoint(i_direction) - i_second.GetFurthestPoint(-i_direction);
  }

  Vector2d GetLineNormal(const Vector2d& i_frist_point,
                         const Vector2d& i_second_point,
                         const bool i_towards_origin = true)
  {
    Vector2d normal(i_second_point[1] - i_frist_point[1], i_frist_point[0] - i_second_point[0]);
    if (normal.Dot(-i_frist_point) < 0.0 && i_towards_origin)
      normal *= -1;
    return normal;
  }

  bool LineStep(const Simplex2D& i_simplex, Vector2d& io_direction)
  {
    const auto& a = i_simplex.m_points[2];
    const auto& b = i_simplex.m_points[1];

    io_direction = GetLineNormal(a, b, true);
    if (a.Dot(a - b) < 0.0)
      io_direction = -a;

    return false;
  }

  bool TriangleStep(Simplex2D& io_simplex, Vector2d& io_direction)
  {
    const auto& a = io_simplex.m_points[2];
    const auto& b = io_simplex.m_points[1];
    const auto& c = io_simplex.m_points[0];

    io_direction = GetLineNormal(a, c, true);
    if (io_direction.Dot(b) < 0) {
      std::swap(io_simplex.m_points[1], io_simplex.m_points[0]);
      return false;
    }

    io_direction = GetLineNormal(a, b, true);
    if (io_direction.Dot(c) < 0)
      return false;

    return true;
  }

  bool ProcessSimplex(Simplex2D& io_simplex, Vector2d& io_direction)
  {
    if (io_simplex.m_size < 2)
      return false;

    if (io_simplex.m_size == 2)
      return LineStep(io_simplex, io_direction);

    return TriangleStep(io_simplex, io_direction);
  }

  std::pair<bool, Simplex2D> GJK(const GJKConvex2D& i_first, const GJKConvex2D& i_second)
  {
    Simplex2D simplex;
    simplex.Push(GetSupportPoint(i_first, i_second, { 1.0, 0.0 }));

    // Direction towards origin
    Vector2d direction = -simplex.m_points.back();

    while (!ProcessSimplex(simplex, direction)) {
      const auto support = GetSupportPoint(i_first, i_second, direction);

      // Gap exists
      if (support.Dot(direction) <= 0.0)
        return { false, simplex };

      // Same furthest point means that simplex already near origin much as possible that means no collision
      if (support[0] == simplex.m_points[0][0] && support[1] == simplex.m_points[0][1])
        return { false, simplex };

      simplex.Push(support);
    }

    return { true, simplex };
  }

  struct ClosestEdgeRecord
  {
    std::size_t m_start_point_id = 0;
    Vector2d m_normal{ 0.0, 0.0 };
    double m_distance{ Common::Constants::MAX_DOUBLE };
  };

  ClosestEdgeRecord GetClosestEdge(const std::vector<Vector2d>& i_polytope)
  {
    ClosestEdgeRecord closest_edge;
    for (std::size_t point_id = 0; point_id < i_polytope.size(); ++point_id) {
      const auto& a = i_polytope[point_id];
      const auto& b = point_id < i_polytope.size() - 1 ? i_polytope[point_id + 1] : i_polytope.front();

      auto normal = GetLineNormal(a, b, false);
      normal.Normalize();
      const auto distance = normal.Dot(a);
      if (distance < closest_edge.m_distance) {
        closest_edge.m_distance = distance;
        closest_edge.m_normal = normal;
        closest_edge.m_start_point_id = point_id;
      }
    }
    return closest_edge;
  }

  Collision2D EPA(std::vector<Vector2d> i_polytope, const GJKConvex2D& i_first, const GJKConvex2D& i_second)
  {
    if (i_polytope.size() < 3)
      return {};

    while (true) {
      const auto closest_edge = GetClosestEdge(i_polytope);
      const auto support = GetSupportPoint(i_first, i_second, closest_edge.m_normal);
      const auto distance = closest_edge.m_normal.Dot(support);

      if (std::fabs(distance - closest_edge.m_distance) < 1e-6)
        return { closest_edge.m_normal, closest_edge.m_distance, true};

      i_polytope.insert(i_polytope.begin() + static_cast<long long>(closest_edge.m_start_point_id + 1), support);
    }
  }

}

namespace GJKCollisionDetection2D {
  Collision2D GetCollision(const GJKConvex2D& i_first, const GJKConvex2D& i_second)
  {
    const auto gjk = GJK(i_first, i_second);
    if (!gjk.first)
      return {};
    return EPA(std::vector<Vector2d>(gjk.second.m_points.begin(), gjk.second.m_points.end()), i_first, i_second);
  }
}
