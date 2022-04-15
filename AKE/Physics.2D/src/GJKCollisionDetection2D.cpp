#include "Physics.2D/GJKCollisionDetection2D.h"

#include <array>

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

  Vector2d GetLineNormalToOrigin(const Vector2d& a, const Vector2d& b)
  {
    Vector2d normal(b[1] - a[1], a[0] - b[0]);
    if (normal.Dot(-a) < 0.0)
      normal *= -1;
    return normal;
  }

  bool LineStep(const Simplex2D& i_simplex, Vector2d& io_direction)
  {
    const auto& a = i_simplex.m_points[2];
    const auto& b = i_simplex.m_points[1];

    io_direction = GetLineNormalToOrigin(a, b);
    if (a.Dot(a - b) < 0.0)
      io_direction = -a;

    return false;
  }

  bool TriangleStep(Simplex2D& io_simplex, Vector2d& io_direction)
  {
    const auto& a = io_simplex.m_points[2];
    const auto& b = io_simplex.m_points[1];
    const auto& c = io_simplex.m_points[0];

    io_direction = GetLineNormalToOrigin(a, c);
    if (io_direction.Dot(b) < 0) {
      std::swap(io_simplex.m_points[1], io_simplex.m_points[0]);
      return false;
    }

    io_direction = GetLineNormalToOrigin(a, b);
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
}

namespace GJKCollisionDetection2D {
  bool GetCollision(const GJKConvex2D& i_first, const GJKConvex2D& i_second)
  {
    Simplex2D simplex;
    simplex.Push(GetSupportPoint(i_first, i_second, { 1.0, 0.0 }));

    // Direction towards origin
    Vector2d direction = -simplex.m_points.back();

    while (!ProcessSimplex(simplex, direction)) {
      const auto support = GetSupportPoint(i_first, i_second, direction);

      // Gap exists
      if (support.Dot(direction) <= 0.0)
        return false;

      // Same furthest point means that simplex already near origin much as possible that means no collision
      if (support[0] == simplex.m_points[0][0] && support[1] == simplex.m_points[0][1])
        return false;

      simplex.Push(support);
    }

    return true;
  }
}
