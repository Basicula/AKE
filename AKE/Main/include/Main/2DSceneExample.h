#pragma once
#include <cstddef>

namespace Scene2DExamples {
  void Circles(std::size_t i_window_width, std::size_t i_window_height, std::size_t i_circles_count);
  void Rectangles(std::size_t i_window_width, std::size_t i_window_height, std::size_t i_rectangles_count);
  void RotatedRectangles(std::size_t i_window_width, std::size_t i_window_height, std::size_t i_rectangles_count);
  void RotatedTriangles(std::size_t i_window_width, std::size_t i_window_height, std::size_t i_triangles_count);
  void CollisionDetectionExample(std::size_t i_window_width, std::size_t i_window_height, std::size_t i_objects_count);
}