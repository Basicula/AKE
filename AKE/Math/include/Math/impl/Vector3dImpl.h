#pragma once
#include <cmath>

template <>
inline Vector3d::Vector(const double i_elem)
  : m_coords{ i_elem, i_elem, i_elem }
{}

template <>
template <>
inline Vector3d::Vector(const double i_x, const double i_y, const double i_z)
  : m_coords{ i_x, i_y, i_z }
{}

template <>
inline Vector3d Vector3d::operator-(const Vector3d& i_other) const
{
  return { m_coords[0] - i_other.m_coords[0], m_coords[1] - i_other.m_coords[1], m_coords[2] - i_other.m_coords[2] };
}

template <>
inline Vector3d Vector3d::operator+(const Vector3d& i_other) const
{
  return { m_coords[0] + i_other.m_coords[0], m_coords[1] + i_other.m_coords[1], m_coords[2] + i_other.m_coords[2] };
}

template <>
inline double Vector3d::Dot(const Vector3d& i_other) const
{
  return m_coords[0] * i_other.m_coords[0] + m_coords[1] * i_other.m_coords[1] + m_coords[2] * i_other.m_coords[2];
}

template <>
template <>
inline Vector3d Vector3d::operator*<double>(const double i_factor) const
{
  return { m_coords[0] * i_factor, m_coords[1] * i_factor, m_coords[2] * i_factor };
}

template <>
template <>
inline Vector3d& Vector3d::operator*=<double>(const double i_factor)
{
  m_coords[0] *= i_factor;
  m_coords[1] *= i_factor;
  m_coords[2] *= i_factor;
  return *this;
}

template <>
inline double Vector3d::SquareLength() const
{
  return m_coords[0] * m_coords[0] + m_coords[1] * m_coords[1] + m_coords[2] * m_coords[2];
}

template <>
inline double Vector3d::SquareDistance(const Vector3d& i_other) const
{
  const double dx = i_other.m_coords[0] - m_coords[0];
  const double dy = i_other.m_coords[1] - m_coords[1];
  const double dz = i_other.m_coords[2] - m_coords[2];
  return dx * dx + dy * dy + dz * dz;
}

template <>
inline void Vector3d::Normalize()
{
  const double length = Length();
  if (length > 0.0) {
    m_coords[0] /= length;
    m_coords[1] /= length;
    m_coords[2] /= length;
  }
}

template <>
inline Vector3d Vector3d::Normalized() const
{
  const double sqr_length = SquareLength();
  // already normalized or zero
  if (sqr_length == 0.0 || sqr_length - 1 == 0.0)
    return *this;
  // normalizing
  const double length = sqrt(sqr_length);
  return { m_coords[0] / length, m_coords[1] / length, m_coords[2] / length };
}