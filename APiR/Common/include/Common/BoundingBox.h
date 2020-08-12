#pragma once
#include <algorithm>
#include <string>

#include <Math/Vector.h>

class BoundingBox
  {
  public:
    BoundingBox();
    BoundingBox(
      const Vector3d& i_min,
      const Vector3d& i_max);

    const Vector3d& GetMin() const;
    const Vector3d& GetMax() const;
    Vector3d Center() const;
    // iterate from min to max using bitmask
    // i.e. (0,0,0) -> (1,1,1)
    Vector3d GetCorner(std::size_t i_corner_id) const;

    double DeltaX() const;
    double DeltaY() const;
    double DeltaZ() const;

    void Merge(const BoundingBox& i_other);
    void Reset();

    void AddPoint(const Vector3d& i_point);
    bool Contains(const Vector3d& i_point) const;

    bool IsValid() const;

    std::string Serialize() const;

  private:
    Vector3d m_min;
    Vector3d m_max;
  };

inline std::string BoundingBox::Serialize() const
  {
  std::string res = "{ \"BoundingBox\" : { ";
  res += "\"MinCorner\" : " + m_min.Serialize() + ", ";
  res += "\"MaxCorner\" : " + m_max.Serialize();
  res += " } }";
  return res;
  }

inline const Vector3d& BoundingBox::GetMin() const
  {
  return m_min;
  }

inline const Vector3d& BoundingBox::GetMax() const
  {
  return m_max;
  }

inline Vector3d BoundingBox::Center() const
  {
  return (m_min + m_max) / 2;
  }

inline bool BoundingBox::IsValid() const
  {
  return (m_min < m_max);
  }

inline bool BoundingBox::Contains(const Vector3d& i_point) const
  {
  return (m_min <= i_point && i_point <= m_max);
  }

inline double BoundingBox::DeltaX() const
  {
  return (m_max[0] - m_min[0]);
  }

inline double BoundingBox::DeltaY() const
  {
  return (m_max[1] - m_min[1]);
  }

inline double BoundingBox::DeltaZ() const
  {
  return (m_max[2] - m_min[2]);
  }