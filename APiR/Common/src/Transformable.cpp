#include <Common/Transformable.h>

void Transformable::Scale(double i_factor)
  {
  m_transformation.SetScale(m_transformation.GetScale() * i_factor);
  _OnTransformationChange();
  }

void Transformable::Scale(const Vector3d& i_factors)
  {
  const auto old_scale = m_transformation.GetScale();
  m_transformation.SetScale(
    Vector3d(
      old_scale[0] * i_factors[0],
      old_scale[1] * i_factors[1],
      old_scale[2] * i_factors[2]));
  _OnTransformationChange();
  }

Vector3d Transformable::ApplyScaling(const Vector3d& i_point) const
  {
  const auto& scaling = m_transformation.GetScale();
  return Vector3d(
    scaling[0] * i_point[0], 
    scaling[1] * i_point[1], 
    scaling[2] * i_point[2]);
  }

void Transformable::Rotate(const Vector3d& i_axis, double i_degree_in_rad)
  {
  const auto old_rotation = m_transformation.GetRotation();
  m_transformation.SetRotation(i_axis, i_degree_in_rad);
  const auto new_rotation = m_transformation.GetRotation();
  m_transformation.SetRotation(new_rotation * old_rotation);
  _OnTransformationChange();
  }

Vector3d Transformable::ApplyRotation(const Vector3d& i_point) const
  {
  const auto& rotation = m_transformation.GetRotation();
  return rotation * i_point;
  }

void Transformable::Translate(const Vector3d& i_translation)
  {
  m_transformation.SetTranslation(m_transformation.GetTranslation() + i_translation);
  _OnTransformationChange();
  }

Vector3d Transformable::ApplyTranslation(const Vector3d& i_point) const
  {
  return i_point + m_transformation.GetTranslation();
  }

Ray Transformable::RayToLocal(const Ray& i_ray) const
  {
  return Ray(
    m_transformation.PointToLocal(i_ray.GetOrigin()),
    m_transformation.DirectionToLocal(i_ray.GetDirection()));
  }

BoundingBox Transformable::BBoxToWorld(const BoundingBox& i_local_bbox) const
  {
  BoundingBox res;
  for (auto corner_id = 0; corner_id < 8; ++corner_id)
    res.AddPoint(PointToWorld(i_local_bbox.GetCorner(corner_id)));
  return res;
  }

void Transformable::_LocalIntersectionToWorld(IntersectionRecord& io_intersection) const
  {
  io_intersection.m_intersection = PointToWorld(io_intersection.m_intersection);
  io_intersection.m_normal = DirectionToWorld(io_intersection.m_normal);
  }
