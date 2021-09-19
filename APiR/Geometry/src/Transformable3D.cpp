#include <Geometry/Transformable3D.h>

void Transformable3D::Scale(double i_factor)
  {
  m_transformation.SetScale(m_transformation.GetScale() * i_factor);
  _OnTransformationChange();
  }

void Transformable3D::Scale(const Vector3d& i_factors)
  {
  const auto old_scale = m_transformation.GetScale();
  m_transformation.SetScale(
    Vector3d(
      old_scale[0] * i_factors[0],
      old_scale[1] * i_factors[1],
      old_scale[2] * i_factors[2]));
  _OnTransformationChange();
  }

void Transformable3D::Rotate(const Vector3d& i_axis, double i_degree_in_rad)
  {
  const auto old_rotation = m_transformation.GetRotation();
  m_transformation.SetRotation(i_axis, i_degree_in_rad);
  const auto new_rotation = m_transformation.GetRotation();
  m_transformation.SetRotation(new_rotation * old_rotation);
  _OnTransformationChange();
  }

void Transformable3D::Translate(const Vector3d& i_translation)
  {
  m_transformation.SetTranslation(m_transformation.GetTranslation() + i_translation);
  _OnTransformationChange();
  }

Ray Transformable3D::RayToLocal(const Ray& i_ray) const
  {
  return Ray(
    m_transformation.InverseTransform(i_ray.GetOrigin()),
    m_transformation.InverseTransform(i_ray.GetDirection(), true));
  }

BoundingBox3D Transformable3D::BBoxToWorld(const BoundingBox3D& i_local_bbox) const
  {
  BoundingBox3D res;
  for (auto corner_id = 0; corner_id < 8; ++corner_id)
    res.AddPoint(PointToWorld(i_local_bbox.GetCorner(corner_id)));
  return res;
  }

Transformation3D Transformable3D::GetTransformation() const {
  return m_transformation;
}

Vector3d Transformable3D::PointToLocal(const Vector3d& i_world_point) const {
  return m_transformation.InverseTransform(i_world_point);
}

Vector3d Transformable3D::PointToWorld(const Vector3d& i_local_point) const {
  return m_transformation.Transform(i_local_point);
}

Vector3d Transformable3D::DirectionToLocal(const Vector3d& i_world_dir) const {
  return m_transformation.InverseTransform(i_world_dir, true);
}

Vector3d Transformable3D::DirectionToWorld(const Vector3d& i_local_dir) const {
  return m_transformation.Transform(i_local_dir, true);
}

Vector3d Transformable3D::GetScale() const {
  return m_transformation.GetScale();
}

void Transformable3D::SetScale(const Vector3d& i_factors) {
  m_transformation.SetScale(i_factors);
  _OnTransformationChange();
}

Matrix3x3d Transformable3D::GetRotation() const {
  return m_transformation.GetRotation();
}

void Transformable3D::SetRotation(const Vector3d& i_axis, double i_degree_in_rad) {
  m_transformation.SetRotation(i_axis, i_degree_in_rad);
  _OnTransformationChange();
}

Vector3d Transformable3D::GetTranslation() const {
  return m_transformation.GetTranslation();
}

void Transformable3D::SetTranslation(const Vector3d& i_translation) {
  m_transformation.SetTranslation(i_translation);
  _OnTransformationChange();
}

