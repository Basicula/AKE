#pragma once
#include <Geometry/Intersection.h>
#include <Geometry/Transformation3D.h>

#include <Math/Vector.h>

class Transformable3D
  {
  public:
    Transformable3D() = default;
    ~Transformable3D() = default;

    Transformation3D GetTransformation() const;

    Vector3d PointToLocal(const Vector3d& i_world_point) const;
    Vector3d PointToWorld(const Vector3d& i_local_point) const;

    Vector3d DirectionToLocal(const Vector3d& i_world_dir) const;
    Vector3d DirectionToWorld(const Vector3d& i_local_dir) const;

    // Scalings
    Vector3d GetScale() const;
    void SetScale(const Vector3d& i_factors);
    void Scale(double i_factor);
    void Scale(const Vector3d& i_factors);

    // Rotations
    Matrix3x3d GetRotation() const;
    void SetRotation(const Vector3d& i_axis, double i_degree_in_rad);
    void Rotate(const Vector3d& i_axis, double i_degree_in_rad);

    // Translations
    Vector3d GetTranslation() const;
    void SetTranslation(const Vector3d& i_translation);
    void Translate(const Vector3d& i_translation);

    Ray RayToLocal(const Ray& i_world_ray) const;

    BoundingBox3D BBoxToWorld(const BoundingBox3D& i_local_bbox) const;

  protected:
    virtual void _OnTransformationChange() = 0;

  private:
    Transformation3D m_transformation;
  };