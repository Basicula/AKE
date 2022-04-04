#pragma once
#include "Fluid/SPHSimulation.h"
#include "Geometry/BoundingBox.h"
#include "Rendering/Object.h"

class Fluid final : public Object
{
public:
  explicit Fluid(std::size_t i_num_particles);

  bool IntersectWithRay(double& o_distance, const Ray& i_ray, double i_far) const override;
  [[nodiscard]] std::string Serialize() const;
  [[nodiscard]] BoundingBox GetBoundingBox() const override;
  [[nodiscard]] Vector3d GetNormalAtPoint(const Vector3d& i_point) const override;

  void Update();

  [[nodiscard]] double GetTimeStep() const;
  [[nodiscard]] std::size_t GetNumParticles() const;

private:
  void _UpdateBBox();

private:
  BoundingBox m_bbox;
  SPHSimulation m_simulation;
};