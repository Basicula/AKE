#pragma once
#include <Geometry/BoundingBox.h>
#include <Fluid/SPHSimulation.h>
#include <Rendering/Object.h>
#include <Visual/IVisualMaterial.h>

class Fluid : public Object
  {
  public:
    Fluid(std::size_t i_num_particles);

    virtual bool IntersectWithRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far) const override;
    virtual std::string Serialize() const;
    virtual BoundingBox3D GetBoundingBox() const override;
    virtual Vector3d GetNormalAtPoint(const Vector3d& i_point) const override;

    void Update();

    double GetTimeStep();
    std::size_t GetNumParticles() const;

  private:
    void _UpdateBBox();

  private:
    BoundingBox3D m_bbox;
    SPHSimulation m_simulation;
  };

inline std::string Fluid::Serialize() const
  {
  std::string res = "{ \"Fluid\" : { ";
  res += "\"NumOfParticles\" : " + std::to_string(GetNumParticles());
  res += "} }";
  return res;
  }

inline BoundingBox3D Fluid::GetBoundingBox() const
  {
  return m_bbox;
  }

inline double Fluid::GetTimeStep()
  {
  return m_simulation.GetTimeStep();
  }

inline std::size_t Fluid::GetNumParticles() const
  {
  return m_simulation.GetParticleSystem().GetNumOfParticles();
  }

inline void Fluid::Update()
  {
  m_simulation.Update();
  _UpdateBBox();
  }