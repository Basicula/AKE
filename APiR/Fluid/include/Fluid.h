#pragma once
#include <vector>

#include <IRenderable.h>
#include <Common/BoundingBox.h>
#include <StandartParticle.h>
#include <SPHSimulation.h>

class Fluid : public IRenderable
  {
  public:
    Fluid(std::size_t i_num_particles);

    virtual bool IntersectWithRay(
      IntersectionRecord& o_intersection,
      const Ray& i_ray) const override;
    virtual std::string Serialize() const override;
    virtual BoundingBox GetBoundingBox() const override;
    
    virtual void Update() override;

    double GetTimeStep();
    std::size_t GetNumParticles() const;

  private:
    void _UpdateBBox();

  private:
    BoundingBox m_bbox;
    SPHSimulation m_simulation;

    std::shared_ptr<IMaterial> m_material;
  };

inline std::string Fluid::Serialize() const
  {
  std::string res = "{ \"Fluid\" : { ";
  res += "\"NumOfParticles\" : " + std::to_string(GetNumParticles());
  res += "} }";
  return res;
  }

inline BoundingBox Fluid::GetBoundingBox() const
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