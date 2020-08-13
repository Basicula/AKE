#pragma once
#include <Common/PointNeighborSearcher.h>

#include <Fluid/FluidConstants.h>
#include <Fluid/ParticleSystem.h>

class SPHSystem : public ParticleSystem
  {
  public:
    SPHSystem(
      std::size_t i_num_particles = 1024,
      double i_system_density = WATER_DENSITY,
      double i_viscosity = VISCOSITY,
      double i_radius = SMOOTHING_RADIUS,
      double i_mass = PARTICLE_MASS);

    VectorDataIteratorC BeginPositions() const;
    VectorDataIteratorC EndPositions() const;
    VectorDataIterator BeginPositions();
    VectorDataIterator EndPositions();

    VectorDataIteratorC BeginVelocities() const;
    VectorDataIteratorC EndVelocities() const;
    VectorDataIterator BeginVelocities();
    VectorDataIterator EndVelocities();

    VectorDataIteratorC BeginForces() const;
    VectorDataIteratorC EndForces() const;
    VectorDataIterator BeginForces();
    VectorDataIterator EndForces();
    void ClearForces();

    ScalarDataIteratorC BeginDensities() const;
    ScalarDataIteratorC EndDensities() const;
    ScalarDataIterator BeginDensities();
    ScalarDataIterator EndDensities();

    ScalarDataIteratorC BeginPressures() const;
    ScalarDataIteratorC EndPressures() const;
    ScalarDataIterator BeginPressures();
    ScalarDataIterator EndPressures();

    double GetSystemDensity() const;
    void SetSystemDensity(double i_system_density);

    double GetViscosity() const;
    void SetViscosity(double i_viscosity);

    double GetRadius() const;
    void SetRadius(double i_radius);

    double GetMass() const;
    void SetMass(double i_mass);

    void BuildNeighborSearcher();
    void BuildNeighborsList();
    void UpdateDensities();

    std::vector<std::vector<std::size_t>> GetNeighborsList() const;

  private:
    double _SumOfKernelsNearby(const Vector3d& i_origin) const;

  private:
    std::size_t m_positions_id;
    std::size_t m_velocities_id;
    std::size_t m_forces_id;

    std::size_t m_densities_id;
    std::size_t m_pressures_id;

    double m_system_density;
    double m_viscosity;

    double m_radius;
    double m_mass;

    PointNeighborSearcherPtr mp_neighbor_searcher;
    std::vector<std::vector<std::size_t>> m_neighbors_list;
  };

inline ParticleSystem::VectorDataIteratorC SPHSystem::BeginPositions() const
  {
  return _BeginVectorDataAt(m_positions_id);
  }

inline ParticleSystem::VectorDataIteratorC SPHSystem::EndPositions() const
  {
  return _EndVectorDataAt(m_positions_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::BeginPositions()
  {
  return _BeginVectorDataAt(m_positions_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::EndPositions()
  {
  return _EndVectorDataAt(m_positions_id);
  }

inline ParticleSystem::VectorDataIteratorC SPHSystem::BeginVelocities() const
  {
  return _BeginVectorDataAt(m_velocities_id);
  }

inline ParticleSystem::VectorDataIteratorC SPHSystem::EndVelocities() const
  {
  return _EndVectorDataAt(m_velocities_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::BeginVelocities()
  {
  return _BeginVectorDataAt(m_velocities_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::EndVelocities()
  {
  return _EndVectorDataAt(m_velocities_id);
  }

inline ParticleSystem::VectorDataIteratorC SPHSystem::BeginForces() const
  {
  return _BeginVectorDataAt(m_forces_id);
  }

inline ParticleSystem::VectorDataIteratorC SPHSystem::EndForces() const
  {
  return _EndVectorDataAt(m_forces_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::BeginForces()
  {
  return _BeginVectorDataAt(m_forces_id);
  }

inline ParticleSystem::VectorDataIterator SPHSystem::EndForces()
  {
  return _EndVectorDataAt(m_forces_id);
  }

inline ParticleSystem::ScalarDataIteratorC SPHSystem::BeginDensities() const
  {
  return _BeginScalarDataAt(m_densities_id);
  }

inline ParticleSystem::ScalarDataIteratorC SPHSystem::EndDensities() const
  {
  return _EndScalarDataAt(m_densities_id);
  }

inline ParticleSystem::ScalarDataIterator SPHSystem::BeginDensities()
  {
  return _BeginScalarDataAt(m_densities_id);
  }

  inline ParticleSystem::ScalarDataIterator SPHSystem::EndDensities()
  {
  return _EndScalarDataAt(m_densities_id);
  }

inline ParticleSystem::ScalarDataIteratorC SPHSystem::BeginPressures() const
  {
  return _BeginScalarDataAt(m_pressures_id);
  }

inline ParticleSystem::ScalarDataIteratorC SPHSystem::EndPressures() const
  {
  return _EndScalarDataAt(m_pressures_id);
  }

inline ParticleSystem::ScalarDataIterator SPHSystem::BeginPressures()
  {
  return _BeginScalarDataAt(m_pressures_id);
  }

inline ParticleSystem::ScalarDataIterator SPHSystem::EndPressures()
  {
  return _EndScalarDataAt(m_pressures_id);
  }

inline double SPHSystem::GetSystemDensity() const
  {
  return m_system_density;
  }

inline void SPHSystem::SetSystemDensity(double i_system_density)
  {
  m_system_density = i_system_density;
  }

inline double SPHSystem::GetViscosity() const
  {
  return m_viscosity;
  }

inline void SPHSystem::SetViscosity(double i_viscosity)
  {
  m_viscosity = i_viscosity;
  }

inline double SPHSystem::GetMass() const
  {
  return m_mass;
  }

inline void SPHSystem::SetMass(double i_mass)
  {
  m_mass = i_mass;
  }

inline double SPHSystem::GetRadius() const
  {
  return m_radius;
  }

inline void SPHSystem::SetRadius(double i_radius)
  {
  m_radius = i_radius;
  }

inline std::vector<std::vector<std::size_t>> SPHSystem::GetNeighborsList() const
  {
  return m_neighbors_list;
  }