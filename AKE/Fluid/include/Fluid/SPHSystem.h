#pragma once
#include "Fluid/FluidConstants.h"
#include "Fluid/ParticleSystem.h"
#include "Fluid/PointNeighborSearcher.h"

class SPHSystem final : public ParticleSystem
{
public:
  explicit SPHSystem(std::size_t i_num_particles = 1024,
                     double i_system_density = WATER_DENSITY,
                     double i_viscosity = VISCOSITY,
                     double i_radius = SMOOTHING_RADIUS,
                     double i_mass = PARTICLE_MASS);
  ~SPHSystem();

  [[nodiscard]] VectorDataIteratorC BeginPositions() const;
  [[nodiscard]] VectorDataIteratorC EndPositions() const;
  VectorDataIterator BeginPositions();
  VectorDataIterator EndPositions();

  [[nodiscard]] VectorDataIteratorC BeginVelocities() const;
  [[nodiscard]] VectorDataIteratorC EndVelocities() const;
  VectorDataIterator BeginVelocities();
  VectorDataIterator EndVelocities();

  [[nodiscard]] VectorDataIteratorC BeginForces() const;
  [[nodiscard]] VectorDataIteratorC EndForces() const;
  VectorDataIterator BeginForces();
  VectorDataIterator EndForces();
  void ClearForces();

  [[nodiscard]] ScalarDataIteratorC BeginDensities() const;
  [[nodiscard]] ScalarDataIteratorC EndDensities() const;
  ScalarDataIterator BeginDensities();
  ScalarDataIterator EndDensities();

  [[nodiscard]] ScalarDataIteratorC BeginPressures() const;
  [[nodiscard]] ScalarDataIteratorC EndPressures() const;
  ScalarDataIterator BeginPressures();
  ScalarDataIterator EndPressures();

  [[nodiscard]] double GetSystemDensity() const;
  void SetSystemDensity(double i_system_density);

  [[nodiscard]] double GetViscosity() const;
  void SetViscosity(double i_viscosity);

  [[nodiscard]] double GetRadius() const;
  void SetRadius(double i_radius);

  [[nodiscard]] double GetMass() const;
  void SetMass(double i_mass);

  void BuildNeighborSearcher();
  void BuildNeighborsList();
  void UpdateDensities();

  [[nodiscard]] std::vector<std::vector<std::size_t>> GetNeighborsList() const;

private:
  [[nodiscard]] double _SumOfKernelsNearby(const Vector3d& i_origin) const;

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

  PointNeighborSearcher* mp_neighbor_searcher;
  std::vector<std::vector<std::size_t>> m_neighbors_list;
};
