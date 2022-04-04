#include "Fluid/SPHSystem.h"

#include "Common/ThreadPool.h"
#include "Fluid/BFPointSearcher.h"
#include "Fluid/SPHStandartKernel.h"

SPHSystem::SPHSystem(const std::size_t i_num_particles,
                     const double i_system_density,
                     const double i_viscosity,
                     const double i_radius,
                     const double i_mass)
  : ParticleSystem(i_num_particles)
  , m_system_density(i_system_density)
  , m_viscosity(i_viscosity)
  , m_radius(i_radius)
  , m_mass(i_mass)
  , mp_neighbor_searcher(nullptr)
  , m_neighbors_list(i_num_particles)
{
  m_positions_id = AddVectorData();
  m_velocities_id = AddVectorData();
  m_forces_id = AddVectorData();

  m_densities_id = AddScalarData();
  m_pressures_id = AddScalarData();
}

SPHSystem::~SPHSystem()
{
  delete mp_neighbor_searcher;
}

void SPHSystem::BuildNeighborSearcher()
{
  mp_neighbor_searcher = new BFPointSearcher(BeginPositions(), m_num_particles);
}

void SPHSystem::BuildNeighborsList()
{
  const auto points = BeginPositions();
  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), m_num_particles, [&](const std::size_t i) {
      const Vector3d& origin = points[i];
      m_neighbors_list[i].clear();

      mp_neighbor_searcher->ForEachNearbyPoint(origin, m_radius, [&](const std::size_t j, const Vector3d&) {
        if (i != j) {
          m_neighbors_list[i].push_back(j);
        }
      });
    });
}

void SPHSystem::UpdateDensities()
{
  const auto positions = BeginPositions();
  auto densities = BeginDensities();

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), m_num_particles, [&](const std::size_t i) {
      const double sum = _SumOfKernelsNearby(positions[i]);
      densities[i] = m_mass * sum;
    });
}

double SPHSystem::_SumOfKernelsNearby(const Vector3d& i_origin) const
{
  double sum = 0.0;
  const SPHStandartKernel kernel(m_radius);
  mp_neighbor_searcher->ForEachNearbyPoint(i_origin, m_radius, [&](std::size_t, const Vector3d& i_neighbor) {
    const double dist = i_origin.SquareDistance(i_neighbor);
    sum += kernel(dist);
  });
  return sum;
}

void SPHSystem::ClearForces()
{
  auto forces = BeginForces();
  for (auto i = 0u; i < m_num_particles; ++i, ++forces)
    *forces = Vector3d(0.0);
}

ParticleSystem::VectorDataIteratorC SPHSystem::BeginPositions() const
{
  return _BeginVectorDataAt(m_positions_id);
}

ParticleSystem::VectorDataIteratorC SPHSystem::EndPositions() const
{
  return _EndVectorDataAt(m_positions_id);
}

ParticleSystem::VectorDataIterator SPHSystem::BeginPositions()
{
  return _BeginVectorDataAt(m_positions_id);
}

ParticleSystem::VectorDataIterator SPHSystem::EndPositions()
{
  return _EndVectorDataAt(m_positions_id);
}

ParticleSystem::VectorDataIteratorC SPHSystem::BeginVelocities() const
{
  return _BeginVectorDataAt(m_velocities_id);
}

ParticleSystem::VectorDataIteratorC SPHSystem::EndVelocities() const
{
  return _EndVectorDataAt(m_velocities_id);
}

ParticleSystem::VectorDataIterator SPHSystem::BeginVelocities()
{
  return _BeginVectorDataAt(m_velocities_id);
}

ParticleSystem::VectorDataIterator SPHSystem::EndVelocities()
{
  return _EndVectorDataAt(m_velocities_id);
}

ParticleSystem::VectorDataIteratorC SPHSystem::BeginForces() const
{
  return _BeginVectorDataAt(m_forces_id);
}

ParticleSystem::VectorDataIteratorC SPHSystem::EndForces() const
{
  return _EndVectorDataAt(m_forces_id);
}

ParticleSystem::VectorDataIterator SPHSystem::BeginForces()
{
  return _BeginVectorDataAt(m_forces_id);
}

ParticleSystem::VectorDataIterator SPHSystem::EndForces()
{
  return _EndVectorDataAt(m_forces_id);
}

ParticleSystem::ScalarDataIteratorC SPHSystem::BeginDensities() const
{
  return _BeginScalarDataAt(m_densities_id);
}

ParticleSystem::ScalarDataIteratorC SPHSystem::EndDensities() const
{
  return _EndScalarDataAt(m_densities_id);
}

ParticleSystem::ScalarDataIterator SPHSystem::BeginDensities()
{
  return _BeginScalarDataAt(m_densities_id);
}

ParticleSystem::ScalarDataIterator SPHSystem::EndDensities()
{
  return _EndScalarDataAt(m_densities_id);
}

ParticleSystem::ScalarDataIteratorC SPHSystem::BeginPressures() const
{
  return _BeginScalarDataAt(m_pressures_id);
}

ParticleSystem::ScalarDataIteratorC SPHSystem::EndPressures() const
{
  return _EndScalarDataAt(m_pressures_id);
}

ParticleSystem::ScalarDataIterator SPHSystem::BeginPressures()
{
  return _BeginScalarDataAt(m_pressures_id);
}

ParticleSystem::ScalarDataIterator SPHSystem::EndPressures()
{
  return _EndScalarDataAt(m_pressures_id);
}

double SPHSystem::GetSystemDensity() const
{
  return m_system_density;
}

void SPHSystem::SetSystemDensity(const double i_system_density)
{
  m_system_density = i_system_density;
}

double SPHSystem::GetViscosity() const
{
  return m_viscosity;
}

void SPHSystem::SetViscosity(const double i_viscosity)
{
  m_viscosity = i_viscosity;
}

double SPHSystem::GetMass() const
{
  return m_mass;
}

void SPHSystem::SetMass(const double i_mass)
{
  m_mass = i_mass;
}

double SPHSystem::GetRadius() const
{
  return m_radius;
}

void SPHSystem::SetRadius(const double i_radius)
{
  m_radius = i_radius;
}

std::vector<std::vector<std::size_t>> SPHSystem::GetNeighborsList() const
{
  return m_neighbors_list;
}
