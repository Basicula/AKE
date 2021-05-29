#include <Common/ThreadPool.h>

#include <Fluid/BFPointSearcher.h>
#include <Fluid/SPHSystem.h>
#include <Fluid/SPHStandartKernel.h>

SPHSystem::SPHSystem(
  std::size_t i_num_particles,
  double i_system_density,
  double i_viscosity,
  double i_radius,
  double i_mass)
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
  if (mp_neighbor_searcher)
    delete mp_neighbor_searcher;
}

void SPHSystem::BuildNeighborSearcher()
  {
  mp_neighbor_searcher = new BFPointSearcher(BeginPositions(), m_num_particles);
  }

void SPHSystem::BuildNeighborsList()
  {
  const auto points = BeginPositions();
  ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0),
    m_num_particles,
    [&](std::size_t i)
    {
    const Vector3d& origin = points[i];
    m_neighbors_list[i].clear();

    mp_neighbor_searcher->ForEachNearbyPoint(
      origin,
      m_radius,
      [&](std::size_t j, const Vector3d&)
      {
      if (i != j)
        {
        m_neighbors_list[i].push_back(j);
        }
      });
    });
  }

void SPHSystem::UpdateDensities()
  {
  const auto positions = BeginPositions();
  auto densities = BeginDensities();

  ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), 
    m_num_particles, 
    [&](std::size_t i)
    {
    const double sum = _SumOfKernelsNearby(positions[i]);
    densities[i] = m_mass * sum;
    });
  }

double SPHSystem::_SumOfKernelsNearby(const Vector3d& i_origin) const
  {
  double sum = 0.0;
  const SPHStandartKernel kernel(m_radius);
  mp_neighbor_searcher->ForEachNearbyPoint(
    i_origin, 
    m_radius, 
    [&](std::size_t, const Vector3d& i_neighbor)
    {
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