#include "Fluid/SPHSimulation.h"

#include "Common/ThreadPool.h"
#include "Fluid/SPHSpikyKernel.h"
#include "Physics/Constants.h"

namespace {
  double ComputePressureFromEOS(const double i_density,
                                const double i_system_density,
                                const double i_eos_scale,
                                const double i_eos_exponent,
                                const double i_negative_pressure_scale)
  {
    // See Murnaghan-Tait equation of state from
    // https://en.wikipedia.org/wiki/Tait_equation
    double pressure = i_eos_scale / i_eos_exponent * (std::pow(i_density / i_system_density, i_eos_exponent) - 1.0);

    // Negative pressure scaling
    if (pressure < 0)
      pressure *= i_negative_pressure_scale;

    return pressure;
  }
}

SPHSimulation::SPHSimulation(const std::size_t i_num_particles)
  : Simulation(0.04)
  , m_particle_system(i_num_particles)
  , m_new_positions(i_num_particles)
  , m_new_velocities(i_num_particles)
  , m_eos_exponent(7.15)
  , m_negative_pressure_scale(0.75)
{
  _InitParticles();
}

void SPHSimulation::_InitParticles()
{
  auto particles_positions = m_particle_system.BeginPositions();
  const auto num_of_particles = m_particle_system.GetNumOfParticles();

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&particles_positions](const std::size_t i) {
      particles_positions[i] = Vector3d(5 * static_cast<double>(rand()) / RAND_MAX - 2.5,
                                        static_cast<double>(rand()) / RAND_MAX,
                                        5 * static_cast<double>(rand()) / RAND_MAX - 2.5);
    });
}

void SPHSimulation::_PreProcessing()
{
  m_particle_system.ClearForces();
  m_particle_system.BuildNeighborSearcher();
  m_particle_system.BuildNeighborsList();
  m_particle_system.UpdateDensities();
}

void SPHSimulation::_Update()
{
  _AccumulateExternalForces();
  _AccumulateViscosityForce();

  _ComputePressure();
  _AccumulatePressureForeces();

  _TimeIntegration();

  _ResolveCollisions();
}

void SPHSimulation::_PostProcessing()
{
  _UpdatePositionsAndVelocities();
}

void SPHSimulation::_AccumulateExternalForces()
{
  const size_t num_of_particles = m_particle_system.GetNumOfParticles();
  auto forces = m_particle_system.BeginForces();
  const double mass = m_particle_system.GetMass();

  const auto gravity_force = Vector3d(0, -Physics::Constants::GRAVITY_CONSTANT, 0) * mass;

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&](const std::size_t i_index) {
      forces[i_index] += gravity_force;
    });
}

void SPHSimulation::_AccumulateViscosityForce()
{
  const size_t num_of_particles = m_particle_system.GetNumOfParticles();
  auto positions = m_particle_system.BeginPositions();
  auto densities = m_particle_system.BeginDensities();
  auto velosities = m_particle_system.BeginVelocities();
  auto forces = m_particle_system.BeginForces();

  const auto mass = m_particle_system.GetMass();
  const auto square_mass = mass * mass;
  const auto& neigbors_list = m_particle_system.GetNeighborsList();
  const SPHSpikyKernel kernel(m_particle_system.GetRadius());

  const double viscosity = m_particle_system.GetViscosity();

  Parallel::ThreadPool::GetInstance()->ParallelFor(static_cast<size_t>(0), num_of_particles, [&](const size_t i) {
    const auto& neighbors = neigbors_list[i];
    for (auto j : neighbors) {
      double dist = positions[i].Distance(positions[j]);

      forces[i] +=
        (velosities[j] - velosities[i]) * kernel.SecondDerivative(dist) * viscosity * square_mass / densities[j];
    }
  });
}

void SPHSimulation::_ComputePressure()
{
  const auto num_of_particles = m_particle_system.GetNumOfParticles();
  auto densities = m_particle_system.BeginDensities();
  auto pressures = m_particle_system.BeginPressures();

  // See Murnaghan-Tait equation of state from
  // https://en.wikipedia.org/wiki/Tait_equation
  const double system_density = m_particle_system.GetSystemDensity();
  const double sqr_speed_of_sound = WATER_SPEED_OF_SOUND * WATER_SPEED_OF_SOUND;
  // TODO : understand this
  const double eos_scale = system_density * sqr_speed_of_sound / 1e6;

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&](const std::size_t i) {
      pressures[i] =
        ComputePressureFromEOS(densities[i], system_density, eos_scale, m_eos_exponent, m_negative_pressure_scale);
    });
}

void SPHSimulation::_AccumulatePressureForeces()
{
  size_t num_of_particles = m_particle_system.GetNumOfParticles();
  auto positions = m_particle_system.BeginPositions();
  auto densities = m_particle_system.BeginDensities();
  auto pressures = m_particle_system.BeginPressures();
  auto forces = m_particle_system.BeginForces();

  const double mass = m_particle_system.GetMass();
  const double square_mass = mass * mass;
  const auto& neigbors_list = m_particle_system.GetNeighborsList();
  const SPHSpikyKernel kernel(m_particle_system.GetRadius());

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&](const std::size_t i) {
      const auto& neighbors = neigbors_list[i];
      for (auto j : neighbors) {
        const double dist = positions[i].Distance(positions[j]);

        if (dist > 0.0) {
          const Vector3d dir = (positions[j] - positions[i]) / dist;
          forces[i] -= kernel.Gradient(dist, dir) * square_mass *
                       (pressures[i] / (densities[i] * densities[i]) + pressures[j] / (densities[j] * densities[j]));
        }
      }
    });
}

void SPHSimulation::_UpdatePositionsAndVelocities()
{
  size_t num_of_particles = m_particle_system.GetNumOfParticles();
  auto positions = m_particle_system.BeginPositions();
  auto velocities = m_particle_system.BeginVelocities();
  Parallel::ThreadPool::GetInstance()->ParallelFor(static_cast<std::size_t>(0), num_of_particles, [&](const size_t i) {
    positions[i] = m_new_positions[i];
    velocities[i] = m_new_velocities[i];
  });
}

void SPHSimulation::_ResolveCollisions()
{
  size_t num_of_particles = m_particle_system.GetNumOfParticles();

  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&](const std::size_t i) {
      if (m_new_positions[i].SquareLength() < 9)
        return;
      const auto& normal = -m_new_positions[i].Normalized();
      const double dot = normal.Dot(-m_new_velocities[i]);
      const auto vel_normal_component = normal * dot;
      const auto vel_tangent_component = m_new_velocities[i] - vel_normal_component;
      m_new_positions[i] = -normal * (3 - m_particle_system.GetRadius());
      m_new_velocities[i] = vel_normal_component * 0.75 + vel_tangent_component * 0.5 +
                            Vector3d(normal[1], -normal[0], 0.1) * 3; // rotation component emulates sphere rotation
    });
}

void SPHSimulation::_TimeIntegration()
{
  size_t num_of_particles = m_particle_system.GetNumOfParticles();
  auto positions = m_particle_system.BeginPositions();
  auto velocities = m_particle_system.BeginVelocities();
  auto forces = m_particle_system.BeginForces();
  Parallel::ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0), num_of_particles, [&](const std::size_t i) {
      m_new_velocities[i] = velocities[i] + forces[i] * GetTimeStep() / m_particle_system.GetMass();

      m_new_positions[i] = positions[i] + m_new_velocities[i] * GetTimeStep();
    });
}

const SPHSystem& SPHSimulation::GetParticleSystem() const
{
  return m_particle_system;
}

double SPHSimulation::GetEOSExponent() const
{
  return m_eos_exponent;
}

void SPHSimulation::SetEOSExponent(const double i_eos_exponent)
{
  m_eos_exponent = i_eos_exponent;
}

double SPHSimulation::GetNegativePressureScale() const
{
  return m_negative_pressure_scale;
}

void SPHSimulation::SetNegativePressureScale(const double i_scale)
{
  m_negative_pressure_scale = i_scale;
}