#include "Fluid/ParticleSystem.h"

ParticleSystem::ParticleSystem()
  : ParticleSystem(0)
  {}

ParticleSystem::ParticleSystem(std::size_t i_num_particles)
  : m_num_particles(i_num_particles)
  {
  }

void ParticleSystem::_Resize(std::size_t i_num_particles)
  {
  m_num_particles = i_num_particles;
  
  for (auto& vector_data : m_vector_data)
    vector_data.resize(m_num_particles, Vector3d(0));

  for (auto& scalar_data : m_scalar_data)
    scalar_data.resize(m_num_particles, 0);
  }
