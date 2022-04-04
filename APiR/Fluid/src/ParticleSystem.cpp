#include "Fluid/ParticleSystem.h"

ParticleSystem::ParticleSystem()
  : ParticleSystem(0)
{}

ParticleSystem::ParticleSystem(const std::size_t i_num_particles)
  : m_num_particles(i_num_particles)
{}

void ParticleSystem::_Resize(const std::size_t i_num_particles)
{
  m_num_particles = i_num_particles;

  for (auto& vector_data : m_vector_data)
    vector_data.resize(m_num_particles, Vector3d(0));

  for (auto& scalar_data : m_scalar_data)
    scalar_data.resize(m_num_particles, 0);
}

std::size_t ParticleSystem::AddVectorData(const Vector3d& i_inittial_value)
{
  const auto id = m_vector_data.size();
  m_vector_data.emplace_back(m_num_particles, i_inittial_value);
  return id;
}

std::size_t ParticleSystem::AddScalarData(double i_initial_value)
{
  const auto id = m_scalar_data.size();
  m_scalar_data.emplace_back(m_num_particles, i_initial_value);
  return id;
}

std::size_t ParticleSystem::GetNumOfParticles() const
{
  return m_num_particles;
}

ParticleSystem::VectorDataIteratorC ParticleSystem::_BeginVectorDataAt(const std::size_t i_index) const
{
  return m_vector_data[i_index].cbegin();
}

ParticleSystem::VectorDataIteratorC ParticleSystem::_EndVectorDataAt(const std::size_t i_index) const
{
  return m_vector_data[i_index].cend();
}

ParticleSystem::VectorDataIterator ParticleSystem::_BeginVectorDataAt(const std::size_t i_index)
{
  return m_vector_data[i_index].begin();
}

ParticleSystem::VectorDataIterator ParticleSystem::_EndVectorDataAt(const std::size_t i_index)
{
  return m_vector_data[i_index].end();
}

ParticleSystem::ScalarDataIteratorC ParticleSystem::_BeginScalarDataAt(const std::size_t i_index) const
{
  return m_scalar_data[i_index].cbegin();
}

ParticleSystem::ScalarDataIteratorC ParticleSystem::_EndScalarDataAt(const std::size_t i_index) const
{
  return m_scalar_data[i_index].cend();
}

ParticleSystem::ScalarDataIterator ParticleSystem::_BeginScalarDataAt(const std::size_t i_index)
{
  return m_scalar_data[i_index].begin();
}

ParticleSystem::ScalarDataIterator ParticleSystem::_EndScalarDataAt(const std::size_t i_index)
{
  return m_scalar_data[i_index].begin();
}