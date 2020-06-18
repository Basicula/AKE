#include <Particle.h>

Particle::Particle()
  : Particle(0, 0)
  {}

Particle::Particle(
  std::size_t i_num_vector_data,
  std::size_t i_num_scalar_data)
  : m_vector_data(i_num_vector_data)
  , m_scalar_data(i_num_scalar_data)
  {}