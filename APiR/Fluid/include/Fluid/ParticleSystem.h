#pragma once
#include <Math/Vector.h>

#include <vector>

class ParticleSystem
  {
  public:
    using VectorData = std::vector<Vector3d>;
    using VectorDataIterator = VectorData::iterator;
    using VectorDataIteratorC = VectorData::const_iterator;

    using ScalarData = std::vector<double>;
    using ScalarDataIterator = ScalarData::iterator;
    using ScalarDataIteratorC = ScalarData::const_iterator;

  public:
    ParticleSystem();
    ParticleSystem(std::size_t i_num_particles);

    std::size_t AddVectorData(const Vector3d& i_inittial_value = Vector3d(0));
    std::size_t AddScalarData(double i_initial_value = 0);

    std::size_t GetNumOfParticles() const;

  protected:
    void _Resize(std::size_t i_num_particles);

    VectorDataIteratorC _BeginVectorDataAt(std::size_t i_index) const;
    VectorDataIteratorC _EndVectorDataAt(std::size_t i_index) const;
    VectorDataIterator _BeginVectorDataAt(std::size_t i_index);
    VectorDataIterator _EndVectorDataAt(std::size_t i_index);

    ScalarDataIteratorC _BeginScalarDataAt(std::size_t i_index) const;
    ScalarDataIteratorC _EndScalarDataAt(std::size_t i_index) const;
    ScalarDataIterator _BeginScalarDataAt(std::size_t i_index);
    ScalarDataIterator _EndScalarDataAt(std::size_t i_index);

  protected:
    std::size_t m_num_particles;

    std::vector<VectorData> m_vector_data;
    std::vector<ScalarData> m_scalar_data;
  };

inline std::size_t ParticleSystem::AddVectorData(const Vector3d& i_inittial_value)
  {
  const auto id = m_vector_data.size();
  m_vector_data.emplace_back(m_num_particles, i_inittial_value);
  return id;
  }

inline std::size_t ParticleSystem::AddScalarData(double i_initial_value)
  {
  const auto id = m_scalar_data.size();
  m_scalar_data.emplace_back(m_num_particles, i_initial_value);
  return id;
  }

inline std::size_t ParticleSystem::GetNumOfParticles() const
  {
  return m_num_particles;
  }

inline auto ParticleSystem::_BeginVectorDataAt(std::size_t i_index) const
-> ParticleSystem::VectorDataIteratorC
  {
  return m_vector_data[i_index].cbegin();
  }

inline auto ParticleSystem::_EndVectorDataAt(std::size_t i_index) const
-> ParticleSystem::VectorDataIteratorC
  {
  return m_vector_data[i_index].cend();
  }

inline auto ParticleSystem::_BeginVectorDataAt(std::size_t i_index)
-> ParticleSystem::VectorDataIterator
  {
  return m_vector_data[i_index].begin();
  }

inline auto ParticleSystem::_EndVectorDataAt(std::size_t i_index)
-> ParticleSystem::VectorDataIterator
  {
  return m_vector_data[i_index].end();
  }

inline auto ParticleSystem::_BeginScalarDataAt(std::size_t i_index) const
-> ParticleSystem::ScalarDataIteratorC
  {
  return m_scalar_data[i_index].cbegin();
  }

inline auto ParticleSystem::_EndScalarDataAt(std::size_t i_index) const
-> ParticleSystem::ScalarDataIteratorC
  {
  return m_scalar_data[i_index].cend();
  }

inline auto ParticleSystem::_BeginScalarDataAt(std::size_t i_index)
-> ParticleSystem::ScalarDataIterator
  {
  return m_scalar_data[i_index].begin();
  }

inline auto ParticleSystem::_EndScalarDataAt(std::size_t i_index)
-> ParticleSystem::ScalarDataIterator
  {
  return m_scalar_data[i_index].begin();
  }