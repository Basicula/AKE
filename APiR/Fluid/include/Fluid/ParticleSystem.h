#pragma once
#include "Math/Vector.h"

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
  explicit ParticleSystem(std::size_t i_num_particles);

  std::size_t AddVectorData(const Vector3d& i_inittial_value = Vector3d(0));
  std::size_t AddScalarData(double i_initial_value = 0);

  [[nodiscard]] std::size_t GetNumOfParticles() const;

protected:
  void _Resize(std::size_t i_num_particles);

  [[nodiscard]] VectorDataIteratorC _BeginVectorDataAt(std::size_t i_index) const;
  [[nodiscard]] VectorDataIteratorC _EndVectorDataAt(std::size_t i_index) const;
  VectorDataIterator _BeginVectorDataAt(std::size_t i_index);
  VectorDataIterator _EndVectorDataAt(std::size_t i_index);

  [[nodiscard]] ScalarDataIteratorC _BeginScalarDataAt(std::size_t i_index) const;
  [[nodiscard]] ScalarDataIteratorC _EndScalarDataAt(std::size_t i_index) const;
  ScalarDataIterator _BeginScalarDataAt(std::size_t i_index);
  ScalarDataIterator _EndScalarDataAt(std::size_t i_index);

protected:
  std::size_t m_num_particles;

  std::vector<VectorData> m_vector_data;
  std::vector<ScalarData> m_scalar_data;
};
