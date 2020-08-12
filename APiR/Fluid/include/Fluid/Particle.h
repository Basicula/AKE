#pragma once
#include <vector>

#include <Math/Vector.h>

class Particle
  {
  public:
    Particle();
    Particle(
      std::size_t i_num_vector_data, 
      std::size_t i_num_scalar_data);

    std::size_t AddVectorData(const Vector3d& i_vector_prop = Vector3d(0));
    std::size_t AddScalarData(double i_scalar_prop = 0);

    //void SetVectorDataAt(std::size_t i_index, const Vector3d& i_data);
    //Vector3d GetVectorDataAt(std::size_t i_index) const;
    //
    //void SetScalarDataAt(std::size_t i_index, double i_data);
    //double GetScalarDataAt(std::size_t i_index) const;

  protected:
    std::vector<Vector3d> m_vector_data;
    std::vector<double> m_scalar_data;
  };

inline std::size_t Particle::AddVectorData(const Vector3d& i_vector_prop)
  {
  m_vector_data.push_back(i_vector_prop);
  return m_vector_data.size()-1;
  }

inline std::size_t Particle::AddScalarData(double i_scalar_prop)
  {
  m_scalar_data.push_back(i_scalar_prop);
  return m_scalar_data.size() - 1;
  }

//inline void Particle::SetVectorDataAt(std::size_t i_index, const Vector3d& i_data)
//  {
//  m_vector_data[i_index] = i_data;
//  }
//
//inline Vector3d Particle::GetVectorDataAt(std::size_t i_index) const
//  {
//  return m_vector_data[i_index];
//  }
//
//inline void Particle::SetScalarDataAt(std::size_t i_index, double i_data)
//  {
//  m_scalar_data[i_index] = i_data;
//  }
//
//
//inline double Particle::GetScalarDataAt(std::size_t i_index) const
//  {
//  return m_scalar_data[i_index];
//  }