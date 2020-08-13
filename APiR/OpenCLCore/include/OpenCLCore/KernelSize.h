#pragma once
#include <cstdlib>

class KernelSize
  {
  public:
    // invalid size
    KernelSize();
    KernelSize(std::size_t i_size);
    KernelSize(std::size_t i_width, std::size_t i_height);
    KernelSize(std::size_t i_width, std::size_t i_height, std::size_t i_depth);
    KernelSize(const KernelSize& i_other);

    const std::size_t* AsArray() const;
    const std::size_t& operator[](std::size_t i_id) const;
    std::size_t GetWorkDimension() const;
    std::size_t Size() const;

  private:
    std::size_t* m_dims;
    std::size_t m_work_dimension;
  };

inline const std::size_t& KernelSize::operator[](std::size_t i_id) const
  {
  return m_dims[i_id];
  }

inline const std::size_t* KernelSize::AsArray() const
  {
  return m_dims;
  }

inline std::size_t KernelSize::GetWorkDimension() const
  {
  return m_work_dimension;
  }
