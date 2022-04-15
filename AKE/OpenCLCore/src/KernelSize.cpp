#include "OpenCLCore/KernelSize.h"

#include <algorithm>

KernelSize::KernelSize()
  : m_work_dimension(0)
  , m_dims(nullptr)
  {
  }

KernelSize::KernelSize(std::size_t i_size)
  : m_work_dimension(1)
  , m_dims(new std::size_t[1]{ i_size })
  {
  }

KernelSize::KernelSize(
  std::size_t i_width,
  std::size_t i_height)
  : m_work_dimension(2)
  , m_dims(new std::size_t[2]{ i_width, i_height })
  {
  }

KernelSize::KernelSize(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_depth)
  : m_work_dimension(3)
  , m_dims(new std::size_t[3]{ i_width, i_height, i_depth })
  {
  }

KernelSize::KernelSize(const KernelSize& i_other)
  : m_work_dimension(i_other.m_work_dimension)
  {
  std::copy(i_other.m_dims, i_other.m_dims + m_work_dimension, m_dims);
  }

std::size_t KernelSize::Size() const
  {
  std::size_t res = *m_dims;
  for (std::size_t i = 1; i < m_work_dimension; ++i)
    res *= m_dims[i];
  return res;
  }