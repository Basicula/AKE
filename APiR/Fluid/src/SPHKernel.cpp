#include <SPHKernel.h>

SPHKernel::SPHKernel()
  : m_h(0), m_h2(0), m_h3(0), m_h4(0), m_h5(0)
  {}

SPHKernel::SPHKernel(double i_kernel_radius)
  : m_h(i_kernel_radius)
  , m_h2(m_h * m_h)
  , m_h3(m_h2 * m_h)
  , m_h4(m_h2 * m_h2)
  , m_h5(m_h3 * m_h2)
  {}