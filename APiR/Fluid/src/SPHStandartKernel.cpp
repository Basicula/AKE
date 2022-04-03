#include "Fluid/SPHStandartKernel.h"
#include "Math/Constants.h"

SPHStandartKernel::SPHStandartKernel()
  : SPHKernel()
  {}

SPHStandartKernel::SPHStandartKernel(double i_kernel_radius)
  : SPHKernel(i_kernel_radius)
  {}

double SPHStandartKernel::operator()(double i_sqr_distance) const
  {
  if (i_sqr_distance >= m_h2)
    return 0.0;
  else
    {
    const double x = 1.0 - i_sqr_distance / m_h2;
    return 315.0 / (64.0 * PI * m_h3) * x * x * x;
    }
  }

double SPHStandartKernel::FirstDerivative(double i_distance) const
  {
  if (i_distance >= m_h)
    return 0.0;
  else
    {
    const double x = 1.0 - i_distance * i_distance / m_h2;
    return -945.0 / (32.0 * PI * m_h5) * i_distance * x * x;
    }
  }

double SPHStandartKernel::SecondDerivative(double i_sqr_distance) const
  {
  if (i_sqr_distance >= m_h2)
    return 0.0;
  else
    {
    double x = i_sqr_distance / m_h2;
    return 945.0 / (32.0 * PI * m_h5) * (1 - x) * (5 * x - 1);
    }
  }

Vector3d SPHStandartKernel::Gradient(const Vector3d& i_point) const
  {
  const double dist = i_point.Length();
  if (dist > 0.0)
    return Gradient(dist, i_point / dist);
  else
    return Vector3d(0, 0, 0);
  }

Vector3d SPHStandartKernel::Gradient(double i_distance, const Vector3d& i_direction) const
  {
  return -i_direction * FirstDerivative(i_distance);
  }
