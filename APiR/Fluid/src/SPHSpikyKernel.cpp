#include <DefinesAndConstants.h>
#include <SPHSpikyKernel.h>

SPHSpikyKernel::SPHSpikyKernel()
  : SPHKernel()
  {}

SPHSpikyKernel::SPHSpikyKernel(double i_kernel_radius)
  : SPHKernel(i_kernel_radius)
  {}

double SPHSpikyKernel::operator()(double i_distance) const
  {
  if (i_distance >= m_h)
    return 0.0;
  else
    {
    double x = 1.0 - i_distance / m_h;
    return 15.0 / (PI* m_h3) * x * x * x;
    }
  }

double SPHSpikyKernel::FirstDerivative(double i_distance) const
  {
  if (i_distance >= m_h)
    return 0.0;
  else
    {
    double x = 1.0 - i_distance / m_h;
    return -45.0 * x * x / (PI * m_h4);
    }
  }

double SPHSpikyKernel::SecondDerivative(double i_distance) const
  {
  if (i_distance >= m_h)
    return 0.0;
  else
    {
    double x = 1.0 - i_distance / m_h;
    return 90.0 / (PI * m_h5) * x;
    }
  }

Vector3d SPHSpikyKernel::Gradient(const Vector3d& i_point) const
  {
  const double dist = i_point.Length();
  if (dist > 0.0)
    return Gradient(dist, i_point / dist);
  else
    return Vector3d(0, 0, 0);
  }

Vector3d SPHSpikyKernel::Gradient(double i_distance, const Vector3d& i_direction) const
  {
  return -i_direction * FirstDerivative(i_distance);
  }
