#pragma once
#include "Fluid/SPHKernel.h"

class SPHSpikyKernel : public SPHKernel
{
public:
  SPHSpikyKernel();
  SPHSpikyKernel(double i_kernel_radius);

  virtual double operator()(double i_distance) const override;

  virtual double FirstDerivative(double i_distance) const override;

  virtual double SecondDerivative(double i_distance) const override;

  virtual Vector3d Gradient(const Vector3d& i_point) const override;

  virtual Vector3d Gradient(double i_distance, const Vector3d& i_direction) const override;
};