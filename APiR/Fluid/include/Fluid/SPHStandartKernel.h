#pragma once
#include "Fluid/SPHKernel.h"

class SPHStandartKernel final : public SPHKernel
{
public:
  SPHStandartKernel();
  explicit SPHStandartKernel(double i_kernel_radius);

  double operator()(double i_square_distance) const override;

  [[nodiscard]] double FirstDerivative(double i_distance) const override;
  [[nodiscard]] double SecondDerivative(double i_square_distance) const override;

  [[nodiscard]] Vector3d Gradient(const Vector3d& i_point) const override;
  [[nodiscard]] Vector3d Gradient(double i_distance, const Vector3d& i_direction) const override;
};