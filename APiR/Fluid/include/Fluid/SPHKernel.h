#pragma once
#include "Math/Vector.h"

class SPHKernel
{
public:
  SPHKernel();
  explicit SPHKernel(double i_kernel_radius);
  virtual ~SPHKernel() = default;

  // Returns kernel function value at given distance.
  virtual double operator()(double i_value) const = 0;

  // Returns the first derivative at given distance.
  [[nodiscard]] virtual double FirstDerivative(double i_value) const = 0;

  // Returns the second derivative at given distance.
  [[nodiscard]] virtual double SecondDerivative(double i_value) const = 0;

  // Returns the gradient at a point.
  [[nodiscard]] virtual Vector3d Gradient(const Vector3d& i_point) const = 0;

  //! Returns the gradient at a point defined by distance and direction.
  [[nodiscard]] virtual Vector3d Gradient(double i_value, const Vector3d& i_direction) const = 0;

protected:
  double m_h;
  double m_h2;
  double m_h3;
  double m_h4;
  double m_h5;
};