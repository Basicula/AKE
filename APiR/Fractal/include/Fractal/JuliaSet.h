#pragma once
#include "Fractal/Fractal.h"

class JuliaSet final: public Fractal
{
public:
  enum class JuliaSetType
  {
    SpirallyBlob,
    WhiskeryDragon,
    SeparatedWhorls,
    RomanescoBroccoli
  };

public:
  JuliaSet(std::size_t i_width, std::size_t i_height, std::size_t i_iterations = 1000);

  HOSTDEVICE [[nodiscard]] size_t GetValue(int i_x, int i_y) const override;

  void SetType(JuliaSetType i_type);

  void SetCustomStart(float i_cx, float i_cy);

protected:
  HOSTDEVICE void _InitFractalRange() override;

  void _ResetStart();

private:
  JuliaSetType m_type;

  float m_cx;
  float m_cy;
};
