#pragma once
#include <Fractal/Fractal.h>

class JuliaSet : public Fractal
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
    JuliaSet(
      std::size_t i_width,
      std::size_t i_height,
      std::size_t i_iterations = 1000);

    virtual size_t GetValue(int i_x, int i_y) const override;

    void SetType(JuliaSetType i_type);

    void SetCustomStart(double i_cx, double i_cy);

  protected:
    virtual void _InitFractalRange() override;

    void _ResetStart();

  private:
    JuliaSetType m_type;

    double m_cx;
    double m_cy;
  };

inline void JuliaSet::SetCustomStart(double i_cx, double i_cy)
  {
  m_cx = i_cx;
  m_cy = i_cy;
  }