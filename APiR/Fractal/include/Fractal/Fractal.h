#pragma once
#include <Macro/CudaMacro.h>
#include <Fractal/ColorMap.h>

#include <memory>

class Fractal
  {
  public:
    virtual ~Fractal() = default;

    HOSTDEVICE virtual Color GetColor(int i_x, int i_y) const = 0;

    void SetColorMap(const ColorMap* i_color_map);
    void SetMaxIterations(std::size_t i_max_iterations);
    void SetScale(double i_scale);
    void SetOrigin(double i_origin_x, double i_origin_y);

  protected:
    HOSTDEVICE Fractal(
      std::size_t i_width,
      std::size_t i_height, 
      std::size_t i_max_iterations = 1000);

    HOSTDEVICE void _MapCoordinate(double& o_x, double& o_y, int i_x, int i_y) const;

    HOSTDEVICE virtual void _InitFractalRange() = 0;

  protected:
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_max_iterations;
    double m_origin_x;
    double m_origin_y;
    double m_scale;

    double m_x_min;
    double m_x_max;
    double m_y_min;
    double m_y_max;

    const ColorMap* m_color_map;
  };

inline void Fractal::SetColorMap(const ColorMap* ip_color_map)
  {
  m_color_map = ip_color_map;
  }

inline void Fractal::SetMaxIterations(std::size_t i_max_iterations)
  {
  m_max_iterations = i_max_iterations;
  }

inline void Fractal::SetScale(double i_scale)
  {
  m_scale = i_scale;
  }

inline void Fractal::SetOrigin(double i_origin_x, double i_origin_y)
  {
  m_origin_x = i_origin_x;
  m_origin_y = i_origin_y;
  }