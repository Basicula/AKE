#pragma once
#include <OpenCLCore/Kernel.h>
#include <Visual/Image.h>

#include <CL/cl.h>

class MandelbrotSetKernel : public Kernel
  {
  public:
    struct MandelbrotSet
      {
      static const std::uint8_t m_color_map[17 * 3];
      std::size_t m_width;
      std::size_t m_height;
      std::size_t m_max_iterations = 1000;
      int m_origin_x;
      int m_origin_y;
      double m_scale;

      MandelbrotSet::MandelbrotSet(
        std::size_t i_width,
        std::size_t i_height,
        std::size_t i_iterations);
      };

  public:
    MandelbrotSetKernel(
      std::size_t i_width,
      std::size_t i_height,
      std::size_t i_max_iterations = 1000);
    ~MandelbrotSetKernel();

    virtual bool InitKernelsForProgram() override;
    virtual void UpdateKernelSizeInfo(std::size_t i_device_cnt) override;
    virtual bool SetKernelArgs() const override;
    virtual void UpdateDeviceOffset(std::size_t i_queue_id) override;
    virtual bool WriteBuffers(const cl_command_queue& i_queue) const override;
    virtual bool ExecuteKernels(const cl_command_queue& i_queue) const override;
    virtual bool CollectResults(const cl_command_queue& i_queue) override;

    void SetOutput(Image& o_image);

    void SetMaxIterations(std::size_t i_iterations);

  protected:
    virtual bool _InitBuffers() override;

  private:
    MandelbrotSet m_mandelbrot_set;

    Image* mp_output_image;

    std::size_t m_dimensions;

    cl_kernel mk_mandelbrot;
    cl_mem md_image;
    cl_mem md_color_map;
  };

inline void MandelbrotSetKernel::SetMaxIterations(std::size_t i_iterations)
  {
  m_mandelbrot_set.m_max_iterations = i_iterations;
  }

inline void MandelbrotSetKernel::SetOutput(Image& o_image)
  {
  mp_output_image = &o_image;
  }