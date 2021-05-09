#pragma once
#include <OpenCLCore/Kernel.h>
#include <Image/Image.h>

class MandelbrotSetKernel : public Kernel
  {
  public:
    MandelbrotSetKernel(
      std::size_t i_width,
      std::size_t i_height,
      std::size_t i_max_iterations = 1000);
    ~MandelbrotSetKernel();

    virtual bool InitKernelsForProgram() override;
    virtual bool BuildProgram(
      cl_uint i_num_of_devices, 
      const cl_device_id* i_device_ids) const override;
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
    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_max_iterations;

    Image* mp_output_image;

    cl_kernel mk_mandelbrot;
    cl_mem md_image;
    cl_mem md_color_map;
  };

inline void MandelbrotSetKernel::SetMaxIterations(std::size_t i_iterations)
  {
  m_max_iterations = i_iterations;
  }

inline void MandelbrotSetKernel::SetOutput(Image& o_image)
  {
  mp_output_image = &o_image;
  }