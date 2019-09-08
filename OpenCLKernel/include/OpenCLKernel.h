#pragma once
#include <string>
#include <vector>

#include <CL/cl.h>

class OpenCLKernel
  {
  public:
    enum DeviceMode
      {
      GPU = 0,
      CPU = 1,
      ALL = 2,
      };

  public:
    static const size_t mg_max_platforms = 16;
    static const size_t mg_max_devices_for_platform = 16;
    static const size_t mg_max_devices = 16 * 16;
    static const size_t mg_max_info_size = 256 * 256;

  public:
    OpenCLKernel();
    ~OpenCLKernel();

    // set kernel code source file
    inline void SetKernelSource(const std::string& i_source) { m_source_code = i_source; };

    // init information for correct OpenCL's work
    bool Init();
    bool Build();

    // print info about devices etc
    void PrintInfo();

    // temporary test features
    void Test();
    std::vector<unsigned char> Dummy();
    std::vector<unsigned char> MandelbrotSet(size_t i_width, size_t i_height, size_t i_max_iterations = 1000);
  private:

    // init platforms and set defalt platform
    cl_int _InitPlatforms();
    cl_int _InitDevices();
    cl_int _InitContext();

  private:
    std::string m_source_code;
    DeviceMode m_device_mode;

    cl_uint m_platform_count;
    cl_uint m_platform_idx;
    cl_platform_id m_all_platforms[mg_max_platforms];

    cl_uint m_number_of_devices[mg_max_platforms];
    cl_device_id m_devices[mg_max_platforms][mg_max_devices_for_platform];

    cl_context m_context;
    cl_uint m_queue_count;
    cl_command_queue m_queue[mg_max_devices];
    cl_program m_program;

    cl_kernel mk_mandelbrot;
    cl_mem md_picture;
    bool m_picture_buffer_created;
  };