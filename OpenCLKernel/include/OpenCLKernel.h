#pragma once
#include <string>
#include <vector>

#include <CL/cl.h>

struct Mandelbrot
  {
  int m_width = 1024;
  int m_height = 768;
  int m_origin_x = 0;
  int m_origin_y = 0;
  int m_max_iterations = 256;
  int m_scale = 1;
  const float m_color_map[17 * 3] =
    {
      0.0,  0.0,  0.0,
      0.26, 0.18, 0.06,
      0.1,  0.03, 0.1,
      0.04, 0.0,  0.18,
      0.02, 0.02, 0.29,
      0.0,  0.03, 0.39,
      0.05, 0.17, 0.54,
      0.09, 0.32, 0.69,
      0.22, 0.49, 0.82,
      0.52, 0.71, 0.9,
      0.82, 0.92, 0.97,
      0.94, 0.91, 0.75,
      0.97, 0.79, 0.37,
      1.0,  0.67, 0.0,
      0.8,  0.5,  0.0,
      0.6,  0.34, 0.0,
      0.41, 0.2,  0.01
    };
  };

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
    void MandelbrotSetBegin(size_t i_width, size_t i_height, size_t i_max_iterations = 1000);
    std::vector<unsigned char> MandelbrotSetEnd();
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
    Mandelbrot m_mandelbrot;
    cl_mem md_picture;
    cl_mem md_color_map;
    bool m_mandelbrot_buffers_created;
  };