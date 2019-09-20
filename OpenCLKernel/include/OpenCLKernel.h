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
  const unsigned char m_color_map[17 * 3] =
    {
    0, 0, 0,
    66, 45, 15,
    25, 7, 25,
    10, 0, 45,
    5, 5, 73,
    0, 7, 99,
    12, 43, 137,
    22, 81, 175,
    56, 124, 209,
    132, 181, 229,
    209, 234, 247,
    239, 232, 191,
    247, 201, 94,
    255, 170, 0,
    204, 127, 0,
    153, 86, 0,
    104, 51, 2,
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
    void MandelbrotSetInit(size_t i_width, size_t i_height, size_t i_max_iterations = 1000);
    std::vector<unsigned char> MandelbrotSetRender();
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