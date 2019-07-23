#pragma once
#include <string>
#include <vector>

#include <CL/cl.h>

class OpenCLKernel
  {
  public:
    static const size_t mg_max_platforms = 16;
    static const size_t mg_max_devices = 16;
    static const size_t mg_max_info_size = 256 * 256;

  public:
    OpenCLKernel();
    ~OpenCLKernel();

    // return 1 if OK, and 0 if operation is impossible
    inline bool SetPlatform(cl_uint i_platform);
    inline bool SetDevice(cl_uint i_device);

    // set kernel code source file
    inline void SetKernelSource(const std::string& i_file_name) { m_file_name = i_file_name; };

    // init information for correct OpenCL's work
    bool Init();
    void PrintInfo();
    void Test();
    std::vector<unsigned char> Dummy();
  private:

    // init platforms and set defalt platform
    cl_int _InitPlatforms();
    cl_int _InitDevices();
    cl_int _InitContext();

  private:
    std::string m_file_name;

    cl_uint m_platform_count;
    cl_uint m_platform;
    cl_platform_id m_all_platforms[mg_max_platforms];
    cl_platform_id m_current_platform;

    cl_uint m_number_of_devices[mg_max_platforms];
    cl_uint m_device;
    cl_device_id m_all_devices[mg_max_platforms][mg_max_devices];
    cl_device_id m_current_device;

    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;
  };