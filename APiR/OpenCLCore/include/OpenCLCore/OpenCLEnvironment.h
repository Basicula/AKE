#pragma once
#include "OpenCLCore/Kernel.h"

#include <CL/cl.h>

class OpenCLEnvironment
  {
  public:
    enum class DeviceType
      {
      GPU = 0,
      CPU = 1,
      ALL = 2,
      };

  public:
    OpenCLEnvironment(DeviceType i_device_mode = DeviceType::ALL);
    ~OpenCLEnvironment() = default;

    /// init information for correct OpenCL's work
    bool Init();
    /// Build certain kernel in the environment
    bool Build(Kernel& io_kernel);
    /// Execute kernel after build
    bool Execute(Kernel& io_kernel);

    /// print info about devices etc
    void PrintInfo();
    /// Log some info about success of kernel processing
    void SetLoggingState(bool i_is_on);

  private:
    /// Init platforms and set defalt platform
    cl_int _InitPlatforms();
    /// Init devices for every platform
    cl_int _InitDevices();
    /// Init context for chosen platform for further kernel execution
    cl_int _InitContext();

  private:
    static const std::size_t mg_max_platforms = 16;
    static const std::size_t mg_max_devices_for_platform = 16;
    static const std::size_t mg_max_devices = 16 * 16;
    static const std::size_t mg_max_info_size = 64 * 64;

  private:
    bool m_enable_logging;
    cl_device_type m_device_type;

    cl_uint m_platform_count;
    cl_uint m_platform_idx;
    cl_platform_id m_all_platforms[mg_max_platforms];

    cl_uint m_number_of_devices[mg_max_platforms];
    cl_device_id m_devices[mg_max_platforms][mg_max_devices_for_platform];

    cl_context m_context;
    cl_uint m_queue_count;
    cl_command_queue m_queues[mg_max_devices];
  };

inline void OpenCLEnvironment::SetLoggingState(bool i_is_on)
  {
  m_enable_logging = i_is_on;
  }