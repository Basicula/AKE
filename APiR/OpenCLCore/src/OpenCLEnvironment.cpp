#include <OpenCLCore/OpenCLEnvironment.h>
#include <OpenCLCore/OpenCLUtils.h>

namespace
  {
  cl_device_type MapDeviceType(OpenCLEnvironment::DeviceType i_device_type)
    {
    switch (i_device_type)
      {
      case OpenCLEnvironment::DeviceType::GPU:
        return CL_DEVICE_TYPE_GPU;
      case OpenCLEnvironment::DeviceType::CPU:
        return CL_DEVICE_TYPE_CPU;
      case OpenCLEnvironment::DeviceType::ALL:
      default:
        return CL_DEVICE_TYPE_ALL;
      }
    }
  }

OpenCLEnvironment::OpenCLEnvironment(DeviceType i_device_type)
  : m_enable_logging(false)
  , m_device_type(CL_DEVICE_TYPE_ALL)
  , m_platform_count(0)
  , m_platform_idx(0)
  , m_all_platforms()
  , m_number_of_devices()
  , m_devices()
  , m_context()
  , m_queue_count(0)
  , m_queues()
  {
  m_device_type = MapDeviceType(i_device_type);
  }

void OpenCLEnvironment::PrintInfo()
  {
  for (std::size_t platform_id = 0; platform_id < m_platform_count; ++platform_id)
    {
    const auto& platform = m_all_platforms[platform_id];
    OpenCLUtils::LogPlatformInfo(platform);
    for (std::size_t device_id = 0; device_id < m_number_of_devices[platform_id]; ++device_id)
      {
      const auto& device = m_devices[platform_id][device_id];
      OpenCLUtils::LogDeviceMainInfo(device);
      OpenCLUtils::LogDeviceCharacteristicsInfo(device);
      OpenCLUtils::LogDeviceTypeInfo(device);
      }
    }
  }

cl_int OpenCLEnvironment::_InitPlatforms()
  {
  auto rc = clGetPlatformIDs(mg_max_platforms, m_all_platforms, &m_platform_count);
  m_platform_idx = 0;

  // Log error in finding platforms, some troubles with OpenCL
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Find platforms", rc);

  return rc;
  }

cl_int OpenCLEnvironment::_InitDevices()
  {
  cl_int rc;
  for (auto platform = 0u; platform < m_platform_count; ++platform)
    {
    rc = clGetDeviceIDs(
      m_all_platforms[platform], 
      m_device_type, 
      mg_max_devices, 
      m_devices[platform], 
      &m_number_of_devices[platform]);

    if (m_enable_logging)
      OpenCLUtils::CheckSuccess("Get devices for platform with id = " + std::to_string(platform), rc);

    if (rc != CL_SUCCESS)
      return rc;
    }

  return CL_SUCCESS;
  }

cl_int OpenCLEnvironment::_InitContext()
  {
  cl_int rc;
  m_context = clCreateContext(nullptr, m_number_of_devices[m_platform_idx], m_devices[m_platform_idx], nullptr, nullptr, &rc);
  
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Creating context", rc);
  if (rc != CL_SUCCESS)
    return rc;

  m_queue_count = m_number_of_devices[m_platform_idx];
  for(auto i = 0u; i < m_queue_count; ++i)
    {
    m_queues[i] = clCreateCommandQueue(m_context, m_devices[m_platform_idx][i], 0, &rc);

    if (m_enable_logging)
      OpenCLUtils::CheckSuccess("Create queue", rc);
    if (rc != CL_SUCCESS)
      return rc;
    }

  return CL_SUCCESS;
  }

bool OpenCLEnvironment::Init()
  {
  if (_InitPlatforms() != CL_SUCCESS)
    return false;

  if (_InitDevices() != CL_SUCCESS)
    return false;

  if (_InitContext() != CL_SUCCESS)
    return false;

  return true;
  }

bool OpenCLEnvironment::Build(Kernel& io_kernel)
  {
  io_kernel.InitProgramForContext(m_context);
  const auto& program = io_kernel.GetProgram();

  cl_int rc = clBuildProgram(program, m_number_of_devices[m_platform_idx], m_devices[m_platform_idx], "", 0, nullptr);
  
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Building program", rc);

  io_kernel.InitKernelsForProgram();
  io_kernel.UpdateKernelSizeInfo(m_number_of_devices[m_platform_idx]);

  return (rc == CL_SUCCESS);
  }

bool OpenCLEnvironment::Execute(Kernel& io_kernel)
  {
  if (!io_kernel.SetKernelArgs())
    return false;

  for (std::size_t i = 0; i < m_queue_count; ++i)
    {
    const auto& queue = m_queues[i];
    io_kernel.UpdateDeviceOffset(i);
    if (!io_kernel.WriteBuffers(queue))
      return false;
    if (!io_kernel.ExecuteKernels(queue))
      return false;
    if (!io_kernel.CollectResults(queue))
      return false;
    }

  cl_int rc;
  for (auto i = 0u; i < m_queue_count; ++i)
    {
    rc = clFlush(m_queues[i]);
    if (m_enable_logging)
      OpenCLUtils::CheckSuccess("Flush queue", rc);
    if (rc != CL_SUCCESS)
      return false;

    clFinish(m_queues[i]);
    if (m_enable_logging)
      OpenCLUtils::CheckSuccess("Finish queue", rc);
    if (rc != CL_SUCCESS)
      return false;
    }
  return true;
  }
