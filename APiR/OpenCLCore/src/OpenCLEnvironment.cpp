#include <iostream>
#include <vector>

#include <OpenCLEnvironment.h>
#include <OpenCLUtils.h>

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

std::vector<unsigned char> OpenCLEnvironment::Dummy()
  {
  std::string kernel_code =
    "void kernel create_dummy_picture(int width, int height, global uchar* picture,global int* k){"
    "  int i = get_global_id(0);"
    "  int j = get_global_id(1);"
    "  int x = i * width * 4;"
    "  int y = j * 4;"
    "  int const a1 = 16887;"
    "  int const a2 = 78125;"
    "  int const m = 2147483647;"
    "  int r = (k[i*width+j] * a1 + a2) % m;"
    "  int g = (r * a1 + a2) % m;"
    "  int b = (g * a1 + a2) % m;"
    "  picture[x + y + 0] = r % 256;"
    "  picture[x + y + 1] = g % 256;"
    "  picture[x + y + 2] = b % 256;"
    "  picture[x + y + 3] = 255;"
    "}";
  size_t length = kernel_code.length();
  const char* raw_data = kernel_code.data();
  cl_int rc;
  auto program = clCreateProgramWithSource(m_context, 1, &raw_data, &length, &rc);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Creating kernel program failed\n";
    }
  rc = clBuildProgram(program, 1, &m_devices[m_platform_idx][0], "", 0, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Build kernel program failed\n";
    }

  cl_kernel kernel = clCreateKernel(program, "create_dummy_picture", &rc);

  const int width = 256;
  const int height = 256;
  const int bytes_per_pixel = 4;
  std::vector<int> random_numbers(width*height);
  for(auto& number : random_numbers)
    number = rand() % RAND_MAX;

  cl_mem d_c = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, width * height * bytes_per_pixel * sizeof(cl_uchar), nullptr, &rc);
  cl_mem d_randoms = clCreateBuffer(m_context, CL_MEM_READ_ONLY, width * height * sizeof(int), nullptr, &rc);

  if (!d_c || !d_randoms)
    {
    std::cout << "Error: Failed to allocate device memory!\n";
    }
  
  rc = clEnqueueWriteBuffer(m_queues[0], d_randoms, CL_TRUE, 0, width * height * sizeof(int), random_numbers.data(), 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Error: Failed to write!\n";
    }

  rc = clSetKernelArg(kernel, 0, sizeof(int), &width);
  if (rc != CL_SUCCESS)
    {
    std::cout << "0 kernel program failed\n";
    }
  rc = clSetKernelArg(kernel, 1, sizeof(int), &width);
  if (rc != CL_SUCCESS)
    {
    std::cout << "1 kernel program failed\n";
    }
  rc = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  if (rc != CL_SUCCESS)
    {
    std::cout << "2 kernel program failed\n";
    }
  rc = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_randoms);
  if (rc != CL_SUCCESS)
    {
    std::cout << "2 kernel program failed\n";
    }
  size_t global_size[2] = { width, height };
  size_t local_size[2] = { 1, 1 };
  rc = clEnqueueNDRangeKernel(m_queues[0], kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "ND kernel program failed\n";
    }
  clFinish(m_queues[0]);
  auto picture = new cl_uchar[width * height * bytes_per_pixel];
  clEnqueueReadBuffer(m_queues[0], d_c, CL_TRUE, 0, width * height * bytes_per_pixel * sizeof(cl_uchar), picture, 0, nullptr, nullptr);
  std::vector<unsigned char> res(picture,picture+width*height*bytes_per_pixel);
  delete[] picture;
  picture = nullptr;
  return res;
  }
