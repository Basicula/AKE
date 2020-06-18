#include <OpenCLKernel.h>

#include <iostream>
#include <vector>

namespace
  {
  void CheckSuccess(const char* i_message, cl_int i_rc)
    {
    if(i_rc == 0)
      return;
    std::cout<<(i_message)<<" : ";
    switch (i_rc)
      {
      case 0: std::cout<<("Success\n"); break;
      case -1: std::cout<<("Device not found\n"); break;
      case -2: std::cout<<("Device not available\n"); break;
      case -3: std::cout<<("Compiler not available\n"); break;
      case -4: std::cout<<("Memory object allocation failure\n"); break;
      case -5: std::cout<<("Out of resources\n"); break;
      case -6: std::cout<<("Out of host memory\n"); break;
      case -7: std::cout<<("Profiling info not available\n"); break;
      case -8: std::cout<<("Memory copy overlap\n"); break;
      case -9: std::cout<<("Image format mismatch\n"); break;
      case -10: std::cout<<("Image format not supported\n"); break;
      case -11: std::cout<<("Build program failure\n"); break;
      case -12: std::cout<<("Map failure\n"); break;
      case -30: std::cout<<("Invalid value\n"); break;
      case -31: std::cout<<("Invaid device type\n"); break;
      case -32: std::cout<<("Invalid platform\n"); break;
      case -33: std::cout<<("Invalid device\n"); break;
      case -34: std::cout<<("Invalid context\n"); break;
      case -35: std::cout<<("Invalid queue properties\n"); break;
      case -36: std::cout<<("Invalid command queue\n"); break;
      case -37: std::cout<<("Invalid host pointer\n"); break;
      case -38: std::cout<<("Invalid memory object\n"); break;
      case -39: std::cout<<("Invalid image format descriptor\n"); break;
      case -40: std::cout<<("Invalid image size\n"); break;
      case -41: std::cout<<("Invalid sampler\n"); break;
      case -42: std::cout<<("Invalid binary\n"); break;
      case -43: std::cout<<("Invalid build options\n"); break;
      case -44: std::cout<<("Invalid program\n"); break;
      case -45: std::cout<<("Invalid program executable\n"); break;
      case -46: std::cout<<("Invalid kernel name\n"); break;
      case -47: std::cout<<("Invalid kernel defintion\n"); break;
      case -48: std::cout<<("Invalid kernel\n"); break;
      case -49: std::cout<<("Invalid argument index\n"); break;
      case -50: std::cout<<("Invalid argument value\n"); break;
      case -51: std::cout<<("Invalid argument size\n"); break;
      case -52: std::cout<<("Invalid kernel arguments\n"); break;
      case -53: std::cout<<("Invalid work dimension\n"); break;
      case -54: std::cout<<("Invalid work group size\n"); break;
      case -55: std::cout<<("Invalid work item size\n"); break;
      case -56: std::cout<<("Invalid global offset\n"); break;
      case -57: std::cout<<("Invalid event wait list\n"); break;
      case -58: std::cout<<("Invalid event\n"); break;
      case -59: std::cout<<("Invalid operation\n"); break;
      case -60: std::cout<<("Invalid GL object\n"); break;
      case -61: std::cout<<("Invalid buffer size\n"); break;
      case -62: std::cout<<("Invalid mip level\n"); break;
      case -63: std::cout<<("Invalid global work size\n"); break;
      default:
        break;
      }
    }
  }


OpenCLKernel::OpenCLKernel()
  : m_source_code("")
  , m_device_mode(DeviceMode::ALL)
  , m_platform_count(0)
  , m_queue_count(0)
  , m_mandelbrot_buffers_created(false)
  {

  }

OpenCLKernel::~OpenCLKernel()
  {
  clReleaseMemObject(md_picture);
  }

void OpenCLKernel::PrintInfo()
  {
  char message[mg_max_info_size];
  size_t message_length;
  for (auto platform = 0u; platform < m_platform_count; ++platform)
    {
    std::cout << "Platform " << platform << std::endl;

    clGetPlatformInfo(m_all_platforms[platform], CL_PLATFORM_NAME, mg_max_info_size, message, &message_length);
    std::cout << "  Name..............: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(m_all_platforms[platform], CL_PLATFORM_VERSION, mg_max_info_size, message, &message_length);
    std::cout << "  Version...........: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(m_all_platforms[platform], CL_PLATFORM_VENDOR, mg_max_info_size, message, &message_length);
    std::cout << "  Vendor............: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(m_all_platforms[platform], CL_PLATFORM_PROFILE, mg_max_info_size, message, &message_length);
    std::cout << "  Profile...........: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(m_all_platforms[platform], CL_PLATFORM_EXTENSIONS, mg_max_info_size, message, &message_length);
    std::cout << "  Extensions........: " << std::string(message, message + message_length) << std::endl;

    for (cl_uint device = 0; device < m_number_of_devices[platform]; ++device)
      {
      std::cout << "  --------------------------------------\n";
      std::cout << "  Device " << device << " id=" << m_devices[platform][device];
      std::cout << "  --------------------------------------\n";
      std::string deviceDescription;
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_NAME, sizeof(message), message, NULL);
      std::cout << "    Name............: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_VENDOR, sizeof(message), message, NULL);
      std::cout << "    Vendor..........: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_VERSION, sizeof(message), message, NULL);
      std::cout << "    Version.........: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_devices[platform][device], CL_DRIVER_VERSION, sizeof(message), message, NULL);
      std::cout << "    Driver version..: " << std::string(message, message + message_length) << std::endl;

      cl_uint value;
      cl_uint values[10];
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, NULL);
      std::cout << "    Compute units...: " << value << std::endl;
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), &value, NULL);
      std::cout << "    Work item dims..: " << value << std::endl;
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(values), &values, NULL);
      std::cout << "    Work item size..: " << values[0] << ", " << values[1] << ", " << values[2] << std::endl;
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), &value, NULL);
      std::cout << "    Clock frequency.: " << value << " Hz\n";

      cl_device_type infoType;
      clGetDeviceInfo(m_devices[platform][device], CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL);
      if (infoType & CL_DEVICE_TYPE_DEFAULT)
        {
        infoType &= ~CL_DEVICE_TYPE_DEFAULT;
        std::cout << "    Type............: Default\n";
        }
      if (infoType & CL_DEVICE_TYPE_CPU)
        {
        infoType &= ~CL_DEVICE_TYPE_CPU;
        std::cout << "    Type............: CPU\n";
        }
      if (infoType & CL_DEVICE_TYPE_GPU)
        {
        infoType &= ~CL_DEVICE_TYPE_GPU;
        std::cout << "    Type............: GPU\n";
        }
      if (infoType & CL_DEVICE_TYPE_ACCELERATOR)
        {
        infoType &= ~CL_DEVICE_TYPE_ACCELERATOR;
        std::cout << "    Type............: Accelerator\n";
        }
      if (infoType != 0)
        std::cout << "    Type............: Unknown " << infoType << std::endl;
      }
    }
  }

cl_int OpenCLKernel::_InitPlatforms()
  {
  auto rc = clGetPlatformIDs(mg_max_platforms, m_all_platforms, &m_platform_count);
  m_platform_idx = 0;

  // Log error in finding platforms, some troubles with OpenCL
  CheckSuccess("Find platforms",rc);

  std::cout << "Were found " << m_platform_count << " platforms\n";

  return CL_SUCCESS;
  }

cl_int OpenCLKernel::_InitDevices()
  {
  for (auto platform = 0u; platform < m_platform_count; ++platform)
    clGetDeviceIDs(m_all_platforms[platform], CL_DEVICE_TYPE_ALL, mg_max_devices, m_devices[platform], &m_number_of_devices[platform]);

  std::cout << "Devices were found ans set for each platform\n";

  return CL_SUCCESS;
  }

cl_int OpenCLKernel::_InitContext()
  {
  cl_int rc;
  m_context = clCreateContext(nullptr, m_number_of_devices[m_platform_idx], m_devices[m_platform_idx], nullptr, nullptr, &rc);
  CheckSuccess("Creating context", rc);

  m_queue_count = m_number_of_devices[m_platform_idx];
  for(auto i = 0u; i < m_queue_count; ++i)
    {
    m_queue[i] = clCreateCommandQueue(m_context, m_devices[m_platform_idx][i], 0, &rc);
    CheckSuccess("Create queue", rc);
    }

  return CL_SUCCESS;
  }

bool OpenCLKernel::Init()
  {
  if (_InitPlatforms() != CL_SUCCESS)
    return false;

  if (_InitDevices() != CL_SUCCESS)
    return false;

  if (_InitContext() != CL_SUCCESS)
    return false;

  return true;
  }

bool OpenCLKernel::Build()
  {
  const size_t source_length = m_source_code.length();
  const auto* raw_source_data = m_source_code.data();
  
  cl_int rc;
  m_program = clCreateProgramWithSource(m_context, 1, &raw_source_data, &source_length, &rc);
  CheckSuccess("Create program", rc);

  rc = clBuildProgram(m_program, m_number_of_devices[m_platform_idx], m_devices[m_platform_idx], "", 0, nullptr);
  CheckSuccess("Building program", rc);

  mk_mandelbrot = clCreateKernel(m_program, "mandelbrot_set", &rc);
  CheckSuccess("Mandelbrot kernel creation", rc);

  return true;
  }

void OpenCLKernel::MandelbrotSetInit(std::size_t i_width, std::size_t i_height, std::size_t i_max_iterations)
  {
  cl_int rc;

  const auto bytes_per_pixel = 4;
  const auto one_dim_picture_size = i_width * i_height * bytes_per_pixel;

  m_mandelbrot.m_width = i_width;
  m_mandelbrot.m_height = i_height;
  m_mandelbrot.m_max_iterations = i_max_iterations;

  if(!m_mandelbrot_buffers_created)
    {
    md_picture = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, one_dim_picture_size * sizeof(unsigned char), nullptr, &rc);
    CheckSuccess("Create output buffer",rc);
    md_color_map = clCreateBuffer(m_context, CL_MEM_READ_ONLY, sizeof(m_mandelbrot.m_color_map), nullptr, &rc);
    CheckSuccess("Create color map buffer", rc);
    m_mandelbrot_buffers_created = true;
    }
  }

std::vector<unsigned char> OpenCLKernel::MandelbrotSetRender()
  {
  cl_int rc;

  const int bytes_per_pixel = 4;
  const auto width = m_mandelbrot.m_width;
  const auto height = m_mandelbrot.m_height;
  const auto one_dim_picture_size = width * height * bytes_per_pixel;

  rc = clSetKernelArg(mk_mandelbrot, 0, sizeof(int), &m_mandelbrot.m_max_iterations);
  CheckSuccess("Set max iterations for mandelbrot kernel", rc);

  rc = clSetKernelArg(mk_mandelbrot, 1, sizeof(cl_mem), &md_color_map);
  CheckSuccess("Set color map memory for mandelbrot kernel", rc);

  rc = clSetKernelArg(mk_mandelbrot, 2, sizeof(cl_mem), &md_picture);
  CheckSuccess("Set picture memory for mandelbrot kernel", rc);

  auto picture = new std::uint8_t[one_dim_picture_size];
  size_t global_size[2] = { width, height };
  size_t local_size[2] = { 1,1 };
  size_t device_size[2] = { width, height / m_number_of_devices[m_platform_idx] };
  for (auto i = 0u; i < m_queue_count; ++i)
    {
    rc = clEnqueueWriteBuffer(m_queue[i], md_color_map, CL_FALSE, 0, sizeof(m_mandelbrot.m_color_map), m_mandelbrot.m_color_map, 0, nullptr, nullptr);
    CheckSuccess("Write color map info", rc);

    size_t device_offset[2] = { 0, device_size[1] * i };
    rc = clEnqueueNDRangeKernel(m_queue[i], mk_mandelbrot, 2, device_offset, device_size, local_size, 0, nullptr, nullptr);
    CheckSuccess("Run kernel", rc);

    size_t offset = device_offset[1] * 4 * width;
    rc = clEnqueueReadBuffer(m_queue[i], md_picture, CL_FALSE, offset, one_dim_picture_size * sizeof(std::uint8_t) / m_number_of_devices[m_platform_idx], picture, 0, nullptr, nullptr);
    CheckSuccess("Read picture",rc);
    }
  std::vector<unsigned char> res(picture, picture + one_dim_picture_size);
  delete[] picture;
  picture = nullptr;

  for (auto i = 0u; i < m_queue_count; ++i)
    {
    clFlush(m_queue[i]);
    clFinish(m_queue[i]);
    }

  return res;
  }

std::vector<unsigned char> OpenCLKernel::Dummy()
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
  m_program = clCreateProgramWithSource(m_context, 1, &raw_data, &length, &rc);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Creating kernel program failed\n";
    }
  rc = clBuildProgram(m_program, 1, &m_devices[m_platform_idx][0], "", 0, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Build kernel program failed\n";
    }

  cl_kernel kernel = clCreateKernel(m_program, "create_dummy_picture", &rc);

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
  
  rc = clEnqueueWriteBuffer(m_queue[0], d_randoms, CL_TRUE, 0, width * height * sizeof(int), random_numbers.data(), 0, nullptr, nullptr);
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
  rc = clEnqueueNDRangeKernel(m_queue[0], kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "ND kernel program failed\n";
    }
  clFinish(m_queue[0]);
  auto picture = new cl_uchar[width * height * bytes_per_pixel];
  clEnqueueReadBuffer(m_queue[0], d_c, CL_TRUE, 0, width * height * bytes_per_pixel * sizeof(cl_uchar), picture, 0, nullptr, nullptr);
  std::vector<unsigned char> res(picture,picture+width*height*bytes_per_pixel);
  delete[] picture;
  picture = nullptr;
  return res;
  }
