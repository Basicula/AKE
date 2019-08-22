#include <OpenCLKernel.h>

#include <iostream>
#include <vector>

OpenCLKernel::OpenCLKernel()
  : m_file_name("")
  , m_platform_count(0)
  , m_platform(0)
  , m_device(0)
  {

  }

OpenCLKernel::~OpenCLKernel()
  {

  }

cl_int OpenCLKernel::_InitPlatforms()
  {
  auto rc = clGetPlatformIDs(mg_max_platforms, m_all_platforms, &m_platform_count);
  if (rc != CL_SUCCESS)
    {
    // Log error in finding platforms, some troubles with OpenCL
    std::cout << "No platforms were found\n";
    return rc;
    }
  std::cout << "Were found " << m_platform_count << " platforms\n";
  m_current_platform = m_all_platforms[m_platform];
  return CL_SUCCESS;
  }

cl_int OpenCLKernel::_InitDevices()
  {
  for (auto platform = 0; platform < m_platform_count; ++platform)
    {
    clGetDeviceIDs(m_all_platforms[platform], CL_DEVICE_TYPE_ALL, mg_max_devices, m_all_devices[platform], &m_number_of_devices[platform]);
    // Log for platform was found devices
    std::cout << "Platform " << m_all_platforms[platform] << " has " << m_number_of_devices[platform] << " devices\n";
    }
  if (false)
    {
    std::cout << "No avalible devices for current platform\n";
    return CL_DEVICE_NOT_FOUND;
    }
  std::cout << "Devices were found ans set for each platform\n";
  m_current_device = m_all_devices[m_platform][m_device];
  return CL_SUCCESS;
  }

cl_int OpenCLKernel::_InitContext()
  {
  cl_int rc;
  std::cout << "Context\n";
  m_context = clCreateContext(nullptr, m_number_of_devices[m_platform], m_all_devices[m_platform], nullptr, nullptr, &rc);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Something wrong with creating context\n";
    return rc;
    }
  std::cout << "Context was successfully created\n";
  m_queue = clCreateCommandQueue(m_context, m_current_device, 0, &rc);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Something wrong with creating command queue\n";
    return rc;
    }
  std::cout << "Command queue was successfully created\n";
  return CL_SUCCESS;
  }

void OpenCLKernel::PrintInfo()
  {
  char message[mg_max_info_size];
  size_t message_length;
  for (auto platform = 0; platform < m_platform_count; ++platform)
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
      std::cout << "  Device " << device << " id=" << m_all_devices[platform][device];
      std::cout << "  --------------------------------------\n";
      std::string deviceDescription;
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_NAME, sizeof(message), message, NULL);
      std::cout << "    Name............: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_VENDOR, sizeof(message), message, NULL);
      std::cout << "    Vendor..........: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_VERSION, sizeof(message), message, NULL);
      std::cout << "    Version.........: " << std::string(message, message + message_length) << std::endl;

      clGetDeviceInfo(m_all_devices[platform][device], CL_DRIVER_VERSION, sizeof(message), message, NULL);
      std::cout << "    Driver version..: " << std::string(message, message + message_length) << std::endl;

      cl_uint value;
      cl_uint values[10];
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, NULL);
      std::cout << "    Compute units...: " << value << std::endl;
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), &value, NULL);
      std::cout << "    Work item dims..: " << value << std::endl;
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(values), &values, NULL);
      std::cout << "    Work item size..: " << values[0] << ", " << values[1] << ", " << values[2] << std::endl;
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), &value, NULL);
      std::cout << "    Clock frequency.: " << value << " Hz\n";

      cl_device_type infoType;
      clGetDeviceInfo(m_all_devices[platform][device], CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL);
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

bool OpenCLKernel::Init()
  {
  if (_InitPlatforms() != CL_SUCCESS)
    return false;

  if (_InitDevices() != CL_SUCCESS)
    return false;

  if (_InitContext() != CL_SUCCESS)
    return false;


  //cl::Context context({ default_device });
  //
  //cl::Program::Sources sources;
  //
  //// kernel calculates for each element C=A+B
  //std::string kernel_code =
  //  "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
  //  "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
  //  "   }                                                                               ";
  //sources.push_back({ kernel_code.c_str(),kernel_code.length() });
  //
  //cl::Program program(context, sources);
  //if (program.build({ default_device }) != CL_SUCCESS) {
  //  //std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
  //  //exit(1);
  //  }
  //
  //
  //// create buffers on the device
  //cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  //cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  //cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  //
  //int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  //int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
  //
  ////create queue to which we will push commands for the device.
  //cl::CommandQueue queue(context, default_device);
  //
  ////write arrays A and B to the device
  //queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
  //queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);
  //
  //
  ////run the kernel
  ////cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
  ////simple_add(buffer_A, buffer_B, buffer_C);
  //
  ////alternative way to run the kernel
  //cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
  //kernel_add.setArg(0, buffer_A);
  //kernel_add.setArg(1, buffer_B);
  //kernel_add.setArg(2, buffer_C);
  //queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
  //queue.finish();
  //
  //int C[10];
  ////read result C from the device to array C
  //queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);
  //
  ////std::cout << " result: \n";
  //for (int i = 0; i < 10; i++) {
  //  //std::cout << C[i] << " ";
  //  }
  }

void OpenCLKernel::Test()
  {
  std::string kernel_code =
    "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
    "       int id = get_global_id(0);"
    "       C[id]=A[id]+B[id];                 "
    "   }                                                                               ";
  size_t length = kernel_code.length();
  const char* raw_data = kernel_code.data();
  cl_int rc;
  m_program = clCreateProgramWithSource(m_context, 1, &raw_data, &length, &rc);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Creating kernel program failed\n";
    }
  rc = clBuildProgram(m_program, 1, &m_current_device, "", 0, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Build kernel program failed\n";
    }

  cl_kernel kernel = clCreateKernel(m_program, "simple_add", &rc);

  int n = rand() % 100;
  int* a = new int[n];
  int* b = new int[n];
  for (int i = 0; i < n; ++i)
    {
    a[i] = rand() % 1000;
    b[i] = rand() % 1000;
    }

  cl_mem ag = clCreateBuffer(m_context, CL_MEM_READ_ONLY, n * sizeof(int), nullptr, &rc);
  cl_mem bg = clCreateBuffer(m_context, CL_MEM_READ_ONLY, n * sizeof(int), nullptr, &rc);
  cl_mem cg = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, n * sizeof(int), nullptr, &rc);

  if (!ag || !bg || !cg)
    {
    std::cout<<"Error: Failed to allocate device memory!\n";
    }

  rc = clEnqueueWriteBuffer(m_queue, ag, CL_TRUE, 0, n * sizeof(int), a, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Error: Failed to write to source array a!\n";
    }
  rc = clEnqueueWriteBuffer(m_queue, bg, CL_TRUE, 0, n * sizeof(int), b, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Error: Failed to write to source array b!\n";
    }

  rc = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ag);
  if (rc != CL_SUCCESS)
    {
    std::cout << "0 kernel program failed\n";
    }
  rc = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bg);
  if (rc != CL_SUCCESS)
    {
    std::cout << "1 kernel program failed\n";
    }
  rc = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cg);
  if (rc != CL_SUCCESS)
    {
    std::cout << "2 kernel program failed\n";
    }
  size_t size = 128;
  rc = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &size, &size, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "ND kernel program failed\n";
    }
  clFinish(m_queue);
  int* c_cpu = new int[n];
  clEnqueueReadBuffer(m_queue, cg, CL_TRUE, 0, n * sizeof(int), c_cpu, 0, nullptr, nullptr);
  for (int i = 0; i < n; ++i)
    {
    std::cout << a[i] << " + " << b[i] << " = " << c_cpu[i] << std::endl;
    }
  delete[] c_cpu;
  c_cpu = nullptr;
  }

std::vector<unsigned char> OpenCLKernel::Dummy()
  {
  std::string kernel_code =
    "void kernel create_dummy_picture(int width, int height, global uchar* picture, int k){"
    "  int i = get_global_id(0);"
    "  int j = get_global_id(1);"
    "  int x = i * width * 4;"
    "  int y = j * 4;"
    "  int const a = k;"
    "  int const m = 2147483647;"
    "  int r = ((x + y) * a) % m;"
    "  int g = (r * a) % m;"
    "  int b = (g * a) % m;"
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
  rc = clBuildProgram(m_program, 1, &m_current_device, "", 0, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "Build kernel program failed\n";
    }

  cl_kernel kernel = clCreateKernel(m_program, "create_dummy_picture", &rc);

  const int width = 256;
  const int height = 256;
  const int bytes_per_pixel = 4;
  const int rd = rand() % RAND_MAX;

  cl_mem d_c = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, width * height * bytes_per_pixel * sizeof(cl_uchar), nullptr, &rc);

  if (!d_c)
    {
    std::cout << "Error: Failed to allocate device memory!\n";
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
  rc = clSetKernelArg(kernel, 3, sizeof(int), &rd);
  if (rc != CL_SUCCESS)
    {
    std::cout << "2 kernel program failed\n";
    }
  size_t global_size[2] = { width, height };
  size_t local_size[2] = { 1, 1 };
  rc = clEnqueueNDRangeKernel(m_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
  if (rc != CL_SUCCESS)
    {
    std::cout << "ND kernel program failed\n";
    }
  clFinish(m_queue);
  auto picture = new cl_uchar[width * height * bytes_per_pixel];
  clEnqueueReadBuffer(m_queue, d_c, CL_TRUE, 0, width * height * bytes_per_pixel * sizeof(cl_uchar), picture, 0, nullptr, nullptr);
  std::vector<unsigned char> res(picture,picture+width*height*bytes_per_pixel);
  delete[] picture;
  picture = nullptr;
  return res;
  }
