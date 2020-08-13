#include <OpenCLCore/OpenCLUtils.h>
#include <OpenCLCore/MandelbrotSetKernel.h>

// MandelbrotSet
const std::uint8_t MandelbrotSetKernel::MandelbrotSet::m_color_map[17 * 3] =
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

MandelbrotSetKernel::MandelbrotSet::MandelbrotSet(
  std::size_t i_width, 
  std::size_t i_height, 
  std::size_t i_iterations)
  : m_width(i_width)
  , m_height(i_height)
  , m_max_iterations(i_iterations)
  , m_origin_x(0)
  , m_origin_y(0)
  , m_scale(0.0)
  {
  }

// MandelbrotSet end

MandelbrotSetKernel::MandelbrotSetKernel(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_max_iterations)
  : Kernel(
    MANDELBROT_SET_KERNEL_PATH,
    "mandelbrot_set")
  , m_mandelbrot_set(i_width, i_height, i_max_iterations)
  , mp_output_image(nullptr)
  , m_dimensions(2)
  , mk_mandelbrot()
  , md_image()
  , md_color_map()
  {
  }

MandelbrotSetKernel::~MandelbrotSetKernel()
  {
  clReleaseMemObject(md_image);
  clReleaseMemObject(md_color_map);
  }

bool MandelbrotSetKernel::InitKernelsForProgram()
  {
  cl_int rc;
  mk_mandelbrot = clCreateKernel(m_program, m_main_func_name.c_str(), &rc);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Mandelbrot kernel creation", rc);

  return (rc == CL_SUCCESS);
  }

bool MandelbrotSetKernel::SetKernelArgs() const
  {
  cl_int rc;

  rc = clSetKernelArg(mk_mandelbrot, 0, sizeof(int), &m_mandelbrot_set.m_max_iterations);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Set max iterations for mandelbrot kernel", rc);
  if (rc != CL_SUCCESS)
    return false;

  rc = clSetKernelArg(mk_mandelbrot, 1, sizeof(cl_mem), &md_color_map);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Set color map memory for mandelbrot kernel", rc);
  if (rc != CL_SUCCESS)
    return false;

  rc = clSetKernelArg(mk_mandelbrot, 2, sizeof(cl_mem), &md_image);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Set picture memory for mandelbrot kernel", rc);

  return (rc == CL_SUCCESS);
  }

void MandelbrotSetKernel::UpdateDeviceOffset(std::size_t i_queue_id)
  {
  m_device_offset = KernelSize(0, m_device_size[m_dimensions - 1] * i_queue_id);
  }

bool MandelbrotSetKernel::WriteBuffers(
  const cl_command_queue& i_queue) const
  {
  cl_int rc = clEnqueueWriteBuffer(
    i_queue,
    md_color_map,
    CL_FALSE,
    0,
    sizeof(m_mandelbrot_set.m_color_map),
    m_mandelbrot_set.m_color_map,
    0,
    nullptr,
    nullptr);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Write color map info", rc);

  return (rc == CL_SUCCESS);
  }

bool MandelbrotSetKernel::ExecuteKernels(
  const cl_command_queue& i_queue) const
  {
  cl_int rc = clEnqueueNDRangeKernel(
    i_queue, 
    mk_mandelbrot, 
    static_cast<cl_uint>(m_global_size.GetWorkDimension()), 
    m_device_offset.AsArray(), 
    m_device_size.AsArray(), 
    m_local_size.AsArray(), 
    0, 
    nullptr, 
    nullptr);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Run kernel", rc);

  return (rc == CL_SUCCESS);
  }

bool MandelbrotSetKernel::CollectResults(
  const cl_command_queue& i_queue)
  {
  const size_t offset = 
    m_device_offset[m_dimensions - 1] * 
    mp_output_image->GetDepth() * 
    m_mandelbrot_set.m_width;

  cl_int rc = clEnqueueReadBuffer(
    i_queue, 
    md_image, 
    CL_FALSE, 
    offset, 
    m_device_size.Size() * mp_output_image->GetDepth() * sizeof(std::uint8_t),
    mp_output_image->GetRGBAData() + offset, 
    0, 
    nullptr, 
    nullptr);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Read picture", rc);

  return (rc == CL_SUCCESS);
  }

bool MandelbrotSetKernel::_InitBuffers()
  {
  cl_int rc;

  if (!mp_output_image)
    return false;

  md_image = clCreateBuffer(
    m_context,
    CL_MEM_WRITE_ONLY,
    mp_output_image->GetBytesCount() * sizeof(std::uint8_t),
    nullptr,
    &rc);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Create output buffer", rc);
  if (rc != CL_SUCCESS)
    return false;

  md_color_map = clCreateBuffer(
    m_context, 
    CL_MEM_READ_ONLY, 
    sizeof(m_mandelbrot_set.m_color_map), 
    nullptr, 
    &rc);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Create color map buffer", rc);

  return (rc == CL_SUCCESS);
  }

void MandelbrotSetKernel::UpdateKernelSizeInfo(std::size_t i_device_cnt)
  {
  m_global_size = KernelSize(
    m_mandelbrot_set.m_width, 
    m_mandelbrot_set.m_height);
  m_local_size = KernelSize(32, 32);
  m_device_size = KernelSize(
    m_mandelbrot_set.m_width, 
    m_mandelbrot_set.m_height / i_device_cnt);
  m_device_offset = KernelSize(0, 0);
  }