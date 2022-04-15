#include "OpenCLCore/OpenCLUtils.h"
#include "OpenCLKernels/MandelbrotSetKernel.h"

MandelbrotSetKernel::MandelbrotSetKernel(
  std::size_t i_width,
  std::size_t i_height,
  std::size_t i_max_iterations)
  : Kernel(
    MANDELBROT_SET_KERNEL_PATH,
    "mandelbrot_set")
  , m_width(i_width)
  , m_height(i_height)
  , m_max_iterations(i_max_iterations)
  , mp_output_image(nullptr)
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

bool MandelbrotSetKernel::BuildProgram(
  cl_uint i_num_of_devices, 
  const cl_device_id* i_device_ids) const
  {
  cl_int rc = clBuildProgram(
    m_program, 
    i_num_of_devices, 
    i_device_ids, 
    "",
    0, 
    nullptr);

  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Building program", rc);

  return (rc == CL_SUCCESS);
  }

bool MandelbrotSetKernel::SetKernelArgs() const
  {
  cl_int rc;

  rc = clSetKernelArg(mk_mandelbrot, 0, sizeof(int), &m_max_iterations);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Set max iterations for mandelbrot kernel", rc);
  if (rc != CL_SUCCESS)
    return false;

  rc = clSetKernelArg(mk_mandelbrot, 1, sizeof(cl_mem), &md_image);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Set picture memory for mandelbrot kernel", rc);

  return (rc == CL_SUCCESS);
  }

void MandelbrotSetKernel::UpdateDeviceOffset(std::size_t i_queue_id)
  {
  m_device_offset = KernelSize(0, m_device_size[m_kernel_dimensions - 1] * i_queue_id);
  }

bool MandelbrotSetKernel::WriteBuffers(
  const cl_command_queue& /*i_queue*/) const
  {
  return true;
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
    m_device_offset[m_kernel_dimensions - 1] * 
    mp_output_image->GetDepth() * 
    m_width;

  cl_int rc = clEnqueueReadBuffer(
    i_queue, 
    md_image, 
    CL_FALSE, 
    offset, 
    m_device_size.Size() * mp_output_image->GetDepth() * sizeof(uint8_t),
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
    mp_output_image->GetBytesCount() * sizeof(uint8_t),
    nullptr,
    &rc);
  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Create output buffer", rc);
  if (rc != CL_SUCCESS)
    return false;

  return (rc == CL_SUCCESS);
  }

void MandelbrotSetKernel::UpdateKernelSizeInfo(std::size_t i_device_cnt)
  {
  m_kernel_dimensions = 2;
  m_global_size = KernelSize(
    m_width, 
    m_height);
  m_local_size = KernelSize(32, 32);
  m_device_size = KernelSize(
    m_width, 
    m_height / i_device_cnt);
  m_device_offset = KernelSize(0, 0);
  }