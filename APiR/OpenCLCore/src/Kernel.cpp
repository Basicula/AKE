#include <OpenCLCore/Kernel.h>
#include <OpenCLCore/OpenCLUtils.h>

#include <fstream>
#include <streambuf>

Kernel::Kernel(
  const std::string& i_file_path, 
  const std::string& i_main_func_name)
  : m_enable_logging(false)
  , m_file_path(i_file_path)
  , m_main_func_name(i_main_func_name)
  {
  _GetSourceDataFromFile();
  }

bool Kernel::InitProgramForContext(
  const cl_context& io_context)
  {
  m_context = io_context;

  const size_t source_length = m_source_code.length();
  const auto* raw_source_data = m_source_code.data();

  cl_int rc;
  m_program = clCreateProgramWithSource(m_context, 1, &raw_source_data, &source_length, &rc);

  if (m_enable_logging)
    OpenCLUtils::CheckSuccess("Create program", rc);

  _InitBuffers();

  return (rc == CL_SUCCESS);
  }

void Kernel::_GetSourceDataFromFile()
  {
  std::ifstream file(m_file_path);
  m_source_code.assign(
    std::istreambuf_iterator<char>(file),
    std::istreambuf_iterator<char>());
  }
