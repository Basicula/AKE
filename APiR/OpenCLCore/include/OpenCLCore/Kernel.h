#pragma once
#include <string>

#include <OpenCLCore/KernelSize.h>

class Kernel
  {
  public:
    /// Creates kernel with path to source code file 
    /// and main function name for further kernel processing
    Kernel(
      const std::string& i_file_path,
      const std::string& i_main_func_name);

    /// Preprocessing for kernel in given context from environment
    bool InitProgramForContext(const cl_context& io_context);
    /// Get kernel program to environment 
    /// Need for building on specific platform and devices
    cl_program GetProgram() const;
    /// Creating kernels from given source code
    virtual bool InitKernelsForProgram() = 0;
    /// Updates all information about different sizes for kernels
    /// i_device_cnt given by environment
    /// Must be derived and specified for correct work
    virtual void UpdateKernelSizeInfo(std::size_t i_device_cnt) = 0;
    /// Create connection between device and host data
    /// Specify input/output arguments for every kernel
    virtual bool SetKernelArgs() const = 0;
    /// Update offset by given queue id
    virtual void UpdateDeviceOffset(std::size_t i_queue_id) = 0;
    /// Writes input buffers to kernel
    virtual bool WriteBuffers(const cl_command_queue& i_queue) const = 0;
    /// Execute kernels from queue given by environment 
    virtual bool ExecuteKernels(const cl_command_queue& i_queue) const = 0;
    /// Collect results after kernel execution
    virtual bool CollectResults(const cl_command_queue& i_queue) = 0;

    /// Log some info about success of kernel processing
    void SetLoggingState(bool i_is_on);

  private:
    void _GetSourceDataFromFile();

  protected:
    virtual bool _InitBuffers() = 0;

  protected:
    bool m_enable_logging;

    cl_context m_context;
    cl_program m_program;

    std::string m_file_path;
    std::string m_main_func_name;
    std::string m_source_code;

    KernelSize m_global_size;
    KernelSize m_local_size;
    KernelSize m_device_size;
    KernelSize m_device_offset;
  };

inline cl_program Kernel::GetProgram() const
  {
  return m_program;
  }

inline void Kernel::SetLoggingState(bool i_is_on)
  {
  m_enable_logging = i_is_on;
  }
