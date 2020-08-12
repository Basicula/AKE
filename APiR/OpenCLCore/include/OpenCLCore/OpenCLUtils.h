#pragma once
#include <string>
#include <iostream>

#include <CL/cl.h>

namespace OpenCLUtils
  {
  void CheckSuccess(
    const std::string& i_message, 
    cl_int i_return_code, 
    std::ostream& o_log_stream = std::cout);

  void LogPlatformInfo(
    cl_platform_id i_platform,
    std::ostream& o_log_stream = std::cout);
  void LogDeviceMainInfo(
    cl_device_id i_device,
    std::ostream& o_log_stream = std::cout);
  void LogDeviceCharacteristicsInfo(
    cl_device_id i_device,
    std::ostream& o_log_stream = std::cout);
  void LogDeviceTypeInfo(
    cl_device_id i_device,
    std::ostream& o_log_stream = std::cout);
  }