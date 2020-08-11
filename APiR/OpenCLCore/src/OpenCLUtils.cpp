#include <OpenCLUtils.h>

namespace OpenCLUtils
  {
  const std::size_t g_max_message_size = 4096;

  void CheckSuccess(
    const std::string& i_message, 
    cl_int i_return_code,
    std::ostream& o_log_stream)
    {
    std::cout << (i_message) << " : ";
    switch (i_return_code)
      {
      case 0:   o_log_stream << ("Success\n"); break;
      case -1:  o_log_stream << ("Device not found\n"); break;
      case -2:  o_log_stream << ("Device not available\n"); break;
      case -3:  o_log_stream << ("Compiler not available\n"); break;
      case -4:  o_log_stream << ("Memory object allocation failure\n"); break;
      case -5:  o_log_stream << ("Out of resources\n"); break;
      case -6:  o_log_stream << ("Out of host memory\n"); break;
      case -7:  o_log_stream << ("Profiling info not available\n"); break;
      case -8:  o_log_stream << ("Memory copy overlap\n"); break;
      case -9:  o_log_stream << ("Image format mismatch\n"); break;
      case -10: o_log_stream << ("Image format not supported\n"); break;
      case -11: o_log_stream << ("Build program failure\n"); break;
      case -12: o_log_stream << ("Map failure\n"); break;
      case -30: o_log_stream << ("Invalid value\n"); break;
      case -31: o_log_stream << ("Invaid device type\n"); break;
      case -32: o_log_stream << ("Invalid platform\n"); break;
      case -33: o_log_stream << ("Invalid device\n"); break;
      case -34: o_log_stream << ("Invalid context\n"); break;
      case -35: o_log_stream << ("Invalid queue properties\n"); break;
      case -36: o_log_stream << ("Invalid command queue\n"); break;
      case -37: o_log_stream << ("Invalid host pointer\n"); break;
      case -38: o_log_stream << ("Invalid memory object\n"); break;
      case -39: o_log_stream << ("Invalid image format descriptor\n"); break;
      case -40: o_log_stream << ("Invalid image size\n"); break;
      case -41: o_log_stream << ("Invalid sampler\n"); break;
      case -42: o_log_stream << ("Invalid binary\n"); break;
      case -43: o_log_stream << ("Invalid build options\n"); break;
      case -44: o_log_stream << ("Invalid program\n"); break;
      case -45: o_log_stream << ("Invalid program executable\n"); break;
      case -46: o_log_stream << ("Invalid kernel name\n"); break;
      case -47: o_log_stream << ("Invalid kernel defintion\n"); break;
      case -48: o_log_stream << ("Invalid kernel\n"); break;
      case -49: o_log_stream << ("Invalid argument index\n"); break;
      case -50: o_log_stream << ("Invalid argument value\n"); break;
      case -51: o_log_stream << ("Invalid argument size\n"); break;
      case -52: o_log_stream << ("Invalid kernel arguments\n"); break;
      case -53: o_log_stream << ("Invalid work dimension\n"); break;
      case -54: o_log_stream << ("Invalid work group size\n"); break;
      case -55: o_log_stream << ("Invalid work item size\n"); break;
      case -56: o_log_stream << ("Invalid global offset\n"); break;
      case -57: o_log_stream << ("Invalid event wait list\n"); break;
      case -58: o_log_stream << ("Invalid event\n"); break;
      case -59: o_log_stream << ("Invalid operation\n"); break;
      case -60: o_log_stream << ("Invalid GL object\n"); break;
      case -61: o_log_stream << ("Invalid buffer size\n"); break;
      case -62: o_log_stream << ("Invalid mip level\n"); break;
      case -63: o_log_stream << ("Invalid global work size\n"); break;
      default:
        break;
      }
    }

  void LogPlatformInfo(
    cl_platform_id i_platform,
    std::ostream& o_log_stream)
    {
    char message[g_max_message_size];
    size_t message_length;
    o_log_stream << "Platform " << i_platform << std::endl;

    clGetPlatformInfo(i_platform, CL_PLATFORM_NAME, g_max_message_size, message, &message_length);
    o_log_stream << "  Name..............: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(i_platform, CL_PLATFORM_VERSION, g_max_message_size, message, &message_length);
    o_log_stream << "  Version...........: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(i_platform, CL_PLATFORM_VENDOR, g_max_message_size, message, &message_length);
    o_log_stream << "  Vendor............: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(i_platform, CL_PLATFORM_PROFILE, g_max_message_size, message, &message_length);
    o_log_stream << "  Profile...........: " << std::string(message, message + message_length) << std::endl;

    clGetPlatformInfo(i_platform, CL_PLATFORM_EXTENSIONS, g_max_message_size, message, &message_length);
    o_log_stream << "  Extensions........: " << std::string(message, message + message_length) << std::endl;
    }

  void LogDeviceMainInfo(
    cl_device_id i_device,
    std::ostream& o_log_stream)
    {
    char message[g_max_message_size];
    size_t message_length;
    o_log_stream << "  Device " << i_device << std::endl;
    std::string deviceDescription;
    clGetDeviceInfo(i_device, CL_DEVICE_NAME, g_max_message_size, message, &message_length);
    o_log_stream << "    Name............: " << std::string(message, message + message_length) << std::endl;

    clGetDeviceInfo(i_device, CL_DEVICE_VENDOR, g_max_message_size, message, &message_length);
    o_log_stream << "    Vendor..........: " << std::string(message, message + message_length) << std::endl;

    clGetDeviceInfo(i_device, CL_DEVICE_VERSION, g_max_message_size, message, &message_length);
    o_log_stream << "    Version.........: " << std::string(message, message + message_length) << std::endl;

    clGetDeviceInfo(i_device, CL_DRIVER_VERSION, g_max_message_size, message, &message_length);
    o_log_stream << "    Driver version..: " << std::string(message, message + message_length) << std::endl;
    }

  void LogDeviceCharacteristicsInfo(
    cl_device_id i_device, 
    std::ostream& o_log_stream)
    {
    cl_uint compute_units;
    clGetDeviceInfo(i_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    o_log_stream << "    Compute units...: " << compute_units << std::endl;

    cl_uint max_dimension;
    clGetDeviceInfo(i_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_dimension), &max_dimension, NULL);
    o_log_stream << "    Work item dims..: " << max_dimension << std::endl;

    std::size_t* work_item_sizes = new size_t[max_dimension];
    clGetDeviceInfo(i_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_dimension * sizeof(size_t), work_item_sizes, NULL);
    o_log_stream << "    Work item sizes.: ";
    for (cl_uint i = 0; i < max_dimension; ++i)
      o_log_stream << work_item_sizes[i] << (i == max_dimension - 1 ? "\n" : ", ");

    std::size_t work_group_size;
    clGetDeviceInfo(i_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
    o_log_stream << "    Work group size.: " << work_group_size << std::endl;

    cl_uint max_clock_frequency;
    clGetDeviceInfo(i_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, NULL);
    o_log_stream << "    Clock frequency.: " << max_clock_frequency << " Hz\n";
    }

  void LogDeviceTypeInfo(
    cl_device_id i_device,
    std::ostream& o_log_stream)
    {
    cl_device_type infoType;
    clGetDeviceInfo(i_device, CL_DEVICE_TYPE, sizeof(infoType), &infoType, NULL);
    if (infoType & CL_DEVICE_TYPE_DEFAULT)
      {
      infoType &= ~CL_DEVICE_TYPE_DEFAULT;
      o_log_stream << "    Type: Default\n";
      }
    if (infoType & CL_DEVICE_TYPE_CPU)
      {
      infoType &= ~CL_DEVICE_TYPE_CPU;
      o_log_stream << "    Type: CPU\n";
      }
    if (infoType & CL_DEVICE_TYPE_GPU)
      {
      infoType &= ~CL_DEVICE_TYPE_GPU;
      o_log_stream << "    Type: GPU\n";
      }
    if (infoType & CL_DEVICE_TYPE_ACCELERATOR)
      {
      infoType &= ~CL_DEVICE_TYPE_ACCELERATOR;
      o_log_stream << "    Type: Accelerator\n";
      }
    if (infoType != 0)
      o_log_stream << "    Type: Unknown " << infoType << std::endl;
    }
  }