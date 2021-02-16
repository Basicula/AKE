#pragma once
#include <Macro/CudaMacro.h>

#include <cuda_runtime.h>

#define CheckCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(
  cudaError_t result, 
  char const* const func, 
  const char* const file, 
  int const line);

void GPUInfo();