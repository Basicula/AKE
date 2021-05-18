#pragma once
#include <cuda_runtime.h>

#if defined(_DEBUG)
#define CheckCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#else
#define CheckCudaErrors(val) (val)
#endif

void check_cuda(
  cudaError_t result,
  char const* const func,
  const char* const file,
  int const line);
void GPUInfo();

template<class T, class F, class...Args>
__global__ void _execute(T* ip_object, F i_function, Args...i_args) {
  (ip_object->*i_function)(i_args...);
  }

template<class T, class F, class...Args>
void CUDA_execute(T* ip_object, F i_function, Args...i_args) {
  _execute << <1, 1 >> > ( ip_object, i_function, i_args... );
  //( ip_object->*function )( i_args... );
  }

#define CUDATEST(pointer_to_object, function_name, ...) \
  pointer_to_object->function_name(__VA_ARGS__)     \
