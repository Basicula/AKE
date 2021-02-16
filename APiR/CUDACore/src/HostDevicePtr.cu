#include <CUDACore/HostDevicePtr.cuh>

template<class T, class... Args>
__global__ void cudaNew(T* iop_object, Args... i_args)
  {
  new(iop_object) T(i_args...);
  }

template<class T, class... Args>
void cudaDeviceNew(T* iop_data, Args... i_args)
  {
  cudaNew<T><<<1, 1>>>(iop_data, i_args...);
  }