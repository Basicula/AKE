#include <CUDACore/CUDAUtils.h>

namespace {
#ifdef __CUDACC__
  template<class T, class... Args>
  __global__ void _cuda_new(T* iop_object, Args... i_args) {
    new( iop_object ) T(i_args...);
    }

  template<class T, class... Args>
  inline void cudaNew(T* iop_object, Args&&... i_args) {
    _cuda_new<T> << <1, 1 >> > (iop_object, i_args...);
    }
#else
  template<class T, class... Args>
  inline void cudaNew(T*, Args&&...) {
    }
#endif // __CUDACC__
  }

template<class T>
inline T* device_allocator<T>::allocate(size_t i_count) {
  T* ptr;
  CheckCudaErrors(cudaMalloc(&ptr, i_count * sizeof(T)));
  CheckCudaErrors(cudaDeviceSynchronize());
  return ptr;
  }

template<class T>
template<class... Args>
inline T* device_allocator<T>::create_new(Args&&...i_args) {
  T* ptr = allocate(1); 
  create_new(ptr, i_args...);
  return ptr;
  }

template<class T>
template<class... Args>
void device_allocator<T>::create_new(T* iop_at_pointer, Args&&...i_args) {
  cudaNew(iop_at_pointer, i_args...);
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline void device_allocator<T>::clean(T* ip_pointer) {
  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cudaFree(ip_pointer));
  }