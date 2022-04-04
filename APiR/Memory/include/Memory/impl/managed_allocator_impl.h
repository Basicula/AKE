#include "CUDACore/CUDAUtils.h"

template<class T>
inline T* managed_allocator<T>::allocate(size_t i_count) {
  T* ptr;
  CheckCudaErrors(cudaMallocManaged(&ptr, i_count * sizeof(T)));
  CheckCudaErrors(cudaDeviceSynchronize());
  return ptr;
  }

template<class T>
template<class ...Args>
inline T* managed_allocator<T>::create_new(Args&& ...i_args) {
  T* ptr = allocate(1);
  create_new(ptr, i_args...);
  return ptr;
  }

template<class T>
template<class... Args>
void managed_allocator<T>::create_new(T* iop_at_pointer, Args&&...i_args) {
  new( iop_at_pointer ) T(i_args...);
  }

template<class T>
inline void managed_allocator<T>::clean(T* iop_pointer) {
  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cudaFree(iop_pointer));
  }