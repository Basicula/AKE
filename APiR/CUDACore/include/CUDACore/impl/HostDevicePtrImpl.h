#include <type_traits>
#include <iostream>


template<class T, class... Args>
__global__ void cudaNew(T* iop_object, Args... i_args)
  {
  new(iop_object) T(i_args...);
  }

template<class T>
T* base_ptr<T>::operator->()
  {
  return mp_data;
  }

template<class T>
inline T* base_ptr<T>::get() const
  {
  return mp_data;
  }

template<class T>
inline T& base_ptr<T>::operator*() const
  {
  return *mp_data;
  }

template<class T>
template<class ...Args>
device_ptr<T>::device_ptr(Args ...i_args)
  {
  CheckCudaErrors(cudaMalloc(&mp_data, sizeof(T)));
  cudaNew<T> << <1, 1 >> > (mp_data, i_args...);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline device_ptr<T>::~device_ptr()
  {
  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cudaFree(mp_data));
  }

template<class T>
template<class ...Args>
managed_ptr<T>::managed_ptr(Args ...i_args)
  {
  static_assert(
    !std::is_polymorphic<T>::value == 1,
    "Object contains vtable and can't be located in shared memory");
  CheckCudaErrors(cudaMallocManaged(&mp_data, sizeof(T)));
  new(mp_data) T(i_args...);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline managed_ptr<T>::~managed_ptr()
  {
  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cudaFree(mp_data));
  }

template<class T>
template<class ...Args>
host_ptr<T>::host_ptr(Args ...i_args)
  {
  mp_data = new T(i_args...);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline host_ptr<T>::~host_ptr()
  {
  CheckCudaErrors(cudaDeviceSynchronize());
  delete mp_data;
  }