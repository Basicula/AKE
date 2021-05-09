#include <Memory/MemoryManager.h>

template<class T>
device_ptr<T>::device_ptr()
  : base_ptr<T>() {
  mp_data = MemoryManager::allocate<T>(MemoryManager::AllocateType::CudaDevice).mp_data;
  }

template<class T>
template<class ...Args>
device_ptr<T>::device_ptr(Args&&... i_args) {
  mp_data = MemoryManager::create_new<T>(MemoryManager::AllocateType::CudaDevice, i_args...).mp_data;
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline device_ptr<T>::~device_ptr() {
  MemoryManager::clean<T>(MemoryManager::AllocateType::CudaDevice, mp_data);
  }

template<class T>
T device_ptr<T>::get_host_copy() const {
  T* copy = MemoryManager::allocate<T>(MemoryManager::AllocateType::Default, 1).mp_data;
  CheckCudaErrors(cudaMemcpy(copy, mp_data, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CheckCudaErrors(cudaDeviceSynchronize());
  return std::move(*copy);
  }

template<class T>
void device_ptr<T>::make_device_copy(const T& i_object) {
  mp_data = MemoryManager::allocate<T>(MemoryManager::AllocateType::CudaDevice, 1).mp_data;
  CheckCudaErrors(cudaMemcpy(mp_data, &i_object, sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
  CheckCudaErrors(cudaDeviceSynchronize());
  }