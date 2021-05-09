#include <Memory/default_allocator.h>
#if ENABLED_CUDA
#include <Memory/device_allocator.h>
#include <Memory/managed_allocator.h>
#endif

#include <algorithm>

template<class T>
inline MemoryManager::pointer<T> MemoryManager::allocate(AllocateType i_allocate_type, const size_t i_count) {
  pointer<T> pointer;
  pointer.m_allocate_type = i_allocate_type;
  allocate(pointer, i_count);
  return pointer;
  }

template<class T>
inline MemoryManager::pointer<T> MemoryManager::allocate(const size_t i_count) {
  pointer<T> pointer;
  pointer.m_allocate_type = m_current_allocate_type;
  allocate(pointer, i_count);
  return pointer;
  }

template<class T>
inline void MemoryManager::allocate(pointer<T>& io_pointer, const size_t i_count) {
  switch (io_pointer.m_allocate_type) {
      case MemoryManager::AllocateType::CudaDevice:
        io_pointer.mp_data = device_allocator<T>::allocate(i_count);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::CudaManaged:
        io_pointer.mp_data = managed_allocator<T>::allocate(i_count);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::Default:
        io_pointer.mp_data = default_allocator<T>::allocate(i_count);
        break;
      case MemoryManager::AllocateType::Undefined:
      default:
        return; // "Can't allocate such type of information"
    }
  }

template<class T, class...Args>
MemoryManager::pointer<T> MemoryManager::create_new(AllocateType i_allocate_type, Args&&...i_args) {
  pointer<T> pointer;
  pointer.m_allocate_type = i_allocate_type;
  const auto old_allocate_type = m_current_allocate_type;
  m_current_allocate_type = i_allocate_type;
  switch (i_allocate_type) {
      case MemoryManager::AllocateType::CudaDevice:
        pointer.mp_data = device_allocator<T>::create_new(i_args...);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::CudaManaged:
        pointer.mp_data = managed_allocator<T>::create_new(i_args...);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::Default:
        pointer.mp_data = default_allocator<T>::create_new(i_args...);
        break;
      default:
        break; // "Can't create new object with such type of information"
    }
  m_current_allocate_type = old_allocate_type;
  return pointer;
  }

template<class T, class ...Args>
inline void MemoryManager::create_new(AllocateType i_allocate_type, T* iop_at_pointer, Args&&...i_args) {
  // for deep allocation in case if some classes will allocate memory inside
  const auto old_allocate_type = m_current_allocate_type;
  m_current_allocate_type = i_allocate_type;
  switch (i_allocate_type) {
      case MemoryManager::AllocateType::CudaDevice:
        device_allocator<T>::create_new(iop_at_pointer, i_args...);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::CudaManaged:
        managed_allocator<T>::create_new(iop_at_pointer, i_args...);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::Default:
        default_allocator<T>::create_new(iop_at_pointer, i_args...);
        break;
      default:
        break; // "Can't create new object with such type of information"
    }
  m_current_allocate_type = old_allocate_type;
  }

template<class T>
inline void MemoryManager::clean(MemoryManager::pointer<T>& ip_data) {
  clean(ip_data.m_allocate_type, ip_data.mp_data);
  }

template<class T>
inline void MemoryManager::clean(AllocateType i_allocate_type, T* ip_data) {
  switch (i_allocate_type) {
      case MemoryManager::AllocateType::CudaDevice:
        return device_allocator<T>::clean(ip_data);
      case MemoryManager::AllocateType::CudaManaged:
        return managed_allocator<T>::clean(ip_data);
      case MemoryManager::AllocateType::Default:
        return default_allocator<T>::clean(ip_data);
      default:
        return; // "Can't free such type of information"
    }
  }

template<class T>
inline void MemoryManager::copy(const pointer<T>& i_begin, size_t i_size, pointer<T>& o_out) {
  if (i_begin.m_allocate_type != o_out.m_allocate_type)
    return; // "Wrong pointers for copiyng"
  switch (o_out.m_allocate_type) {
      case MemoryManager::AllocateType::CudaDevice:
#ifdef __CUDA_ARCH__
#else
        cudaMemcpy(o_out.mp_data, i_begin.mp_data, i_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
#endif
        break;
      case MemoryManager::AllocateType::CudaManaged:
        cudaMemcpy(o_out.mp_data, i_begin.mp_data, i_size, cudaMemcpyKind::cudaMemcpyDefault);
        cudaDeviceSynchronize();
        break;
      case MemoryManager::AllocateType::Default:
        std::copy(i_begin.mp_data, i_begin.mp_data + i_size, o_out.mp_data);
        break;
      default:
        return; // "Can't copy such type of information"
    }
  }