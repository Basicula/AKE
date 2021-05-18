#include <Memory/managed_allocator.h>
#include <Memory/MemoryManager.h>

template<class T>
managed_ptr<T>::managed_ptr()
  : base_ptr<T>() {
  }

template<class T>
template<class ...Args>
managed_ptr<T>::managed_ptr(Args ...i_args) {
  static_assert(
    !std::is_polymorphic<T>::value == 1,
    "Object contains vtable and can't be located in shared memory" );
  const auto old_allocate_type = MemoryManager::GetCurrentAllocateType();
  MemoryManager::SetCurrentAllocateType(MemoryManager::AllocateType::CudaManaged);
  mp_data = managed_allocator<T>::create_new(i_args...);
  MemoryManager::SetCurrentAllocateType(old_allocate_type);
  }

template<class T>
inline managed_ptr<T>::~managed_ptr() {
  managed_allocator<T>::clean(mp_data);
  }