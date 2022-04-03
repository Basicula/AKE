#pragma once
#include "Macros.h"

class MemoryManager
  {
  public:
    enum class AllocateType
      {
      Default,
#if ENABLED_CUDA
      CudaDevice,
      CudaManaged,
#endif
      Undefined
      };

    template<class T>
    struct pointer
      {
      AllocateType m_allocate_type;
      T* mp_data;
      };

  public:
    static AllocateType GetCurrentAllocateType();
    static void SetCurrentAllocateType(AllocateType i_allocate_type);

    // Uses current allocate type
    template<class T>
    static pointer<T> allocate(const size_t i_count = 1);
    template<class T>
    static pointer<T> allocate(AllocateType i_allocate_type, const size_t i_count = 1);
    template<class T>
    static void allocate(pointer<T>& io_pointer, const size_t i_count = 1);

    // create_new functions have deep allocation
    // i.e. if class has dynamic memory inside it will be created with following allocate type
    template<class T, class...Args>
    static pointer<T> create_new(AllocateType i_allocate_type, Args&&...i_args);
    template<class T, class...Args>
    static void create_new(AllocateType i_allocate_type, T* iop_at_pointer, Args&&...i_args);

    template<class T>
    static void clean(pointer<T>& ip_data);
    template<class T>
    static void clean(AllocateType i_allocate_type, T* ip_data);

    template<class T>
    static void copy(const pointer<T>& i_begin, size_t i_size, pointer<T>& o_out);

  private:
    static AllocateType m_current_allocate_type;
  };

#include "impl/MemoryManager_impl.h"