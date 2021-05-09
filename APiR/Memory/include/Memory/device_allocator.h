#pragma once
#include <Macros.h>

template<class T>
class device_allocator
  {
  public:
    static T* allocate(size_t i_count);
    template<class... Args>
    static T* create_new(Args&&...i_args);
    template<class... Args>
    static void create_new(T* iop_at_pointer, Args&&...i_args);
    static void clean(T* ip_pointer);
  };

#include "impl/device_allocator_impl.h"