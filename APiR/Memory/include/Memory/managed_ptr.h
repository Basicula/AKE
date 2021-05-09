#pragma once
#include <Memory/base_ptr.h>

template<class T>
class managed_ptr : public base_ptr<T>
  {
  public:
    managed_ptr();
    template<class... Args>
    managed_ptr(Args... i_args);
    ~managed_ptr();
  };
  
#include "impl/managed_ptr_impl.h"