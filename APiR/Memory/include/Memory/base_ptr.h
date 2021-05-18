#pragma once

template<class T>
class base_ptr
  {
  public:
    base_ptr();
    T* operator->();
    T* get() const;
    T& operator*() const;

  protected:
    T* mp_data;
  };
  
#include "impl/base_ptr_impl.h"