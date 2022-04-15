#pragma once
#include <Memory/base_ptr.h>

template<class T>
class device_ptr : public base_ptr<T>
  {
  public:
    explicit device_ptr();
    template<class... Args>
    explicit device_ptr(Args&&... i_args);
    ~device_ptr();

    T get_host_copy() const;
    void make_device_copy(const T& i_object);
  };
  
#include "impl/device_ptr_impl.h"