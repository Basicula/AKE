#pragma once
#include <cuda_runtime.h>

template<class T>
class base_ptr
  {
  public:
    T* operator->();
    T* get() const;
    T& operator*() const;

  protected:
    T* mp_data;
  };

template<class T>
class device_ptr : public base_ptr<T>
  {
  public:
    template<class... Args>
    device_ptr(Args... i_args);
    ~device_ptr();
  };

template<class T>
class managed_ptr : public base_ptr<T>
  {
  public:
    template<class... Args>
    managed_ptr(Args... i_args);
    ~managed_ptr();
  };

template<class T>
class host_ptr : public base_ptr<T>
  {
  public:
    template<class... Args>
    host_ptr(Args... i_args);
    ~host_ptr();
  };

#include <CUDACore/impl/HostDevicePtrImpl.h>