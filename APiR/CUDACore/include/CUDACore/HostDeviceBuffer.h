#pragma once
#include <Macro/CudaMacro.h>

#include <CUDACore/CUDAUtils.h>

#include <initializer_list>

template<class T>
class HostDeviceBuffer
  {
  public:
    HostDeviceBuffer(size_t i_size);
    HostDeviceBuffer(
      size_t i_size,
      const T& i_init_value);
    HostDeviceBuffer(const HostDeviceBuffer& i_other);
    HostDeviceBuffer(std::initializer_list<T> i_list);
    ~HostDeviceBuffer();

    HOSTDEVICE T& operator[](size_t i);
    HOSTDEVICE T& operator[](size_t i) const;

    T* data() const;
    size_t size() const;
    bool empty() const;
    void push_back(const T& i_elem);

    template<class... Args>
    void emplace_back(Args... i_args);

  private:
    void resize(size_t i_new_size);

  private:
    size_t m_size;
    size_t m_capacity;
    T* mp_buffer;

    const double m_size_multiplier = 1.5;
  };

#include <CUDACore/impl/HostDeviceBufferImpl.h>

template<class T>
template<class ...Args>
inline void HostDeviceBuffer<T>::emplace_back(Args ...i_args)
  {
  resize(m_size + 1);
  new(mp_buffer + m_size - 1) T(i_args...);
  }
