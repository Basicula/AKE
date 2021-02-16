#include <CUDACore/HostDeviceBuffer.h>

#include <algorithm>

template<class T>
inline HostDeviceBuffer<T>::HostDeviceBuffer(size_t i_size)
  : m_size(i_size)
  , m_capacity(i_size)
  , mp_buffer(nullptr)
  {
  CheckCudaErrors(cudaMallocManaged(&mp_buffer, m_capacity * sizeof(T)));
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline HostDeviceBuffer<T>::HostDeviceBuffer(
  size_t i_size,
  const T& i_init_value)
  : HostDeviceBuffer(i_size)
  {
  std::fill_n(mp_buffer, m_size, i_init_value);
  }

template<class T>
inline HostDeviceBuffer<T>::HostDeviceBuffer(const HostDeviceBuffer<T>& i_other)
  : HostDeviceBuffer<T>(i_other.m_size)
  {
  CheckCudaErrors(cudaMemcpy(&mp_buffer, i_other.mp_buffer, m_size * sizeof(T), cudaMemcpyDefault));
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class T>
inline HostDeviceBuffer<T>::HostDeviceBuffer(std::initializer_list<T> i_list)
  : HostDeviceBuffer<T>(i_list.size())
  {
  //std::copy(i_list.begin(), i_list.end(), mp_buffer);
  }

template<class T>
inline HostDeviceBuffer<T>::~HostDeviceBuffer()
  {
  CheckCudaErrors(cudaFree(mp_buffer));
  }

template<class T>
inline T& HostDeviceBuffer<T>::operator[](size_t i)
  {
  return mp_buffer[i];
  }

template<class T>
inline T& HostDeviceBuffer<T>::operator[](size_t i) const
  {
  return mp_buffer[i];
  }

template<class T>
inline T* HostDeviceBuffer<T>::data() const
  {
  return mp_buffer;
  }

template<class T>
inline size_t HostDeviceBuffer<T>::size() const
  {
  return m_size;
  }

template<class T>
inline bool HostDeviceBuffer<T>::empty() const
  {
  return m_size == 0;
  }

template<class T>
inline void HostDeviceBuffer<T>::push_back(const T& i_elem)
  {
  resize(m_size + 1);
  mp_buffer[m_size - 1] = i_elem;
  }

template<class T>
inline void HostDeviceBuffer<T>::resize(size_t i_new_size)
  {
  if (i_new_size > m_capacity)
    {
    m_capacity = static_cast<size_t>(i_new_size * m_size_multiplier);
    T* temp;
    CheckCudaErrors(cudaMallocManaged(&temp, m_capacity * sizeof(T)));
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaMemcpy(temp, mp_buffer, m_size * sizeof(T), cudaMemcpyDefault));
    CheckCudaErrors(cudaFree(mp_buffer));
    mp_buffer = temp;
    temp = nullptr;
    }
  m_size = i_new_size;
  }
