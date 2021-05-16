#pragma once
#include <Macros.h>

#include <Memory/MemoryManager.h>
#if defined(ENABLED_CUDA)
#include <Memory/device_ptr.h>
#include <Memory/managed_ptr.h>
#endif

#include <initializer_list>

template<class T>
class custom_vector
  {
  public:
    class iterator
      {
      public:
        HOSTDEVICE explicit iterator(T* ip_pointer = nullptr);
        HOSTDEVICE bool operator==(const iterator& i_other) const;
        HOSTDEVICE bool operator!=(const iterator& i_other) const;
        HOSTDEVICE T operator*() const;
        HOSTDEVICE iterator& operator++();
      private:
        T* mp_pointer;
      };

  public:
    explicit custom_vector();
    explicit custom_vector(size_t i_size);
    explicit custom_vector(size_t i_size, const T& i_init_value);
    explicit custom_vector(const std::initializer_list<T>& i_list);
    custom_vector(const custom_vector& i_other);
    custom_vector(custom_vector&& i_other);
    ~custom_vector();

    custom_vector& operator=(const custom_vector& i_other);

    HOSTDEVICE T& operator[](size_t i_index);
    HOSTDEVICE T& operator[](size_t i_index) const;

    template<class... Args>
    T& emplace_back(Args&&... i_args);
    void push_back(const T& i_val);

    void resize(size_t i_new_size);
    void resize(size_t i_new_size, const T& i_init_value);
    void clear();

    HOSTDEVICE T* data() const;
    HOSTDEVICE size_t size() const;
    HOSTDEVICE size_t capacity() const;
    HOSTDEVICE bool empty() const;

    HOSTDEVICE iterator begin() const;
    HOSTDEVICE iterator end() const;

    static device_ptr<custom_vector<T>> device_vector_ptr(size_t i_size);
    static device_ptr<custom_vector<T>> device_vector_ptr(size_t i_size, const T& i_init_value);
    static device_ptr<custom_vector<T>> device_vector_ptr(const std::initializer_list<T>& i_list);

  private:
    void resize(size_t i_new_size, const T* ip_init_value);
    void destroy_range(T* i_start, T* i_end);
    void copy(const custom_vector& i_other);

  private:
    size_t m_size;
    size_t m_capacity;
    MemoryManager::pointer<T> m_data;
  };

#if defined(ENABLED_CUDA)
template<class T>
using device_vector_ptr = device_ptr<custom_vector<T>>;

template<class T>
using managed_vector_ptr = managed_ptr<custom_vector<T>>;

#endif

#include "impl/custom_vector_impl.h"