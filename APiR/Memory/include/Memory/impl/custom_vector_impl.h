#include <algorithm>

#if defined(__CUDACC__)
#include <device_launch_parameters.h>
#endif

// Iterator start
template<class T>
inline custom_vector<T>::iterator::iterator(T* ip_pointer)
  : mp_pointer(ip_pointer) {
  }

template<class T>
inline bool custom_vector<T>::iterator::operator==(const custom_vector<T>::iterator& i_other) const {
  return mp_pointer == i_other.mp_pointer;
  }

template<class T>
inline bool custom_vector<T>::iterator::operator!=(const custom_vector<T>::iterator& i_other) const {
  return mp_pointer != i_other.mp_pointer;
  }

template<class T>
inline T custom_vector<T>::iterator::operator*() const {
  return *mp_pointer;
  }

template<class T>
inline typename custom_vector<T>::iterator& custom_vector<T>::iterator::operator++() {
  ++mp_pointer;
  return *this;
  }
// Iterator end

// vector start
template<class T>
inline custom_vector<T>::custom_vector()
  : m_size(0)
  , m_capacity(0)
  , m_data{ MemoryManager::GetCurrentAllocateType() , nullptr } {
  }

template<class T>
inline custom_vector<T>::custom_vector(size_t i_size)
  : custom_vector() {
  resize(i_size);
  }

template<class T>
inline custom_vector<T>::custom_vector(size_t i_size, const T& i_init_value)
  : custom_vector() {
  resize(i_size, i_init_value);
  }

template<class T>
inline custom_vector<T>::custom_vector(const std::initializer_list<T>& i_list)
  : custom_vector() {
  resize(i_list.size());
  std::copy(i_list.begin(), i_list.end(), m_data.mp_data);
  }

template<class T>
inline custom_vector<T>::custom_vector(const custom_vector& i_other)
  : custom_vector() {
  copy(i_other);
  }

template<class T>
inline custom_vector<T>::custom_vector(custom_vector&& i_other)
  : m_size(std::move(i_other.m_size))
  , m_capacity(std::move(i_other.m_capacity))
  , m_data(std::move(i_other.m_data)) {
  }

template<class T>
inline custom_vector<T>::~custom_vector() {
  if (m_data.mp_data != nullptr) {
    MemoryManager::clean(m_data);
    m_data.mp_data = nullptr;
    }
  }

#if defined(__CUDACC__)
template<class T>
__global__ void set_up_vector(MemoryManager::pointer<T>* iop_pointer, size_t* iop_size, size_t* iop_capacity, size_t i_size, T* ip_pointer_to_data) {
  iop_pointer->m_allocate_type = MemoryManager::AllocateType::CudaDevice;
  iop_pointer->mp_data = ip_pointer_to_data;
  *iop_size = i_size;
  *iop_capacity = i_size;
  }

template<class T>
__global__ void fill_data(T* iop_data, const T* ip_init_value) {
  int i = threadIdx.x;
  iop_data[i] = *ip_init_value;
  }

template<class T>
inline device_ptr<custom_vector<T>> custom_vector<T>::device_vector_ptr(size_t i_size) {
  device_ptr<custom_vector<T>> result;
  T* data_ptr = MemoryManager::allocate<T>(MemoryManager::AllocateType::CudaDevice, i_size).mp_data;
  set_up_vector<T> << <1, 1 >> > (&result->m_data, &result->m_size, &result->m_capacity, i_size, data_ptr);
  CheckCudaErrors(cudaDeviceSynchronize());
  return result;
  }

template<class T>
inline device_ptr<custom_vector<T>> custom_vector<T>::device_vector_ptr(size_t i_size, const T& i_init_value) {
  device_ptr<custom_vector<T>> result;
  T* data_ptr = MemoryManager::allocate<T>(MemoryManager::AllocateType::CudaDevice, i_size).mp_data;
  T* d_init_value;
  CheckCudaErrors(cudaMemcpy(d_init_value, &i_init_value, sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
  fill_data<T> << <1, i_size >> > (data_ptr, d_init_value);
  CheckCudaErrors(cudaDeviceSynchronize());
  set_up_vector<T> << <1, 1 >> > (&result->m_data, &result->m_size, &result->m_capacity, i_size, data_ptr);
  CheckCudaErrors(cudaDeviceSynchronize());
  return result;
  }

template<class T>
inline device_ptr<custom_vector<T>> custom_vector<T>::device_vector_ptr(const std::initializer_list<T>& i_list) {
  device_ptr<custom_vector<T>> result;
  const size_t size = i_list.size();
  T* data_ptr = MemoryManager::allocate<T>(MemoryManager::AllocateType::CudaDevice, size).mp_data;
  CheckCudaErrors(cudaMemcpy(data_ptr, i_list.begin(), sizeof(T) * size, cudaMemcpyKind::cudaMemcpyHostToDevice));
  set_up_vector<T> << <1, 1 >> > (&result->m_data, &result->m_size, &result->m_capacity, size, data_ptr);
  CheckCudaErrors(cudaDeviceSynchronize());
  return result;
  }
#endif

template<class T>
inline custom_vector<T>& custom_vector<T>::operator=(const custom_vector<T>& i_other) {
  if (this != &i_other)
    copy(i_other);
  return *this;
  }

template<class T>
inline T& custom_vector<T>::operator[](size_t i_index) {
  return m_data.mp_data[i_index];
  }

template<class T>
inline T& custom_vector<T>::operator[](size_t i_index) const {
  return m_data.mp_data[i_index];
  }

template<class T>
template<class... Args>
inline T& custom_vector<T>::emplace_back(Args&& ...i_args) {
  if (m_size == m_capacity)
    resize(m_size + 1);
  else
    ++m_size;
  auto* obj_ptr = m_data.mp_data + m_size - 1;
  MemoryManager::create_new(m_data.m_allocate_type, obj_ptr, i_args...);
  return *obj_ptr;
  }

template<class T>
inline void custom_vector<T>::push_back(const T& i_val) {
  if (m_size == m_capacity)
    resize(m_size + 1);
  else
    ++m_size;
  m_data.mp_data[m_size - 1] = i_val;
  }

template<class T>
inline void custom_vector<T>::resize(size_t i_new_size) {
  resize(i_new_size, nullptr);
  }

template<class T>
inline void custom_vector<T>::resize(size_t i_new_size, const T& i_init_value) {
  resize(i_new_size, &i_init_value);
  }

template<class T>
inline void custom_vector<T>::clear() {
  destroy_range(m_data.mp_data, m_data.mp_data + m_size);
  m_size = 0;
  }

template<class T>
inline T* custom_vector<T>::data() const {
    return m_data.mp_data;
    }

template<class T>
inline size_t custom_vector<T>::size() const {
  return m_size;
  }

template<class T>
inline size_t custom_vector<T>::capacity() const {
  return m_capacity;
  }

template<class T>
inline bool custom_vector<T>::empty() const {
  return m_size == 0;
  }

template<class T>
inline typename custom_vector<T>::iterator custom_vector<T>::begin() const {
  return iterator(m_data.mp_data);
  }

template<class T>
inline typename custom_vector<T>::iterator custom_vector<T>::end() const {
  return iterator(m_data.mp_data + m_size);
  }

template<class T>
inline void custom_vector<T>::resize(size_t i_new_size, const T* ip_init_value) {
  if (i_new_size < m_size)
    destroy_range(m_data.mp_data + i_new_size, m_data.mp_data + m_size);

  if (i_new_size > m_size) {
    if (i_new_size > m_capacity) {
      m_capacity += m_capacity / 2;
      if (i_new_size > m_capacity)
        m_capacity = i_new_size;
      auto new_data = MemoryManager::allocate<T>(m_data.m_allocate_type, m_capacity);
      auto* obj_ptr = m_data.mp_data;
      auto* new_obj_ptr = new_data.mp_data;
      const auto* last = m_data.mp_data + m_size;
      if (std::is_move_constructible<T>::value) {
        for (; obj_ptr < last; ++obj_ptr, ++new_obj_ptr)
          *new_obj_ptr = std::move(*obj_ptr);
        }
      else {
        for (; obj_ptr < last; ++obj_ptr, ++new_obj_ptr)
          *new_obj_ptr = *obj_ptr;
        }
      MemoryManager::clean<T>(m_data);
      m_data = new_data;
      }
    if (ip_init_value != nullptr) {
      auto* obj_ptr = m_data.mp_data + m_size;
      const auto* last = m_data.mp_data + i_new_size;
      for (; obj_ptr < last; ++obj_ptr)
        *obj_ptr = *ip_init_value;
      }
    }
  m_size = i_new_size;
  }

template<class T>
inline void custom_vector<T>::destroy_range(T* i_start, T* i_end) {
  for (; i_start < i_end; ++i_start)
    i_start->~T();
  }

template<class T>
inline void custom_vector<T>::copy(const custom_vector& i_other) {
  MemoryManager::clean<T>(m_data);
  m_data.m_allocate_type = i_other.m_data.m_allocate_type;
  MemoryManager::allocate<T>(m_data, i_other.m_capacity);
  MemoryManager::copy(i_other.m_data, i_other.m_size, m_data);
  m_size = i_other.m_size;
  m_capacity = i_other.m_capacity;
  }
// vector end