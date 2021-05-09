
template<class T>
inline T* default_allocator<T>::allocate(size_t i_count) {
  const size_t memory_block_size = i_count * sizeof(T);
  T* p_data = static_cast<T*>( malloc(memory_block_size) );
  memset(p_data, 0, memory_block_size);
  return p_data;
  }

template<class T>
template<class... Args>
inline T* default_allocator<T>::create_new(Args&&...i_args) {
  return new T(i_args...);
  }

template<class T>
template<class... Args>
void default_allocator<T>::create_new(T* iop_at_pointer, Args&&...i_args) {
  new( iop_at_pointer )T(i_args...);
  }

template<class T>
inline void default_allocator<T>::clean(T* ip_pointer) {
  free(ip_pointer);
  }
