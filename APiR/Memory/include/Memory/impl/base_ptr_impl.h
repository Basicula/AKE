template<class T>
base_ptr<T>::base_ptr()
  : mp_data(nullptr) {
  }

template<class T>
T* base_ptr<T>::operator->() {
  return mp_data;
  }

template<class T>
inline T* base_ptr<T>::get() const {
  return mp_data;
  }

template<class T>
inline T& base_ptr<T>::operator*() const {
  return *mp_data;
  }