#pragma once

class MemoryManager
  {
  public:
#ifndef __CUDDAC__
    void* operator new(size_t i_len);
    void operator delete(void* ptr);
#endif
  };