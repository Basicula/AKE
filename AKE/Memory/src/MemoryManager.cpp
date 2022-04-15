#include "Memory/MemoryManager.h"

MemoryManager::AllocateType MemoryManager::m_current_allocate_type = MemoryManager::AllocateType::Default;

MemoryManager::AllocateType MemoryManager::GetCurrentAllocateType()
{
#ifdef __CUDA_ARCH__
  // we are calling this function from cuda device so use this type of allocation
  return MemoryManager::AllocateType::CudaDevice;
#else
  return m_current_allocate_type;
#endif
}

void MemoryManager::SetCurrentAllocateType(AllocateType i_allocate_type)
{
  m_current_allocate_type = i_allocate_type;
}
