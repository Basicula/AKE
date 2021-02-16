#include <CUDACore/CUDAUtils.h>
#include <CUDACore/MemoryManager.h>

#include <cuda_runtime.h>

void* MemoryManager::operator new(size_t i_len)
  {
  void* ptr;
  CheckCudaErrors(cudaMallocManaged(&ptr, i_len));
  CheckCudaErrors(cudaDeviceSynchronize());
  return ptr;
  }

void MemoryManager::operator delete(void* ptr) 
  {
  CheckCudaErrors(cudaDeviceSynchronize());
  CheckCudaErrors(cudaFree(ptr));
  }