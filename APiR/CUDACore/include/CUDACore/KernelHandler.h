#pragma once
#include <cuda_runtime.h>

template<class Kernel>
class KernelHandler
  {
  public:
    KernelHandler(Kernel& i_kernel);

    template<class... Args>
    void Run(Args&&...i_args);

    void SetNumberOfBlocks(dim3 i_number_of_blocks);
    void SetThreadsPerBlock(dim3 i_threads_per_block);

  private:
    dim3 m_number_of_blocks;
    dim3 m_threads_per_block;
    Kernel& m_kernel;
  };

template<class Kernel, class...Args>
void LaunchCUDAKernel(Kernel& i_kernel, Args&&... i_args, dim3 i_number_of_blocks, dim3 i_threads_per_block);

#include "impl/KernelHandler_impl.h"