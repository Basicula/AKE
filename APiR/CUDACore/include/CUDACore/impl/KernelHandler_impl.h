#include <CUDACore/CUDAUtils.h>

#include <utility>

template<class Kernel>
KernelHandler<Kernel>::KernelHandler(Kernel& i_kernel) 
  : m_kernel(i_kernel) {
  }

template<class Kernel>
template<class... Args>
void KernelHandler<Kernel>::Run(Args&&...i_args) {
  m_kernel<<<m_number_of_blocks, m_threads_per_block>>>(std::forward<Args>(i_args)...);
  CheckCudaErrors(cudaDeviceSynchronize());
  }

template<class Kernel>
void KernelHandler<Kernel>::SetNumberOfBlocks(dim3 i_number_of_blocks) {
  m_number_of_blocks = i_number_of_blocks;
  };

template<class Kernel>
void KernelHandler<Kernel>::SetThreadsPerBlock(dim3 i_threads_per_block) {
  m_threads_per_block = i_threads_per_block;
  };

template<class Kernel, class ...Args>
inline void LaunchCUDAKernel(Kernel& i_kernel, Args&&...i_args, dim3 i_number_of_blocks, dim3 i_threads_per_block) {
  m_kernel << <i_number_of_blocks, i_threads_per_block >> > (std::forward<Args>(i_args)...);
  CheckCudaErrors(cudaDeviceSynchronize());
  }