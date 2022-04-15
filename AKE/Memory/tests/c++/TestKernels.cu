#include "TestKernels.h"

#include <Memory/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add(int* a, int* b, int* c) {
  *c = *a + *b;
  }

__global__ void _set(B* b) {
  b->setb(-2);
  b->seta(2);
  }

__global__ void _fill_vector(custom_vector<int>* iop_vector) {
  ( *iop_vector )[threadIdx.x] = threadIdx.x;
  }

void add(int* a, int* b, int& c) {
  device_ptr<int> d_c(0);
  add << <1, 1 >> > ( a, b, d_c.get() );
  cudaDeviceSynchronize();
  c = d_c.get_host_copy();
  }

void set(B* b) {
  _set << <1, 1 >> > ( b );
  cudaDeviceSynchronize();
  }

void fill_vector(custom_vector<int>* iop_vector, size_t i_size) {
  _fill_vector << <1, i_size >> > ( iop_vector );
  cudaDeviceSynchronize();
  }
