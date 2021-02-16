#pragma once
#include <cuda_runtime.h>

template<class T, class... Args>
void cudaDeviceNew(T* iop_data, Args... i_args);