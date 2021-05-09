#pragma once

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define HOST 
#define DEVICE 
#define HOSTDEVICE 
#endif