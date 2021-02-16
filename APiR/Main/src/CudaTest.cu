#include "CudaTest.h"
#include <CUDACore/HostDevicePtr.h>

#include <device_launch_parameters.h>

#include <iostream>

__global__ void process_fractal(Image* iop_image, const MandelbrotSet* ip_fractal)
  {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  iop_image->SetPixel(x, y, ip_fractal->GetColor(x, y));
  }

void cuda_fractal(
  Image* iop_image,
  const MandelbrotSet* ip_fractal)
  {
  dim3 threads(16, 16);
  dim3 blocks(iop_image->GetWidth() / threads.x, iop_image->GetHeight() / threads.y);
  device_ptr<MandelbrotSet> set(iop_image->GetWidth(), iop_image->GetHeight(), 100);
  process_fractal << <blocks, threads >> > (iop_image, set.get());
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  }

class A
  {
  public:
    HOSTDEVICE A(int x) : x(x) {}
    HOSTDEVICE virtual int getx() const { return x; };
  private:
    int x;
  };

class B : public A
  {
  public:
    HOSTDEVICE B(int x, int y) : A(x), y(y) {}
    HOSTDEVICE int gety() const { return y; };
    HOSTDEVICE virtual int getx() const { return y * 2; };
  private:
    int y;
  };

__global__ void test_kernel(A* ip_b, int* x, int* y)
  {
  auto b = static_cast<B*>(ip_b);
  *x = b->getx();
  *y = b->gety();
  }

void test_cuda_kernel()
  {
  const std::size_t width = 1024;
  const std::size_t height = 768;
  dim3 threads(16, 16);
  dim3 blocks(width / threads.x, height / threads.y);
  int* x, * y;
  CheckCudaErrors(cudaMallocManaged(&x, sizeof(int)));
  CheckCudaErrors(cudaMallocManaged(&y, sizeof(int)));
  device_ptr<B> b(21, 22);
  test_kernel << <blocks, threads >> > (b.get(), x, y);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  std::cout << *x << " " << *y << std::endl;
  }