#include <CUDACore/HostDeviceBuffer.h>
#include <Rendering/Image.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

__global__ void Process(Image* iop_image)
  {
  int x = threadIdx.x;
  int y = threadIdx.y;
  iop_image->SetPixel(x, y, x + y);
  }

template<class T>
__global__ void ProcessTemplate(HostDeviceBuffer<T>* io_buffer)
  {
  int x = threadIdx.x;
  (*io_buffer)[x] = x;
  }

int main()
  {
  const std::size_t width = 13;
  const std::size_t height = 11;
  Image* image_ptr = new Image(width, height);
  dim3 threads(width, height);
  Process<<<1, threads>>>(image_ptr);
  CheckCudaErrors(cudaGetLastError());
  CheckCudaErrors(cudaDeviceSynchronize());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      std::cout << image_ptr->GetPixel(x, y) << (x == width - 1 ? "\n":" ");
  CheckCudaErrors(cudaFree(image_ptr));

  //const int n = 10;
  //int* data;
  //CheckCudaErrors(cudaMallocManaged((void**)&data, n * sizeof(int)));
  //test<<<1, n>>>(data);
  //CheckCudaErrors(cudaGetLastError());
  //CheckCudaErrors(cudaDeviceSynchronize());
  //for (int i = 0; i < n; ++i)
  //  std::cout << data[i] << std::endl;

  //const auto n = 123;
  //auto buffer = new HostDeviceBuffer<std::uint32_t>(n);
  //ProcessTemplate<<<1, n>>>(buffer);
  //CheckCudaErrors(cudaGetLastError());
  //CheckCudaErrors(cudaDeviceSynchronize());
  //for (auto i = 0; i < n; ++i)
  //  std::cout << (*buffer)[i] << std::endl;
  //CheckCudaErrors(cudaFree(buffer));
  return 0;
  }