#include "Common/ThreadPool.h"

#ifdef ENABLED_CUDA
#include "CUDACore/CUDAUtils.h"
#include "CUDACore/KernelHandler.h"

#include <Memory/device_ptr.h>
#include <Memory/managed_ptr.h>
#endif

#include "Main/2DSceneExample.h"

int main()
{
  SceneExample2D();
  return 0;
}