#include "Common/ThreadPool.h"

#ifdef ENABLED_CUDA
#include "CUDACore/CUDAUtils.h"
#include "CUDACore/KernelHandler.h"

#include <Memory/device_ptr.h>
#include <Memory/managed_ptr.h>
#endif

#include "Main/2DSceneExample.h"
#include "Main/3DSceneExample.h"

int main()
{
  Scene2DExamples::RotatedRectangles(800, 800, 10);
  //SceneExample3D();
  return 0;
}