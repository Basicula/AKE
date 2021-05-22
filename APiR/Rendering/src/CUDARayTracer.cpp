#include <Rendering/CUDARayTracer.h>

#include <device_launch_parameters.h>

namespace {
  __global__ void RayTracing(Image* iop_image, const Scene* ip_scene) {
    //double x = threadIdx.x;
    //double y = blockIdx.x;
    //const auto& camera = ip_scene->GetActiveCamera();
    //Ray camera_ray(camera.GetLocation(), camera.GetDirection(x / iop_image->GetWidth(), y / iop_image->GetHeight()));
    //IntersectionRecord hit;
    //ip_scene->TraceRay(hit, camera_ray);
    }
  }

CUDARayTracer::CUDARayTracer() 
  : m_kernel(RayTracing) {
  }

void CUDARayTracer::Render() {
  m_kernel.Run(mp_frame_image, mp_scene);
  }

void CUDARayTracer::_OutputImageWasSet() {
  m_kernel.SetNumberOfBlocks(static_cast<unsigned int>(mp_frame_image->GetHeight()));
  m_kernel.SetThreadsPerBlock(static_cast<unsigned int>(mp_frame_image->GetWidth()));
  }

void CUDARayTracer::_SceneWasSet() {
  }
