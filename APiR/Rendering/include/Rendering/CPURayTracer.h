#pragma once
#include "Rendering/ImageRenderer.h"

class CPURayTracer : public ImageRenderer
{
public:
  explicit CPURayTracer(const Scene& i_scene);

private:
  void _GenerateFrameImage() override;

  Color _TraceRay(const Ray& i_ray);
  Color _ProcessIntersection(const Object* ip_intersected_object, double i_distance, const Ray& i_camera_ray);
  Color _ProcessReflection(const Object* ip_intersected_object, double i_distance, const Ray& i_camera_ray);
  Color _ProcessRefraction(const Object* ip_intersected_object, double i_distance, const Ray& i_camera_ray);
  Color _ProcessLightInfluence(const Object* ip_intersected_object, double i_distance, const Ray& i_camera_ray);

private:
  const Camera* mp_active_camera;
  const Scene& m_scene_source;
  std::size_t m_reflection_depth;
};