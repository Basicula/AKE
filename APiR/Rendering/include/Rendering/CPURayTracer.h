#pragma once
#include "Rendering/IRenderer.h"

class CPURayTracer : public IRenderer
{
public:
  CPURayTracer();

  virtual void Render() override;

private:
  virtual void _OutputImageWasSet() override;
  virtual void _SceneWasSet() override;

  Color _TraceRay(const Ray& i_ray);
  Color _ProcessIntersection(const Object* ip_intersected_object, const double i_distance, const Ray& i_camera_ray);
  Color _ProcessReflection(const Object* ip_intersected_object, const double i_distance, const Ray& i_camera_ray);
  Color _ProcessRefraction(const Object* ip_intersected_object, const double i_distance, const Ray& i_camera_ray);
  Color _ProcessLightInfluence(const Object* ip_intersected_object, const double i_distance, const Ray& i_camera_ray);

private:
  const Camera* mp_active_camera;
};