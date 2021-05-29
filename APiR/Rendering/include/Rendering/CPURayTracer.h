#pragma once
#include <Rendering/IRenderer.h>

class CPURayTracer : public IRenderer
  {
  public:
    CPURayTracer();

    virtual void Render() override;

  private:
    virtual void _OutputImageWasSet() override;
    virtual void _SceneWasSet() override;

    Color _TraceRay(std::size_t i_ray_id);
    Color _ProcessIntersection(
      const Object* ip_intersected_object,
      const double i_distance,
      const Ray& i_camera_ray);
    Color _ProcessReflection(
      const Object* ip_intersected_object,
      const double i_distance,
      const Ray& i_camera_ray);
    Color _ProcessRefraction(
      const Object* ip_intersected_object,
      const double i_distance,
      const Ray& i_camera_ray);
    Color _ProcessLightInfluence(
      const Object* ip_intersected_object,
      const double i_distance,
      const Ray& i_camera_ray);

  private:
    const Camera* m_active_camera;
    // we can create all rays for camera image
    // only once and then if it needed
    // reset their origin and dirs
    std::vector<Ray> m_rays;
  };