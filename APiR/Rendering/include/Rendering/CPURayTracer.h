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
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessReflection(
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessRefraction(
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessLightInfluence(
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);

  private:
    const Camera* m_active_camera;
    // we can create all rays for camera image
    // only once and then if it needed
    // reset their origin and dirs
    std::vector<Ray> m_rays;
    std::vector<IntersectionRecord> m_intersection_records;
  };