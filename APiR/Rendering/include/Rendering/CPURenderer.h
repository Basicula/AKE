#pragma once
#include <Rendering/IRenderer.h>

class CPURenderer : public IRenderer
  {
  public:
    CPURenderer();

    virtual void Render(const Scene& i_scene) override;

  private:
    virtual void _OutputImageWasSet() override;
    void _UpdateRaysForActiveCamera(const Scene& i_scene);

    Color _TraceRay(const Scene& i_scene, std::size_t i_ray_id);
    Color _ProcessIntersection(
      const Scene& i_scene,
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessReflection(
      const Scene& i_scene,
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessRefraction(
      const Scene& i_scene,
      const IntersectionRecord& i_intersection,
      const Ray& i_camera_ray);
    Color _ProcessLightInfluence(
      const Scene& i_scene,
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