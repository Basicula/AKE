#pragma once
#include <Image/Image.h>

#include <Rendering/Camera.h>
#include <Rendering/IRenderable.h>
#include <Rendering/KDTree.h>

#include <Visual/IMaterial.h>
#include <Visual/ILight.h>

#include <vector>
#include <memory>
#include <string>

class Scene
  {
  public:
    Scene(const std::string& i_name = "Unnamed");

    void AddObject(IRenderableSPtr i_object);
    void AddCamera(const Camera& i_camera, bool i_set_active = false);
    void AddLight(std::shared_ptr<ILight> i_light);

    std::size_t GetNumObjects() const;
    std::size_t GetNumLights() const;
    std::size_t GetNumCameras() const;

    std::shared_ptr<ILight> GetLight(size_t i_id) const;

    bool TraceRay(IntersectionRecord& o_hit, const Ray& i_ray) const;

    bool SetActiveCamera(std::size_t i_id);
    std::size_t GetActiveCameraId() const;
    const Camera& GetActiveCamera() const;

    bool SetOnOffLight(std::size_t i_id, bool i_state);

    std::string GetName() const;
    void SetName(const std::string& i_name);

    Color GetBackGroundColor() const;
    void SetBackGroundColor(const Color& i_color);

    void Update();

    std::string Serialize() const;

  private:
    std::string m_name;

    std::size_t m_active_camera;

    Color m_background_color;
    KDTree m_object_tree;

    std::vector<Camera> m_cameras;
    std::vector<std::shared_ptr<ILight>> m_lights;
  };
