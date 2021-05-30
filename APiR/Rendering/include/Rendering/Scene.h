#pragma once
#include <Image/Image.h>

#include <Rendering/Camera.h>
#include <Rendering/Object.h>
#include <Rendering/KDTree.h>

#include <Visual/IVisualMaterial.h>
#include <Visual/ILight.h>

#include <vector>
#include <memory>
#include <string>

class Scene final
  {
  public:
    Scene(const std::string& i_name = "Unnamed");
    Scene(Scene&& i_other) noexcept;
    ~Scene();

    void AddObject(Object* i_object);
    void AddCamera(const Camera& i_camera, bool i_set_active = false);
    void AddLight(ILight* i_light);

    std::size_t GetNumObjects() const;
    std::size_t GetNumLights() const;
    std::size_t GetNumCameras() const;

    const ILight* GetLight(size_t i_id) const;

    HOSTDEVICE const Object* TraceRay(double& o_distance, const Ray& i_ray) const;

    bool SetActiveCamera(std::size_t i_id);
    std::size_t GetActiveCameraId() const;
    HOSTDEVICE Camera& GetActiveCamera();
    HOSTDEVICE const Camera& GetActiveCamera() const;

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
    Container* mp_object_container;
    std::vector<Camera> m_cameras;
    std::vector<ILight*> m_lights;
  };
