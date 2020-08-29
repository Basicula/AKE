#pragma once
#include <Rendering/Camera.h>
#include <Rendering/Image.h>
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
    Scene(
      const std::string& i_name = "Unnamed", 
      std::size_t i_frame_width = 800, 
      std::size_t i_frame_height = 600);

    void AddObject(IRenderableSPtr i_object);
    void AddCamera(const Camera& i_camera, bool i_set_active = false);
    void AddLight(std::shared_ptr<ILight> i_light);

    std::size_t GetNumObjects() const;
    std::size_t GetNumLights() const;
    std::size_t GetNumCameras() const;

    void ClearObjects();
    void ClearCameras();
    void ClearLights();

    bool SetActiveCamera(std::size_t i_id);
    std::size_t GetActiveCamera() const;

    bool SetOnOffLight(std::size_t i_id, bool i_state);

    std::string GetName() const;
    void SetName(const std::string& i_name);

    std::size_t GetFrameWidth() const;
    void SetFrameWidth(std::size_t i_frame_width);
    std::size_t GetFrameHeight() const;
    void SetFrameHeight(std::size_t i_frame_height);

    Color GetBackGroundColor() const;
    void SetBackGroundColor(const Color& i_color);

    std::size_t GetRenderingDepth() const;
    void SetRenderingDepth(std::size_t i_depth);

    bool RenderFrame(
      Image& o_image,
      int i_offset_x = 0,
      int i_offset_y = 0);
    bool RenderCameraFrame(
      Image& o_image, 
      std::size_t i_camera,
      int i_offset_x = 0,
      int i_offset_y = 0);

    void Update();

    std::string Serialize() const;
  private:
    bool _Render(
      Image& o_image, 
      std::size_t i_camera, 
      int i_offset_x, 
      int i_offset_y);

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

    void _UpdateRaysForActiveCamera();

  private:
    std::string m_name;

    std::size_t m_active_camera;
    std::size_t m_frame_width;
    std::size_t m_frame_height;
    std::size_t m_max_depth;

    Color m_background_color;
    KDTree m_object_tree;

    std::vector<Camera> m_cameras;
    std::vector<std::shared_ptr<ILight>> m_lights;

    // we can create all rays for camera image
    // only once and then if it needed
    // reset their origin and dirs
    std::vector<Ray> m_rays;
    std::vector<IntersectionRecord> m_intersection_records;
  };

inline void Scene::AddObject(IRenderableSPtr i_object)
  {
  m_object_tree.AddObject(i_object);
  }

inline void Scene::AddCamera(const Camera& i_camera, bool i_set_active)
  {
  m_cameras.push_back(i_camera);
  if (i_set_active)
    SetActiveCamera(m_cameras.size()-1);
  }

inline void Scene::AddLight(std::shared_ptr<ILight> i_light)
  {
  m_lights.push_back(i_light);
  }

inline std::size_t Scene::GetNumObjects() const
  {
  return m_object_tree.Size();
  }

inline std::size_t Scene::GetNumLights() const
  {
  return m_lights.size();
  }

inline std::size_t Scene::GetNumCameras() const
  {
  return m_cameras.size();
  }

inline bool Scene::SetActiveCamera(std::size_t i_id)
  {
  if (i_id < m_cameras.size())
    {
    m_active_camera = i_id;
    _UpdateRaysForActiveCamera();
    return true;
    }
  return false;
  }

inline std::size_t Scene::GetActiveCamera() const
  {
  return m_active_camera;
  }

inline std::string Scene::GetName() const
  {
  return m_name;
  }

inline void Scene::SetName(const std::string& i_name)
  {
  m_name = i_name;
  }

inline void Scene::ClearObjects()
  {
  m_object_tree.Clear();
  }

inline void Scene::ClearCameras()
  {
  m_cameras.clear();
  }

inline void Scene::ClearLights()
  {
  m_lights.clear();
  }

inline bool Scene::SetOnOffLight(std::size_t i_id, bool i_state)
  {
  if (i_id < m_lights.size())
    {
    m_lights[i_id]->SetState(i_state);
    return true;
    }
  return false;
  }

inline std::string Scene::Serialize() const
  {
  std::string res = "{ \"Scene\" : { ";
  res += "\"Name\" : \"" + m_name + "\", ";

  // TODO
  //m_object_tree.Serialize();

  const auto cameras_cnt = m_cameras.size();
  if (cameras_cnt != 0)
    {
    res += "\"Cameras\" : [ ";
    for (auto i = 0u; i < cameras_cnt; ++i)
      res += m_cameras[i].Serialize() + (i == cameras_cnt - 1 ? " ], " : ", ");
    }

  const auto lights_cnt = m_lights.size();
  if (lights_cnt != 0)
    {
    res += "\"Lights\" : [ ";
    for (auto i = 0u; i < lights_cnt; ++i)
      res += m_lights[i]->Serialize() + (i == lights_cnt - 1 ? " ], " : ", ");
    }

  res += "\"ActiveCamera\" : " + std::to_string(m_active_camera) + ", ";
  res += "\"FrameWidth\" : " + std::to_string(m_frame_width) + ", ";
  res += "\"FrameHeight\" : " + std::to_string(m_frame_height);
  res += "} }";
  return res;
  }

inline std::size_t Scene::GetFrameWidth() const
  {
  return m_frame_width;
  }

inline void Scene::SetFrameWidth(std::size_t i_frame_width)
  {
  m_frame_width = i_frame_width;
  }

inline std::size_t Scene::GetFrameHeight() const
  {
  return m_frame_height;
  }

inline void Scene::SetFrameHeight(std::size_t i_frame_height)
  {
  m_frame_height = i_frame_height;
  }

inline void Scene::SetBackGroundColor(const Color& i_color)
  {
  m_background_color = i_color;
  }

inline std::size_t Scene::GetRenderingDepth() const
  {
  return m_max_depth;
  }

inline void Scene::SetRenderingDepth(std::size_t i_depth)
  {
  m_max_depth = i_depth;
  }

inline Color Scene::GetBackGroundColor() const
  {
  return m_background_color;
  }
