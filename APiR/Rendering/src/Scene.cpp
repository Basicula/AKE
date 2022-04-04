#include "Rendering/Scene.h"

#include "Common/ThreadPool.h"

Scene::Scene(const std::string& i_name)
  : m_name(i_name)
  , m_active_camera(static_cast<std::size_t>(-1))
  , m_background_color(0xffffccaa)
  , mp_object_container(nullptr)
{
  // mp_object_container = new Container;
  mp_object_container = new KDTree;
}

Scene::Scene(Scene&& i_other) noexcept
  : m_name(i_other.m_name)
  , m_active_camera(i_other.m_active_camera)
  , m_background_color(i_other.m_background_color)
  , mp_object_container(i_other.mp_object_container)
  , m_cameras(std::move(i_other.m_cameras))
  , m_lights(std::move(i_other.m_lights))
{
  i_other.mp_object_container = nullptr;
}

Scene::~Scene()
{
  if (mp_object_container)
    delete mp_object_container;
  for (auto& light : m_lights)
    delete light;
  for (auto& camera : m_cameras)
    delete camera;
}

void Scene::AddObject(Object* ip_object)
{
  mp_object_container->AddObject(ip_object);
}

void Scene::AddCamera(Camera* ip_camera, bool i_set_active)
{
  m_cameras.push_back(ip_camera);
  if (i_set_active)
    SetActiveCamera(m_cameras.size() - 1);
}

void Scene::AddLight(ILight* ip_light)
{
  m_lights.push_back(ip_light);
}

std::size_t Scene::GetNumObjects() const
{
  return mp_object_container->Size();
}

std::size_t Scene::GetNumLights() const
{
  return m_lights.size();
}

std::size_t Scene::GetNumCameras() const
{
  return m_cameras.size();
}

bool Scene::SetActiveCamera(std::size_t i_id)
{
  if (i_id < m_cameras.size()) {
    m_active_camera = i_id;
    return true;
  }
  return false;
}

std::size_t Scene::GetActiveCameraId() const
{
  return m_active_camera;
}

Camera* Scene::GetActiveCamera()
{
  return m_cameras[m_active_camera];
}

HOSTDEVICE const Camera* Scene::GetActiveCamera() const
{
  return m_cameras[m_active_camera];
}

std::string Scene::GetName() const
{
  return m_name;
}

void Scene::SetName(const std::string& i_name)
{
  m_name = i_name;
}

bool Scene::SetOnOffLight(std::size_t i_id, bool i_state)
{
  if (i_id < m_lights.size()) {
    m_lights[i_id]->SetState(i_state);
    return true;
  }
  return false;
}

std::string Scene::Serialize() const
{
  std::string res = "{ \"Scene\" : { ";
  res += "\"Name\" : \"" + m_name + "\", ";

  // TODO
  // mp_object_container->Serialize();
  // const auto cameras_cnt = m_cameras.size();
  // if (cameras_cnt != 0)     {
  //  res += "\"Cameras\" : [ ";
  //  for (auto i = 0u; i < cameras_cnt; ++i)
  //    res += m_cameras[i].Serialize() + (i == cameras_cnt - 1 ? " ], " : ", ");
  //  }

  const auto lights_cnt = m_lights.size();
  if (lights_cnt != 0) {
    res += "\"Lights\" : [ ";
    for (auto i = 0u; i < lights_cnt; ++i)
      res += m_lights[i]->Serialize() + (i == lights_cnt - 1 ? " ], " : ", ");
  }

  res += "\"ActiveCamera\" : " + std::to_string(m_active_camera);
  res += "} }";
  return res;
}

void Scene::SetBackGroundColor(const Color& i_color)
{
  m_background_color = i_color;
}

Color Scene::GetBackGroundColor() const
{
  return m_background_color;
}

const ILight* Scene::GetLight(size_t i_id) const
{
  return m_lights[i_id];
}

const Object* Scene::TraceRay(double& o_hit, const Ray& i_ray) const
{
  // TODO: implement far distance logic
  return mp_object_container->TraceRay(o_hit, i_ray, 1000);
}

void Scene::Update()
{
  mp_object_container->Update();
}