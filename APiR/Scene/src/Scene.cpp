#include <Scene/Scene.h>
#include <Common/ThreadPool.h>

Scene::Scene(
  const std::string& i_name,
  std::size_t i_frame_width,
  std::size_t i_frame_height)
  : m_name(i_name)
  , m_active_camera(static_cast<std::size_t>(-1))
  , m_frame_width(i_frame_width)
  , m_frame_height(i_frame_height)
  , m_max_depth(3)
  , m_background_color(0xffffccaa)
  , m_object_tree()
  , m_rays()
  , m_intersection_records(m_frame_height* m_frame_width)
  {
  }

bool Scene::RenderFrame(
  Image& o_image,
  int i_offset_x,
  int i_offset_y)
  {
  if (m_active_camera == static_cast<std::size_t>(-1))
    throw "No active camera has been set";
  return _Render(
    o_image,
    m_active_camera,
    i_offset_x,
    i_offset_y);
  }

bool Scene::RenderCameraFrame(
  Image& o_image,
  std::size_t i_camera,
  int i_offset_x,
  int i_offset_y)
  {
  if (i_camera >= m_cameras.size())
    return false;
  return _Render(
    o_image,
    i_camera,
    i_offset_x,
    i_offset_y);
  }

bool Scene::_Render(
  Image& o_image,
  std::size_t i_camera,
  int i_offset_x,
  int i_offset_y)
  {
  // TODO : UPDATE THIS VERY BAD CODE)
  if (i_camera != m_active_camera)
    return false;
  //for (std::size_t i_pixel_id = 0; i_pixel_id < m_frame_height * m_frame_width; ++i_pixel_id)
  ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0),
    m_frame_width * m_frame_height,
    [&](std::size_t i_pixel_id)
    {
    auto x = i_pixel_id % m_frame_width + i_offset_x;
    auto y = i_pixel_id / m_frame_width + i_offset_y;
    o_image.SetPixel(x, y, _TraceRay(i_pixel_id));
    }
  );
  return true;
  }

void Scene::_UpdateRaysForActiveCamera()
  {
  m_rays.clear();
  const auto& camera = m_cameras[m_active_camera];

  const auto& ray_origin = camera.GetLocation();
  for (auto y = 0.0; y < m_frame_height; ++y)
    for (auto x = 0.0; x < m_frame_width; ++x)
    {
    const auto& ray_dir = camera.GetDirection(
      x / m_frame_width,
      y / m_frame_height);
    m_rays.emplace_back(ray_origin, ray_dir);
    }
  }

Color Scene::_TraceRay(std::size_t i_ray_id)
  {
  auto& hit = m_intersection_records[i_ray_id];
  const auto& camera_ray = m_rays[i_ray_id];

  hit.Reset();
  bool intersected = m_object_tree.IntersectWithRay(hit, camera_ray);

  if (!intersected || !hit.m_material)
    return m_background_color;

  return _ProcessIntersection(hit, camera_ray);
  }

Color Scene::_ProcessIntersection(
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)
  {
  Color reflected_color;
  if (i_intersection.m_material->IsReflectable())
    reflected_color = _ProcessReflection(i_intersection, i_camera_ray);

  Color refracted_color;
  if (i_intersection.m_material->IsRefractable())
    refracted_color = _ProcessRefraction(i_intersection, i_camera_ray);

  Color light_influence = _ProcessLightInfluence(i_intersection, i_camera_ray);
  return light_influence + reflected_color;
  }

Color Scene::_ProcessReflection(
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)
  {
  std::size_t depth = 0;
  IntersectionRecord local_record(i_intersection);
  Ray local_ray(i_camera_ray);
  while (local_record.m_material->IsReflectable() && depth < m_max_depth)
    {
    const auto reflected_direction =
      local_record.m_material->ReflectedDirection(local_record.m_normal, i_camera_ray.GetDirection());
    local_ray.SetOrigin(local_record.m_intersection);
    local_ray.SetDirection(reflected_direction);
    local_record.Reset();
    if (m_object_tree.IntersectWithRay(local_record, local_ray))
      ++depth;
    else
      return m_background_color;
    }
  return _ProcessLightInfluence(local_record, i_camera_ray) * i_intersection.m_material->ReflectionInfluence();
  }

Color Scene::_ProcessRefraction(
  const IntersectionRecord& /*i_intersection*/,
  const Ray& /*i_camera_ray*/)
  {
  //double refraction_coef = i_ray.GetEnvironment() / object->GetMaterial().GetRefraction();
  //double cos_n_ray = i_ray.GetDirection().Dot(-normal);
  //bool is_refracted = sqrt(1 - cos_n_ray * cos_n_ray) < (1.0 / refraction_coef);
  //if (is_refracted)
  //  {
  //  Vector3d refraction_parallel_part = (i_ray.GetDirection() + normal * cos_n_ray) * refraction_coef;
  //  Vector3d refraction_perpendicular_part = -normal * sqrt(1 - refraction_parallel_part.SquareLength());
  //  Vector3d refraction_dir = refraction_parallel_part + refraction_perpendicular_part;
  //  Ray refraction_ray(intersection, refraction_dir, object->GetMaterial().GetRefraction());
  //  refracted_color = CastRay(refraction_ray, i_objects, i_lights, depth - 1);
  //  }
  return Color();
  }

Color Scene::_ProcessLightInfluence(
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)
  {
  const auto& view_direction = i_camera_ray.GetDirection();
  Color result_pixel_color = i_intersection.m_material->GetPrimitiveColor();
  std::size_t red, green, blue, active_lights_cnt;
  red = green = blue = active_lights_cnt = 0;
  Ray to_light(i_intersection.m_intersection + i_intersection.m_normal * 1e-10, Vector3d(0, 1, 0));
  for (auto i = 0u; i < m_lights.size(); ++i)
    {
    const auto& light = m_lights[i];
    if (!light->GetState())
      continue;
    ++active_lights_cnt;
    const auto light_direction = light->GetDirection(i_intersection.m_intersection);
    to_light.SetDirection(-light_direction);
    IntersectionRecord temp_intersection;
    const bool is_intersected = m_object_tree.IntersectWithRay(temp_intersection, to_light);
    if (!is_intersected || light_direction.Dot(light->GetDirection(temp_intersection.m_intersection)) < 0.0)
      {
      Color light_influence =
        i_intersection.m_material->GetLightInfluence(
          i_intersection.m_intersection,
          i_intersection.m_normal,
          view_direction,
          light);
      red += light_influence.GetRed();
      green += light_influence.GetGreen();
      blue += light_influence.GetBlue();
      }
    }
  if (active_lights_cnt > 0)
    {
    //result_pixel_color += Color(
    //  static_cast<uint8_t>(red / active_lights_cnt),
    //  static_cast<uint8_t>(green / active_lights_cnt),
    //  static_cast<uint8_t>(blue / active_lights_cnt));
    result_pixel_color += Color(
      static_cast<uint8_t>(red > 255 ? 255 : red),
      static_cast<uint8_t>(green > 255 ? 255 : green),
      static_cast<uint8_t>(blue > 255 ? 255 : blue));
    }
  return result_pixel_color;
  }

void Scene::Update()
  {
  m_object_tree.Update();
  }