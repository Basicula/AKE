#include <Rendering/CPURayTracer.h>

#include <Common/ThreadPool.h>

CPURayTracer::CPURayTracer() 
  : m_active_camera(nullptr){
  }

void CPURayTracer::Render(const Scene& i_scene) {
  if (m_active_camera == nullptr || &i_scene.GetActiveCamera() != m_active_camera)
    _UpdateRaysForActiveCamera(i_scene);
  //for(auto i_pixel_id = 0; i_pixel_id < mp_frame_image->GetSize(); ++i_pixel_id)
  ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0),
    mp_frame_image->GetSize(),
    [&](std::size_t i_pixel_id)
    {
    auto x = i_pixel_id % mp_frame_image->GetWidth();
    auto y = i_pixel_id / mp_frame_image->GetWidth();
    mp_frame_image->SetPixel(x, y, _TraceRay(i_scene, i_pixel_id));
    }
  );
  }

void CPURayTracer::_OutputImageWasSet() {
  m_rays.resize(mp_frame_image->GetSize(), Ray(Vector3d(0), Vector3d(0)));
  m_intersection_records.resize(mp_frame_image->GetSize());
  }

void CPURayTracer::_UpdateRaysForActiveCamera(const Scene& i_scene)   {
  m_rays.clear();
  m_active_camera = &i_scene.GetActiveCamera();

  const auto& ray_origin = m_active_camera->GetLocation();
  for (auto y = 0.0; y < mp_frame_image->GetHeight(); ++y)
    for (auto x = 0.0; x < mp_frame_image->GetWidth(); ++x)     {
      const auto& ray_dir = m_active_camera->GetDirection(
        x / mp_frame_image->GetWidth(),
        y / mp_frame_image->GetHeight());
      m_rays.emplace_back(ray_origin, ray_dir);
      }
  }

Color CPURayTracer::_TraceRay(const Scene& i_scene, std::size_t i_ray_id)   {
  auto& hit = m_intersection_records[i_ray_id];
  const auto& camera_ray = m_rays[i_ray_id];

  hit.Reset();
  bool intersected = i_scene.TraceRay(hit, camera_ray);

  if (!intersected || !hit.m_material)
    return i_scene.GetBackGroundColor();

  return _ProcessIntersection(i_scene, hit, camera_ray);
  }

Color CPURayTracer::_ProcessIntersection(
  const Scene& i_scene,
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)   {
  Color reflected_color;
  if (i_intersection.m_material->IsReflectable())
    reflected_color = _ProcessReflection(i_scene, i_intersection, i_camera_ray);

  Color refracted_color;
  if (i_intersection.m_material->IsRefractable())
    refracted_color = _ProcessRefraction(i_scene, i_intersection, i_camera_ray);

  Color light_influence = _ProcessLightInfluence(i_scene, i_intersection, i_camera_ray);
  return light_influence + reflected_color;
  }

Color CPURayTracer::_ProcessReflection(
  const Scene& i_scene,
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)   {
  std::size_t depth = 0;
  IntersectionRecord local_record(i_intersection);
  Ray local_ray(i_camera_ray);
  while (local_record.m_material->IsReflectable() && depth < m_depth)     {
    const auto reflected_direction =
      local_record.m_material->ReflectedDirection(local_record.m_normal, i_camera_ray.GetDirection());
    local_ray.SetOrigin(local_record.m_intersection);
    local_ray.SetDirection(reflected_direction);
    local_record.Reset();
    if (i_scene.TraceRay(local_record, local_ray))
      ++depth;
    else
      return i_scene.GetBackGroundColor();
    }
  return _ProcessLightInfluence(i_scene, local_record, i_camera_ray) * i_intersection.m_material->ReflectionInfluence();
  }

Color CPURayTracer::_ProcessRefraction(
  const Scene& /*i_scene*/,
  const IntersectionRecord& /*i_intersection*/,
  const Ray& /*i_camera_ray*/)   {
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

Color CPURayTracer::_ProcessLightInfluence(
  const Scene& i_scene,
  const IntersectionRecord& i_intersection,
  const Ray& i_camera_ray)   {
  const auto& view_direction = i_camera_ray.GetDirection();
  Color result_pixel_color = i_intersection.m_material->GetPrimitiveColor();
  std::size_t red, green, blue, active_lights_cnt;
  red = green = blue = active_lights_cnt = 0;
  Ray to_light(i_intersection.m_intersection + i_intersection.m_normal * 1e-10, Vector3d(0, 1, 0));
  for (auto i = 0u; i < i_scene.GetNumLights(); ++i)     {
    const auto& light = i_scene.GetLight(i);
    if (!light->GetState())
      continue;
    ++active_lights_cnt;
    const auto light_direction = light->GetDirection(i_intersection.m_intersection);
    to_light.SetDirection(-light_direction);
    IntersectionRecord temp_intersection;
    const bool is_intersected = i_scene.TraceRay(temp_intersection, to_light);
    if (!is_intersected || light_direction.Dot(light->GetDirection(temp_intersection.m_intersection)) < 0.0)       {
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
  if (active_lights_cnt > 0)     {
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