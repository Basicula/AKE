#include <Rendering/CPURayTracer.h>
#include <Rendering/RenderableObject.h>

#include <Common/ThreadPool.h>

CPURayTracer::CPURayTracer() 
  : m_active_camera(nullptr){
  }

void CPURayTracer::Render() {
#if true
  ThreadPool::GetInstance()->ParallelFor(
    static_cast<std::size_t>(0),
    mp_frame_image->GetSize(),
    [&](std::size_t i_pixel_id)
    {
    auto x = i_pixel_id % mp_frame_image->GetWidth();
    auto y = i_pixel_id / mp_frame_image->GetWidth();
    //if (x == mp_frame_image->GetWidth() / 2 && y == mp_frame_image->GetHeight() / 2)
    mp_frame_image->SetPixel(x, y, _TraceRay(i_pixel_id));
    }
  );
#else
  for (auto y = 0; y < mp_frame_image->GetHeight(); ++y) {
    for (auto x = 0; x < mp_frame_image->GetWidth(); ++x) {
      const auto pixel_id = y * mp_frame_image->GetWidth() + x;
      //if (!(x == mp_frame_image->GetWidth() / 2 - 20 && y == mp_frame_image->GetHeight() / 2 - 20))
      //  continue;
      mp_frame_image->SetPixel(x, y, _TraceRay(pixel_id));
      }
  }
#endif
  }

void CPURayTracer::_OutputImageWasSet() {
  m_rays.resize(mp_frame_image->GetSize(), Ray(Vector3d(0), Vector3d(0)));
  }

void CPURayTracer::_SceneWasSet() {
  m_rays.clear();
  m_active_camera = &mp_scene->GetActiveCamera();

  const auto& ray_origin = m_active_camera->GetLocation();
  for (auto y = 0.0; y < mp_frame_image->GetHeight(); ++y)
    for (auto x = 0.0; x < mp_frame_image->GetWidth(); ++x) {
      const auto& ray_dir = m_active_camera->GetDirection(
        x / mp_frame_image->GetWidth(),
        y / mp_frame_image->GetHeight());
      m_rays.emplace_back(ray_origin, ray_dir);
      }
  }

Color CPURayTracer::_TraceRay(std::size_t i_ray_id)   {
  const auto& camera_ray = m_rays[i_ray_id];

  double distance;
  auto* p_intersected_object = mp_scene->TraceRay(distance, camera_ray);

  if (!p_intersected_object)
    return mp_scene->GetBackGroundColor();

  return _ProcessIntersection(p_intersected_object, distance, camera_ray);
  }

Color CPURayTracer::_ProcessIntersection(
  const Object* ip_intersected_object,
  const double i_distance,
  const Ray& i_camera_ray)   {
  Color reflected_color;
  if (ip_intersected_object->VisualRepresentation()->IsReflectable())
    reflected_color = _ProcessReflection(ip_intersected_object, i_distance, i_camera_ray);

  Color refracted_color;
  if (ip_intersected_object->VisualRepresentation()->IsRefractable())
    refracted_color = _ProcessRefraction(ip_intersected_object, i_distance, i_camera_ray);

  Color light_influence = _ProcessLightInfluence(ip_intersected_object, i_distance, i_camera_ray);
  return light_influence + reflected_color;
  }

Color CPURayTracer::_ProcessReflection(
  const Object* ip_intersected_object,
  const double i_distance,
  const Ray& i_camera_ray)   {
  std::size_t depth = 0;
  Ray local_ray(i_camera_ray);
  auto* p_intersected_object = ip_intersected_object;
  auto distance = i_distance;
  while (p_intersected_object->VisualRepresentation()->IsReflectable() && depth < m_depth) {
    const auto intersection_point = local_ray.GetPoint(distance);
    const auto normal = p_intersected_object->GetNormalAtPoint(intersection_point);
    const auto reflected_direction =
      p_intersected_object->VisualRepresentation()->ReflectedDirection(normal, local_ray.GetDirection());
    local_ray.SetOrigin(intersection_point);
    local_ray.SetDirection(reflected_direction);
    p_intersected_object = mp_scene->TraceRay(distance, local_ray);
    if (p_intersected_object)
      ++depth;
    else
      return mp_scene->GetBackGroundColor();
    }
  return _ProcessLightInfluence(p_intersected_object, distance, local_ray) * ip_intersected_object->VisualRepresentation()->ReflectionInfluence();
  }

Color CPURayTracer::_ProcessRefraction(
  const Object* /*ip_intersected_object*/,
  const double /*i_distance*/,
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
  const Object* ip_intersected_object,
  const double i_distance,
  const Ray& i_camera_ray)   {
  const auto& view_direction = i_camera_ray.GetDirection();
  Color result_pixel_color = ip_intersected_object->VisualRepresentation()->GetPrimitiveColor();
  std::size_t red, green, blue, active_lights_cnt;
  red = green = blue = active_lights_cnt = 0;
  const auto intersection_point = i_camera_ray.GetPoint(i_distance);
  const auto normal = ip_intersected_object->GetNormalAtPoint(intersection_point);
  Ray to_light(intersection_point + normal * 1e-6, Vector3d(0, 1, 0));
  for (auto i = 0u; i < mp_scene->GetNumLights(); ++i)     {
    const auto& light = mp_scene->GetLight(i);
    if (!light->GetState())
      continue;
    ++active_lights_cnt;
    const auto light_direction = light->GetDirection(intersection_point);
    to_light.SetDirection(-light_direction);
    double distance;
    auto* p_intersected_object = mp_scene->TraceRay(distance, to_light);
    const auto local_intersection = to_light.GetPoint(distance);
    if (!p_intersected_object || light_direction.Dot(light->GetDirection(local_intersection)) < 0.0) {
      Color light_influence =
        ip_intersected_object->VisualRepresentation()->CalculateColor(
          normal,
          view_direction,
          light_direction) * light->GetIntensityAtPoint(intersection_point);
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
    result_pixel_color = Color(
      static_cast<uint8_t>(red > 255 ? 255 : red),
      static_cast<uint8_t>(green > 255 ? 255 : green),
      static_cast<uint8_t>(blue > 255 ? 255 : blue));
    }
  return result_pixel_color;
  }