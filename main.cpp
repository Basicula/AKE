#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>

#include <BMPWriter.h>
#include <Vector.h>
#include <Ray.h>
#include <IntersectionUtilities.h>
#include <SpotLight.h>
#include <ColorMaterial.h>
#include <TransformationMatrix.h>
#include <SolveEquations.h>

#define mpi 0

std::string GetPath(int i_num)
  {
  std::string path = "D:\\Study\\RayTracing\\ResultsOutputs\\";
  std::string num;
  while (i_num)
    {
    num += '0' + i_num % 10;
    i_num /= 10;
    }
  std::reverse(num.begin(), num.end());
  return path + num + ".bmp";
  }

Color CastRay(const Ray& i_ray, const std::vector<std::unique_ptr<IObject>>& i_objects, const std::vector<std::unique_ptr<SpotLight>>& i_lights, int depth = 3)
  {
  Color res = Color(0, 0, 0);
  double distance = INFINITY;
  Vector3d intersection;
  /*if (IntersectRayWithWave(intersection, ray))
    {
    distances[y][x] = Vector3d(0, 0, -1).Distance(intersection);
    ColorMaterial material2(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.6, 0.6, 0.6), Vector3d(0.75, 0.75, 0.75), 6);
    double sqrt_sum_sqr = sqrt(intersection[0] * intersection[0] + intersection[1] * intersection[1]);
    double cos_sqrt_sum_sqr = cos(sqrt_sum_sqr);
    Vector3d normal(intersection[0] * cos_sqrt_sum_sqr / sqrt_sum_sqr, intersection[1] * cos_sqrt_sum_sqr / sqrt_sum_sqr, 1);
    normal.Normalize();
    res[y][x] = material2.GetResultColor(normal, (light.GetLocation() - intersection).Normalized(), (Vector3d(0, 0, -1) - intersection).Normalized());
    }*/
  for (const auto& object : i_objects)
    {
    double dist_to_obj;
    if (IntersectRayWithObject(intersection, dist_to_obj, i_ray, object.get()))
      {
      if (dist_to_obj > distance) continue;
      distance = dist_to_obj;

      //cast ray again if reflection part > 0
      if (object->GetMaterial().GetReflection() > 0 && depth >= 0)
        {
        Vector3d normal;
        object->GetNormalInPoint(normal, intersection);
        Vector3d reflected_dir = normal * normal.Dot(-i_ray.GetDirection()) * 2 + i_ray.GetDirection();
        res = CastRay(Ray(intersection, reflected_dir), i_objects, i_lights, depth - 1);
        }
      //cast ray inside (-normal) if refraction > 0
      Color refracted_color(0, 0, 0);
      if (object->GetMaterial().GetRefraction() > 0.0 && depth >= 0)
        {
        Vector3d normal;
        object->GetNormalInPoint(normal, intersection);
        double refraction_coef = i_ray.GetEnvironment() / object->GetMaterial().GetRefraction();
        double cos_n_ray = i_ray.GetDirection().Dot(-normal);
        bool is_refracted = sqrt(1 - cos_n_ray * cos_n_ray) < (1.0 / refraction_coef);
        if (is_refracted)
          {
          Vector3d refraction_parallel_part = (i_ray.GetDirection() + normal * cos_n_ray) * refraction_coef;
          Vector3d refraction_perpendicular_part = -normal * sqrt(1 - refraction_parallel_part.SquareLength());
          Vector3d refraction_dir = refraction_parallel_part + refraction_perpendicular_part;
          Ray refraction_ray(intersection, refraction_dir, object->GetMaterial().GetRefraction());
          refracted_color = CastRay(refraction_ray, i_objects, i_lights, depth - 1);
          }
        }

      //find lights that have infuence on intersection point
      std::vector<SpotLight*> active_light;
      active_light.reserve(i_lights.size());
      for (const auto& light : i_lights)
        {
        bool is_active = true;
        Ray to_light(intersection, light->GetLocation() - intersection);
        Vector3d temp_intersection;
        double temp_dist;
        for (const auto& shadow_obj : i_objects)
          if (object.get() != shadow_obj.get() && IntersectRayWithObject(temp_intersection, temp_dist, to_light, shadow_obj.get()))
            {
            if (intersection.SquareDistance(light->GetLocation()) > temp_dist*temp_dist)
              {
              is_active = false;
              break;
              }
            }
        if (is_active)
          active_light.emplace_back(light.get());
        }
      res = object->GetColorInPoint(intersection, active_light, (i_ray.GetStart() - intersection).Normalized()) * (1 - object->GetMaterial().GetReflection()) + res * object->GetMaterial().GetReflection() + refracted_color;
      }
    }
  return res;
  }

Picture TestSphere(int w, int h)
  {
  Picture res = Picture(w, h);
  std::vector<std::unique_ptr<IObject>> objects;
  std::vector<std::unique_ptr<SpotLight>> lights;
  double cx = -20;
  double cy = -20;

  ColorMaterial pure_mirror(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0), Vector3d(1.0, 1.0, 1.0), 1, 1);
  ColorMaterial more_real_mirror(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.75, 0.75, 0.75), Vector3d(1.0, 1.0, 1.0), 1, 0.75);
  ColorMaterial half_mirror(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5), Vector3d(1.0, 1.0, 1.0), 1, 0.5);
  ColorMaterial ruby(Color(255, 0, 0), Vector3d(0.1745, 0.01175, 0.01175), Vector3d(0.61424, 0.04136, 0.04136), Vector3d(0.727811, 0.626959, 0.626959), 76.8);//ruby
  ColorMaterial green_plastic(Color(0, 255, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.1, 0.35, 0.1), Vector3d(0.45, 0.55, 0.45), 32);//green plastic
  ColorMaterial blue_plastic(Color(0, 0, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.1, 0.1, 0.35), Vector3d(0.45, 0.45, 0.55), 32);
  ColorMaterial red_plastic(Color(255, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);//red plastic
  ColorMaterial pure_glass(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0, 1.5);
  ColorMaterial water(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0.25, 1.33);

  for (size_t i = 0; i < 9; ++i)
    if (i != 4) objects.emplace_back(new Sphere(Vector3d(cx + 20 * (i % 3), cy + 20 * (i / 3), 150), 9, pure_mirror));
  objects.emplace_back(new Sphere(Vector3d(-25, -25, 100), 9, pure_glass));
  objects.emplace_back(new Plane(Vector3d(0, -30, 1), Vector3d(1, -30, 0), Vector3d(0, -30, 0), half_mirror));
  objects.emplace_back(new Plane(Vector3d(30, 1, 0), Vector3d(30, 0, 0), Vector3d(30, 0, 1), red_plastic));
  objects.emplace_back(new Plane(Vector3d(-30, 0, 1), Vector3d(-30, 0, 0), Vector3d(-30, 1, 0), green_plastic));
  objects.emplace_back(new Plane(Vector3d(1, 30, 0), Vector3d(0, 30, 1), Vector3d(0, 30, 0), half_mirror));
  objects.emplace_back(new Torus(Vector3d(0, 0, 150), 10, 4, blue_plastic));
  objects.emplace_back(new Cylinder(Vector3d(-25, -25, 100), 5, 40, half_mirror));
  objects.emplace_back(new Sphere(Vector3d(0, -27, 60), 3, ruby));
  lights.emplace_back(new SpotLight(Vector3d(-25, -25, 55), Color(255, 255, 255), 4));
  lights.emplace_back(new SpotLight(Vector3d(-20, -20, 130), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(20, -20, 130), Color(255, 255, 255), 2));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 130), Color(255, 255, 255), 3));
  lights.emplace_back(new SpotLight(Vector3d(-20, 20, 130), Color(255, 255, 255), 4));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 160), Color(255, 255, 255), 5));
  Vector3d eye(0, 0, -1);
  for (int y = 0; y < h; ++y)
    {
    for (int x = 0; x < w; ++x)
      {
      Ray ray(eye, Vector3d(1.*(x - w / 2) / w, 1.*(y - h / 2) / h, 1));
      res[y][x] = CastRay(ray, objects, lights);
      }
    }
  return res;
  }


void LabMandelbrot()
  {
  const size_t width = 600, height = 600;
  auto t1 = std::chrono::system_clock::now();
  auto t2 = std::chrono::system_clock::now();
  Picture mand;
  BMPWriter writer(width, height);
  mand = TestSphere(width, height);
  writer.SetPicture(mand);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\sphere.bmp");
#if mpi == 0
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height, OMP);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
#else
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height, MPI);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
#endif
  writer.SetPicture(mand);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\test.bmp");
  }

int main()
  {
  Vector3d vec(1, 0.5, 0), vec2(132, 1, 0);
  Vector3d norm = vec.CrossProduct(vec2);
  Vector3d res = vec - vec2;
  const size_t width = 1280, height = 720;
  Picture picture;
  BMPWriter writer(width, height);
  auto t1 = std::chrono::system_clock::now();
  picture = TestSphere(width, height);
  auto t2 = std::chrono::system_clock::now();
  std::cout << "Frame rendering takes " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
  writer.SetPicture(picture);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\sphere.bmp");
  return 0;
  }