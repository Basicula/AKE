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

#define mpi 0

Color CastRay(const Ray& i_ray, const std::vector<std::unique_ptr<IObject>>& i_objects, const std::vector<std::unique_ptr<SpotLight>>& i_lights, int depth = 5)
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
  for (size_t i = 0; i < i_objects.size(); ++i)
    if (IntersectRayWithObject(intersection, i_ray, i_objects[i].get()))
      {
      double dist_to_obj = i_ray.GetStart().Distance(intersection);
      if (dist_to_obj > distance) continue;
      distance = dist_to_obj;

      //cast ray again if reflection part > 0
      if (i_objects[i]->GetMaterial().GetReflection() > 0 && depth>=0)
        {
        Vector3d normal;
        i_objects[i]->GetNormalInPoint(normal,intersection);
        Vector3d reflected_dir = normal * normal.Dot(-i_ray.GetDirection()) * 2 + i_ray.GetDirection();
        res = CastRay(Ray(intersection+reflected_dir,reflected_dir),i_objects,i_lights,depth-1);
        }
      //cast ray inside (-normal) if refraction > 0

      //find lights that have infuence on intersection point
      std::vector<SpotLight*> active_light;
      for (const auto& light : i_lights)
        {
        bool is_active = true;
        Ray to_light(intersection, light->GetLocation() - intersection);
        Vector3d temp_intersection;
        for (size_t j = 0; j < i_objects.size(); ++j)
          if (i != j && IntersectRayWithObject(temp_intersection, to_light, i_objects[j].get()))
            {
            if (intersection.Distance(light->GetLocation()) > intersection.Distance(temp_intersection))
              {
              is_active = false;
              break;
              }
            }
        if (is_active)
          active_light.emplace_back(light.get());
        }
      res = i_objects[i]->GetColorInPoint(intersection, active_light, (i_ray.GetStart() - intersection).Normalized()) * (1-i_objects[i]->GetMaterial().GetReflection()) + res * i_objects[i]->GetMaterial().GetReflection();
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
  ColorMaterial blue_plastic(Color(0,0,255),Vector3d(0.0,0.0,0.0),Vector3d(0.1,0.1,0.35),Vector3d(0.45,0.45,0.55),32);
  ColorMaterial red_plastic(Color(255,0,0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);//red plastic

  for (size_t i = 0; i < 9; ++i)
    if(i!=4) objects.emplace_back(new Sphere(Vector3d(cx + 20 * (i % 3), cy + 20 * (i / 3), 200), 9, pure_mirror));
  objects.emplace_back(new Sphere(Vector3d(0, 0, 200), 9, ruby));
  objects.emplace_back(new Plane(Vector3d(0, -30, 1), Vector3d(1, -30, 0), Vector3d(0, -30, 0), half_mirror));
  objects.emplace_back(new Plane(Vector3d(30, 1, 0), Vector3d(30, 0, 0), Vector3d(30, 0, 1), red_plastic));
  objects.emplace_back(new Plane(Vector3d(-30, 0, 1), Vector3d(-30, 0, 0), Vector3d(-30, 1, 0), green_plastic));
  objects.emplace_back(new Plane(Vector3d(1, 30, 0), Vector3d(0, 30, 1), Vector3d(0, 30, 0), blue_plastic));
  lights.emplace_back(new SpotLight(Vector3d(-20, -20, 190), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(20, -20, 190), Color(255, 255, 255), 2));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 190), Color(255, 255, 255), 3));
  lights.emplace_back(new SpotLight(Vector3d(-20, 20, 190), Color(255, 255, 255), 4));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 210), Color(255, 255, 255), 5));
  Vector3d eye(0, 0, -1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      {
      Ray ray(eye, Vector3d(1.*(x - w / 2) / w, 1.*(y - h / 2) / h, 1) - eye);
      res[y][x] = CastRay(ray, objects, lights);
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