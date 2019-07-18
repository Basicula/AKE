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
#include <IObject.h>
#include <Sphere.h>
#include <Torus.h>
#include <Plane.h>
#include <Cylinder.h>
#include <GLUTWindow.h>
#include <CL/cl.hpp>

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
  for (const auto& object : i_objects)
    {
    double dist_to_obj;
    if (object->IntersectWithRay(intersection, dist_to_obj, i_ray))
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
          if (object.get() != shadow_obj.get() && shadow_obj->IntersectWithRay(temp_intersection, temp_dist, to_light))
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
  ColorMaterial test(Color(0,255,0),Vector3d(0,0,0),Vector3d(0.5,0.5,0.5),Vector3d(0.5,0.5,0.5),1);

  for (size_t i = 0; i < 9; ++i)
    if (i != 4) objects.emplace_back(new Sphere(Vector3d(cx + 20 * (i % 3), cy + 20 * (i / 3), 150), 9, pure_mirror));
  objects.emplace_back(new Sphere(Vector3d(-10, 10, 100), 5, pure_glass));
  objects.emplace_back(new Sphere(Vector3d(23, -23, 140), 7, test));
  objects.emplace_back(new Plane(Vector3d(0, -30, 1), Vector3d(1, -30, 0), Vector3d(0, -30, 0), half_mirror));
  objects.emplace_back(new Plane(Vector3d(30, 1, 0), Vector3d(30, 0, 0), Vector3d(30, 0, 1), red_plastic));
  objects.emplace_back(new Plane(Vector3d(-30, 0, 1), Vector3d(-30, 0, 0), Vector3d(-30, 1, 0), green_plastic));
  objects.emplace_back(new Plane(Vector3d(1, 30, 0), Vector3d(0, 30, 1), Vector3d(0, 30, 0), half_mirror));
  objects.emplace_back(new Torus(Vector3d(0, 0, 150), 10, 4, blue_plastic));
  objects.emplace_back(new Cylinder(Vector3d(-25, -25, 100), 5, 40, half_mirror));
  objects.emplace_back(new Cylinder(Vector3d(-27, -20, 100), 1, -1, half_mirror));
  lights.emplace_back(new SpotLight(Vector3d(-25, -25, 55), Color(255, 255, 255), 2));
  lights.emplace_back(new SpotLight(Vector3d(-20, -20, 120), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(20, -20, 120), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 120), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(-20, 20, 120), Color(255, 255, 255), 1));
  lights.emplace_back(new SpotLight(Vector3d(20, 20, 160), Color(255, 255, 255), 2));
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

void OpenCLTest()
  {
  // get all platforms(drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    exit(1);
    }
  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  //get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
    }
  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";


  cl::Context context({ default_device });

  cl::Program::Sources sources;

  // kernel calculates for each element C=A+B
  std::string kernel_code =
    "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
    "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
    "   }                                                                               ";
  sources.push_back({ kernel_code.c_str(),kernel_code.length() });

  cl::Program program(context, sources);
  if (program.build({ default_device }) != CL_SUCCESS) {
    std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
    exit(1);
    }


  // create buffers on the device
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

  int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);

  //write arrays A and B to the device
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);


  //run the kernel
  //cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
  //simple_add(buffer_A, buffer_B, buffer_C);

  //alternative way to run the kernel
  cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
  kernel_add.setArg(0,buffer_A);
  kernel_add.setArg(1,buffer_B);
  kernel_add.setArg(2,buffer_C);
  queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
  queue.finish();

  int C[10];
  //read result C from the device to array C
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

  std::cout << " result: \n";
  for (int i = 0; i < 10; i++) {
    std::cout << C[i] << " ";
    }
  }

int main()
  {
  OpenCLTest();
  GLUTWindow window(800,600,"Test");
  window.Open();
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