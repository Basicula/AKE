#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>

#include <BMPWriter.h>
#include <Vector.h>
#include <Plane.h>
#include <Ray.h>
#include <Sphere.h>
#include <IntersectionUtilities.h>
#include <SpotLight.h>

#define mpi 0

const double g_max_distance = 200;

Picture TestSphere(int w, int h)
  {
  Picture res = Picture(w, h);
  std::vector<std::vector<double>> distances(h, std::vector<double>(w, INFINITY));
  std::vector<std::unique_ptr<IObject>> objects;
  double cx = -20;
  double cy = -20;
  for (size_t i = 0; i < 9; ++i)
    objects.emplace_back(new Sphere(Vector3d(cx + 20 * (i % 3), cy + 20 * (i / 3), 200), 9));
  objects.emplace_back(new Plane(Vector3d(0, -30, 1), Vector3d(1, -30, 0), Vector3d(0, -30, 0)));
  SpotLight light(Vector3d(-20, -10, 200));
  //Sphere sphere(Vector3d(0, 0, 10), 10);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      {
      Ray ray(Vector3d(0, 0, -1), Vector3d(1.*x / w - 0.5, 1.*y / h - 0.5, 1) - Vector3d(0, 0, -1));
      Vector3d intersection;
      for (size_t i = 0; i < objects.size(); ++i)
        if (IntersectRayWithObject(intersection, ray, objects[i].get()))
          {
          double distance = Vector3d(0, 0, -1).Distance(intersection);
          if (distance < distances[y][x])
            {
            distances[y][x] = distance;
            Vector3d to_ligth = (light.GetLocation() - intersection).Normalized();
            Vector3d normal;
            objects[i]->GetNormalInPoint(normal,intersection);
            double col = std::max(0.0, to_ligth.Dot(normal)) * 255;
            res[y][x] = Color(col, 0, 0);
            }
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
  const size_t width = 800, height = 600;
  Picture picture;
  BMPWriter writer(width, height);
  picture = TestSphere(width, height);
  writer.SetPicture(picture);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\sphere.bmp");
  return 0;
  }