#include "Main/SceneExamples.h"

#include "Rendering/SimpleCamera.h"

namespace ExampleMaterials {
  PhongMaterial* pure_mirror()
  {
    return new PhongMaterial(
      Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0), Vector3d(1.0, 1.0, 1.0), 1, 1);
  }
  PhongMaterial* more_real_mirror()
  {
    return new PhongMaterial(
      Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.05, 0.05, 0.05), Vector3d(0.1, 0.1, 0.1), 50, 0.75);
  }
  PhongMaterial* half_mirror()
  {
    return new PhongMaterial(
      Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5), Vector3d(1.0, 1.0, 1.0), 1, 0.5);
  }

  PhongMaterial* ruby()
  {
    return new PhongMaterial(Color(255, 0, 0),
                             Vector3d(0.1745, 0.01175, 0.01175),
                             Vector3d(0.61424, 0.04136, 0.04136),
                             Vector3d(0.727811, 0.626959, 0.626959),
                             76.8);
  }

  PhongMaterial* green_plastic()
  {
    return new PhongMaterial(
      Color(0, 255, 0), Vector3d(0.0, 0.05, 0.0), Vector3d(0.1, 0.35, 0.1), Vector3d(0.45, 0.55, 0.45), 32);
  }
  PhongMaterial* blue_plastic()
  {
    return new PhongMaterial(
      Color(0, 0, 255), Vector3d(0.0, 0.0, 0.05), Vector3d(0.1, 0.1, 0.35), Vector3d(0.45, 0.45, 0.55), 32);
  }
  PhongMaterial* red_plastic()
  {
    return new PhongMaterial(
      Color(255, 0, 0), Vector3d(0.05, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  }
  PhongMaterial* yellow_plastic()
  {
    return new PhongMaterial(
      Color(255, 255, 0), Vector3d(0.05, 0.05, 0.0), Vector3d(0.5, 0.5, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  }

  PhongMaterial* pure_glass()
  {
    return new PhongMaterial(
      Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0, 1.5);
  }

  PhongMaterial* water()
  {
    return new PhongMaterial(
      Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0.25, 1.33);
  }
}

namespace ExampleScene {
  Scene OneSphere()
  {
    Scene scene("One sphere");

    scene.AddObject(new RenderableObject(new Sphere(Vector3d(0, 0, 0), 0.5), ExampleMaterials::ruby()));

    scene.AddLight(new SpotLight(Vector3d(0, 3, 0)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -1), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene NineSpheres()
  {
    Scene scene("Nine spheres");

    double cx = -7;
    double cy = -7;
    for (size_t i = 0; i < 9; ++i)
      scene.AddObject(
        new RenderableObject(new Sphere(Vector3d(cx + 7 * (i % 3), cy + 7 * (i / 3), 5), 2), ExampleMaterials::ruby()));

    scene.AddLight(new SpotLight(Vector3d(9, 9, -5)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -15), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 90, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene RandomSpheres(const size_t i_count)
  {
    Scene scene("Random spheres");

    for (size_t i = 0; i < i_count; ++i)
      scene.AddObject(new RenderableObject(
        new Sphere(Vector3d(rand() % 100 - 50, rand() % 100 - 50, rand() % 100 - 50), 2), ExampleMaterials::ruby()));

    scene.AddLight(new SpotLight(Vector3d(9, 9, -5)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -15), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 90, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene EmptyRoom()
  {
    Scene scene("Empty room");

    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, 0, 10), Vector3d(0, 0, -1)), ExampleMaterials::blue_plastic()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), ExampleMaterials::red_plastic()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), ExampleMaterials::green_plastic()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), ExampleMaterials::yellow_plastic()));

    scene.AddLight(new SpotLight(Vector3d(0, 0, 0)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -5), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene OnePlane()
  {
    Scene scene("Plane");

    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), ExampleMaterials::yellow_plastic()));

    scene.AddLight(new SpotLight(Vector3d(0, 0, 0)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -5), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene OneCylinder()
  {
    Scene scene("One cylinder");

    const auto cylinder = new Cylinder(Vector3d(0, 0, 0), 0.5, 5);
    scene.AddObject(new RenderableObject(cylinder, ExampleMaterials::ruby()));

    scene.AddLight(new SpotLight(Vector3d(0, 3, -2.4)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -3), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    return scene;
  }

  Scene OneTorus()
  {
    Scene scene("Torus");

    auto torus = new Torus(Vector3d(0, 0, 0), 1, 0.5);
    scene.AddObject(new RenderableObject(torus, ExampleMaterials::blue_plastic()));

    scene.AddLight(new SpotLight(Vector3d(0, 3, -3)));

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -5), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    return std::move(scene);
  }

  Scene ComplexScene()
  {
    Scene scene("Complex scene");

    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, 0, 10), Vector3d(0, 0, -1)), ExampleMaterials::blue_plastic()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), ExampleMaterials::pure_mirror()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), ExampleMaterials::green_plastic()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), ExampleMaterials::yellow_plastic()));

    auto torus = new Torus(Vector3d(0, 0, 0), 1, 0.5);
    scene.AddObject(new RenderableObject(torus, ExampleMaterials::blue_plastic()));

    const auto cylinder = new Cylinder(Vector3d(0, -2, 0), 0.5, 5);
    scene.AddObject(new RenderableObject(cylinder, ExampleMaterials::ruby()));

    // scene.AddCamera(new SimpleCamera(Vector3d(-5, 5, -5), Vector3d(0, 0, 0), Vector3d(1 / SQRT_3, 1 / SQRT_3, 1 /
    // SQRT_3), 75, 16.0 / 9.0, 0.5), true);

    scene.AddCamera(new SimpleCamera(Vector3d(0, 0, -5), Vector3d(0, 0, 1), Vector3d(0, 1, 0), 75, 16.0 / 9.0), true);

    scene.AddLight(new SpotLight(Vector3d(3, 3, -3), Color::White, 0.75));

    return std::move(scene);
  }

  Scene InfinityMirror()
  {
    Scene scene("Complex scene");

    scene.AddObject(new RenderableObject(new Sphere(Vector3d(2, 0, 0), 0.5), ExampleMaterials::ruby()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, -3, 0), Vector3d(0, 1, 0)), ExampleMaterials::blue_plastic()));

    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, 0, 2), Vector3d(0, 0, -1)), ExampleMaterials::more_real_mirror()));
    scene.AddObject(
      new RenderableObject(new Plane(Vector3d(0, 0, -2), Vector3d(0, 0, 1)), ExampleMaterials::more_real_mirror()));

    scene.AddLight(new SpotLight(Vector3d(0, 3, 0), Color::White, 0.25));

    scene.AddCamera(
      new SimpleCamera(
        Vector3d(-1, 1, -1), Vector3d(1, -1, 1), Vector3d(1 / SQRT_3, 1 / SQRT_3, 1 / SQRT_3), 75, 16.0 / 9.0),
      true);

    return scene;
  }
}