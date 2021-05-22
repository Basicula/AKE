#include <Main/SceneExamples.h>

namespace ExampleMaterials {
  auto pure_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.0, 1.0, 1.0), Vector3d(1.0, 1.0, 1.0), 1, 1);
  auto more_real_mirror = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.75, 0.75, 0.75), Vector3d(1.0, 1.0, 1.0), 1, 0.75);
  auto half_mirror = std::make_shared<ColorMaterial>(Color(0, 0, 0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, 0.5, 0.5), Vector3d(1.0, 1.0, 1.0), 1, 0.5);

  auto ruby = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.1745, 0.01175, 0.01175), Vector3d(0.61424, 0.04136, 0.04136), Vector3d(0.727811, 0.626959, 0.626959), 76.8);

  auto green_plastic = std::make_shared<ColorMaterial>(Color(0, 255, 0), Vector3d(0.0, 0.05, 0.0), Vector3d(0.1, 0.35, 0.1), Vector3d(0.45, 0.55, 0.45), 32);
  auto blue_plastic = std::make_shared<ColorMaterial>(Color(0, 0, 255), Vector3d(0.0, 0.0, 0.05), Vector3d(0.1, 0.1, 0.35), Vector3d(0.45, 0.45, 0.55), 32);
  auto red_plastic = std::make_shared<ColorMaterial>(Color(255, 0, 0), Vector3d(0.05, 0.0, 0.0), Vector3d(0.5, 0.0, 0.0), Vector3d(0.7, 0.6, 0.6), 32);
  auto yellow_plastic = std::make_shared<ColorMaterial>(Color(255, 255, 0), Vector3d(0.05, 0.05, 0.0), Vector3d(0.5, 0.5, 0.0), Vector3d(0.7, 0.6, 0.6), 32);

  auto pure_glass = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0, 1.5);

  auto water = std::make_shared<ColorMaterial>(Color(255, 255, 255), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), 1, 0.25, 1.33);
  }

namespace ExampleScene {
  Scene OneSphere() {
    Scene scene("One sphere");

    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Sphere>(Vector3d(0, 0, 0), 0.5), ExampleMaterials::ruby));

    scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 3, 0)));

    scene.AddCamera(Camera(Vector3d(0, 0, -1), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 75, 16.0 / 9.0, 0.5), true);

    return std::move(scene);
    }

  Scene NineSpheres() {
    Scene scene("Nine spheres");

    double cx = -7;
    double cy = -7;
    for (size_t i = 0; i < 9; ++i)
      scene.AddObject(
        std::make_shared<RenderableObject>(
          std::make_shared<Sphere>(
            Vector3d(
              cx + 7 * (i % 3),
              cy + 7 * (i / 3),
              5),
            2),
          ExampleMaterials::ruby));

    scene.AddLight(std::make_shared<SpotLight>(Vector3d(9, 9, -5)));

    scene.AddCamera(Camera(Vector3d(0, 0, -5), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 75, 16.0 / 9.0, 0.5), true);

    return std::move(scene);
    }

  Scene EmptyRoom() {
    Scene scene("Empty room");

    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, 0, 10), Vector3d(0, 0, -1)), ExampleMaterials::blue_plastic));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), ExampleMaterials::red_plastic));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), ExampleMaterials::green_plastic));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), ExampleMaterials::yellow_plastic));

    scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 0, 0)));

    scene.AddCamera(Camera(Vector3d(0, 0, -5), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 75, 16.0 / 9.0, 0.5), true);

    return std::move(scene);
    }

  Scene RotatableTorus() {
    Scene scene("Torus");

    auto torus = std::make_shared<Torus>(Vector3d(0, 0, 0), 1, 0.5);
    scene.AddObject(std::make_shared<RenderableObject>(torus, ExampleMaterials::blue_plastic));

    scene.AddLight(std::make_shared<SpotLight>(Vector3d(0, 3, -3)));

    scene.AddCamera(Camera(Vector3d(0, 0, -5), Vector3d(0, 0, 0), Vector3d(0, 1, 0), 75, 16.0 / 9.0, 0.5), true);

    return std::move(scene);
    }

  Scene ComplexScene() {
    Scene scene("Complex scene");

    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, 0, 10), Vector3d(0, 0, -1)), ExampleMaterials::blue_plastic));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(10, 0, 0), Vector3d(-1, 0, 0)), ExampleMaterials::pure_mirror));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(-10, 0, 0), Vector3d(1, 0, 0)), ExampleMaterials::green_plastic));
    scene.AddObject(std::make_shared<RenderableObject>(std::make_shared<Plane>(Vector3d(0, -10, 0), Vector3d(0, 1, 0)), ExampleMaterials::yellow_plastic));

    auto torus = std::make_shared<Torus>(Vector3d(0, 0, 0), 1, 0.5);
    scene.AddObject(std::make_shared<RenderableObject>(torus, ExampleMaterials::blue_plastic));

    const auto cylinder = std::make_shared<Cylinder>(Vector3d(0, -2, 0), 0.5, 5);
    scene.AddObject(std::make_shared<RenderableObject>(cylinder, ExampleMaterials::ruby));

    scene.AddCamera(Camera(Vector3d(-5, 5, -5), Vector3d(0, 0, 0), Vector3d(1 / SQRT_3, 1 / SQRT_3, 1 / SQRT_3), 75, 16.0 / 9.0, 0.5), true);

    scene.AddLight(std::make_shared<SpotLight>(Vector3d(3, 3, -3)));

    return std::move(scene);
    }
  }