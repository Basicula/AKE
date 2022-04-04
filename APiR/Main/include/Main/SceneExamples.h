#pragma once
#include "Geometry/Cylinder.h"
#include "Geometry/Sphere.h"
#include "Geometry/Torus.h"
#include "Math/Constants.h"
#include "Rendering/RenderableObject.h"
#include "Rendering/Scene.h"
#include "Visual/PhongMaterial.h"
#include "Visual/SpotLight.h"

namespace ExampleScene {
  Scene OneSphere();
  Scene OnePlane();
  Scene OneCylinder();
  Scene OneTorus();

  Scene NineSpheres();
  Scene RandomSpheres(const size_t i_count);
  Scene EmptyRoom();
  Scene ComplexScene();
  Scene InfinityMirror();
}