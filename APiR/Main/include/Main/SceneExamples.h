#pragma once
#include <Rendering/Scene.h>

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